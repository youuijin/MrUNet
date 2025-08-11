from utils.loss_utils import *
import torch.nn.functional as F

class Train_Loss:
    def __init__(self, loss, reg, alpha, alpha_scale=1.0, scale_fn='exp'):
        if loss == 'MSE':
            self.loss_fn_sim = MSE_loss
        elif loss == 'NCC':
            self.loss_fn_sim = NCC_loss
        
        if reg is None:
            self.loss_fn_reg, self.reg_alpha = [], []
        else:
            assert len(reg.split("_")) == len(alpha.split("_"))
            self.loss_fn_reg, self.reg_alpha = [], [float(a) for a in alpha.split("_")]
            for r in reg.split('_'):
                if r == "tv":
                    loss_fn = tv_loss
                elif r == "l2":
                    loss_fn = l2_loss
                elif r == "jac":
                    loss_fn = jac_det_loss
                self.loss_fn_reg += [loss_fn]
        
        # scale factor for multi-resolution method
        self.alpha_scale = alpha_scale
        self.scale_fn = scale_fn

    def __call__(self, moved, fixed, disp, idx=0):
        sim_loss = self.loss_fn_sim(moved, fixed)
        reg_loss = torch.tensor(0.0, device=moved.device)
        for loss_fn, alpha in zip(self.loss_fn_reg, self.reg_alpha):
            # exponential
            if self.scale_fn == 'exp':
                alpha = (self.alpha_scale ** idx) * alpha
            # linear
            elif self.scale_fn == 'linear':
                alpha = alpha*(self.alpha_scale-1)/2*idx+alpha

            cur_loss = loss_fn(disp)
            reg_loss += alpha * cur_loss

        tot_loss = sim_loss + reg_loss
        return tot_loss, sim_loss.item(), reg_loss.item()
    
class Uncert_Loss:
    def __init__(self, reg, image_sigma, prior_lambda):
        # TODO: Select smoothness loss
        # if reg == 'tv':
        #     self.reg_fn = tv_loss_l2
        # elif reg == 'atv':
        #     self.reg_fn = adaptive_tv_loss_l2
        self.reg_fn = self.smoothness_laplacian
        # TODO: Select smoothness loss

        self.image_sigma = image_sigma
        self.prior_lambda = prior_lambda
        self.degree_matrices = dict()

    def smoothness_laplacian(self, mu):
        """
        Compute μ^T Λ μ = 0.5 * λ * sum_i sum_j∈N(i) (μ[i] - μ[j])^2
        mu: [B, 3, D, H, W]
        """
        loss = torch.tensor(0.0, device=mu.device)

        for dim, shift in enumerate([(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]):
            # shift mu for neighbor
            shifted_mu = torch.roll(mu, shifts=shift, dims=(2,3,4))
            
            # Mask to remove wrap-around effect
            mask = torch.ones_like(mu)
            if shift[0] == 1:
                mask[:, :, 0, :, :] = 0
            elif shift[0] == -1:
                mask[:, :, -1, :, :] = 0
            if shift[1] == 1:
                mask[:, :, :, 0, :] = 0
            elif shift[1] == -1:
                mask[:, :, :, -1, :] = 0
            if shift[2] == 1:
                mask[:, :, :, :, 0] = 0
            elif shift[2] == -1:
                mask[:, :, :, :, -1] = 0

            diff = (mu - shifted_mu) ** 2 * mask
            loss += diff.sum()

        return self.prior_lambda * 0.5 * loss / mu.numel()  # mean over batch

    def _adj_filt_3d(self):
        """
        Build 3D adjacency filter for computing degree matrix.
        Output: Tensor of shape [3, 1, 3, 3, 3] for grouped conv
        """
        filt_inner = torch.zeros(3, 3, 3)
        filt_inner[1, 1, [0, 2]] = 1  # x neighbors
        filt_inner[1, [0, 2], 1] = 1  # y neighbors
        filt_inner[[0, 2], 1, 1] = 1  # z neighbors

        filt = torch.zeros(3, 1, 3, 3, 3)
        for i in range(3):
            filt[i, 0] = filt_inner
        return filt 
    
    def _compute_degree_matrix(self, device, vol_shape):
        key = tuple(vol_shape)
        if key not in self.degree_matrices:
            filt = self._adj_filt_3d().to(device)
            z = torch.ones(1, 3, *vol_shape, device=device)
            D = F.conv3d(z, filt, padding=1, stride=1, groups=3)
            self.degree_matrices[key] = D
        return self.degree_matrices[key]

    def kl_loss(self, mean, std):
        """
        mean, std: [B, 3, D, H, W]
        """
        self.degree_matrix = self._compute_degree_matrix(mean.device, mean.shape[2:])

        sigma_term = self.prior_lambda * self.degree_matrix * (std**2) - torch.log(std**2)
        sigma_term = torch.mean(sigma_term)

        # prec_term = self.prior_lambda * self._precision_loss(mu)
        prec_term = self.prior_lambda * 0.5 * self.reg_fn(mean)

        return 0.5 * (sigma_term + prec_term)

    def recon_loss(self, y_true, y_pred):
        """
        MSE reconstruction loss.
        """
        return 1. / (2* self.image_sigma ** 2) * torch.mean((y_true - y_pred) ** 2)

    def __call__(self, warped, fixed, mean, std):
        """
        fixed: ground truth image [B, 1, D, H, W]
        warped: warped image [B, 1, D, H, W]
        mean, std: output of model [B, 3, D, H, W]
        """
        recon_loss = self.recon_loss(fixed, warped)
        kl_loss = self.kl_loss(mean, std)

        sigma_term = torch.mean(torch.log(std**2))
        sigma_var = torch.var(torch.log(std**2)) # for logging

        return recon_loss + kl_loss, recon_loss.item(), kl_loss.item(), sigma_term.item(), sigma_var.item()
    
class Aleatoric_Uncert_Loss:
    def __init__(self, reg, prior_lambda):
        if reg == 'tv':
            self.reg_fn = tv_loss_l2
        elif reg == 'atv':
            self.reg_fn = adaptive_tv_loss_l2

        self.prior_lambda = prior_lambda
        self.degree_matrices = dict()


    def kl_loss(self, mean, std):
        """
        mean, std: [B, 3, D, H, W]
        kl_loss = log(σ^2) + λ * tv
        """
        # self.degree_matrix = self._compute_degree_matrix(mean.device, mean.shape[2:])

        # sigma_term = self.prior_lambda * self.degree_matrix * (std**2) - torch.log(std**2)
        # sigma_term = torch.mean(sigma_term)
        sigma_term = torch.mean(torch.log(std**2))
        sigma_var = torch.var(torch.log(std**2)) # for logging

        # prec_term = self.prior_lambda * self._precision_loss(mu)
        smooth_term = self.reg_fn(mean, std)

        return sigma_term, smooth_term, sigma_var

    def recon_loss(self, y_true, y_pred, std, eps=1e-6):
        """
        MSE reconstruction loss.
        """
        return torch.mean(1. / (std ** 2 + 1e-6) * ((y_true - y_pred) ** 2))

    def __call__(self, warped, fixed, mean, std):
        """
        fixed: ground truth image [B, 1, D, H, W]
        warped: warped image [B, 1, D, H, W]
        mean, std: output of model [B, 3, D, H, W]
        """
        std = torch.clamp(std, min=1e-3) # add clamp to 안정화
        recon_loss = self.recon_loss(fixed, warped, std)
        sigma_loss, smooth_loss, sigma_var = self.kl_loss(mean, std)

        return recon_loss + sigma_loss + self.prior_lambda * smooth_loss, recon_loss.item(), sigma_loss.item(), smooth_loss.item(), sigma_var.item()

class MultiSampleLoss:
    def __init__(self, loss, reg, alpha):
        if loss == 'MSE':
            self.loss_fn_sim = MSE_loss
        elif loss == 'NCC':
            self.loss_fn_sim = NCC_loss

        self.reg = reg
        self.alpha = float(alpha)
        self.gamma = 1e-3

    def __call__(self, warped_imgs, fixed, mean, std):
        """
        fixed: ground truth image [B, 1, D, H, W]
        warped_imgs: [warped image [B, 1, D, H, W]] x N
        mean, std: output of model [B, 3, D, H, W]
        """
        if len(warped_imgs) == 1:
            warped_mean = warped_imgs[0]
            warped_var = torch.zeros_like(warped_mean)
        else:
            stacked = torch.stack(warped_imgs, dim=0)
            warped_mean = torch.mean(stacked, dim=0)
            warped_var = torch.var(stacked, dim=0, unbiased=False)
        
        sim_loss = self.loss_fn_sim(fixed, warped_mean)
        var_loss = -1 * torch.mean(warped_var)

        # KL Loss (prior penalty)
        kl_loss = torch.mean(0.5 * (mean**2 + std**2 - torch.log(std**2 + 1e-6) - 1))

        std_mean = torch.mean(torch.log(std+1e-6))
        std_var = torch.var(torch.log(std+1e-6)) # for logging

        return sim_loss + self.alpha * var_loss + self.gamma * kl_loss, sim_loss.item(), var_loss.item(), kl_loss.item(), std_mean.item(), std_var.item()
    
class MultiSampleEachLoss:
    def __init__(self, loss, reg, alpha, p, sig, beta=1e-3):
        assert loss in ['MSE', 'NCC']
        assert reg in ['none', 'atv-const', 'atv-linear']
        assert sig in ['L1', 'L1tv', 'logL1', 'logL1tv']

        if loss == 'MSE':
            self.loss_fn_sim = MSE_loss
        elif loss == 'NCC':
            self.loss_fn_sim = NCC_loss

        self.alpha = float(alpha)
        self.p_max = float(p)
        self.beta = beta

        # if tv, atv-const + p=0
        self.reg = reg
        self.reg_fn = weighted_tv_loss_l2
        if self.reg == 'none':
            self.reg_fn = self.none_fn
        
        self.sig = sig
        if sig == 'L1' or sig == 'logL1':
            self.sig_fn = l1_loss
        elif sig == 'L1tv' or sig == 'logL1tv':
            self.sig_fn = tv_loss

    def none_fn(self, mean, std, p):
        return torch.tensor(0.,).cuda()

    def __call__(self, warped_imgs, fixed, mean, std, epoch):
        """
        fixed: ground truth image [B, 1, D, H, W]
        warped_imgs: [warped image [B, 1, D, H, W]] x N
        mean, std: output of model [B, 3, D, H, W]
        """
        # stacked = torch.stack(warped_imgs, dim=0)

        if self.reg == 'atv-const':
            p = self.p_max
        elif self.reg == 'atv-linear':
            p = self.p_max * epoch / 400
        elif self.reg == 'none':
            p = 0.0
        
        sim_loss = torch.tensor(0.0).cuda()
        for w in warped_imgs:
            sim_loss += self.loss_fn_sim(fixed, w)
        sim_loss /= len(warped_imgs)
        reg_loss = self.reg_fn(mean, std, p)
        if 'log' in self.sig:
            log_std = torch.log(std+1e-6)
            sig_loss = self.sig_fn(log_std)
        else:
            sig_loss = self.sig_fn(std)

        std_mean = torch.mean(torch.log(std+1e-6))
        std_var = torch.var(torch.log(std+1e-6)) # for logging

        return sim_loss + self.alpha * reg_loss + self.beta * sig_loss, sim_loss.item(), reg_loss.item(), sig_loss.item(), std_mean.item(), std_var.item()

