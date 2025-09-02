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

    def kl_loss(self, mean, std, eps=1e-6):
        """
        mean, std: [B, 3, D, H, W]
        """
        self.degree_matrix = self._compute_degree_matrix(mean.device, mean.shape[2:])

        sigma_term = self.prior_lambda * self.degree_matrix * (std**2) - torch.log(std**2 + eps)
        sigma_term = torch.mean(sigma_term)

        # prec_term = self.prior_lambda * self._precision_loss(mu)
        prec_term = self.prior_lambda * 0.5 * self.reg_fn(mean)

        return 0.5 * (sigma_term + prec_term)

    def recon_loss(self, y_true, y_pred):
        """
        MSE reconstruction loss.
        """
        return 1. / (2* self.image_sigma ** 2) * torch.mean((y_true - y_pred) ** 2)

    def __call__(self, warped, fixed, mean, std, only_kl=False, eps=1e-6):
        """
        fixed: ground truth image [B, 1, D, H, W]
        warped: warped image [B, 1, D, H, W]
        mean, std: output of model [B, 3, D, H, W]
        """
        recon_loss = self.recon_loss(fixed, warped)
        kl_loss = self.kl_loss(mean, std)

        sigma_term = torch.mean(torch.log(std**2 + eps))
        sigma_var = torch.var(torch.log(std**2 + eps)) # for logging

        if only_kl:
            tot_loss = kl_loss
        else:
            tot_loss = recon_loss + kl_loss

        return tot_loss, recon_loss.item(), kl_loss.item(), sigma_term.item(), sigma_var.item()
    
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
    # Now Doing in SFA
    def __init__(self, loss, reg, alpha, p, sig, beta=1e-3, alpha_scale=1.0):
        assert loss in ['MSE', 'NCC']
        assert reg in ['none', 'atv-const', 'atv-linear', 'wsmooth']
        assert sig in ['L1', 'L1tv', 'logL1', 'logL1tv']

        if loss == 'MSE':
            self.loss_fn_sim = MSE_loss
        elif loss == 'NCC':
            self.loss_fn_sim = NCC_loss

        self.alpha = float(alpha)
        self.p_max = float(p)
        self.beta = beta
        self.alpha_scale = 1.0

        # if tv, atv-const + p=0
        self.reg = reg
        if 'atv' in self.reg:
            self.reg_fn = self.weighted_tv_loss_l2
        elif self.reg == 'wsmooth':
            self.reg_fn = self.consistency_loss
        elif self.reg == 'none':
            self.reg_fn = self.none_fn
        
        self.sig = sig
        if sig == 'L1' or sig == 'logL1':
            self.sig_fn = l1_loss
        elif sig == 'L1tv' or sig == 'logL1tv':
            self.sig_fn = tv_loss

    def none_fn(self, mean, std, p):
        return torch.tensor(0.,).cuda()

    def consistency_loss(self, warped_imgs, smoothed_imgs, fixed_img):
        const_loss = torch.tensor(0.0).cuda()
        for w, s in zip(warped_imgs, smoothed_imgs):
            s_det = s.detach()
            # L(I, I')
            const_loss += self.loss_fn_sim(w, s_det)

            # (max(0, L(phi, f)-L(phi', f)))**2
            with torch.no_grad():
                loss_phi_t = self.loss_fn_sim(s_det, fixed_img)
            # loss_phi = self.loss_fn_sim(w, fixed_img)
            # hinge = torch.clamp(loss_phi - loss_phi_t, min=0.0)
            # L_anch = hinge * hinge

            # const_loss += L_anch

        const_loss /= len(warped_imgs)

        return const_loss

    def weighted_tv_loss_l2(self, phi, std, p, eps=1e-6):
        """
        Adaptive total variation loss based on inverse σ².
        phi: [B, 3, D, H, W]
        std: [B, 3, D, H, W]  (std = exp(0.5 * log_sigma))
        """
        dz = (phi[:, :, 1:, :, :] - phi[:, :, :-1, :, :])**2 # (z[i+1] - z[i])**2
        dy = (phi[:, :, :, 1:, :] - phi[:, :, :, :-1, :])**2
        dx = (phi[:, :, :, :, 1:] - phi[:, :, :, :, :-1])**2

        weight_z = torch.norm(sigma_z, p=p, )
        weight_y = 1.0 / (sigma_y**2 + eps)
        weight_x = 1.0 / (sigma_x**2 + eps)

        loss_z = (weight_z * dz).mean()
        loss_y = (weight_y * dy).mean()
        loss_x = (weight_x * dx).mean()

        return (loss_z + loss_y + loss_x) / 3.0

    def conf_from_std(self, std, sigma0=1.0, p=1.0, wmin=1e-3, wmax=10.0):
        # std: (B,1,H,W,D) or (B,3,H,W,D)
        # 네 식: exp(-(std-1)) == exp(-((std/sigma0)**p)) with sigma0=1,p=1 (단순화)
        conf = torch.exp(- (std - sigma0).clamp_min(1e-6).pow(p))
        # return conf.clamp(wmin, wmax)  # 너무 큰/작은 값 컷
        return conf

    def weighted_interp_1d(self, u, conf, dim):
        """
        u    : (B,1,H,W,D)  # 해당 채널만
        conf : (B,1,H,W,D)  # 같은 위치의 신뢰도(가중)
        dim  : 2(H)/3(W)/4(D) 중 하나
        새 u를 반환: (B,1,H,W,D)
        """
        # replicate pad로 양끝 이웃 보정
        pad = [0,0, 0,0, 0,0]  # (D_l, D_r, W_l, W_r, H_l, H_r)
        if dim == 4: pad[0:2] = [1,1]       # D 방향 패딩
        if dim == 3: pad[2:4] = [1,1]       # W 방향 패딩
        if dim == 2: pad[4:6] = [1,1]       # H 방향 패딩
        u_pad    = F.pad(u, pad, mode='replicate')
        conf_pad = F.pad(conf, pad, mode='replicate')

        # 이웃 값/가중 추출 (i-1, i, i+1)
        # pad 했으므로 중앙 슬라이스는 원래 위치
        slicer_c = [slice(None)]*5
        slicer_l = [slice(None)]*5
        slicer_r = [slice(None)]*5
        slicer_c[dim] = slice(1, -1)        # 중앙
        slicer_l[dim] = slice(0, -2)        # 왼쪽(i-1)
        slicer_r[dim] = slice(2,  None)     # 오른쪽(i+1)

        u_c    = u_pad[tuple(slicer_c)]
        u_l    = u_pad[tuple(slicer_l)]
        u_r    = u_pad[tuple(slicer_r)]
        w_c    = conf_pad[tuple(slicer_c)]
        w_l    = conf_pad[tuple(slicer_l)]
        w_r    = conf_pad[tuple(slicer_r)]

        num = w_l * u_l + w_c * u_c + w_r * u_r
        den = w_l + w_c + w_r + 1e-8
        return num / den

    def build_phi_smooth(self, mean, std, sigma0=1.0, p=1.0):
        # 채널별로 "해당 축"에 1D 보간
        # conf는 std의 역함수(크면 덜 믿음)
        conf_all = self.conf_from_std(std, sigma0=sigma0, p=p)  # (B,3,H,W,D)

        # x채널(0): W축(dim=3)만 보간
        ux  = mean[:, 0:1]                  # (B,1,H,W,D)
        cx  = conf_all[:, 0:1]
        ux_s = self.weighted_interp_1d(ux, cx, dim=3)

        # y채널(1): H축(dim=2)
        uy  = mean[:, 1:1+1]
        cy  = conf_all[:, 1:1+1]
        uy_s = self.weighted_interp_1d(uy, cy, dim=2)

        # z채널(2): D축(dim=4)
        uz  = mean[:, 2:3]
        cz  = conf_all[:, 2:3]
        uz_s = self.weighted_interp_1d(uz, cz, dim=4)

        return torch.cat([ux_s, uy_s, uz_s], dim=1)  # (B,3,H,W,D)

    def __call__(self, warped_imgs, smoothed_imgs, fixed, mean, std, epoch, idx=0):
        """
        fixed: ground truth image [B, 1, D, H, W]
        warped_imgs: [warped image [B, 1, D, H, W]] x N
        mean, std: output of model [B, 3, D, H, W]
        idx = layer index, smaller = lower resolution
        """
        # stacked = torch.stack(warped_imgs, dim=0)
        if self.reg == 'atv-const':
            p = self.p_max
        elif self.reg == 'atv-linear':
            p = self.p_max * epoch / 400
        elif self.reg == 'none':
            p = 0.0
        elif self.reg == 'wsmooth':
            p = 1.0

        # mean of similairty
        sim_loss = torch.tensor(0.0).cuda()
        for w in warped_imgs:
            sim_loss += self.loss_fn_sim(fixed, w)
        sim_loss /= len(warped_imgs)

        if self.reg == 'wsmooth':
            reg_loss = self.reg_fn(warped_imgs, smoothed_imgs, fixed)
        else:
            reg_loss = self.reg_fn(mean, std, p)
        
        if 'log' in self.sig:
            log_std = torch.log(std+1e-6)
            sig_loss = self.sig_fn(log_std)/log_std.shape.numel()
        else:
            sig_loss = self.sig_fn(std)/std.shape.numel()

        std_mean = torch.mean(torch.log(std+1e-6))
        std_var = torch.var(torch.log(std+1e-6)) # for logging
        
        return sim_loss + self.alpha * reg_loss + self.beta * sig_loss, sim_loss.item(), reg_loss.item(), sig_loss.item(), std_mean.item(), std_var.item()

