from utils.loss_utils import *

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
        if reg == 'tv':
            self.reg_fn = tv_loss_l2
        elif reg == 'atv':
            self.reg_fn = adaptive_tv_loss_l2

        self.image_sigma = image_sigma
        self.prior_lambda = prior_lambda
        self.degree_matrices = dict()

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

        return recon_loss + kl_loss, recon_loss.item(), kl_loss.item()


if __name__ == "__main__":
    UL = Uncert_Loss('tv', 0.02, 10.0)
    mean, std = torch.rand((1, 3, 36, 38, 40)), torch.rand((1, 3, 36, 38, 40))
    UL.kl_loss(mean, std)
