from Trainer.Trainer_base import Trainer
from utils.loss import Uncert_Loss

import torch

import matplotlib.pyplot as plt
from utils.utils import save_middle_slices, save_middle_slices_mfm, apply_deformation_using_disp, save_grid_spline, print_with_timestamp
from networks.VecInt import VecInt

class VoxelMorph_Uncert_Trainer(Trainer):
    def __init__(self, args):
        assert args.reg in ['tv', 'atv']
        assert args.method in ['VM-Un', 'VM-Un-diff']
        # setting log name first!
        self.log_name = f'{args.method}_{args.loss}({args.reg}_{args.image_sigma}_{args.prior_lambda})'
        self.num_samples = args.num_samples
        if self.num_samples>1:
            self.log_name = f'{args.method}_{args.loss}({args.reg}_{args.image_sigma}_{args.prior_lambda}_N{self.num_samples})'
        self.method = args.method

        config={
            "method": args.method,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "reg": args.reg,
            "image_sigma": args.image_sigma,
            "prior_lambda": args.prior_lambda,
        }

        self.args = args
        self.out_channels = 6
        self.out_layers = 1

        self.loss_fn = Uncert_Loss(args.loss, args.reg, args.image_sigma, args.prior_lambda)
        self.reset_logs()

        if 'diff' in args.method:
            self.integrate = VecInt(inshape=(160, 192, 160), nsteps=7)


        super().__init__(args, config)

    def forward(self, img, template, stacked_input, epoch=0, val=False, return_uncert=False):
        mean_list, std_list, _, _ = self.model(stacked_input)
        mean = mean_list[-1] # use only last one
        std = std_list[-1]

        tot_loss = None
        log_losses = [0, 0, 0, 0, 0]
        for _ in range(self.num_samples):
            if val==False:
                # sample in Gaussian distribution
                eps_r = torch.randn_like(mean)
                sampled_disp = mean + eps_r * std
            else:
                sampled_disp = mean

            if self.method == 'VM-Un':
                deformed_img = apply_deformation_using_disp(img, sampled_disp)
                self.disp_field = sampled_disp
            elif self.method == 'VM-Un-diff':
                # velocity field to deformation field
                accumulate_disp = self.integrate(sampled_disp)
                deformed_img = apply_deformation_using_disp(img, accumulate_disp)
                self.disp_field = accumulate_disp

            loss, sim_loss, smoo_loss, sigma_loss, sigma_var = self.loss_fn(deformed_img, template, mean, std)

            if tot_loss is None:
                tot_loss = loss
            else:
                tot_loss += loss

            log_losses[0] += loss.item()
            log_losses[1] += sigma_loss
            log_losses[2] += sigma_var
            log_losses[3] += sim_loss
            log_losses[4] += smoo_loss

        tot_loss /= self.num_samples
        self.log_dict['Loss_tot'] += log_losses[0]/self.num_samples
        self.log_dict['Std_mean'] += log_losses[1]/self.num_samples
        self.log_dict['Std_var'] += log_losses[2]/self.num_samples
        self.log_dict['Loss_sim'] += log_losses[3]/self.num_samples
        self.log_dict['Loss_reg'] += log_losses[4]/self.num_samples

        if return_uncert:
            return tot_loss, deformed_img, std
        
        return tot_loss, deformed_img
    
    def get_disp(self):
        return self.disp_field
            
    def reset_logs(self):
        # for single layer, deterministic version (VM)
        self.log_dict = {
            'Loss_tot':0.0,
            'Loss_sim':0.0,
            'Loss_reg':0.0,
            'Std_mean':0.0,
            'Std_var':0.0
        }

    def save_imgs(self, epoch, num):
        self.model.eval()
        for idx, (img, template, _, _, _) in enumerate(self.save_loader):
            if idx >= num:
                break
            img, template = img.unsqueeze(1).cuda(), template.unsqueeze(1).cuda() # [B, D, H, W] -> [B, 1, D, H, W]
            stacked_input = torch.cat([img, template], dim=1) # [B, 2, D, H, W]

            # forward & calculate loss in child trainer
            _, deformed_img, std = self.forward(img, template, stacked_input, val=True, return_uncert=True)
            disp = self.get_disp()

            std_magnitude = torch.norm(std, dim=1)
            fig = save_middle_slices(std_magnitude, epoch, idx)
            self.log_single_img(f'std_img{idx}', fig, epoch)
            plt.close(fig)

            if self.pair_train:
                fig = save_middle_slices_mfm(img, template, deformed_img, epoch, idx)
                self.log_single_img(f'deformed_slices_img{idx}', fig, epoch)
                plt.close(fig)
            else:
                fig = save_middle_slices(deformed_img, epoch, idx)
                self.log_single_img(f'deformed_slices_img{idx}', fig, epoch)
                plt.close(fig)

                if epoch == 0 and idx == 0:
                    fig = save_middle_slices(template, epoch, idx)
                    self.log_single_img(f'Template', fig, epoch)
                    plt.close(fig)

            fig = save_grid_spline(disp)
            self.log_single_img(f'disps_img{idx}', fig, epoch)
            plt.close(fig)

        print_with_timestamp(f'Epoch {epoch}: Successfully saved {num} images')