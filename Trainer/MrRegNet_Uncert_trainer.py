from Trainer.Trainer_base import Trainer
from utils.loss import Uncert_Loss

import matplotlib.pyplot as plt
from utils.utils import save_middle_slices, save_middle_slices_mfm, apply_deformation_using_disp, save_grid_spline, print_with_timestamp
from networks.VecInt import VecInt
from datetime import datetime

import torch
import torch.nn.functional as F

class MrRegNet_Uncert_Trainer(Trainer):
    def __init__(self, args):
        assert args.method in ['Mr-Un', 'Mr-Un-diff']
        # setting log name first!

        # Add: only kl options
        self.only_kl = args.only_kl
        if self.only_kl:
            prefix = 'resscakl'
        else:
            prefix = 'ressca'

        if args.reg is None:
            self.log_name = f'{args.method}-{prefix}_{args.loss}'
        else:
            self.log_name = f'{args.method}-{prefix}_{args.loss}({args.reg}_{args.alpha}_{args.sca_fn}_{args.alp_sca})'
        self.method = args.method

        config={
            "method": args.method,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "loss":args.loss,
            "reg": args.reg,
            "image_sigma": args.image_sigma,
            "prior_lambda": args.prior_lambda
        }

        self.args = args
        self.out_channels = 6
        self.out_layers = 3

        self.loss_fn = Uncert_Loss(args.loss, args.reg, args.image_sigma, args.prior_lambda)
        self.reset_logs()

        if 'diff' in args.method:
            self.integrate = VecInt(inshape=(160, 192, 160), nsteps=7, multi=True)

        super().__init__(args, config)

    def forward(self, img, template, stacked_input, epoch=0, val=False, return_uncert=False):
        mean_list, std_list, res_mean_list, res_std_list = self.model(stacked_input)
        if val:
            mean_list = mean_list[-1:]
            std_list = std_list[-1:]
            res_mean_list = res_mean_list[-1:]
            res_std_list = res_std_list[-1:]
        
        tot_loss = torch.tensor(0.0).to(img.device)
        # iteration accross resolution level
        for i, (mean, std, res_mean, res_std) in enumerate(zip(mean_list, std_list, res_mean_list, res_std_list)):
            cur_img = F.interpolate(img, size=mean.shape[2:], mode='nearest')
            cur_template = F.interpolate(template, size=mean.shape[2:], mode='nearest') 

            if val == False:
                eps_r = torch.randn_like(mean)
                sampled_disp = mean + eps_r * std
            else:
                sampled_disp = mean
            
            if self.method == 'Mr-Un':
                deformed_img = apply_deformation_using_disp(cur_img, sampled_disp)
                self.disp_field = sampled_disp
            elif self.method == 'Mr-Un-diff':
                # velocity field to deformation field
                accumulate_disp = self.integrate(sampled_disp)
                deformed_img = apply_deformation_using_disp(cur_img, accumulate_disp)
                self.disp_field = accumulate_disp

            # loss, sim_loss, smoo_loss = self.loss_fn(deformed_img, cur_template, out)
            if i < 2:
                loss, sim_loss, smoo_loss, sigma_loss, sigma_var = self.loss_fn(deformed_img, cur_template, res_mean, res_std, only_kl=self.only_kl)
            else:
                # in highest layer, use all loss (recon + kl)
                loss, sim_loss, smoo_loss, sigma_loss, sigma_var = self.loss_fn(deformed_img, cur_template, res_mean, res_std, only_kl=False)

            tot_loss += loss

            self.log_dict['Loss_tot'] += loss.item()
            self.log_dict['Std_mean'] += sigma_loss
            self.log_dict['Std_var'] += sigma_var
            self.log_dict['Loss_sim'] += sim_loss
            self.log_dict['Loss_reg'] += smoo_loss

            self.log_dict[f'Std_mean/res{i+1}'] += sigma_loss
            self.log_dict[f'Std_var/res{i+1}'] += sigma_var
            self.log_dict[f'Loss_sim/res{i+1}'] += sim_loss
            self.log_dict[f'Loss_reg/res{i+1}'] += smoo_loss
        
        if return_uncert:
            return tot_loss, deformed_img, std
        
        return tot_loss, deformed_img

    def reset_logs(self):
        # for multi-resolution layer, deterministic version (Mr)
        self.log_dict = {
            'Loss_tot':0.0,
            'Loss_sim':0.0,
            'Loss_reg':0.0,
            'Std_mean':0.0,
            'Std_var':0.0,
            'Loss_sim/res1':0.0,
            'Loss_sim/res2':0.0,
            'Loss_sim/res3':0.0,
            'Loss_reg/res1':0.0,
            'Loss_reg/res2':0.0,
            'Loss_reg/res3':0.0,
            'Std_mean/res1':0.0,
            'Std_mean/res2':0.0,
            'Std_mean/res3':0.0,
            'Std_var/res1':0.0,
            'Std_var/res2':0.0,
            'Std_var/res3':0.0
        }

    def get_disp(self):
        return self.disp_field
    
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