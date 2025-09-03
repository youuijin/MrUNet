from Trainer.Trainer_base import Trainer
from utils.loss import MultiSampleEachLoss

import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt
from utils.utils import save_middle_slices, save_middle_slices_mfm, apply_deformation_using_disp
from networks.VecInt import VecInt

class MrRegNet_Semantic_Free_Aware_Each_Trainer(Trainer):
    def __init__(self, args):
        assert args.reg in ['none', 'atv-const', 'atv-linear', 'wsmooth']
        assert args.method in ['Mr-SFAeach', 'Mr-SFAeach-diff']
        # setting log name first!
        args.sig = 'logL1'
        args.beta = 1e-2
        self.log_name = f'{args.method}_{args.loss}({args.reg}-detach_{args.alpha}_sca{args.alp_sca}_{args.sig}_{args.beta}_N{args.num_samples})'
        self.method = args.method

        config={
            "method": args.method,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "loss": args.loss,
            "reg": args.reg,
            "alpha": args.alpha
        }

        self.args = args
        self.out_channels = 6
        self.out_layers = 3

        self.loss_fn = MultiSampleEachLoss(args.loss, args.reg, args.alpha, args.p, args.sig, args.beta, alpha_scale=args.alp_sca) #TODO: add sig_fn term into argparser
        self.N = args.num_samples
        self.reset_logs()

        if 'diff' in args.method:
            self.integrate = VecInt(inshape=(160, 192, 160), nsteps=7)

        super().__init__(args, config)

    def forward(self, img, template, stacked_input, epoch, val=False, return_uncert=False):
        mean_list, std_list, res_mean_list, res_std_list = self.model(stacked_input)
        if val:
            mean_list = mean_list[-1:]
            std_list = std_list[-1:]
            res_mean_list = res_mean_list[-1:]
            res_std_list = res_std_list[-1:]

        tot_loss = torch.tensor(0.0).to(img.device)
        for i, (mean, std, res_mean, res_std) in enumerate(zip(mean_list, std_list, res_mean_list, res_std_list)):
            cur_img = F.interpolate(img, size=mean.shape[2:], mode='nearest')
            cur_template = F.interpolate(template, size=mean.shape[2:], mode='nearest') 

            if val == False:
                # sample in Gaussian distribution x N
                disps = []
                for _ in range(self.N):
                    eps_r = torch.randn_like(mean)
                    sampled_disp = mean + eps_r * std
                    disps.append(sampled_disp)
            else:
                disps = [mean]
            
            deformed_imgs, smoothed_imgs = [], []
            for sampled_disp in disps:
                if 'diff' not in self.method:
                    deformed_img = apply_deformation_using_disp(cur_img, sampled_disp)
                elif 'diff' in self.method:
                    # velocity field to deformation field
                    accumulate_disp = self.integrate(sampled_disp)
                    deformed_img = apply_deformation_using_disp(cur_img, accumulate_disp)
                deformed_imgs.append(deformed_img)

                if self.loss_fn.reg == 'wsmooth':
                    smoothed_disp = self.loss_fn.build_phi_smooth(sampled_disp.detach(), std.detach()) #TODO: res_std or std
                    smoothed_img = apply_deformation_using_disp(cur_img, smoothed_disp)
                    smoothed_imgs.append(smoothed_img)

            loss, sim_loss, reg_loss, sig_loss, std_mean, std_var = self.loss_fn(deformed_imgs, smoothed_imgs, cur_template, mean, res_std, epoch) # TODO: check res_std or std
            # print("RES", i)
            # print(loss)
            # print(sim_loss)
            # print(reg_loss)
            # print(sig_loss)
            # print(std_mean)
            # print(std_var)
            tot_loss += loss

            if not torch.isfinite(loss):
                print(f"[NaN detected] epoch={epoch}, "
                    f"loss={loss.item()}, ncc_loss={sim_loss}, sig_loss={sig_loss.item()}")
                # 필요시 sim_loss
                raise RuntimeError("Stopping training due to NaN/Inf in loss")


            self.log_dict['Loss_tot'] += loss.item()
            self.log_dict['Loss_sim'] += sim_loss
            self.log_dict['Loss_reg'] += reg_loss
            self.log_dict['Loss_sig'] += sig_loss
            self.log_dict['Std_mean'] += std_mean
            self.log_dict['Std_var'] += std_var

            self.log_dict[f'Loss_sim/res{i+1}'] += sim_loss
            self.log_dict[f'Loss_reg/res{i+1}'] += reg_loss
            self.log_dict[f'Loss_sig/res{i+1}'] += sig_loss
            self.log_dict[f'Std_mean/res{i+1}'] += std_mean
            self.log_dict[f'Std_var/res{i+1}'] += std_var

        # print()
        if return_uncert:
            return loss, deformed_img, std
        
       
        
        return loss, deformed_img

    def log(self, epoch, phase=None):
        if phase not in ['train', 'valid']:
            raise ValueError("Trainer's log function can only get phase ['train', 'valid'], but received", phase)

        if phase == 'train':
            num = len(self.train_loader)
            tag = 'Train'
        elif phase == 'valid':
            num = len(self.val_loader)
            tag = 'Val'
        
        for key, value in self.log_dict.items():
            self.writer.add_scalar(f"{tag}/{key}", value/num, epoch)

    def reset_logs(self):
        # for single layer, deterministic version (VM)
        self.log_dict = {
            'Loss_tot':0.0,
            'Loss_sim':0.0,
            'Loss_reg':0.0,
            'Loss_sig':0.0,
            'Std_mean':0.0,
            'Std_var':0.0,
            'Loss_sim/res1':0.0,
            'Loss_reg/res1':0.0,
            'Loss_sig/res1':0.0,
            'Std_mean/res1':0.0,
            'Std_var/res1':0.0,
            'Loss_sim/res2':0.0,
            'Loss_reg/res2':0.0,
            'Loss_sig/res2':0.0,
            'Std_mean/res2':0.0,
            'Std_var/res2':0.0,
            'Loss_sim/res3':0.0,
            'Loss_reg/res3':0.0,
            'Loss_sig/res3':0.0,
            'Std_mean/res3':0.0,
            'Std_var/res3':0.0,
        }

    def save_imgs(self, epoch, num):
        self.model.eval()
        for idx, (img, template, _, _, _) in enumerate(self.save_loader):
            if idx >= num:
                break
            img, template = img.unsqueeze(1).cuda(), template.unsqueeze(1).cuda() # [B, D, H, W] -> [B, 1, D, H, W]
            stacked_input = torch.cat([img, template], dim=1) # [B, 2, D, H, W]

            # forward & calculate loss in child trainer
            _, deformed_img, std = self.forward(img, template, stacked_input, epoch, val=True, return_uncert=True)

            std_magnitude = torch.norm(std, dim=1)
            fig = save_middle_slices(std_magnitude, epoch, idx)
            # wandb.log({f"std_img{idx}": wandb.Image(fig)}, step=epoch)
            self.writer.add_figure(f'std_img{idx}', fig, epoch)
            plt.close(fig)

            if self.pair_train:
                fig = save_middle_slices_mfm(img, template, deformed_img, epoch, idx)
                # wandb.log({f"deformed_slices_img{idx}": wandb.Image(fig)}, step=epoch)
                self.writer.add_figure(f'deformed_slices_img{idx}', fig, epoch)
                plt.close(fig)
            else:
                fig = save_middle_slices(deformed_img, epoch, idx)
                # wandb.log({f"deformed_slices_img{idx}": wandb.Image(fig)}, step=epoch)
                self.writer.add_figure(f'deformed_slices_img{idx}', fig, epoch)
                plt.close(fig)

                if epoch == 0 and idx == 0:
                    fig = save_middle_slices(template, epoch, idx)
                    # wandb.log({f"Template": wandb.Image(fig)}, step=epoch)
                    self.writer.add_figure(f'Template', fig, epoch)
                    plt.close(fig)

