from Trainer.Trainer_base import Trainer
from utils.loss import Train_Loss

from utils.utils import apply_deformation_using_disp
from networks.VecInt import VecInt
import wandb
from datetime import datetime

import torch.nn as nn
import torch
import torch.nn.functional as F


class MrRegNet_Trainer(Trainer):
    def __init__(self, args):
        assert args.method in ['Mr', 'Mr-diff']
        # setting log name first!
        if args.reg is None:
            self.log_name = f'{args.method}_{args.loss}(tv_0.0)'
        else:
            self.log_name = f'{args.method}_{args.loss}({args.reg}_{args.alpha}_sca{args.alp_sca})'
        
        self.method = args.method

        config={
            "method": args.method,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "loss":args.loss,
            "reg": args.reg,
            "alpha": args.alpha,
            "alp_sca": args.alp_sca,
            "sca_fn": args.sca_fn
        }

        self.args = args
        self.out_channels = 3
        self.out_layers = 3

        self.loss_fn = Train_Loss(args.loss, args.reg, args.alpha, alpha_scale=args.alp_sca, scale_fn=args.sca_fn)
        self.reset_logs()

        if 'diff' in args.method:
            self.integrate = VecInt(inshape=(160, 192, 160), nsteps=7, multi=True)

        super().__init__(args, config)

    def forward(self, img, template, stacked_input, val=False):
        out_list, _ = self.model(stacked_input)
        if val:
            out_list = out_list[-1:]
        
        tot_loss = torch.tensor(0.0).to(img.device)
        # iteration accross resolution level
        for i, out in enumerate(out_list):
            cur_img = F.interpolate(img, size=out.shape[2:], mode='nearest')
            cur_template = F.interpolate(template, size=out.shape[2:], mode='nearest') 
            if self.method == 'Mr':
                deformed_img = apply_deformation_using_disp(cur_img, out)
            elif self.method == 'Mr-diff':
                # velocity field to deformation field
                accumulate_disp = self.integrate(out)
                deformed_img = apply_deformation_using_disp(cur_img, accumulate_disp)
            
            loss, sim_loss, smoo_loss = self.loss_fn(deformed_img, cur_template, out)

            tot_loss += loss

            self.log_dict['Loss_tot'] += loss.item()
            self.log_dict['Loss_sim'] += sim_loss
            self.log_dict['Loss_reg'] += smoo_loss

            self.log_dict[f'Loss_sim/res{i+1}'] += sim_loss
            self.log_dict[f'Loss_reg/res{i+1}'] += smoo_loss
        
        return tot_loss, deformed_img

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
            wandb.log({f"{tag}/{key}": value / num}, step=epoch)
        
        wandb.log({"Epoch": epoch}, step=epoch)

    def reset_logs(self):
        # for multi-resolution layer, deterministic version (Mr)
        self.log_dict = {
            'Loss_tot':0.0,
            'Loss_sim':0.0,
            'Loss_reg':0.0,
            'Loss_sim/res1':0.0,
            'Loss_sim/res2':0.0,
            'Loss_sim/res3':0.0,
            'Loss_reg/res1':0.0,
            'Loss_reg/res2':0.0,
            'Loss_reg/res3':0.0
        }
