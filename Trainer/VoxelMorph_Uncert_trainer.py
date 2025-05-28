from Trainer.Trainer_base import Trainer
from utils.loss import Uncert_Loss

import torch

from utils.utils import apply_deformation_using_disp
from networks.VecInt import VecInt

import wandb
from datetime import datetime

class VoxelMorph_Uncert_Trainer(Trainer):
    def __init__(self, args):
        assert args.reg in ['tv', 'atv']
        assert args.method in ['VM-Un', 'VM-Un-diff']
        # setting log name first!
        self.log_name = f'{args.method}_({args.reg}_{args.image_sigma}_{args.prior_lambda})'
        # add start time
        now = datetime.now().strftime("%m-%d_%H-%M")
        self.log_name = f'{self.log_name}_{now}'
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

        self.loss_fn = Uncert_Loss(args.reg, args.image_sigma, args.prior_lambda)
        self.reset_logs()

        if 'diff' in args.method:
            self.integrate = VecInt(inshape=(160, 192, 160), nsteps=7)


        super().__init__(args, config)

    def forward(self, img, template, stacked_input, val=False):
        mean_list, std_list, _, _ = self.model(stacked_input)
        mean = mean_list[-1] # use only last one
        std = std_list[-1]

        # sample in Gaussian distribution
        eps_r = torch.randn_like(mean)
        sampled_disp = mean + eps_r * std

        if self.method == 'VM-Un':
            deformed_img = apply_deformation_using_disp(img, sampled_disp)
        elif self.method == 'VM-Un-diff':
            # velocity field to deformation field
            accumulate_disp = self.integrate(sampled_disp)
            deformed_img = apply_deformation_using_disp(img, accumulate_disp)
        
        loss, sim_loss, smoo_loss = self.loss_fn(deformed_img, template, mean, std)

        self.log_dict['Loss_tot'] += loss.item()
        self.log_dict['Loss_sim'] += sim_loss
        self.log_dict['Loss_reg'] += smoo_loss
        
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
            wandb.log({f"{tag}/{key}": value / num}, step=epoch)
        
        wandb.log({"Epoch": epoch}, step=epoch)

    def reset_logs(self):
        # for single layer, deterministic version (VM)
        self.log_dict = {
            'Loss_tot':0.0,
            'Loss_sim':0.0,
            'Loss_reg':0.0
        }
