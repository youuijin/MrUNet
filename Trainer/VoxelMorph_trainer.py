from Trainer.Trainer_base import Trainer
from utils.loss import Train_Loss

from utils.utils import apply_deformation_using_disp
from networks.VecInt import VecInt
import wandb
from datetime import datetime

class VoxelMorph_Trainer(Trainer):
    def __init__(self, args):
        assert args.method in ['VM', 'VM-diff']
        # setting log name first!
        if args.reg is None:
            self.log_name = f'{args.method}_{args.loss}(tv_0.0)'
        else:
            self.log_name = f'{args.method}_{args.loss}({args.reg}_{args.alpha})'
        # add start time
        now = datetime.now().strftime("%m-%d_%H-%M")
        self.log_name = f'{self.log_name}_{now}'
        self.method = args.method

        config={
            "method": args.method,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "loss":args.loss,
            "reg": args.reg,
            "alpha": args.alpha,
        }

        self.args = args
        self.out_channels = 3
        self.out_layers = 1

        self.loss_fn = Train_Loss(args.loss, args.reg, args.alpha)
        self.reset_logs()

        if 'diff' in args.method:
            self.integrate = VecInt(inshape=(160, 192, 160), nsteps=7)

        super().__init__(args, config)

    def forward(self, img, template, stacked_input, val=False):
        out_list, _ = self.model(stacked_input)
        out = out_list[-1] # use only last one
        if self.method == 'VM':
            deformed_img = apply_deformation_using_disp(img, out)
        elif self.method == 'VM-diff':
            # velocity field to deformation field
            accumulate_disp = self.integrate(out)
            deformed_img = apply_deformation_using_disp(img, accumulate_disp)
        
        loss, sim_loss, smoo_loss = self.loss_fn(deformed_img, template, out)

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
