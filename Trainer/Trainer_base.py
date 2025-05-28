# import torch, os
import torch.optim as optim

# from utils.utils import set_seed
# from utils.loss import TrainLoss
from utils.dataset import set_dataloader
# from utils.model import R2Net_UNet

# logging - wandb
import wandb, os, torch
wandb.login(key="87539aeaa75ad2d8a28ec87d70e5d6ce1277c544")

import matplotlib.pyplot as plt
from networks.U_Net import U_Net

from utils.utils import set_seed, save_middle_slices
from tqdm import tqdm

class Trainer:
    def __init__(self, args, config=None):
        set_seed(seed=args.seed)
        # os.makedirs(f'saved_images/{self.log_name}', exist_ok=True)
        wandb.init(
            project=args.wandb_name,
            name=self.log_name,
            config=config
        )
        
        # Setting Model
        self.model = U_Net(out_channels=self.out_channels, out_layers=self.out_layers)
        if args.start_epoch>0:
            self.model.load_state_dict(torch.load(args.saved_path, weights_only=True))
        self.model = self.model.cuda()

        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)
        self.train_loader, self.val_loader, self.save_loader = set_dataloader(args.data_path, args.template_path, args.batch_size)

        # train options
        self.epochs = args.epochs
        self.start_epoch = args.start_epoch
        self.val_interval = args.val_interval
        self.save_interval = args.save_interval
        self.save_num = args.save_num
        # self.val_detail = args.val_detail

    def train(self):
        best_loss = 1e+9
        cnt = 0
        # save template img
        for epoch in tqdm(range(self.epochs), position=0, desc='Epoch', leave=True):
            if epoch < self.start_epoch:
                continue
            self.train_1_epoch(epoch)
            if epoch % self.val_interval == 0:
                with torch.no_grad():
                    self.valid(epoch)
                    cur_loss = self.log_dict['Loss_tot']
                    if best_loss > cur_loss:
                        cnt = 0
                        best_loss = cur_loss
                        torch.save(self.model.state_dict(), f'./saved_models/{self.log_name}.pt')
                    else: 
                        cnt+=1

                    if cnt >= 3:
                        # early stop
                        break
            if epoch % self.save_interval == 0:
                with torch.no_grad():
                    self.save_imgs(epoch, self.save_num)

        wandb.finish()

    def train_1_epoch(self, epoch):
        self.reset_logs()
        self.model.train()
        for (img, template, _, _, _) in tqdm(self.train_loader, desc=f"Train", position=1, leave=False):
            img, template = img.unsqueeze(1).cuda(), template.unsqueeze(1).cuda() # [B, D, H, W] -> [B, 1, D, H, W]
            stacked_input = torch.cat([img, template], dim=1) # [B, 2, D, H, W]

            # forward & calculate loss in child trainer
            loss, _ = self.forward(img, template, stacked_input)

            # backward & update model
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # log into wandb
        self.log(epoch, phase='train')

    def valid(self, epoch):
        self.reset_logs()
        self.model.eval()
        for (img, template, _, _, _) in tqdm(self.val_loader, desc=f"Valid", position=1, leave=False):
            img, template = img.unsqueeze(1).cuda(), template.unsqueeze(1).cuda() # [B, D, H, W] -> [B, 1, D, H, W]
            stacked_input = torch.cat([img, template], dim=1) # [B, 2, D, H, W]

            # forward & calculate loss in child trainer
            _, _ = self.forward(img, template, stacked_input, val=True)

        self.log(epoch, phase='valid')

    def save_imgs(self, epoch, num):
        self.model.eval()
        for idx, (img, template, _, _, _) in enumerate(self.save_loader):
            if idx >= num:
                break
            img, template = img.unsqueeze(1).cuda(), template.unsqueeze(1).cuda() # [B, D, H, W] -> [B, 1, D, H, W]
            stacked_input = torch.cat([img, template], dim=1) # [B, 2, D, H, W]

            # forward & calculate loss in child trainer
            _, deformed_img = self.forward(img, template, stacked_input, val=True)

            fig = save_middle_slices(deformed_img, epoch, idx)
            wandb.log({f"deformed_slices_img{idx}": wandb.Image(fig)}, step=epoch)
            plt.close(fig)

            if epoch == 0 and idx == 0:
                fig = save_middle_slices(template, epoch, idx)
                wandb.log({f"Template": wandb.Image(fig)}, step=epoch)
                plt.close(fig)