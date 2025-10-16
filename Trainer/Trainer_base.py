import wandb, os, torch, shutil, warnings
import torch.optim as optim
import matplotlib.pyplot as plt
from networks.network_utils import set_model
from utils.dataset import set_dataloader, set_paired_dataloader, set_datapath, set_paired_dataloader_usingcsv, set_dataloader_usingcsv
from utils.utils import set_seed, save_middle_slices, save_middle_slices_mfm, print_with_timestamp, save_grid_spline

from datetime import datetime
from zoneinfo import ZoneInfo

from torch.utils.tensorboard import SummaryWriter

class Trainer:
    def __init__(self, args, config=None):
        set_seed(seed=args.seed)

        # train options
        self.epochs = args.epochs
        self.start_epoch = args.start_epoch
        self.val_interval = args.val_interval
        self.save_interval = args.save_interval
        self.save_num = args.save_num
        self.pair_train = args.pair_train
        self.log_method = args.log_method
        
        if args.lr != 1e-4:
            self.log_name = f'{self.log_name}_lr{args.lr}'

        if args.epochs != 200:
            self.log_name = f'{self.log_name}_epochs{args.epochs}'
        if args.lr_scheduler == 'multistep':
            self.log_name = f'{self.log_name}_sche(multi_{args.lr_milestones})'

        # add start time
        now = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%m-%d_%H-%M")
        if args.data_aug:
            self.log_name = f'{self.log_name}_aug{args.data_aug}_geo{args.data_aug_geo}_{now}'
        else:
            self.log_name = f'{self.log_name}_aug{args.data_aug}_{now}'
            
        config['dataset']=args.dataset
        config['model']=args.model
        config['pair']=args.pair_train

        if self.log_method == 'wandb':
            wandb.init(
                project=args.wandb_name,
                name=self.log_name,
                config=config
            )
        else:
            self.writer = SummaryWriter(log_dir=f'{args.log_dir}/{args.dataset}/{args.model}/pair_{args.pair_train}/{self.log_name}')
        
        # Setting Model
        self.model = set_model(args.model, out_channels=self.out_channels, out_layers=self.out_layers)
        if args.start_epoch>0:
            self.model.load_state_dict(torch.load(args.saved_path, weights_only=True))
        self.model = self.model.cuda()

        self.train_data_path, self.val_data_path = set_datapath(args.dataset, args.numpy)
        # if args.pair_train:
        #     self.train_loader, self.val_loader, self.save_loader = set_paired_dataloader(train_dir=self.train_data_path, val_dir=self.val_data_path, batch_size=args.batch_size, numpy=args.numpy)
        # else:
        #     self.train_loader, self.val_loader, self.save_loader = set_dataloader(self.train_data_path, args.template_path, args.batch_size, numpy=args.numpy)
        if args.pair_train:
            self.train_loader, self.val_loader, self.save_loader = set_paired_dataloader_usingcsv(args.dataset, 'data/data_list', batch_size=args.batch_size, numpy=args.numpy, transform=args.data_aug, geo=args.data_aug_geo)
        else:
            self.train_loader, self.val_loader, self.save_loader = set_dataloader_usingcsv(args.dataset, 'data/data_list', args.template_path, args.batch_size, numpy=args.numpy, transform=args.data_aug, geo=args.data_aug_geo)
        
        if args.pair_train:
            self.save_dir = f'./results/pair/saved_models/{args.dataset}/{args.model}'
        else:
            self.save_dir = f'./results/template/saved_models/{args.dataset}/{args.model}'
        os.makedirs(f'{self.save_dir}/completed', exist_ok=True)
        os.makedirs(f'{self.save_dir}/not_finished', exist_ok=True)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)
        # set learning rate scheduler 
        if args.lr_scheduler == 'none':
            self.lr_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda epoch: 1.0)
        elif args.lr_scheduler == 'multistep':
            milestones = [int(i)*len(self.train_loader) for i in args.lr_milestones.split(',')]
            self.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=milestones, gamma=0.1)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.lr_scheduler.step(args.start_epoch * len(self.train_loader))

    def train(self):
        best_loss = 1e+9
        cnt = 0
        # save template img
        for epoch in range(self.start_epoch, self.epochs):
            self.train_1_epoch(epoch)
            if epoch % self.val_interval == 0:
                with torch.no_grad():
                    self.valid(epoch)
                    cur_loss = self.log_dict['Loss_tot']
                    if best_loss > cur_loss:
                        cnt = 0
                        best_loss = cur_loss
                        torch.save(self.model.state_dict(), f'{self.save_dir}/not_finished/{self.log_name}_best.pt')
                    else: 
                        cnt+=1

                    # if cnt >= 3:
                    #     # early stop
                    #     break
                torch.save(self.model.state_dict(), f'{self.save_dir}/not_finished/{self.log_name}_last.pt')
            if epoch % self.save_interval == 0:
                with torch.no_grad():
                    self.save_imgs(epoch, self.save_num)

        # move trained model to complete folder 
        try:
            shutil.move(f'{self.save_dir}/not_finished/{self.log_name}_best.pt', f'{self.save_dir}/completed/{self.log_name}_best.pt')
            shutil.move(f'{self.save_dir}/not_finished/{self.log_name}_last.pt', f'{self.save_dir}/completed/{self.log_name}_last.pt')
        except Exception as e:
            print_with_timestamp(f"Failed to move {self.save_dir}/not_finished/{self.log_name}.pt: {e}")

        # wandb.finish()

    def train_1_epoch(self, epoch):
        self.reset_logs()
        self.model.train()
        tot_loss = 0.
        for (img, template, _, _, _) in self.train_loader:
            img, template = img.unsqueeze(1).cuda(), template.unsqueeze(1).cuda() # [B, D, H, W] -> [B, 1, D, H, W]

            stacked_input = torch.cat([img, template], dim=1) # [B, 2, D, H, W]

            # forward & calculate loss in child trainer
            loss, _ = self.forward(img, template, stacked_input, epoch)
            tot_loss += loss.item()

            # backward & update model
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step()

        print_with_timestamp(f'Epoch {epoch}: train loss {round(tot_loss/len(self.train_loader), 4)}')

        # log into wandb
        self.log(epoch, phase='train')

    def valid(self, epoch):
        self.reset_logs()
        self.model.eval()
        tot_loss = 0.
        for (img, template, _, _, _) in self.val_loader:
            img, template = img.unsqueeze(1).cuda(), template.unsqueeze(1).cuda() # [B, D, H, W] -> [B, 1, D, H, W]
            stacked_input = torch.cat([img, template], dim=1) # [B, 2, D, H, W]

            # forward & calculate loss in child trainer
            loss, _ = self.forward(img, template, stacked_input, epoch, val=True)
            tot_loss += loss.item()

        print_with_timestamp(f'Epoch {epoch}: valid loss {round(tot_loss/len(self.val_loader), 4)}')

        self.log(epoch, phase='valid')

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
            if self.log_method == "tensorboard":
                self.writer.add_scalar(f"{tag}/{key}", value/num, epoch)
            else:
                wandb.log({f"{tag}/{key}": value/num}, step=epoch)

    def log_single_img(self, name, fig, epoch):
        if self.log_method == 'tensorboard':
            self.writer.add_figure(name, fig, epoch)
        else:
            wandb.log({name: wandb.Image(fig)}, step=epoch)

    def save_imgs(self, epoch, num):
        self.model.eval()
        for idx, (img, template, _, _, _) in enumerate(self.save_loader):
            if idx >= num:
                break
            img, template = img.unsqueeze(1).cuda(), template.unsqueeze(1).cuda() # [B, D, H, W] -> [B, 1, D, H, W]
            stacked_input = torch.cat([img, template], dim=1) # [B, 2, D, H, W]

            # forward & calculate loss in child trainer
            _, deformed_img = self.forward(img, template, stacked_input, epoch, val=True)
            disp = self.get_disp()

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