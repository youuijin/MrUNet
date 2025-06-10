import wandb, os, torch, shutil
import torch.optim as optim
import matplotlib.pyplot as plt
from utils.dataset import set_dataloader, set_paired_dataloader, set_datapath
from networks.network_utils import set_model
from utils.utils import set_seed, save_middle_slices, save_middle_slices_mfm
from tqdm import tqdm

from datetime import datetime

# logging - wandb
wandb.login(key="87539aeaa75ad2d8a28ec87d70e5d6ce1277c544")

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
        
        if args.epochs != 200:
            self.log_name = f'{self.log_name}_epochs{args.epochs}'
        if args.lr_scheduler == 'multistep':
            self.log_name = f'{self.log_name}_sche(multi_{args.lr_milestones})'

        # add start time
        now = datetime.now().strftime("%m-%d_%H-%M")
        self.log_name = f'{self.log_name}_{now}'

        wandb.init(
            project=args.wandb_name,
            name=self.log_name,
            dataset=args.dataset,
            model=args.model,
            pair=args.pair_train,
            config=config
        )
        
        # Setting Model
        self.model = set_model(args.model, out_channels=self.out_channels, out_layers=self.out_layers)
        if args.start_epoch>0:
            self.model.load_state_dict(torch.load(args.saved_path, weights_only=True))
        self.model = self.model.cuda()

        self.train_data_path, self.val_data_path = set_datapath(args.dataset)
        if args.pair_train:
            self.train_loader, self.val_loader, self.save_loader = set_paired_dataloader(train_dir=self.train_data_path, val_dir=self.val_data_path, batch_size=args.batch_size)
        else:
            self.train_loader, self.val_loader, self.save_loader = set_dataloader(self.train_data_path, args.template_path, args.batch_size)
        
        if args.pair_train:
            self.save_dir = f'./results/pair/saved_models/{args.dataset}/{args.model}'
        else:
            self.save_dir = f'./results/template/saved_models/{args.dataset}/{args.model}'
        os.makedirs(f'{self.save_dir}/completed', exist_ok=True)
        os.makedirs(f'{self.save_dir}/not_finished', exist_ok=True)
        
        # self.val_detail = args.val_detail

        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)
        # set learning rate scheduler 
        tot_step_number = args.epochs * len(self.train_loader)
        if args.lr_scheduler == 'none':
            self.lr_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda epoch: 1.0)
        elif args.lr_scheduler == 'multistep':
            # milestones = [i*tot_step_number for i in args.lr_milestones.split(',')]
            milestones = [int(i)*len(self.train_loader) for i in args.lr_milestones.split(',')]
            self.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=milestones, gamma=0.1)

        self.lr_scheduler.step(args.start_epoch*len(self.train_loader))

    def train(self):
        best_loss = 1e+9
        cnt = 0
        # save template img
        for epoch in tqdm(range(self.start_epoch, self.epochs), position=0, desc='Epoch', leave=True):
            self.train_1_epoch(epoch)
            if epoch % self.val_interval == 0:
                with torch.no_grad():
                    self.valid(epoch)
                    cur_loss = self.log_dict['Loss_tot']
                    if best_loss > cur_loss:
                        cnt = 0
                        best_loss = cur_loss
                        torch.save(self.model.state_dict(), f'{self.save_dir}/not_finished/{self.log_name}.pt')
                    else: 
                        cnt+=1

                    if cnt >= 3:
                        # early stop
                        break
            if epoch % self.save_interval == 0:
                with torch.no_grad():
                    self.save_imgs(epoch, self.save_num)

        # move trained model to complete folder 
        try:
            
            shutil.move(f'{self.save_dir}/not_finished/{self.log_name}.pt', f'{self.save_dir}/completed/{self.log_name}.pt')
        except Exception as e:
            print(f"Failed to move {self.save_dir}/not_finished/{self.log_name}.pt: {e}")

        wandb.finish()

    def train_1_epoch(self, epoch):
        self.reset_logs()
        self.model.train()
        for (img, template, _, _, _) in tqdm(self.train_loader, desc=f"Train [lr {self.optimizer.param_groups[0]['lr']}]", position=1, leave=False):
            img, template = img.unsqueeze(1).cuda(), template.unsqueeze(1).cuda() # [B, D, H, W] -> [B, 1, D, H, W]
            stacked_input = torch.cat([img, template], dim=1) # [B, 2, D, H, W]

            # forward & calculate loss in child trainer
            loss, _ = self.forward(img, template, stacked_input)

            # backward & update model
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step()

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

            if self.pair_train:
                fig = save_middle_slices_mfm(img, template, deformed_img, epoch, idx)
                wandb.log({f"deformed_slices_img{idx}": wandb.Image(fig)}, step=epoch)
                plt.close(fig)
            else:
                fig = save_middle_slices(deformed_img, epoch, idx)
                wandb.log({f"deformed_slices_img{idx}": wandb.Image(fig)}, step=epoch)
                plt.close(fig)

                if epoch == 0 and idx == 0:
                    fig = save_middle_slices(template, epoch, idx)
                    wandb.log({f"Template": wandb.Image(fig)}, step=epoch)
                    plt.close(fig)