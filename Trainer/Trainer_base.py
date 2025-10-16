import wandb, os, torch, shutil, warnings
import torch.optim as optim
import matplotlib.pyplot as plt
from networks.network_utils import set_model
from utils.dataset import set_dataloader, set_paired_dataloader, set_datapath, set_paired_dataloader_usingcsv, set_dataloader_usingcsv
from utils.utils import set_seed, save_middle_slices, save_middle_slices_mfm, print_with_timestamp, save_grid_spline, apply_deformation_using_disp

from utils.utils import add_identity_to_deformation, compute_jacobian_determinant

from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import nibabel as nib

from torch.utils.tensorboard import SummaryWriter

class Trainer:
    def __init__(self, args, config=None):
        set_seed(seed=args.seed)
        os.environ["WANDB_SYMLINK"] = "false"

        # train options
        self.epochs = args.epochs
        self.start_epoch = args.start_epoch
        self.val_interval = args.val_interval
        self.save_interval = args.save_interval
        self.save_num = args.save_num
        self.pair_train = args.pair_train
        self.log_method = args.log_method
        self.val_detail = args.val_detail
        
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
                config=config,
                settings=wandb.Settings(api_key="87539aeaa75ad2d8a28ec87d70e5d6ce1277c544")
            )
        else:
            self.writer = SummaryWriter(log_dir=f'{args.log_dir}/{args.dataset}/{args.model}/pair_{args.pair_train}/{self.log_name}')
        
        # Setting Model
        self.model = set_model(args.model, out_channels=self.out_channels, out_layers=self.out_layers)
        if args.start_epoch>0:
            self.model.load_state_dict(torch.load(args.saved_path, weights_only=True))
        self.model = self.model.cuda()

        self.train_data_path, self.val_data_path = set_datapath(args.dataset, args.numpy)
        if args.pair_train:
            self.train_loader, self.val_loader, self.save_loader = set_paired_dataloader_usingcsv(args.dataset, 'data/data_list', batch_size=args.batch_size, numpy=args.numpy, transform=args.data_aug, geo=args.data_aug_geo)
        else:
            self.train_loader, self.val_loader, self.save_loader = set_dataloader_usingcsv(args.dataset, 'data/data_list', args.template_path, args.batch_size, numpy=args.numpy, transform=args.data_aug, geo=args.data_aug_geo)
        
        if self.val_detail:
            _, _, self.DSC_loader = set_dataloader_usingcsv(args.dataset, 'data/data_list', args.template_path, args.batch_size, numpy=args.numpy, return_path=True, transform=False, geo=False)

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
        best_loss, best_DSC = 1e+9, 0
        for epoch in range(self.start_epoch, self.epochs):
            self.train_1_epoch(epoch)
            if epoch % self.val_interval == 0:
                with torch.no_grad():
                    self.valid(epoch)
                    cur_loss = self.log_dict['Loss_tot']
                    if best_loss > cur_loss:
                        best_loss = cur_loss
                        torch.save(self.model.state_dict(), f'{self.save_dir}/not_finished/{self.log_name}_best.pt')
                    
                    if self.val_detail:
                        cur_DSC_loss = self.valid_detail(epoch)
                        if best_DSC < cur_DSC_loss:
                            best_DSC = cur_DSC_loss
                            torch.save(self.model.state_dict(), f'{self.save_dir}/not_finished/{self.log_name}_bestDSC.pt')

                torch.save(self.model.state_dict(), f'{self.save_dir}/not_finished/{self.log_name}_last.pt')
            if epoch % self.save_interval == 0:
                with torch.no_grad():
                    self.save_imgs(epoch, self.save_num)

            if self.log_method == 'wandb':
                wandb.save(f'{self.save_dir}/not_finished/{self.log_name}_best.pt')
                wandb.save(f'{self.save_dir}/not_finished/{self.log_name}_bestDSC.pt')
                wandb.save(f'{self.save_dir}/not_finished/{self.log_name}_last.pt')

        # move trained model to complete folder 
        try:
            shutil.move(f'{self.save_dir}/not_finished/{self.log_name}_best.pt', f'{self.save_dir}/completed/{self.log_name}_best.pt')
            shutil.move(f'{self.save_dir}/not_finished/{self.log_name}_last.pt', f'{self.save_dir}/completed/{self.log_name}_last.pt')
            shutil.move(f'{self.save_dir}/not_finished/{self.log_name}_bestDSC.pt', f'{self.save_dir}/completed/{self.log_name}_bestDSC.pt')
        except Exception as e:
            print_with_timestamp(f"Failed to move {self.save_dir}/not_finished/{self.log_name}.pt: {e}")

        wandb.finish()

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

    def valid_detail(self, epoch):
        # check DSCs and folding rates
        self.model.eval()
        tot_folding_rates = 0.
        
        temp_seg = nib.load('data/mni152_label.nii').get_fdata()
        temp_seg = torch.tensor(temp_seg).unsqueeze(0).unsqueeze(0).cuda()
        
        dices = [0.0 for _ in range(35)]
        cnt = 0
        for (img, template, _, _, _, path) in self.DSC_loader:
            path = path[0]
            img, template = img.unsqueeze(1).cuda(), template.unsqueeze(1).cuda() # [B, D, H, W] -> [B, 1, D, H, W]
            stacked_input = torch.cat([img, template], dim=1) # [B, 2, D, H, W]

            # forward & calculate loss in child trainer
            _, _ = self.forward(img, template, stacked_input, epoch, val=True)
            disp = self.get_disp()
            
            # calculate DSCs
            sub_name = path.split('/')[-1].split('.')[0]
            seg_path = f'data/OASIS_label_core/{sub_name}.nii.gz'
            if not os.path.exists(seg_path):
                continue
            cnt += 1 

            seg = nib.load(seg_path).get_fdata().astype(np.float32)
            seg = torch.tensor(seg).unsqueeze(0).unsqueeze(0).cuda()
            deformed_seg = apply_deformation_using_disp(seg, disp, mode='nearest')

            # del seg, disp
            torch.cuda.empty_cache()

            for label in range(35):
                if label+1 in [18, 34]:
                    continue
                dice = self.dice_score(deformed_seg, temp_seg, label+1)
                dices[label] += dice.item()

            del img, template, stacked_input, deformed_seg
            torch.cuda.empty_cache()  # (선택) 메모리 여유를 위해

            # Calculate folding rates
            jacobian = compute_jacobian_determinant(disp)
            negative_mask = jacobian <= 0.
            neg_num = negative_mask.sum().item()
            tot_num = np.prod(jacobian.shape).item()

            tot_folding_rates += neg_num/tot_num

        avg_dices = [d/cnt for d in dices]
        avg_dices = np.array(avg_dices)
        avg_dices = np.delete(avg_dices, [17, 33])
        tot_DSCs = avg_dices.mean()

        # extracted DSCs
        avg_dices = np.delete(avg_dices, [2,3,4,9,13,14,15,16,17,20,21,22,27,28,29,30,31,32])
        ext_DSCs = avg_dices.mean()

        print_with_timestamp(f'Epoch {epoch}: valid loss {round((tot_DSCs), 4)}')

        if self.log_method == "tensorboard":
            self.writer.add_scalar(f"valid_detail/DSCs", tot_DSCs, epoch)
            self.writer.add_scalar(f"valid_detail/DSCs_extracted", ext_DSCs, epoch)
            self.writer.add_scalar(f"valid_detail/folding", tot_folding_rates/len(self.DSC_loader), epoch)
        else:
            wandb.log({f"valid_detail/DSCs": tot_DSCs}, step=epoch)
            wandb.log({f"valid_detail/DSCs_extracted": ext_DSCs}, step=epoch)
            wandb.log({f"valid_detail/folding": tot_folding_rates/len(self.DSC_loader)}, step=epoch)
        
        return ext_DSCs
    
    def dice_score(self, seg1, seg2, label):
        seg1 = seg1.int()
        seg2 = seg2.int()
        mask1 = (seg1 == label)
        mask2 = (seg2 == label)
        intersection = (mask1 & mask2).sum().float()
        size1 = mask1.sum().float()
        size2 = mask2.sum().float()
        return torch.tensor(1.0) if (size1 + size2 == 0) else 2.0 * intersection / (size1 + size2)

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