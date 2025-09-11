from Tester.Tester_base import Tester
from utils.dataset import set_dataloader_usingcsv, set_paired_dataloader_usingcsv
from utils.utils import apply_deformation_using_disp, save_deformed_image_nii

import torch, os
import numpy as np
import nibabel as nib
from tqdm import tqdm
import matplotlib.pyplot as plt

class DSC_Tester_PET(Tester):
    def __init__(self, model_path, args):
        # self.set_dataset(args)
        self.test_dataset = "FDG_PET"
        self.label_path = args.label_path
        if args.label_path == "data/FDG_label":
            dice_type = 'dice35'
            self.seg_num = 35
        elif args.label_path == "data/FDG_label_cortex":
            dice_type = 'dice6'
            self.seg_num = 6

        self.trained_dataset = model_path.split("/")[-4]

        self.inverse_test = args.inverse_test
        if args.inverse_test:
            dice_type = f'{dice_type}_inverse'
        self.csv_path = f'{args.csv_dir}/{self.test_dataset}/{dice_type}_results.csv'
        self.visualize = True
        self.save_num = 3
        self.dice_type = dice_type

        super().__init__(model_path, args)

        self.pair_test = args.pair_test
        if args.pair_test:
            _, _, self.save_loader = set_paired_dataloader_usingcsv(self.test_dataset, 'data/data_list', batch_size=1, numpy=False, return_path=True)
        else:
            _, _, self.save_loader = set_dataloader_usingcsv(self.test_dataset, 'data/data_list', args.template_path, batch_size=1, numpy=False, return_path=True)
        
        self.lut = self.load_freesurfer_lut()

        if self.inverse_test:
            self.visualize_save_dir = f"visualization/PET_dice_inverse/{'/'.join(model_path.split('/')[1:])}"
        else:
            self.visualize_save_dir = f"visualization/PET_dice/{'/'.join(model_path.split('/')[1:])}"
        os.makedirs(self.visualize_save_dir, exist_ok=True)

    # LUT 불러오기 함수
    def load_freesurfer_lut(self, lut_path="utils/FreeSurferColorLUT.txt"):
        lut = {}
        with open(lut_path, 'r') as f:
            for line in f:
                if line.startswith('#') or line.strip() == '':
                    continue
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                try:
                    label_id = int(parts[0])
                    label_name = parts[1]
                    lut[label_id] = label_name
                except:
                    continue
        return lut

    def dice_score(self, seg1, seg2, label):
        seg1 = seg1.int()
        seg2 = seg2.int()
        mask1 = (seg1 == label)
        mask2 = (seg2 == label)
        intersection = (mask1 & mask2).sum().float()
        size1 = mask1.sum().float()
        size2 = mask2.sum().float()
        return torch.tensor(1.0) if (size1 + size2 == 0) else 2.0 * intersection / (size1 + size2)
    
    def transform_slice(self, img):
        # apply 90-degree CCW rotation + horizontal flip
        return np.fliplr(np.rot90(img, k=1))
    
    def save_middle_slices(self, img_3d, save_path, cmap='gray'):
        """
        img_3d: [D, H, W] or [1, D, H, W] or [B, 1, D, H, W] (e.g., torch.Tensor)
        Returns: matplotlib Figure with x, y, z middle slices side-by-side
        """
        img_3d = img_3d.squeeze().detach().cpu().numpy()

        D, H, W = img_3d.shape

        slice_x = self.transform_slice(img_3d[D // 2, :, :])
        slice_y =self. transform_slice(img_3d[:, H // 2, :])
        slice_z = self.transform_slice(img_3d[:, :, W // 2])

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(slice_z, cmap=cmap)
        axes[0].set_title('Axial (X)')
        axes[1].imshow(slice_y, cmap=cmap)
        axes[1].set_title('Coronal (Y)')
        axes[2].imshow(slice_x, cmap=cmap)
        axes[2].set_title('Sagittal (Z)')

        for ax in axes:
            ax.axis('off')

        plt.tight_layout()
        plt.savefig(save_path)    
        plt.close(fig)

        return fig

    def test(self):
        if 'dice35' in self.dice_type:
            temp_seg = nib.load('data/mni152_label.nii').get_fdata().astype(np.float32)
        else:
            temp_seg = nib.load('data/FDG_label_cortex/template_T1w_MRI.nii.gz').get_fdata().astype(np.float32)
        temp_seg = torch.tensor(temp_seg).unsqueeze(0).unsqueeze(0).cuda()

        dices = [0.0 for _ in range(self.seg_num)]
        cnt = 0
        for i, (img, template, _, _, aff, path) in enumerate(tqdm(self.save_loader, position=0, leave=True, ascii=True)):
            img_path = path[0]
            if self.pair_test:
                img_path, template_path = path[0][0], path[1][0]
                temp_seg = nib.load(f"{self.label_path}/{template_path.split('/')[-1]}").get_fdata().astype(np.float32)
                temp_seg = torch.tensor(temp_seg).unsqueeze(0).unsqueeze(0).cuda()
            sub_name = img_path.split('/')[-1].split("_")[1]
            seg_path = f"{self.label_path}/{sub_name}_T1w_MRI.nii.gz"
            if not os.path.exists(seg_path):
                continue
            cnt += 1

            seg = nib.load(seg_path).get_fdata().astype(np.float32)
            seg = torch.tensor(seg).unsqueeze(0).unsqueeze(0).cuda()

            img, template = img.unsqueeze(1).cuda(), template.unsqueeze(1).cuda() # [B, D, H, W] -> [B, 1, D, H, W]
            
            if self.inverse_test:
                moving_img = template
                fixed_img = img
                moving_seg = temp_seg
                fixed_seg = seg
            else:
                moving_img = img
                fixed_img = template
                moving_seg = seg
                fixed_seg = temp_seg
            
            stacked_input = torch.cat([moving_img, fixed_img], dim=1) # [B, 2, D, H, W]

            disp = self.model(stacked_input)[0][-1]
            if 'diff' in self.dice_type:
                disp = self.integrate(disp)

            deformed_seg = apply_deformation_using_disp(moving_seg, disp, mode='nearest')
            deformed_img = apply_deformation_using_disp(moving_img, disp, mode='bilinear')

            if self.visualize and i < self.save_num:
                save_deformed_image_nii(deformed_seg, f'{self.visualize_save_dir}/img{i}_{self.dice_type}_deformed.nii.gz', torch.tensor(0.0), torch.tensor(1.0), aff.squeeze())
                self.save_middle_slices(deformed_seg, f'{self.visualize_save_dir}/img{i}_{self.dice_type}_deformed.png', cmap='jet')
                save_deformed_image_nii(moving_seg, f'{self.visualize_save_dir}/img{i}_{self.dice_type}_origin.nii.gz', torch.tensor(0.0), torch.tensor(1.0), aff.squeeze())
                self.save_middle_slices(moving_seg, f'{self.visualize_save_dir}/img{i}_{self.dice_type}_origin.png', cmap='jet')
                save_deformed_image_nii(fixed_seg, f'{self.visualize_save_dir}/img{i}_{self.dice_type}_template.nii.gz', torch.tensor(0.0), torch.tensor(1.0), aff.squeeze())
                self.save_middle_slices(fixed_seg, f'{self.visualize_save_dir}/img{i}_{self.dice_type}_template.png', cmap='jet')
                save_deformed_image_nii(deformed_img, f'{self.visualize_save_dir}/img{i}_img_deformed.nii.gz', torch.tensor(0.0), torch.tensor(1.0), aff.squeeze())
                self.save_middle_slices(deformed_img, f'{self.visualize_save_dir}/img{i}_img_deformed.png')
                save_deformed_image_nii(moving_img, f'{self.visualize_save_dir}/img{i}_img_origin.nii.gz', torch.tensor(0.0), torch.tensor(1.0), aff.squeeze())
                self.save_middle_slices(moving_img, f'{self.visualize_save_dir}/img{i}_img_origin.png')
                save_deformed_image_nii(fixed_img, f'{self.visualize_save_dir}/img{i}_img_template.nii.gz', torch.tensor(0.0), torch.tensor(1.0), aff.squeeze())
                self.save_middle_slices(fixed_img, f'{self.visualize_save_dir}/img{i}_img_template.png')

            if 'dice35' in self.dice_type:
                for label in tqdm(range(35), position=1, leave=False):
                    if label+1 in [18, 34]:
                        continue
                    dice = self.dice_score(deformed_seg, fixed_seg, label+1)
                    dices[label] += dice.item()
            else:
                for label in tqdm(range(6), position=1, leave=False):
                    dice = self.dice_score(deformed_seg, fixed_seg, label+1)
                    dices[label] += dice.item()
            
        avg_dices = [d/cnt for d in dices]

        avg_dices = np.array(avg_dices)
        if "dice35" in self.dice_type:
            avg_dices = np.delete(avg_dices, [17, 33])

        results = [self.model_type, self.log_name, avg_dices.mean(), avg_dices.std()] + [a for a in avg_dices]
        self.save_results(self.csv_path, results)
