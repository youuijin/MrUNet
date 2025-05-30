from Tester.Tester_base import Tester
from utils.dataset import set_dataloader
from utils.utils import apply_deformation_using_disp

import torch
import numpy as np
import nibabel as nib
from tqdm import tqdm

class DSC_Tester(Tester):
    def __init__(self, model_path, args):
        self.csv_path = 'results/csvs/dice_results.csv'
        super().__init__(model_path, args)
        _, _, self.save_loader = set_dataloader(args.image_path, args.template_path, batch_size=1, return_path=True)
        self.label_path = args.label_path
        self.lut = self.load_freesurfer_lut()

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
        return 1.0 if (size1 + size2 == 0) else 2.0 * intersection / (size1 + size2)

    def test(self):
        temp_seg = nib.load('data/mni152_label.nii').get_fdata()
        temp_seg = torch.tensor(temp_seg).unsqueeze(0).unsqueeze(0).cuda()

        dices = [0.0 for _ in range(35)]
        for i, (img, template, _, _, affine, path) in enumerate(tqdm(self.save_loader, position=0, leave=True, ascii=True)):
            path = path[0]
            seg = nib.load(f"{self.label_path}/{path.split('/')[-1]}").get_fdata().astype(np.float32)
            seg = torch.tensor(seg).unsqueeze(0).unsqueeze(0).cuda()

            img, template = img.unsqueeze(1).cuda(), template.unsqueeze(1).cuda() # [B, D, H, W] -> [B, 1, D, H, W]
            stacked_input = torch.cat([img, template], dim=1) # [B, 2, D, H, W]

            disp = self.model(stacked_input)[0][-1]
            if 'diff' in self.method:
                disp = self.integrate(disp)

            deformed_seg = apply_deformation_using_disp(seg, disp, mode='nearest')

            del seg, disp
            torch.cuda.empty_cache()

            for label in tqdm(range(35), position=1, leave=False):
                if label+1 in [18, 34]:
                    continue
                dice = self.dice_score(deformed_seg, temp_seg, label+1)
                dices[label] += dice.item()

            del img, template, stacked_input, deformed_seg
            torch.cuda.empty_cache()  # (선택) 메모리 여유를 위해

        avg_dices = [d/len(self.save_loader) for d in dices]

        avg_dices = np.array(avg_dices)
        avg_dices = np.delete(avg_dices, [17, 33])

        results = [self.log_name, avg_dices.mean(), avg_dices.std()] + [a for a in avg_dices]
        self.save_results(self.csv_path, results)
