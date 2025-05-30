from Tester.Tester_base import Tester
from utils.utils import apply_deformation_using_disp
from utils.loss import MSE_loss, NCC_loss, SSIM_loss

import torch
import numpy as np

from tqdm import tqdm

class Similarity_Tester(Tester):
    def __init__(self, args):
        self.csv_path = 'results/csvs/similar_results.csv'
        super().__init__(args)

    def test(self):
        mse, ncc, ssim = [], [], []
        for img, template, _, _, _ in tqdm(self.save_loader):
            img, template = img.unsqueeze(1).cuda(), template.unsqueeze(1).cuda() # [B, D, H, W] -> [B, 1, D, H, W]
            stacked_input = torch.cat([img, template], dim=1) # [B, 2, D, H, W]

            if self.method == 'VM' or self.method == 'Mr':
                disp, _ = self.model(stacked_input)
                disp = disp[-1]
            elif self.method == 'VM-diff' or self.method == 'Mr-diff':
                disp, _ = self.model(stacked_input)
                out = disp[-1]
                disp = self.integrate(out)
                # deformed_img = apply_deformation_using_disp(img, accumulate_disp)
            elif self.method == 'VM-Un' or self.method == 'Mr-Un':
                disp, _, _, _ = self.model(stacked_input)
                disp = disp[-1]
            elif self.method == 'VM-Un-diff' or self.method == 'Mr-Un-diff':
                disp, _, _, _ = self.model(stacked_input)
                out = disp[-1]
                disp = self.integrate(out)
            
            deformed_img = apply_deformation_using_disp(img, disp)

            # mse
            mse.append(MSE_loss(deformed_img, template).item())

            # ncc
            ncc.append(-NCC_loss(deformed_img, template).item())

            # ssim
            ssim.append(SSIM_loss(deformed_img, template).item())

        mse, ncc, ssim = np.array(mse), np.array(ncc), np.array(ssim)

        results = [self.log_name, mse.mean(), mse.std(), ncc.mean(), ncc.std(), ssim.mean(), ssim.std()]
        self.save_results(self.csv_path, results)
