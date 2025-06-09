from Tester.Tester_base import Tester
from utils.utils import apply_deformation_using_disp
from utils.loss import MSE_loss, NCC_loss, SSIM_loss

import torch
import numpy as np

from tqdm import tqdm

class Similarity_Tester(Tester):
    def __init__(self, model_path, args):
        self.csv_path = 'results/csvs/similar_results.csv'
        if args.external:
            self.csv_path = 'results/csvs/similar_results_external.csv'
        super().__init__(model_path, args)

    def test(self):
        mse, ncc, ssim = [], [], []
        for img, template, _, _, _ in tqdm(self.save_loader):
            img, template = img.unsqueeze(1).cuda(), template.unsqueeze(1).cuda() # [B, D, H, W] -> [B, 1, D, H, W]
            stacked_input = torch.cat([img, template], dim=1) # [B, 2, D, H, W]

            disp = self.model(stacked_input)[0][-1]
            if 'diff' in self.method:
                disp = self.integrate(disp)
            
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
