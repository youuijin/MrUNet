from Tester.Tester_base import Tester

import torch
import numpy as np

from tqdm import tqdm

class Folding_Tester(Tester):
    def __init__(self, args):
        self.csv_path = 'results/csvs/folding_results.csv'
        super().__init__(args)

    def test(self):
        fr = []
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
            
            # calculate jacobian matrix & 
            jacobian = self.compute_jacobian_determinant(disp)
            negative_mask = jacobian <= 0.
            neg_num = negative_mask.sum().item()
            tot_num = np.prod(jacobian.shape).item()

            fr.append(neg_num/tot_num)

        fr = np.array(fr)

        results = [self.log_name, fr.mean(), fr.std()]
        self.save_results(self.csv_path, results)

    def add_identity_to_deformation(self, deformation_field):
        D, H, W, _ = deformation_field.shape
        identity = np.stack(np.meshgrid(
            np.arange(D), np.arange(H), np.arange(W), indexing='ij'), axis=-1)
        return deformation_field + identity

    # --- 2. Jacobian Determinant 계산 ---
    def compute_jacobian_determinant(self, displacement_field):
        """
        변형장의 Jacobian Determinant를 계산하여 반환
        displacement_field: (X, Y, Z, 3) 형태의 변형장 (displacement field를 의미)
        """
        displacement_field = displacement_field.squeeze(0).detach().cpu().numpy()
        if displacement_field.shape[-1] !=3:
            displacement_field = np.transpose(displacement_field, (1, 2, 3, 0)) # (X, Y, Z, 3)

        def_voxel = displacement_field.copy()
        deformation_field = self.add_identity_to_deformation(def_voxel)

        dx = np.gradient(deformation_field[..., 0], axis=0)  # dφ_x/dx
        dy = np.gradient(deformation_field[..., 0], axis=1)  # dφ_x/dy
        dz = np.gradient(deformation_field[..., 0], axis=2)  # dφ_x/dz

        ex = np.gradient(deformation_field[..., 1], axis=0)  # dφ_y/dx
        ey = np.gradient(deformation_field[..., 1], axis=1)  # dφ_y/dy
        ez = np.gradient(deformation_field[..., 1], axis=2)  # dφ_y/dz

        fx = np.gradient(deformation_field[..., 2], axis=0)  # dφ_z/dx
        fy = np.gradient(deformation_field[..., 2], axis=1)  # dφ_z/dy
        fz = np.gradient(deformation_field[..., 2], axis=2)  # dφ_z/dz

        # Jacobian 행렬 구성
        jacobian = np.zeros(deformation_field.shape[:-1] + (3, 3))
        jacobian[..., 0, 0] = dx
        jacobian[..., 0, 1] = dy
        jacobian[..., 0, 2] = dz

        jacobian[..., 1, 0] = ex
        jacobian[..., 1, 1] = ey
        jacobian[..., 1, 2] = ez

        jacobian[..., 2, 0] = fx
        jacobian[..., 2, 1] = fy
        jacobian[..., 2, 2] = fz

        # # Determinant 계산
        jacobian_det = np.linalg.det(jacobian)
        return jacobian_det
