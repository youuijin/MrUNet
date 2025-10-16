from utils.dataset import set_dataloader, set_dataloader_usingcsv, set_paired_dataloader_usingcsv
import csv, torch, os
from tqdm import tqdm

import cv2

from utils.loss import MSE_loss, NCC_loss, SSIM_loss

import nibabel as nib
import numpy as np

import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

class SyN_tester:
    def __init__(self, model_path, args):
        # check already tested)
        self.log_name = model_path
        self.pair_test = args.pair_test
        self.test_dataset = args.test_dataset
        
        self.csv_subdir = f'{args.csv_dir}/train_OASIS/test_{self.test_dataset}' # save in OASIS 
        
        if self.test_dataset == 'OASIS':
            self.data_path = 'data/OASIS_SyN'
            self.original_path = 'data/OASIS_brain_core_percent'
            self.label_path = 'data/OASIS_label_core'
        elif self.test_dataset == 'DLBS':
            self.data_path = 'data/DLBS_SyN'
            self.original_path = 'data/DLBS_core_percent'
            self.label_path = 'data/DLBS_label_core'
        elif self.test_dataset == 'FDG_MRI':
            self.data_path = 'data/FDG_MRI_SyN'
            self.original_path = 'data/FDG_MRI_percent'
            self.label_path = 'data/FDG_label'
        self.test_method = args.test_method

        if self.test_method == 'quant':
            assert args.label_path is not None
            self.pet_dir = "data/FDG_PET_percent"
            self.label_path = args.label_path
            if self.label_path == "data/FDG_label_cortex":
                self.dice_type = 'dice6'
                self.seg_num = 6
            elif self.label_path == "data/FDG_label":
                self.dice_type = 'dice35'
                self.seg_num = 35
            else:
                self.dice_type = 'dice70'
                self.seg_num = 70

        if args.pair_test:
            _, _, self.save_loader = set_paired_dataloader_usingcsv(self.test_dataset, 'data/data_list', batch_size=1, return_path=True, numpy=False)
        else:
            _, _, self.save_loader = set_dataloader_usingcsv(self.test_dataset, 'data/data_list', args.template_path, 1, return_path=True, numpy=False)

    def save_results(self, csv_path, row):
        with open(csv_path, 'a', encoding='utf-8', newline='') as f:
            wr = csv.writer(f)
            wr.writerow(row)

    def check_tested(self, model_path):
        with open(self.csv_path, 'r', newline='') as f:
            reader = csv.reader(f)
            next(reader)  # 헤더 건너뛰기
            existing_values = set(row[1] for row in reader)

        # 3. 값이 없으면 추가
        if model_path.split('/')[-1] not in existing_values:
            return False
        else:
            print('Already Tested:', model_path, 'in', self.csv_path)
            return True

    def load_single_image(self, path, normalize=True):
        img = nib.load(path)
        aff = img.affine
        img = img.get_fdata()
        
        if normalize:
            #TODO: Delete or not
            # Template normalize - percentile
            t_data = img.flatten()
            p1_temp = np.percentile(t_data, 1)
            p99_temp = np.percentile(t_data, 99)
            img = np.clip(img, p1_temp, p99_temp)
            #TODO: Delete or not

            img_min, img_max = img.min(), img.max()
            img = (img - img_min) / (img_max - img_min)  # Normalize to [0,1]#

        return torch.tensor(img, dtype=torch.float32), torch.tensor(aff, dtype=torch.float32)
    
    def test(self):
        if self.test_method == 'dice':
            self.test_dice()
        elif self.test_method == 'folding':
            self.test_folding()
        elif self.test_method == 'similar':
            self.test_similar()
        elif self.test_method == 'blur':
            self.test_blur()
        elif self.test_method == 'quant':
            self.test_quant()
        
    def test_blur(self):
        self.save_dir = f'visualization/{self.test_dataset}/avg_template'
        self.csv_path = f'{self.csv_subdir}/blur_results.csv'
        
        tested = self.check_tested(self.log_name)
        self.already_tested = False
        if tested:
            self.already_tested = True
            return
        
        aff = None
        deformed_imgs = []  # 리스트에 저장

        for _, _, _, _, _, path in tqdm(self.save_loader):
            path = path[0]
            img_path = f'{self.data_path}/{path.split("/")[-1].split(".")[0].split("_")[0]}_warped.nii.gz'
            deformed_img, affine = self.load_single_image(img_path)
            # img, template = img.unsqueeze(1).cuda(), template.unsqueeze(1).cuda() # [B, D, H, W] -> [B, 1, D, H, W]
            aff = affine.squeeze()

            deformed_imgs.append(deformed_img.unsqueeze(1).cpu())

        # Stack: shape [N, D, H, W]
        deformed_stack = torch.stack(deformed_imgs, dim=0)

        # 평균 및 분산 계산
        mean_img = torch.mean(deformed_stack, dim=0).squeeze()
        var_img = torch.var(deformed_stack, dim=0).squeeze()

        overall_variance = var_img.mean().item()
        save_deformed_image(mean_img, f'{self.save_dir}/{self.log_name}_mean.nii.gz', torch.tensor(0.0), torch.tensor(1.0), aff)
        save_deformed_image(var_img, f'{self.save_dir}/{self.log_name}_std.nii.gz', torch.tensor(0.0), torch.tensor(1.0), aff)

        mean_img = mean_img.detach().cpu().numpy() 

        laplacian = measure_blur_laplacian_3d(mean_img, save_path=f'{self.save_dir}/{self.log_name}_laplacian.png')
        tenengrad = measure_blur_tenengrad_3d(mean_img, save_path=f'{self.save_dir}/{self.log_name}_tenegrad.png')
        fft = measure_blur_fft_3d(mean_img, save_path=f'{self.save_dir}/{self.log_name}_fft.png')

        results = ['None', self.log_name, round(overall_variance, 5), round(laplacian,5), round(tenengrad,5), round(fft, 5)]
        self.save_results(self.csv_path, results)

    def dice_score(self, seg1, seg2, label):
        seg1 = seg1.int()
        seg2 = seg2.int()
        mask1 = (seg1 == label)
        mask2 = (seg2 == label)
        intersection = (mask1 & mask2).sum().float()
        size1 = mask1.sum().float()
        size2 = mask2.sum().float()
        return torch.tensor(1.0) if (size1 + size2 == 0) else 2.0 * intersection / (size1 + size2)

    def test_dice(self):
        self.csv_path = f'{self.csv_subdir}/dice_results.csv'
        
        tested = self.check_tested(self.log_name)
        self.already_tested = False
        if tested:
            self.already_tested = True
            return
        
        temp_seg = nib.load('data/mni152_label.nii').get_fdata()
        temp_seg = torch.tensor(temp_seg).unsqueeze(0).unsqueeze(0).cuda()

        dices = [0.0 for _ in range(35)]
        cnt = 0
        for i, (_, _, _, _, _, path) in enumerate(tqdm(self.save_loader, position=0, leave=True, ascii=True)):
            if self.pair_test:
                img_path, template_path = path[0][0].split("/")[-1].split(".")[0], path[1][0].split("/")[-1].split(".")[0]
                seg_path = f"{self.data_path}/{img_path}_{template_path}_seg.nii.gz"
                temp_seg = nib.load(f"{self.label_path}/{template_path}.nii.gz").get_fdata().astype(np.float32)
                temp_seg = torch.tensor(temp_seg).unsqueeze(0).unsqueeze(0).cuda()
            else:
                path = path[0]
                seg_path = f'{self.data_path}/{path.split("/")[-1].split(".")[0].split("_")[0]}_seg.nii.gz'
                # print(seg_path)
                # exit()
            if not os.path.exists(seg_path):
                print("NO segments:", seg_path)
                continue
            cnt += 1
            deformed_seg, _ = self.load_single_image(seg_path, normalize=False)
            deformed_seg = deformed_seg.unsqueeze(0).unsqueeze(0).cuda()

            torch.cuda.empty_cache()

            for label in tqdm(range(35), position=1, leave=False):
                if label+1 in [18, 34]:
                    continue
                dice = self.dice_score(deformed_seg, temp_seg, label+1)
                dices[label] += dice.item()

            torch.cuda.empty_cache()  # (선택) 메모리 여유를 위해

        avg_dices = [d/cnt for d in dices]

        avg_dices = np.array(avg_dices)
        avg_dices = np.delete(avg_dices, [17, 33])

        results = ['None', self.log_name, avg_dices.mean(), avg_dices.std()] + [a for a in avg_dices]
        self.save_results(self.csv_path, results)

    def test_similar(self):
        self.csv_path = f'{self.csv_subdir}/similar_results.csv'
        mse, ncc, ssim = [], [], []

        tested = self.check_tested(self.log_name)
        self.already_tested = False
        if tested:
            self.already_tested = True
            return

        for _, template, _, _, _, path in tqdm(self.save_loader):
            template = template.unsqueeze(1).cuda()
            if self.pair_test:
                img_path, template_path = path[0][0].split("/")[-1].split(".")[0], path[1][0].split("/")[-1].split(".")[0]
                img_path = f'{self.data_path}/{img_path}_{template_path}_warped.nii.gz'
                template = nib.load(f"{self.original_path}/{template_path}.nii.gz").get_fdata().astype(np.float32)
                template = torch.tensor(template).unsqueeze(0).unsqueeze(0).cuda()
            else:
                path = path[0]
                img_path = f'{self.data_path}/{path.split("/")[-1].split(".")[0].split("_")[0]}_warped.nii.gz'
            
            deformed_img, affine = self.load_single_image(img_path)
            deformed_img = deformed_img.unsqueeze(0).unsqueeze(0).cuda()
            # mse
            mse.append(MSE_loss(deformed_img, template).item())

            # ncc
            ncc.append(-NCC_loss(deformed_img, template).item())

            # ssim
            ssim.append(SSIM_loss(deformed_img, template).item())

        mse, ncc, ssim = np.array(mse), np.array(ncc), np.array(ssim)

        results = ['None', self.log_name, mse.mean(), mse.std(), ncc.mean(), ncc.std(), ssim.mean(), ssim.std()]
        self.save_results(self.csv_path, results)

    def test_folding(self):
        self.csv_path = f'{self.csv_subdir}/folding_results.csv'
        fr = []
        for _, _, _, _, _, path in tqdm(self.save_loader):
            path = path[0]
            disp_path = f'{self.data_path}/{path.split("/")[-1].split(".")[0]}_deform.nii.gz'

            if self.pair_test:
                img_path, template_path = path[0][0].split("/")[-1].split(".")[0], path[1][0].split("/")[-1].split(".")[0]
                disp_path = f'{self.data_path}/{img_path}_{template_path}_deform.nii.gz'
            
            disp, affine = self.load_single_image(disp_path, normalize=False)
            disp = disp.squeeze(3) # [D, H, W, 1, 3] ->  [D, H, W, 3]

            # calculate jacobian matrix & 
            jacobian = self.compute_jacobian_determinant(disp)
            negative_mask = jacobian <= 0.
            neg_num = negative_mask.sum().item()
            tot_num = np.prod(jacobian.shape).item()

            fr.append(neg_num/tot_num)

        fr = np.array(fr)

        results = ['None', self.log_name, fr.mean(), fr.std()]
        self.save_results(self.csv_path, results)

    def add_identity_to_deformation(self, deformation_field):
        D, H, W, _ = deformation_field.shape
        identity = np.stack(np.meshgrid(
            np.arange(D), np.arange(H), np.arange(W), indexing='ij'), axis=-1)
        return deformation_field + identity

    # --- 2. Jacobian Determinant 계산 ---
    def compute_jacobian_determinant(self, disp):
        disp = np.array(disp)
        deformation_field = self.add_identity_to_deformation(disp)

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

    def test_quant(self):
        self.csv_subdir = f'{self.csv_subdir}/quant/quant_{self.seg_num}'
        self.csv_path = f'{self.csv_subdir}/quant_global.csv'
        tested = self.check_tested(self.log_name)
        self.already_tested = False
        if tested:
            self.already_tested = True
            return

        if self.dice_type == 'dice6':
            self.label_name = {
                1: 'frontal',
                2: 'parietal',
                3: 'temporal',
                4: 'occipital',
                5: 'subcortical',
                6: 'cerebellum'
            }
        elif self.dice_type == 'dice35':
            # cerebellum cortex : 6, 24
            # remove : 18, 34
            self.label_name = {
                1: 'Left-Cerebral-White-Matter',
                2: 'Left-Cerebral-Cortex',
                3: 'Left-Lateral-Ventricle',
                4: 'Left-Inf-Lat-Vent',
                5: 'Left-Cerebellum-Exterior',
                6: 'Left-Cerebellum-Cortex',
                7: 'Left-Thalamus',
                8: 'Left-Caudate',
                9: 'Left-Putamen',
                10: 'Left-Pallidum',
                11: '3rd-Ventricle',
                12: '4th-Ventricle',
                13: 'Brain-Stem',
                14: 'Left-Hippocampus',
                15: 'Left-Amygdala',
                16: 'Left-Accumbens-area',
                17: 'Left-VentralDC',
                19: 'Left-choroid-plexus',
                20: 'Right-Cerebral-White-Matter',
                21: 'Right-Cerebral-Cortex',
                22: 'Right-Lateral-Ventricle',
                23: 'Right-Inf-Lat-Vent',
                24: 'Right-Cerebellum-White-Matter',
                25: 'Right-Cerebellum-Cortex',
                26: 'Right-Thalamus',
                27: 'Right-Caudate',
                28: 'Right-Putamen',
                29: 'Right-Pallidum',
                30: 'Right-Hippocampus',
                31: 'Right-Amygdala',
                32: 'Right-Accumbens-area',
                33: 'Right-VentralDC',
                35: 'Right-choroid-plexus'
            }
        else:
            # cerebellum cortex : 69, 70
            self.label_name={
                1: 'ctx-lh-bankssts',
                2: 'ctx-lh-caudalanteriorcingulate',
                3: 'ctx-lh-caudalmiddlefrontal',
                4: 'ctx-lh-cuneus',
                5: 'ctx-lh-entorhinal',
                6: 'ctx-lh-fusiform',
                7: 'ctx-lh-inferiorparietal',
                8: 'ctx-lh-inferiortemporal',
                9: 'ctx-lh-isthmuscingulate',
                10: 'ctx-lh-lateraloccipital',
                11: 'ctx-lh-lateralorbitofrontal',
                12: 'ctx-lh-lingual',
                13: 'ctx-lh-medialorbitofrontal',
                14: 'ctx-lh-middletemporal',
                15: 'ctx-lh-parahippocampal',
                16: 'ctx-lh-paracentral',
                17: 'ctx-lh-parsopercularis',
                18: 'ctx-lh-parsorbitalis',
                19: 'ctx-lh-parstriangularis',
                20: 'ctx-lh-pericalcarine',
                21: 'ctx-lh-postcentral',
                22: 'ctx-lh-posteriorcingulate',
                23: 'ctx-lh-precentral',
                24: 'ctx-lh-precuneus',
                25: 'ctx-lh-rostralanteriorcingulate',
                26: 'ctx-lh-rostralmiddlefrontal',
                27: 'ctx-lh-superiorfrontal',
                28: 'ctx-lh-superiorparietal',
                29: 'ctx-lh-superiortemporal',
                30: 'ctx-lh-supramarginal',
                31: 'ctx-lh-frontalpole',
                32: 'ctx-lh-temporalpole',
                33: 'ctx-lh-transversetemporal',
                34: 'ctx-lh-insula',
                35: 'ctx-rh-bankssts',
                36: 'ctx-rh-caudalanteriorcingulate',
                37: 'ctx-rh-caudalmiddlefrontal',
                38: 'ctx-rh-cuneus',
                39: 'ctx-rh-entorhinal',
                40: 'ctx-rh-fusiform',
                41: 'ctx-rh-inferiorparietal',
                42: 'ctx-rh-inferiortemporal',
                43: 'ctx-rh-isthmuscingulate',
                44: 'ctx-rh-lateraloccipital',
                45: 'ctx-rh-lateralorbitofrontal',
                46: 'ctx-rh-lingual',
                47: 'ctx-rh-medialorbitofrontal',
                48: 'ctx-rh-middletemporal',
                49: 'ctx-rh-parahippocampal',
                50: 'ctx-rh-paracentral',
                51: 'ctx-rh-parsopercularis',
                52: 'ctx-rh-parsorbitalis',
                53: 'ctx-rh-parstriangularis',
                54: 'ctx-rh-pericalcarine',
                55: 'ctx-rh-postcentral',
                56: 'ctx-rh-posteriorcingulate',
                57: 'ctx-rh-precentral',
                58: 'ctx-rh-precuneus',
                59: 'ctx-rh-rostralanteriorcingulate',
                60: 'ctx-rh-rostralmiddlefrontal',
                61: 'ctx-rh-superiorfrontal',
                62: 'ctx-rh-superiorparietal',
                63: 'ctx-rh-superiortemporal',
                64: 'ctx-rh-supramarginal',
                65: 'ctx-rh-frontalpole',
                66: 'ctx-rh-temporalpol',
                67: 'ctx-rh-transversetemporal',
                68: 'ctx-rh-insula',
                69: 'Left-Cerebellum-Cortex',
                70: 'Right-Cerebellum-Cortex'
            }
        
        self.plot_save_dir = f"visualization/R2_plot/SyN"
        if self.dice_type == 'dice35':
            self.plot_save_dir = f"visualization/R2_plot_35/SyN"
        elif self.dice_type == 'dice70':
            self.plot_save_dir = f"visualization/R2_plot_70/SyN"
        os.makedirs(self.plot_save_dir, exist_ok=True)

        if self.dice_type == 'dice35':
            temp_seg = nib.load('data/mni152_label.nii').get_fdata()
        else:
            temp_seg = nib.load(f'{self.label_path}/template_T1w_MRI.nii.gz').get_fdata()
        temp_seg = torch.tensor(temp_seg).unsqueeze(0).unsqueeze(0).cuda()

        origin_SUVrs = [[] for _ in range(self.seg_num)] # for all labels + global
        moved_SUVrs = [[] for _ in range(self.seg_num)] # for all labels + global
        cnt = 0

        for img, template, _, _, _, path in tqdm(self.save_loader):
            # img, template: MR images
            path = path[0]
            sub_name = path.split('/')[-1].split("_")[0] # sub-N
            seg_path = f"{self.label_path}/{sub_name}_T1w_MRI.nii.gz"
            pet_path = f"{self.pet_dir}/core_{sub_name}_FDG_PET.nii.gz"

            if not os.path.exists(seg_path) or not os.path.exists(pet_path):
                print(seg_path, pet_path)
                print("No Segments or PET scans")
                continue
            cnt += 1

            seg = nib.load(seg_path).get_fdata().astype(np.float32)
            seg = torch.tensor(seg).unsqueeze(0).unsqueeze(0).cuda()
            pet = nib.load(pet_path).get_fdata().astype(np.float32)
            pet = torch.tensor(pet).unsqueeze(0).unsqueeze(0).cuda()

            deformed_seg_path = f'{self.data_path}/{path.split("/")[-1].split(".")[0].split("_")[0]}_seg.nii.gz'
            deformed_seg, _ = self.load_single_image(deformed_seg_path, normalize=False)
            deformed_seg = deformed_seg.unsqueeze(0).unsqueeze(0).cuda()

            deformed_pet_path = f'{self.data_path}/{path.split("/")[-1].split(".")[0].split("_")[0]}_warped_PET.nii.gz'
            deformed_pet, _ = self.load_single_image(deformed_pet_path, normalize=False)
            deformed_pet = deformed_pet.unsqueeze(0).unsqueeze(0).cuda()

            # calc reference region (6)
            if self.seg_num == 6:
                ref_num = [6]
            elif self.seg_num == 35:
                ref_num = [6, 24]
            else:
                ref_num = [69, 70]
            origin_suv_ref = self.calc_suv(seg, pet, ref_num)
            moved_suv_ref = self.calc_suv(temp_seg, deformed_pet, ref_num)

            for label in tqdm(self.label_name, position=1, leave=False): # label: 1, 2, 3, 4, 5
                origin_suv = self.calc_suv(seg, pet, label)
                moved_suv = self.calc_suv(temp_seg, deformed_pet, label)
                origin_SUVrs[label-1].append((origin_suv/origin_suv_ref).item())
                moved_SUVrs[label-1].append((moved_suv/moved_suv_ref).item())

            # calculate global
            origin_suv = self.calc_suv(seg, pet, [i for i in self.label_name if i not in ref_num])
            moved_suv = self.calc_suv(temp_seg, deformed_pet, [i for i in self.label_name if i not in ref_num])
            origin_SUVrs[self.seg_num-1].append((origin_suv/origin_suv_ref).item())
            moved_SUVrs[self.seg_num-1].append((moved_suv/moved_suv_ref).item())

        iccs = []

        if self.seg_num == 6:
            # calculate regression parameter
            for idx, label_name in enumerate(['frontal', 'parietal', 'temporal', 'occipital', 'subcortical', 'global']):
                o, m = origin_SUVrs[idx], moved_SUVrs[idx]
                slope, y_inter, r2_value = self.regression_params(o, m)
                icc = self.icc_two_vectors(o, m)
                iccs.append(icc)

                results = ['None', self.log_name, round(icc, 5), round(slope, 5), round(y_inter, 5), round(r2_value, 5)]
                if not os.path.exists(f'{self.csv_subdir}/quant_{label_name}.csv'):
                    header = ['model','log_name','ICC','slope','y-intercept','R2']
                    with open(f'{self.csv_subdir}/quant_{label_name}.csv', mode="w", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow(header)
                self.save_results(f'{self.csv_subdir}/quant_{label_name}.csv', results)

            iccs = np.array(iccs)

            for label, (o, m, icc) in enumerate(zip(origin_SUVrs, moved_SUVrs, iccs)):
                if label == 5:
                    region_name = 'global'
                else:
                    region_name = self.label_name[label+1]
                self.save_plot(o, m, region_name, icc)
            
        else:
            iccs = [0. for _ in range(self.seg_num)]
            # calculate regression parameter
            for idx, label_name in self.label_name.items():
                if idx in ref_num:
                    continue
                o, m = origin_SUVrs[idx-1], moved_SUVrs[idx-1]
                slope, y_inter, r2_value = self.regression_params(o, m)
                icc = self.icc_two_vectors(o, m)
                iccs[idx-1] = icc

                results = ['None', self.log_name, round(icc, 5), round(slope, 5), round(y_inter, 5), round(r2_value, 5)]
                if not os.path.exists(f'{self.csv_subdir}/quant_{label_name}.csv'):
                    header = ['model','log_name','ICC','slope','y-intercept','R2']
                    with open(f'{self.csv_subdir}/quant_{label_name}.csv', mode="w", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow(header)
                self.save_results(f'{self.csv_subdir}/quant_{label_name}.csv', results)
           
            # for global  
            o, m = origin_SUVrs[self.seg_num-1], moved_SUVrs[self.seg_num-1]
            slope, y_inter, r2_value = self.regression_params(o, m)
            icc = self.icc_two_vectors(o, m)
            iccs[self.seg_num-1] = icc

            results = ['None', self.log_name, round(icc, 5), round(slope, 5), round(y_inter, 5), round(r2_value, 5)]
            if not os.path.exists(f'{self.csv_subdir}/quant_global.csv'):
                header = ['model','log_name','ICC','slope','y-intercept','R2']
                with open(f'{self.csv_subdir}/quant_global.csv', mode="w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(header)
            self.save_results(f'{self.csv_subdir}/quant_global.csv', results)

            iccs = np.array(iccs)

            for idx, label_name in self.label_name.items():
                if idx in ref_num:
                    continue
                self.save_plot(origin_SUVrs[idx-1], moved_SUVrs[idx-1], label_name, iccs[idx-1])

            self.save_plot(origin_SUVrs[self.seg_num-1], moved_SUVrs[self.seg_num-1], 'global', iccs[self.seg_num-1])
    
    def save_plot(self, origin, moved, region_name, icc):
        origin = np.asarray(origin, dtype=float)
        moved = np.asarray(moved, dtype=float)
        plt.figure(figsize=(6,6))
        # (1) Scatter plot
        plt.scatter(origin, moved, color='blue', s=50, alpha=0.7, label="SUVr values")

        # (2) Linear regression line
        coef = np.polyfit(origin, moved, 1)
        poly_fn = np.poly1d(coef)
        x_line = np.linspace(origin.min(), origin.max(), 100)
        plt.plot(x_line, poly_fn(x_line), color='red', linewidth=2, label=f"Fit line (ICC={icc:.2f})")

        # (3) y=x 점선
        plt.plot(x_line, x_line, 'k--', linewidth=1.5, label="y=x")

        # 축 레이블 및 범례
        plt.xlabel("SUVr in individual space")
        plt.ylabel("SUVr in template space")
        plt.title(f"Correlation of SUVr {region_name}")
        plt.legend()
        plt.axis("equal")   # x, y 축 스케일 같게
        plt.grid(True, linestyle="--", alpha=0.6)

        plt.savefig(f'{self.plot_save_dir}/{region_name}.png')
        plt.close()
    
    def calc_suv(self, seg, img, labels):
        if type(labels) == int:
            labels = [labels]
        seg = seg.int()
        mask = torch.zeros_like(seg, dtype=torch.bool)
        for l in labels:
            mask |= (seg == l)
        intersection = (mask * img).sum().float()
        return intersection / mask.sum()
    
    def regression_params(self, x, y):
        x = np.asarray(x).reshape(-1,1)
        y = np.asarray(y)
        m = LinearRegression().fit(x, y)
        return m.coef_[0], m.intercept_, m.score(x, y)  # slope, intercept, R²
    
    def icc_matrix(self, X, icc_type="ICC2_1"):
        """
        X: shape [N, k]  (N=피험자 수, k=방법/측정자 수)
        icc_type: "ICC2_1" (absolute agreement), "ICC3_1" (consistency)
        반환: float (ICC 값)
        참고: McGraw & Wong (1996), Shrout & Fleiss 표기
        """
        X = np.asarray(X, dtype=float)
        # NaN이 있는 행은 제거 (complete-case)
        mask = ~np.isnan(X).any(axis=1)
        X = X[mask]
        n, k = X.shape
        if n < 2 or k < 2:
            raise ValueError("피험자>=2, 방법>=2 필요합니다.")

        # 평균들
        mean_subject = X.mean(axis=1, keepdims=True)   # [n,1]
        mean_rater   = X.mean(axis=0, keepdims=True)   # [1,k]
        grand_mean   = X.mean()

        # 제곱합(2원 분산분석)
        ss_total = ((X - grand_mean) ** 2).sum()
        ss_rows  = (k * ((mean_subject - grand_mean) ** 2)).sum()  # subjects 효과
        ss_cols  = (n * ((mean_rater   - grand_mean) ** 2)).sum()  # raters/방법 효과
        ss_err   = ss_total - ss_rows - ss_cols

        # 자유도
        df_rows = n - 1
        df_cols = k - 1
        df_err  = (n - 1) * (k - 1)

        # 평균제곱
        ms_rows = ss_rows / df_rows     # MSR (subjects)
        ms_cols = ss_cols / df_cols     # MSC (raters)
        ms_err  = ss_err  / df_err      # MSE (residual)

        t = icc_type.upper()
        if t in ["ICC2_1", "ICC(2,1)", "A,1", "ICC2"]:
            # Two-way random, absolute agreement, single rater
            icc = (ms_rows - ms_err) / (ms_rows + (k - 1)*ms_err + (k*(ms_cols - ms_err)/n))
        elif t in ["ICC3_1", "ICC(3,1)", "C,1", "ICC3"]:
            # Two-way mixed, consistency, single rater
            icc = (ms_rows - ms_err) / (ms_rows + (k - 1)*ms_err)
        else:
            raise ValueError(f"지원하지 않는 icc_type: {icc_type}")
        return float(icc)

    def icc_two_vectors(self, a, b, icc_type="ICC2_1"):
        """
        편의함수: 두 방법(예: individual vs template)만 있을 때
        a, b: 길이 N의 1D 배열
        """
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        X = np.stack([a, b], axis=1)  # [N,2]
        return self.icc_matrix(X, icc_type=icc_type)


def denormalize_image(normalized_img, img_min, img_max):
    img_min, img_max = img_min.to(normalized_img.device), img_max.to(normalized_img.device)
    return normalized_img * (img_max - img_min) + img_min

def save_deformed_image(deformed_tensor, output_path, img_min, img_max, affine):
    """
    변형된 PyTorch Tensor 이미지를 .nii.gz 형식으로 저장하는 함수.

    Parameters:
    - deformed_tensor: 변형된 3D 텐서 (torch.Tensor) (shape: 1, 1, D, H, W)
    - original_nifti_path: 원본 NIfTI 파일 경로 (메타데이터 유지 목적)
    - output_path: 저장할 NIfTI 파일 경로
    """
    # 1. 텐서를 NumPy 배열로 변환 (CPU로 이동 및 차원 축소)
    deformed_tensor = denormalize_image(deformed_tensor, img_min, img_max)
    deformed_img = deformed_tensor.squeeze().cpu().detach().numpy()

    # 4. 새로운 NIfTI 객체 생성 및 저장
    deformed_nifti = nib.Nifti1Image(deformed_img, affine=affine)
    nib.save(deformed_nifti, output_path)

def measure_blur_laplacian_3d(volume, save_path=None, axis=2):
    scores = []
    for i in range(volume.shape[0]):  # D (axial slices)
        slice_img = volume[i, :, :]
        # slice_img = (slice_img * 255).clip(0, 255).astype(np.uint8)
        slice_img = cv2.normalize(slice_img, None, 0, 1.0, cv2.NORM_MINMAX)
        slice_img = slice_img.astype(np.float32)  # 꼭 float32로!
        lap = cv2.Laplacian(slice_img, cv2.CV_32F)

        # lap = cv2.Laplacian(slice_img.astype(np.float32), cv2.CV_32F)
        scores.append(lap.var())
    # 저장 옵션
    if save_path is not None:
        # save slice 
        slice_index = volume.shape[axis] // 2

        # 슬라이스 선택
        if axis == 0:
            slice_img = volume[slice_index, :, :]
        elif axis == 1:
            slice_img = volume[:, slice_index, :]
        else:
            slice_img = volume[:, :, slice_index]

        slice_img = slice_img.astype(np.float32)
        lap = cv2.Laplacian(slice_img, cv2.CV_32F)

        lap_vis = cv2.normalize(lap, None, 0, 255, cv2.NORM_MINMAX)
        lap_vis = lap_vis.astype(np.uint8)
        cv2.imwrite(save_path, lap_vis)
        print(f"[Saved Laplacian] {save_path}")

    return np.mean(scores)

def measure_blur_tenengrad_3d(volume, save_path=None, axis=2):
    scores = []
    for i in range(volume.shape[0]):
        slice_img = volume[i, :, :]
        gx = cv2.Sobel(slice_img.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(slice_img.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
        grad_magnitude = np.sqrt(gx ** 2 + gy ** 2)
        scores.append(np.mean(grad_magnitude))

    # 저장 옵션
    if save_path is not None:
        # save slice 
        slice_index = volume.shape[axis] // 2

        # 슬라이스 선택
        if axis == 0:
            slice_img = volume[slice_index, :, :]
        elif axis == 1:
            slice_img = volume[:, slice_index, :]
        else:
            slice_img = volume[:, :, slice_index]

        slice_img = slice_img.astype(np.float32)
        gx = cv2.Sobel(slice_img, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(slice_img, cv2.CV_32F, 0, 1, ksize=3)
        grad_magnitude = np.sqrt(gx ** 2 + gy ** 2)

        norm_mag = cv2.normalize(grad_magnitude, None, 0, 255, cv2.NORM_MINMAX)
        cv2.imwrite(save_path, norm_mag.astype(np.uint8))
        print(f"[Saved Sobel] {save_path}")

    return np.mean(scores)

def measure_blur_fft_3d(volume, save_path=None, axis=2):
    scores = []
    for i in range(volume.shape[0]):
        slice_img = volume[i, :, :]
        f = np.fft.fft2(slice_img)
        fshift = np.fft.fftshift(f)
        # magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-8)
        magnitude_spectrum = np.abs(fshift)

        h, w = magnitude_spectrum.shape
        center = (h // 2, w // 2)
        radius = min(h, w) // 4
        mask = np.zeros_like(magnitude_spectrum)
        mask = mask.copy()

        cv2.circle(mask, center, radius, 1, thickness=-1)
        high_freq = magnitude_spectrum * (1 - mask)
        scores.append(np.mean(high_freq))

    # 저장 옵션
    if save_path is not None:
        # save slice 
        slice_index = volume.shape[axis] // 2

        # 슬라이스 선택
        if axis == 0:
            slice_img = volume[slice_index, :, :]
        elif axis == 1:
            slice_img = volume[:, slice_index, :]
        else:
            slice_img = volume[:, :, slice_index]

        f = np.fft.fft2(slice_img)
        fshift = np.fft.fftshift(f)
        # magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-8)

        h, w = fshift.shape
        # center = (h // 2, w // 2)
        center = (w//2, h//2)
        radius = min(h, w) // 4
        mask = np.zeros(fshift.shape, dtype=np.float32)

        cv2.circle(mask, center, radius, 1, thickness=-1)
        high_freq = fshift * (1 - mask)

        # 역 FFT로 복원
        f_ishift = np.fft.ifftshift(high_freq)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)  # 복소수이므로 절댓값

        # 저장용 정규화
        img_vis = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
        cv2.imwrite(save_path, img_vis.astype(np.uint8))
        print(f"[Saved High-freq] {save_path}")


    return np.mean(scores)

