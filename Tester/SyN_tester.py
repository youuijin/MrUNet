from utils.dataset import set_dataloader, set_dataloader_usingcsv, set_paired_dataloader_usingcsv
import csv, torch, os
from tqdm import tqdm

import cv2

from utils.loss import MSE_loss, NCC_loss, SSIM_loss

import nibabel as nib
import numpy as np

class SyN_tester:
    def __init__(self, model_path, args):
        # check already tested)
        self.log_name = model_path
        self.pair_test = args.pair_test
        
        self.csv_dir = args.csv_dir
        self.test_dataset = args.dataset
        if self.test_dataset == 'OASIS':
            self.data_path = 'data/OASIS_SyN'
            self.original_path = 'data/OASIS_brain_core_percent'
            self.label_path = 'data/OASIS_label_core'
        elif self.test_dataset == 'DLBS':
            self.data_path = 'data/DLBS_SyN'
            self.original_path = 'data/DLBS_core_percent'
            self.label_path = 'data/DLBS_label_35'
        self.test_method = args.test_method

        if args.pair_test:
            _, _, self.save_loader = set_paired_dataloader_usingcsv(self.test_dataset, 'data/data_list', batch_size=1, return_path=True, numpy=False)
            self.data_path += "_paired"
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
        
    def test_blur(self):
        self.save_dir = f'visualization/{self.test_dataset}/avg_template'
        self.csv_path = f'{self.csv_dir}/{self.test_dataset}/blur_results.csv'
        
        tested = self.check_tested(self.log_name)
        self.already_tested = False
        if tested:
            self.already_tested = True
            return
        
        aff = None
        deformed_imgs = []  # 리스트에 저장

        for _, _, _, _, _, path in tqdm(self.save_loader):
            path = path[0]
            img_path = f'{self.data_path}/{path.split("/")[-1].split(".")[0]}_warped.nii.gz'
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
        self.csv_path = f'{self.csv_dir}/{self.test_dataset}/dice_results.csv'
        
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
                seg_path = f'{self.data_path}/{path.split("/")[-1].split(".")[0]}_seg.nii.gz'
            if not os.path.exists(seg_path):
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
        self.csv_path = f'{self.csv_dir}/{self.test_dataset}/similar_results.csv'
        mse, ncc, ssim = [], [], []

        for _, template, _, _, _, path in tqdm(self.save_loader):
            template = template.unsqueeze(1).cuda()
            if self.pair_test:
                img_path, template_path = path[0][0].split("/")[-1].split(".")[0], path[1][0].split("/")[-1].split(".")[0]
                img_path = f'{self.data_path}/{img_path}_{template_path}_warped.nii.gz'
                template = nib.load(f"{self.original_path}/{template_path}.nii.gz").get_fdata().astype(np.float32)
                template = torch.tensor(template).unsqueeze(0).unsqueeze(0).cuda()
            else:
                path = path[0]
                img_path = f'{self.data_path}/{path.split("/")[-1].split(".")[0]}_warped.nii.gz'
            
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
        self.csv_path = f'{self.csv_dir}/{self.test_dataset}/folding_results.csv'
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

