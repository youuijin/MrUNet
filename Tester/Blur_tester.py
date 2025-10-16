from Tester.Tester_base import Tester
from utils.utils import apply_deformation_using_disp
import torch, cv2, os, csv
import nibabel as nib
import numpy as np

from tqdm import tqdm

class Blur_Tester(Tester):
    def __init__(self, model_path, args):
        self.set_dataset(args)
        self.csv_path = f'{self.csv_dir}/blur_results.csv'
        if not os.path.exists(self.csv_path):
            os.makedirs(self.csv_dir, exist_ok=True)
            with open(self.csv_path, 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(['model','log_name','std','laplacian','tenegrad','fft'])
        self.save_dir = f'visualization/{self.train_dataset}/avg_template'
        os.makedirs(self.save_dir, exist_ok=True)
        super().__init__(model_path, args)

    def test(self):
        aff = None
        deformed_imgs = []  # 리스트에 저장

        for img, template, _, _, affine in tqdm(self.save_loader):
            img, template = img.unsqueeze(1).cuda(), template.unsqueeze(1).cuda() # [B, D, H, W] -> [B, 1, D, H, W]
            stacked_input = torch.cat([img, template], dim=1) # [B, 2, D, H, W]
            aff = affine.squeeze()

            disp = self.model(stacked_input)[0][-1]
            if 'diff' in self.method:
                disp = self.integrate(disp)
            
            deformed_img = apply_deformation_using_disp(img, disp)

            # tot_warped += deformed_img.squeeze()
            deformed_imgs.append(deformed_img.squeeze(1).cpu())

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

        results = [self.model_type, self.log_name, round(overall_variance, 5), round(laplacian,5), round(tenengrad,5), round(fft, 5)]
        self.save_results(self.csv_path, results)

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

