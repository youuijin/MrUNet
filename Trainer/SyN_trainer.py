# import ants
import ants
from tqdm import tqdm
import os
from utils.dataset import set_paired_dataloader_usingcsv, set_dataloader_usingcsv

from utils.utils import *

def register_lddmm(fixed_img, moving_img, moving_seg=None):
    """
    LDDMM (Diffeomorphic Metric Mapping) 수행
    """
    # LDDMM 적용 (Elastic 또는 Diffeomorphic 옵션 사용)
    # transform = ants.registration(fixed=fixed_img, moving=moving_img, type_of_transform='Elastic', flow_sigma=5, total_sigma=1)
    transform = ants.registration(
        fixed=fixed_img, moving=moving_img, type_of_transform='SyN',
        # flow_sigma=5, total_sigma=1,  # 속도장 정칙화 강화
        # reg_iterations=(1, 1, 1)  # 최적화 반복 횟수 증가
    )

    # 변형된 이미지
    warped_img = transform['warpedmovout']

    warped_seg = None
    if moving_seg is not None:
        warped_seg = ants.apply_transforms(
            fixed=fixed_img, moving=moving_seg, transformlist=transform['fwdtransforms'], interpolator='nearestNeighbor'
        )

    return warped_img, warped_seg, transform['fwdtransforms']

def save_nifti(data, path, affine):
    nii_img = nib.Nifti1Image(data, affine)
    nib.save(nii_img, path)
    
def save_numpy(data, path):
    np.save(path, data)

def get_image(image_path, normalize=True, return_info=False):
    img = nib.load(image_path)
    affine = img.affine
    img = img.get_fdata()
    m = img.min()
    M = img.max()
    
    if normalize:
        img = (img - m) / (M - m)  # Normalize to [0,1]
    
    if return_info:
        return torch.tensor(img, dtype=torch.float32), m, M, affine

    return torch.tensor(img, dtype=torch.float32)

def create_identity_grid(image):
    shape = image.shape
    spacing = image.spacing
    origin = image.origin
    direction = image.direction

    dz, dy, dx = shape
    z = np.arange(dz) * spacing[2] + origin[2]
    y = np.arange(dy) * spacing[1] + origin[1]
    x = np.arange(dx) * spacing[0] + origin[0]

    grid = np.stack(np.meshgrid(x, y, z, indexing='ij'), axis=-1)  # (D, H, W, 3)

    # direction 적용 (optional)
    grid = np.einsum('ij,xyzj->xyzi', direction, grid)

    return grid.astype(np.float32)


# apply optimization
def optimize(dataset, image_paths, seg_paths, save_dir, template_path):
    _, _, save_loader = set_dataloader_usingcsv(dataset, 'data/data_list', template_path, batch_size=1, return_path=True, numpy=False)
    os.makedirs(save_dir, exist_ok=True)

    for _, _, _, _, _, path in tqdm(save_loader):
        img_path = path[0]
        img_name = img_path.split('/')[-1] # file name
        save_name = img_name.split('.')[0] #moving
        seg_name = f'{save_name.split("_")[1]}_T1w_MRI.nii.gz'

        moving_ants = ants.image_read(img_path)
        fixed_ants = ants.image_read(template_path)
        
        if os.path.exists(f'{seg_paths}/{seg_name}'):
            moving_seg = ants.image_read(f'{seg_paths}/{seg_name}')
        else:
            moving_seg = None
        
        # LDDMM (SyN)
        warped_ants, warped_seg, transform = register_lddmm(fixed_ants, moving_ants, moving_seg)

        warped_ants.to_filename(f"{save_dir}/{save_name}_warped.nii.gz")
        if warped_seg is not None:
            warped_seg.to_filename(f"{save_dir}/{save_name}_seg.nii.gz")
        
        # deformation field
        # 변형장(Deformation Field) 저장
        deformation_field = ants.image_read(transform[0])
        deformation_field.to_filename(f'{save_dir}/{save_name}_deform.nii.gz')

        # displacement field
        identity = create_identity_grid(fixed_ants)
        displacement_field = deformation_field.numpy() - identity

        np.save(f'{save_dir}/{save_name}_displace.npy', displacement_field)

# apply optimization
def paired_optimize(dataset, image_paths, seg_paths, save_dir):
    _, _, save_loader = set_paired_dataloader_usingcsv(dataset, 'data/data_list', batch_size=1, return_path=True, numpy=False)
    os.makedirs(save_dir, exist_ok=True)

    for _, _, _, _, _, path in tqdm(save_loader):
        image_paths = "/".join(path[0][0].split('/')[:-1])
        img_path, template_path = path[0][0], path[1][0]
        img_name, template_name = img_path.split('/')[-1], template_path.split('/')[-1]
        save_name = f"{img_name.split('.')[0]}" #moving_fixed

        moving_ants = ants.image_read(f'{image_paths}/{img_name}')
        fixed_ants = ants.image_read(f'{image_paths}/{template_name}')
        
        if os.path.exists(f'{seg_paths}/{img_name}'):
            moving_seg = ants.image_read(f'{seg_paths}/{img_name}')
        else:
            moving_seg = None
        
        # LDDMM (SyN)
        warped_ants, warped_seg, transform = register_lddmm(fixed_ants, moving_ants, moving_seg)

        warped_ants.to_filename(f"{save_dir}/{save_name}_warped.nii.gz")
        if warped_seg is not None:
            warped_seg.to_filename(f"{save_dir}/{save_name}_seg.nii.gz")
        
        # deformation field
        # 변형장(Deformation Field) 저장
        deformation_field = ants.image_read(transform[0])
        deformation_field.to_filename(f'{save_dir}/{save_name}_deform.nii.gz')

        # displacement field
        identity = create_identity_grid(fixed_ants)
        displacement_field = deformation_field.numpy() - identity

        np.save(f'{save_dir}/{save_name}_displace.npy', displacement_field)
        
if __name__ == '__main__':
    # Example usage
    set_seed(seed=0)
    # dataset = 'DLBS'
    # root_dir = "data/DLBS_core_percent"
    # seg_dir = "data/DLBS_label_35"
    # save_dir = "data/DLBS_SyN_paired"
    # template_path = "data/mni152_resample.nii"
    # optimize(dataset, root_dir, seg_dir, save_dir, template_path)

    dataset = 'FDG_PET'
    root_dir = "data/FDG_PET"
    seg_dir = "data/FDG_label"
    save_dir = "data/FDG_PET_SyN_MRtemplate"
    template_path = "data/mni152_resample.nii"
    optimize(dataset, root_dir, seg_dir, save_dir, template_path)

    dataset = 'FDG_PET'
    root_dir = "data/FDG_PET"
    seg_dir = "data/FDG_label"
    save_dir = "data/FDG_PET_SyN_PETtemplate"
    template_path = "data/core_MNI152_PET_1mm.nii"
    optimize(dataset, root_dir, seg_dir, save_dir, template_path)