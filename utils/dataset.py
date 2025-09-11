import torch, os
import nibabel as nib
from torch.utils.data import random_split
from torch.utils.data import Dataset, DataLoader
import numpy as np

def set_datapath(dataset, numpy):
    # Update: use numpy file
    if dataset == 'DLBS':
        if numpy:
            return './data/DLBS_numpy', None
        else:
            return './data/DLBS_core_percent', None
    elif dataset == 'OASIS':
        if numpy:
            return './data/OASIS_numpy', None
        else:
            return './data/OASIS_brain_core_percent', None
    elif dataset == 'LUMIR':
        return './data/LUMIR_train', './data/LUMIR_val'
    elif dataset == 'FDG_MRI':
        if numpy:
            return './data/FDG_MRI_numpy', None
        else:
            return './data/FDG_MRI_brain_percent', None #TODO: upload
    elif dataset == 'FDG_PET':
        if numpy:
            return './data/FDG_PET_percent_numpy', None
        else:
            return './data/FDG_PET_percent', None

def set_dataloader(image_paths, template_path, batch_size, numpy=True, return_path=False, return_mask=False, mask_path=None):
    dataset = MedicalImageDataset(image_paths, template_path, numpy=numpy, return_path=return_path, return_mask=return_mask, mask_path=mask_path)
    dataset_size = len(dataset)
    train_size = int(dataset_size * 0.8)  # 80%
    val_size = int(dataset_size * 0.1)    # 10%
    test_size = dataset_size - train_size - val_size  # remain

    # split dataset
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )

    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    save_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    return train_loader, val_loader, save_loader

# Define dataset class
class MedicalImageDataset(Dataset):
    def __init__(self, root_dir, template_path, numpy=True, mask_path=None, transform=None, return_path=False, return_mask=False):
        if return_mask and mask_path is None:
            return ValueError('If you want to use brain mask, Enter mask path.')
        
        self.image_paths = [f'{root_dir}/{f}' for f in os.listdir(root_dir)]
        self.mask_path = mask_path
        self.numpy = numpy
        if numpy:
            template = np.load(template_path)
        else:
            template = nib.load(template_path).get_fdata().astype(np.float32)
        
        # Template normalize - percentile
        t_data = template.flatten()
        p1_temp = np.percentile(t_data, 1)
        p99_temp = np.percentile(t_data, 99)
        template = np.clip(template, p1_temp, p99_temp)
        
        template_min, template_max = template.min(), template.max()
        self.template = (template - template_min) / (template_max - template_min)

        self.transform = transform
        self.return_path = return_path
        self.return_mask = return_mask

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        if self.numpy:
            img = np.load(f"{self.image_paths[idx]}")
            affine = np.load(f"data/affine.npy")
        else:
            img = nib.load(self.image_paths[idx])
            affine = img.affine
            img = img.get_fdata().astype(np.float32)

        if self.return_mask:
            mask = nib.load(f'{self.mask_path}/{self.image_paths[idx].split("/")[-1]}').get_fdata().astype(np.float32)

        img_min, img_max = img.min(), img.max()
        img = (img - img_min) / (img_max - img_min)  # Normalize to [0,1]#
        
        if self.transform is not None:
            img = self.transform(img)

        if self.return_path:
            if self.return_mask:
                return torch.from_numpy(img), torch.from_numpy(self.template), img_min, img_max, affine, self.image_paths[idx], mask
            return torch.from_numpy(img), torch.from_numpy(self.template), img_min, img_max, affine, self.image_paths[idx]
 
        return torch.from_numpy(img), torch.from_numpy(self.template), img_min, img_max, affine

# Load pair-wise data
import random

def set_paired_dataloader(train_dir, val_dir=None, batch_size=1, numpy=True):
    def get_image_paths(directory):
        if numpy:
            return sorted([
                os.path.join(directory, f)
                for f in os.listdir(directory)
                if f.endswith(".npy")
            ])
        else:
            return sorted([
                os.path.join(directory, f)
                for f in os.listdir(directory)
                if f.endswith(".nii") or f.endswith(".nii.gz")
            ])

    if val_dir is None:
        all_paths = get_image_paths(train_dir)
        dataset_size = len(all_paths)
        train_size = int(dataset_size * 0.8)
        val_size = int(dataset_size * 0.1)

        train_image_paths = all_paths[:train_size]
        val_image_paths = all_paths[train_size:train_size+val_size]

        if len(val_image_paths) % 2:
            val_image_paths = val_image_paths[:-1]
    else:
        train_image_paths = get_image_paths(train_dir)
        val_image_paths = get_image_paths(val_dir)
    
    train_dataset = RandomInterPatientDataset(train_image_paths, numpy=numpy)
    val_dataset = FixedPairDataset(val_image_paths, numpy=numpy)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    save_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    return train_loader, val_loader, save_loader

class RandomInterPatientDataset(Dataset):
    def __init__(self, image_paths, numpy=True, return_path=False):
        """
        image_dir: directory of all training images (e.g., NIfTI)
        num_pairs_per_epoch: how many random pairs to draw per epoch
        """
        self.image_paths = image_paths
        self.num_pairs = len(self.image_paths)
        self.numpy = numpy
        self.return_path = return_path

    def __len__(self):
        return self.num_pairs

    def __getitem__(self, idx):
        idx1, idx2 = random.sample(range(len(self.image_paths)), 2)
        img1, affine = self.load_image(self.image_paths[idx1])
        img2, affine = self.load_image(self.image_paths[idx2])
        if self.return_path:
            return img1, img2, 0, 0, affine, [self.image_paths[idx1], self.image_paths[idx2]]
        return img1, img2, 0, 0, affine

    def load_image(self, path):
        if self.numpy:
            img = np.load(path)
            affine = np.load('data/affine.npy')
        else:
            nifti = nib.load(path)
            img = nifti.get_fdata().astype(np.float32)
            affine = nifti.affine

        img_min, img_max = img.min(), img.max()
        img = (img - img_min) / (img_max - img_min)  # Normalize to [0,1]#
        return torch.from_numpy(img), affine

class FixedPairDataset(Dataset):
    def __init__(self, image_paths, numpy=True, return_path=False):
        self.image_paths = image_paths
        self.return_path = return_path
        self.numpy = numpy

        assert len(self.image_paths) % 2 == 0, "Number of images must be even to form pairs"

        self.pairs = [
            (self.image_paths[i], self.image_paths[i + 1])
            for i in range(0, len(self.image_paths), 2)
        ]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        moving_path, fixed_path = self.pairs[idx]
        moving, affine = self.load_image(moving_path)
        fixed, affine = self.load_image(fixed_path)
        if self.return_path:
            return moving, fixed, 0, 0, affine, [moving_path, fixed_path]
        return moving, fixed, 0, 0, affine

    def load_image(self, path):
        if self.numpy:
            img = np.load(path)
            affine = np.load('data/affine.npy')
        else:
            nifti = nib.load(path)
            img = nifti.get_fdata().astype(np.float32)
            affine = nifti.affine

        img_min, img_max = img.min(), img.max()
        img = (img - img_min) / (img_max - img_min)  # Normalize to [0,1]#
        return torch.from_numpy(img), affine

import csv

def set_dataloader_usingcsv(dataset, csv_dir, template_path, batch_size, numpy=True, return_path=False, return_mask=False, mask_path=None):
    if numpy:
        train_file = f'{csv_dir}/{dataset}/{dataset}_train_numpy.csv'
        valid_file = f'{csv_dir}/{dataset}/{dataset}_valid_numpy.csv'
    else:
        train_file = f'{csv_dir}/{dataset}/{dataset}_train.csv'
        valid_file = f'{csv_dir}/{dataset}/{dataset}_valid.csv'
    train_dataset = MedicalImageDatasetCSV(train_file, template_path, numpy=numpy, return_path=return_path, return_mask=return_mask, mask_path=mask_path)
    val_dataset = MedicalImageDatasetCSV(valid_file, template_path, numpy=numpy, return_path=return_path, return_mask=return_mask, mask_path=mask_path)

    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    save_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    return train_loader, val_loader, save_loader

# Define dataset class
class MedicalImageDatasetCSV(Dataset):
    def __init__(self, csv_path, template_path, numpy=True, mask_path=None, transform=None, return_path=False, return_mask=False):
        if return_mask and mask_path is None:
            return ValueError('If you want to use brain mask, Enter mask path.')
        
        with open(csv_path, newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)
            rows = list(reader)
        self.image_paths = [s[0] for s in rows]
        self.mask_path = mask_path
        self.numpy = numpy
        if numpy:
            template = np.load(template_path)
        else:
            template = nib.load(template_path).get_fdata().astype(np.float32)
        
        # Template normalize - percentile
        t_data = template.flatten()
        p1_temp = np.percentile(t_data, 1)
        p99_temp = np.percentile(t_data, 99)
        template = np.clip(template, p1_temp, p99_temp)
        
        template_min, template_max = template.min(), template.max()
        self.template = (template - template_min) / (template_max - template_min)

        self.transform = transform
        self.return_path = return_path
        self.return_mask = return_mask

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        if self.numpy:
            img = np.load(self.image_paths[idx])
            affine = np.load('data/affine.npy')
        else:
            img = nib.load(self.image_paths[idx])
            affine = img.affine
            img = img.get_fdata().astype(np.float32)

        if self.return_mask:
            mask = nib.load(f'{self.mask_path}/{self.image_paths[idx].split("/")[-1]}').get_fdata().astype(np.float32)

        img_min, img_max = img.min(), img.max()
        img = (img - img_min) / (img_max - img_min)  # Normalize to [0,1]#
        
        if self.transform is not None:
            img = self.transform(img)

        if self.return_path:
            if self.return_mask:
                return torch.from_numpy(img), torch.from_numpy(self.template), img_min, img_max, affine, self.image_paths[idx], mask
            return torch.from_numpy(img), torch.from_numpy(self.template), img_min, img_max, affine, self.image_paths[idx]
 
        return torch.from_numpy(img), torch.from_numpy(self.template), img_min, img_max, affine

def set_paired_dataloader_usingcsv(dataset, csv_dir, batch_size=1, numpy=True, return_path=False, return_mask=False, mask_path=None):
    if numpy:
        train_file = f'{csv_dir}/{dataset}/{dataset}_train_numpy.csv'
        valid_file = f'{csv_dir}/{dataset}/{dataset}_valid_pair_numpy.csv'
    else:
        train_file = f'{csv_dir}/{dataset}/{dataset}_train.csv'
        valid_file = f'{csv_dir}/{dataset}/{dataset}_valid_pair.csv'

    with open(train_file, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)
        rows = list(reader)
    train_image_paths = [s[0] for s in rows]
    train_dataset = RandomInterPatientDataset(train_image_paths, numpy=numpy, return_path=return_path)
    
    with open(valid_file, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)
        valid_image_paths = list(reader)
    val_dataset = FixedPairCSVDataset(valid_image_paths, numpy=numpy, return_path=return_path)

    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    save_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    return train_loader, val_loader, save_loader

class FixedPairCSVDataset(Dataset):
    def __init__(self, image_paths, numpy=True, return_path=False):
        self.pairs = image_paths
        self.numpy = numpy
        self.return_path = return_path

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        moving_path, fixed_path = self.pairs[idx]
        moving, affine = self.load_image(moving_path)
        fixed, affine = self.load_image(fixed_path)
        if self.return_path:
            return moving, fixed, 0, 0, affine, [moving_path, fixed_path]
        return moving, fixed, 0, 0, affine

    def load_image(self, path):
        if self.numpy:
            img = np.load(path)
            affine = np.load('data/affine.npy')
        else:
            nifti = nib.load(path)
            img = nifti.get_fdata().astype(np.float32)
            affine = nifti.affine

        img_min, img_max = img.min(), img.max()
        img = (img - img_min) / (img_max - img_min)  # Normalize to [0,1]#
        return torch.tensor(img, dtype=torch.float32), affine

if __name__ == '__main__':
    dataset = 'DLBS'
    csv_dir = 'data/data_list'
    set_paired_dataloader_usingcsv(dataset, csv_dir)