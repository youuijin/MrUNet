import torch, os
import nibabel as nib
from torch.utils.data import random_split
from torch.utils.data import Dataset, DataLoader
import numpy as np

def set_datapath(dataset):
    if dataset == 'DLBS':
        return './data/DLBS_core_percent', None
    elif dataset == 'OASIS':
        return './data/OASIS_brain_core_percent', None
    elif dataset == 'LUMIR':
        return './data/LUMIR_train', './data/LUMIR_val'

def set_dataloader(image_paths, template_path, batch_size, numpy=True, return_path=False, return_mask=False, mask_path=None):
    dataset = MedicalImageDataset(image_paths, template_path, return_path=return_path, return_mask=return_mask, mask_path=mask_path)
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
    def __init__(self, root_dir, template_path, mask_path=None, transform=None, return_path=False, return_mask=False):
        if return_mask and mask_path is None:
            return ValueError('If you want to use brain mask, Enter mask path.')
        
        self.image_paths = [f'{root_dir}/{f}' for f in os.listdir(root_dir)]
        self.mask_path = mask_path
        template = nib.load(template_path).get_fdata()
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

        self.numpy = False # TODO: delete or implement

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        if self.numpy:
            img = np.load(f"{self.image_paths[idx]}/data.npy")
            affine = np.load(f"{self.image_paths[idx]}/affine.npy")
        else:
            img = nib.load(self.image_paths[idx])
            affine = img.affine
            img = img.get_fdata()

        if self.return_mask:
            mask = nib.load(f'{self.mask_path}/{self.image_paths[idx].split("/")[-1]}').get_fdata()

        img_min, img_max = img.min(), img.max()
        img = (img - img_min) / (img_max - img_min)  # Normalize to [0,1]#
        
        if self.transform is not None:
            img = self.transform(img)

        if self.return_path:
            if self.return_mask:
                return torch.tensor(img, dtype=torch.float32), torch.tensor(self.template, dtype=torch.float32), img_min, img_max, affine, self.image_paths[idx], mask
            return torch.tensor(img, dtype=torch.float32), torch.tensor(self.template, dtype=torch.float32), img_min, img_max, affine, self.image_paths[idx]
 
        return torch.tensor(img, dtype=torch.float32), torch.tensor(self.template, dtype=torch.float32), img_min, img_max, affine

# Load pair-wise data
import random

def set_paired_dataloader(train_dir, val_dir=None, batch_size=1):
    if val_dir is None:
        all_paths = sorted([
            os.path.join(train_dir, f)
            for f in os.listdir(train_dir)
            if f.endswith(".nii") or f.endswith(".nii.gz")
        ])
        dataset_size = len(all_paths)
        train_size = int(dataset_size * 0.8)  # 80%
        val_size = int(dataset_size * 0.1)    # 10%
        test_size = dataset_size - train_size - val_size  # remain

        train_image_paths = all_paths[:train_size]
        val_image_paths = all_paths[train_size:train_size+val_size]

        if len(val_image_paths) % 2:
            val_image_paths = val_image_paths[:-1]
    else:
        train_image_paths = [
            os.path.join(train_dir, f)
            for f in os.listdir(train_dir)
            if f.endswith(".nii") or f.endswith(".nii.gz")
        ]
        val_image_paths = sorted([
            os.path.join(val_dir, f)
            for f in os.listdir(val_dir)
            if f.endswith(".nii") or f.endswith(".nii.gz")
        ])
    
    train_dataset = RandomInterPatientDataset(train_image_paths)
    val_dataset = FixedPairDataset(val_image_paths)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    save_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    return train_loader, val_loader, save_loader

class RandomInterPatientDataset(Dataset):
    def __init__(self, image_paths):
        """
        image_dir: directory of all training images (e.g., NIfTI)
        num_pairs_per_epoch: how many random pairs to draw per epoch
        """
        self.image_paths = image_paths
        self.num_pairs = len(self.image_paths)

    def __len__(self):
        return self.num_pairs

    def __getitem__(self, idx):
        idx1, idx2 = random.sample(range(len(self.image_paths)), 2)
        img1 = self.load_image(self.image_paths[idx1])
        img2 = self.load_image(self.image_paths[idx2])
        return img1, img2, 0, 0, 0

    def load_image(self, path):
        img = nib.load(path).get_fdata()
        # img = torch.from_numpy(img).float().unsqueeze(0)  # shape: [1, D, H, W]

        img_min, img_max = img.min(), img.max()
        img = (img - img_min) / (img_max - img_min)  # Normalize to [0,1]#
        return torch.tensor(img, dtype=torch.float32)

class FixedPairDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths

        assert len(self.image_paths) % 2 == 0, "Number of images must be even to form pairs"

        self.pairs = [
            (self.image_paths[i], self.image_paths[i + 1])
            for i in range(0, len(self.image_paths), 2)
        ]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        moving_path, fixed_path = self.pairs[idx]
        moving = self.load_image(moving_path)
        fixed = self.load_image(fixed_path)
        return moving, fixed, 0, 0, 0

    def load_image(self, path):
        img = nib.load(path).get_fdata()
        # img = torch.from_numpy(img).float().unsqueeze(0)

        img_min, img_max = img.min(), img.max()
        img = (img - img_min) / (img_max - img_min)  # Normalize to [0,1]#
        return torch.tensor(img, dtype=torch.float32)

import csv

def set_dataloader_usingcsv(dataset, csv_dir, template_path, batch_size, numpy=True, return_path=False, return_mask=False, mask_path=None):
    train_dataset = MedicalImageDatasetCSV(f'{csv_dir}/{dataset}_train.csv', template_path, return_path=return_path, return_mask=return_mask, mask_path=mask_path)
    val_dataset = MedicalImageDatasetCSV(f'{csv_dir}/{dataset}_valid.csv', template_path, return_path=return_path, return_mask=return_mask, mask_path=mask_path)

    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    save_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    return train_loader, val_loader, save_loader

# Define dataset class
class MedicalImageDatasetCSV(Dataset):
    def __init__(self, csv_path, template_path, mask_path=None, transform=None, return_path=False, return_mask=False):
        if return_mask and mask_path is None:
            return ValueError('If you want to use brain mask, Enter mask path.')
        
        with open(csv_path, newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)
            rows = list(reader)
        self.image_paths = [s[0] for s in rows]

        self.mask_path = mask_path
        template = nib.load(template_path).get_fdata()
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
        img = nib.load(self.image_paths[idx])
        affine = img.affine
        img = img.get_fdata()

        if self.return_mask:
            mask = nib.load(f'{self.mask_path}/{self.image_paths[idx].split("/")[-1]}').get_fdata()

        img_min, img_max = img.min(), img.max()
        img = (img - img_min) / (img_max - img_min)  # Normalize to [0,1]#
        
        if self.transform is not None:
            img = self.transform(img)

        if self.return_path:
            if self.return_mask:
                return torch.tensor(img, dtype=torch.float32), torch.tensor(self.template, dtype=torch.float32), img_min, img_max, affine, self.image_paths[idx], mask
            return torch.tensor(img, dtype=torch.float32), torch.tensor(self.template, dtype=torch.float32), img_min, img_max, affine, self.image_paths[idx]
 
        return torch.tensor(img, dtype=torch.float32), torch.tensor(self.template, dtype=torch.float32), img_min, img_max, affine
