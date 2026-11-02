import os
import json
import random
from PIL import Image
import numpy as np

import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

from .boundary_utils import extract_boundary_tensor, compute_lesion_scale_category


class TN3KDataset(Dataset):
    """
    TN3K Thyroid Ultrasound Dataset loader.
    Supports official 5-fold cross validation on trainval set, and official held-out test set.
    Generates online boundary ground-truth and lesion-scale categorization.
    """
    def __init__(self, data_root, split='train', fold=0, img_size=(256, 256), augment=True):
        super().__init__()
        self.data_root = data_root
        self.split = split.lower()
        self.fold = fold
        self.img_size = img_size
        self.augment = augment and (self.split == 'train')

        self.samples = []
        self._load_split_samples()

    def _find_file(self, folder, base_name):
        """Finds the file in folder regardless of whether base_name has an extension."""
        for ext in ['', '.jpg', '.png', '.jpeg', '.bmp']:
            candidate = os.path.join(folder, base_name + ext if not base_name.endswith(ext) else base_name)
            if os.path.exists(candidate):
                return candidate
        # Fallback: check matching basename
        clean_name = os.path.splitext(base_name)[0]
        for f in os.listdir(folder):
            if os.path.splitext(f)[0] == clean_name:
                return os.path.join(folder, f)
        return None

    def _load_split_samples(self):
        if self.split in ['train', 'val']:
            img_dir = os.path.join(self.data_root, 'trainval-image')
            mask_dir = os.path.join(self.data_root, 'trainval-mask')
            fold_file = os.path.join(self.data_root, f'tn3k-trainval-fold{self.fold}.json')

            if not os.path.exists(fold_file):
                # Fallback if specific fold JSON is not found: deterministic 80/20 train/val split
                all_images = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
                random.seed(42 + self.fold)
                shuffled = all_images.copy()
                random.shuffle(shuffled)
                val_count = int(0.2 * len(shuffled))
                target_files = shuffled[val_count:] if self.split == 'train' else shuffled[:val_count]
            else:
                with open(fold_file, 'r') as f:
                    fold_data = json.load(f)
                if isinstance(fold_data, dict):
                    # Try keys 'train'/'val' or 'train_set'/'val_set'
                    keys = [k for k in fold_data.keys() if self.split in k.lower()]
                    target_files = fold_data[keys[0]] if keys else fold_data.get('train' if self.split == 'train' else 'val', [])
                elif isinstance(fold_data, list):
                    target_files = fold_data
                else:
                    target_files = []

            for item in target_files:
                name = item if isinstance(item, str) else str(item)
                img_path = self._find_file(img_dir, name)
                mask_path = self._find_file(mask_dir, name)
                if img_path and mask_path:
                    self.samples.append((img_path, mask_path, os.path.basename(img_path)))

        elif self.split == 'test':
            img_dir = os.path.join(self.data_root, 'test-image')
            mask_dir = os.path.join(self.data_root, 'test-mask')
            if not os.path.exists(img_dir):
                raise FileNotFoundError(f"Test directory not found: {img_dir}")

            all_images = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
            for img_name in all_images:
                img_path = os.path.join(img_dir, img_name)
                mask_path = self._find_file(mask_dir, img_name)
                if mask_path:
                    self.samples.append((img_path, mask_path, img_name))

        if len(self.samples) == 0:
            raise RuntimeError(f"No samples loaded for split '{self.split}' under {self.data_root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path, sample_id = self.samples[idx]

        # 1. Load image and mask
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        # 2. Resize
        image = image.resize(self.img_size, resample=Image.BILINEAR)
        mask = mask.resize(self.img_size, resample=Image.NEAREST)

        # 3. Synchronized augmentations during training
        if self.augment:
            # Random horizontal flip
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            # Random vertical flip
            if random.random() > 0.5:
                image = TF.vflip(image)
                mask = TF.vflip(mask)
            # Random slight rotation (-15 to 15 deg)
            if random.random() > 0.5:
                angle = random.uniform(-15, 15)
                image = TF.rotate(image, angle, interpolation=TF.InterpolationMode.BILINEAR)
                mask = TF.rotate(mask, angle, interpolation=TF.InterpolationMode.NEAREST)

        # 4. Convert to tensors
        image_tensor = TF.to_tensor(image)  # [3, H, W] in [0, 1]
        mask_np = (np.array(mask) > 127).astype(np.float32)
        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0)  # [1, H, W] in {0, 1}

        # 5. Extract boundary ground-truth
        boundary_tensor = extract_boundary_tensor(mask_tensor, radius=1)

        # 6. Compute lesion-scale category
        category, rel_area = compute_lesion_scale_category(mask_tensor)

        return {
            'image': image_tensor,
            'mask': mask_tensor,
            'boundary': boundary_tensor,
            'category': category,
            'rel_area': rel_area,
            'id': sample_id,
        }
