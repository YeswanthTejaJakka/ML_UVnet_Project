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

    P0 fixes:
    - Raises AssertionError (never silent) when fold JSON is missing for train/val split.
    - Logs a counted warning (never silent skip) for unmatched entries.
    - Robust file matching: handles zero-padded IDs, with/without extension, case-insensitive.

    P2 additions:
    - Optional in-RAM cache (decode + resize once).
    - Extended augmentation: hflip, rotation, scale, translate, brightness, gamma.
    - vflip REMOVED (thyroid probe always at top).
    """

    def __init__(
        self,
        data_root,
        split='train',
        fold=0,
        img_size=(256, 256),
        augment=True,
        cache_in_ram=False,
        aug_cfg=None,
    ):
        super().__init__()
        self.data_root = data_root
        self.split = split.lower()
        self.fold = fold
        self.img_size = img_size
        self.augment = augment and (self.split == 'train')
        self.cache_in_ram = cache_in_ram
        self.aug_cfg = aug_cfg or {}

        self._skipped_count = 0
        self.samples = []
        self._cache = {}

        self._load_split_samples()

        if self._skipped_count > 0:
            print(
                f"[TN3KDataset WARNING] {self._skipped_count} fold-JSON entries could not be matched "
                f"(split='{self.split}', fold={fold}, data_root='{data_root}'). "
                f"Check audit/audit_report.md for details."
            )

        if cache_in_ram:
            self._build_cache()

    def _find_file(self, folder, base_name):
        """Robust file lookup: handles zero-pads, extensions, case."""
        stem = os.path.splitext(base_name)[0]
        exts = ['', '.jpg', '.jpeg', '.png', '.bmp']

        for ext in exts:
            candidate = os.path.join(folder, stem + ext)
            if os.path.exists(candidate):
                return candidate
        for ext in exts:
            candidate = os.path.join(folder, base_name + ext) if not base_name.endswith(ext) else os.path.join(folder, base_name)
            if os.path.exists(candidate):
                return candidate

        try:
            entries = {e.lower(): e for e in os.listdir(folder)}
        except Exception:
            return None
        for ext in exts:
            key = (stem + ext).lower()
            if key in entries:
                return os.path.join(folder, entries[key])

        if stem.isdigit():
            for try_stem in [stem.zfill(4), str(int(stem))]:
                for ext in ['.jpg', '.jpeg', '.png', '.bmp', '']:
                    key = (try_stem + ext).lower()
                    if key in entries:
                        return os.path.join(folder, entries[key])

        return None

    def _load_split_samples(self):
        if self.split in ['train', 'val']:
            img_dir  = os.path.join(self.data_root, 'trainval-image')
            mask_dir = os.path.join(self.data_root, 'trainval-mask')
            fold_file = os.path.join(self.data_root, f'tn3k-trainval-fold{self.fold}.json')

            # P0: assert fold JSON exists — never fall back silently
            assert os.path.exists(fold_file), (
                f"[TN3KDataset ERROR] Fold JSON not found: '{fold_file}'. "
                f"Cannot load split='{self.split}'. "
                f"Provide the official fold JSON or set split='test'."
            )

            with open(fold_file, 'r') as f:
                fold_data = json.load(f)

            if isinstance(fold_data, dict):
                keys = [k for k in fold_data.keys() if self.split in k.lower()]
                target_files = fold_data[keys[0]] if keys else fold_data.get(
                    'train' if self.split == 'train' else 'val', []
                )
            elif isinstance(fold_data, list):
                target_files = fold_data
            else:
                target_files = []

            for item in target_files:
                name = item if isinstance(item, str) else str(item)
                img_path  = self._find_file(img_dir, name)
                mask_path = self._find_file(mask_dir, name)
                if img_path and mask_path:
                    self.samples.append((img_path, mask_path, os.path.basename(img_path)))
                else:
                    self._skipped_count += 1  # counted, never silent

        elif self.split == 'test':
            img_dir  = os.path.join(self.data_root, 'test-image')
            mask_dir = os.path.join(self.data_root, 'test-mask')
            if not os.path.exists(img_dir):
                raise FileNotFoundError(f"Test directory not found: {img_dir}")

            all_images = sorted([
                f for f in os.listdir(img_dir)
                if f.lower().endswith(('.jpg', '.png', '.jpeg'))
            ])
            for img_name in all_images:
                img_path  = os.path.join(img_dir, img_name)
                mask_path = self._find_file(mask_dir, img_name)
                if mask_path:
                    self.samples.append((img_path, mask_path, img_name))
        else:
            raise ValueError(f"Unknown split: '{self.split}'. Use 'train', 'val', or 'test'.")

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No samples loaded for split='{self.split}', fold={self.fold} "
                f"under '{self.data_root}'. Check data root and fold JSON."
            )

        print(f"[TN3KDataset] split='{self.split}' fold={self.fold}: "
              f"loaded {len(self.samples)} samples "
              f"(skipped={self._skipped_count})")

    def _build_cache(self):
        """Decode and resize all images/masks once into RAM (uint8 arrays)."""
        print(f"[TN3KDataset] Building in-RAM cache for {len(self.samples)} samples...")
        for idx, (img_path, mask_path, _) in enumerate(self.samples):
            img_np  = np.array(Image.open(img_path).convert('RGB').resize(self.img_size, Image.BILINEAR), dtype=np.uint8)
            mask_np = np.array(Image.open(mask_path).convert('L').resize(self.img_size, Image.NEAREST), dtype=np.uint8)
            self._cache[idx] = (img_np, mask_np)
            if (idx + 1) % 500 == 0:
                print(f"  Cached {idx+1}/{len(self.samples)}")
        print("[TN3KDataset] Cache complete.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path, sample_id = self.samples[idx]

        if self.cache_in_ram and idx in self._cache:
            img_np, mask_np_raw = self._cache[idx]
            image = Image.fromarray(img_np)
            mask  = Image.fromarray(mask_np_raw)
        else:
            image = Image.open(img_path).convert('RGB')
            mask  = Image.open(mask_path).convert('L')
            image = image.resize(self.img_size, Image.BILINEAR)
            mask  = mask.resize(self.img_size, Image.NEAREST)

        if self.augment:
            # Horizontal flip
            hflip_p = self.aug_cfg.get('hflip', 0.5)
            if random.random() < hflip_p:
                image = TF.hflip(image)
                mask  = TF.hflip(mask)

            # NO vertical flip — thyroid probe always at top

            # Random rotation
            rot_deg = self.aug_cfg.get('rotate_deg', 15)
            if random.random() > 0.5:
                angle = random.uniform(-rot_deg, rot_deg)
                image = TF.rotate(image, angle, interpolation=TF.InterpolationMode.BILINEAR)
                mask  = TF.rotate(mask,  angle, interpolation=TF.InterpolationMode.NEAREST)

            # Random scale + translate
            scale_range = self.aug_cfg.get('scale', [0.8, 1.2])
            translate = self.aug_cfg.get('translate', 0.10)
            if random.random() > 0.5:
                scale = random.uniform(scale_range[0], scale_range[1])
                tx = random.uniform(-translate, translate)
                ty = random.uniform(-translate, translate)
                W, H = image.size
                image = TF.affine(image, angle=0, translate=[int(tx*W), int(ty*H)],
                                  scale=scale, shear=0,
                                  interpolation=TF.InterpolationMode.BILINEAR)
                mask  = TF.affine(mask,  angle=0, translate=[int(tx*W), int(ty*H)],
                                  scale=scale, shear=0,
                                  interpolation=TF.InterpolationMode.NEAREST)

            # Brightness jitter
            brightness = self.aug_cfg.get('brightness', 0.10)
            if random.random() > 0.5 and brightness > 0:
                bf = random.uniform(1.0 - brightness, 1.0 + brightness)
                image = TF.adjust_brightness(image, bf)

            # Gamma jitter
            gamma_range = self.aug_cfg.get('gamma', [0.9, 1.1])
            if random.random() > 0.5:
                g = random.uniform(gamma_range[0], gamma_range[1])
                image = TF.adjust_gamma(image, g)

        image_tensor = TF.to_tensor(image)
        mask_np      = (np.array(mask) > 127).astype(np.float32)
        mask_tensor  = torch.from_numpy(mask_np).unsqueeze(0)

        boundary_tensor = extract_boundary_tensor(mask_tensor, radius=2)  # r=2 → 5×5 kernel
        category, rel_area = compute_lesion_scale_category(mask_tensor)

        return {
            'image':    image_tensor,
            'mask':     mask_tensor,
            'boundary': boundary_tensor,
            'category': category,
            'rel_area': rel_area,
            'id':       sample_id,
        }
