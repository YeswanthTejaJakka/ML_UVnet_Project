import os
import argparse
import yaml
import json
import random
import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from models import UltraLight_VM_UNet
from dataset import TN3KDataset
from losses import BceDiceLoss, ScaleAwareBoundaryLoss, PerturbationConsistencyLoss
from utils import apply_ultrasound_corruption, MetricTracker, calculate_dice, calculate_iou


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser(description="UltraLight VM-UNet Training Engine")
    parser.add_argument('--config', type=str, default='configs/e1_baseline.yaml', help='Path to YAML config')
    parser.add_argument('--exp', type=str, default=None, help='Experiment name override (e.g. e1_baseline, e4_combined)')
    parser.add_argument('--data_root', type=str, default=None, help='Dataset root path')
    parser.add_argument('--fold', type=int, default=None, help='Fold index for cross validation')
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate')
    parser.add_argument('--save_dir', type=str, default=None, help='Directory to save checkpoints')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()


def load_config(args):
    # If exp is provided directly, map to default config file
    if args.exp and not os.path.exists(args.config):
        candidate = f"configs/{args.exp}.yaml"
        if os.path.exists(candidate):
            args.config = candidate

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    # CLI overrides
    if args.data_root is not None:
        cfg['data_root'] = args.data_root
    if args.fold is not None:
        cfg['fold'] = args.fold
    if args.epochs is not None:
        cfg['epochs'] = args.epochs
    if args.batch_size is not None:
        cfg['batch_size'] = args.batch_size
    if args.lr is not None:
        cfg['lr'] = args.lr
    if args.save_dir is not None:
        cfg['save_dir'] = args.save_dir
    elif 'save_dir' not in cfg:
        cfg['save_dir'] = os.path.join('checkpoints', cfg.get('exp_name', 'exp'))

    return cfg


def train_one_epoch(model, loader, optimizer, loss_modules, cfg, device):
    model.train()
    bce_dice = loss_modules['seg']
    boundary_loss_fn = loss_modules.get('boundary')
    cons_loss_fn = loss_modules.get('consistency')

    total_loss_accum = 0.0
    seg_loss_accum = 0.0
    b_loss_accum = 0.0
    c_loss_accum = 0.0

    for batch in loader:
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        boundaries = batch['boundary'].to(device)

        optimizer.zero_grad()

        use_boundary = cfg.get('use_boundary_loss', False)
        if use_boundary:
            pred_mask, pred_boundary = model(images, return_boundary=True)
        else:
            pred_mask = model(images, return_boundary=False)

        # Baseline segmentation loss
        loss_seg = bce_dice(pred_mask, masks)
        total_loss = loss_seg
        seg_loss_accum += loss_seg.item()

        # Auxiliary Scale-Aware Boundary loss
        if use_boundary and boundary_loss_fn is not None:
            lambda_b = cfg.get('lambda_boundary', 1.0)
            loss_b = boundary_loss_fn(pred_boundary, boundaries, masks)
            total_loss = total_loss + lambda_b * loss_b
            b_loss_accum += loss_b.item()

        # Ultrasound Corruption-Aware Consistency loss
        if cfg.get('use_consistency_loss', False) and cons_loss_fn is not None:
            sigma = cfg.get('speckle_sigma', 0.15)
            gamma = cfg.get('contrast_factor', 0.75)
            corrupt_images = apply_ultrasound_corruption(images, speckle_sigma=sigma, contrast_factor=gamma)
            pred_corrupt = model(corrupt_images, return_boundary=False)

            lambda_c = cfg.get('lambda_consistency', 1.0)
            loss_c = cons_loss_fn(pred_corrupt, pred_mask)
            total_loss = total_loss + lambda_c * loss_c
            c_loss_accum += loss_c.item()

        total_loss.backward()
        optimizer.step()
        total_loss_accum += total_loss.item()

    num_batches = len(loader)
    return {
        'total_loss': total_loss_accum / num_batches,
        'seg_loss': seg_loss_accum / num_batches,
        'boundary_loss': (b_loss_accum / num_batches) if cfg.get('use_boundary_loss') else 0.0,
        'consistency_loss': (c_loss_accum / num_batches) if cfg.get('use_consistency_loss') else 0.0,
    }


def validate(model, loader, device):
    model.eval()
    tracker = MetricTracker()

    with torch.no_grad():
        for batch in loader:
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            categories = batch['category']

            preds = model(images, return_boundary=False)
            preds_np = preds.cpu().numpy()
            masks_np = masks.cpu().numpy()

            for i in range(len(preds_np)):
                pred_i = preds_np[i, 0]
                mask_i = masks_np[i, 0]
                cat_i = categories[i]
                tracker.update(pred_i, mask_i, category=cat_i)

    return tracker.summary()


def main():
    args = parse_args()
    cfg = load_config(args)
    set_seed(42)

    device = torch.device(args.device)
    print(f"=== Starting Experiment: {cfg['exp_name']} on {device} ===")
    print(f"Dataset root: {cfg['data_root']} | Fold: {cfg.get('fold', 0)}")

    os.makedirs(cfg['save_dir'], exist_ok=True)
    with open(os.path.join(cfg['save_dir'], 'config_used.yaml'), 'w') as f:
        yaml.dump(cfg, f)

    # 1. Dataset & Loaders
    train_dataset = TN3KDataset(
        data_root=cfg['data_root'],
        split='train',
        fold=cfg.get('fold', 0),
        img_size=tuple(cfg.get('img_size', [256, 256])),
        augment=True,
    )
    val_dataset = TN3KDataset(
        data_root=cfg['data_root'],
        split='val',
        fold=cfg.get('fold', 0),
        img_size=tuple(cfg.get('img_size', [256, 256])),
        augment=False,
    )

    batch_size = cfg.get('batch_size', 16)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")

    # 2. Model Initialization
    use_boundary = cfg.get('use_boundary_loss', False)
    model = UltraLight_VM_UNet(
        num_classes=1,
        input_channels=3,
        c_list=[8, 16, 24, 32, 48, 64],
        split_att='fc',
        bridge=True,
        aux_boundary_head=use_boundary,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model initialized: {total_params / 1e6:.4f}M parameters")

    # 3. Optimizer & Scheduler
    epochs = cfg.get('epochs', 150)
    optimizer = AdamW(model.parameters(), lr=cfg.get('lr', 1e-3), weight_decay=cfg.get('weight_decay', 1e-2))
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=cfg.get('min_lr', 1e-5))

    # 4. Losses
    loss_modules = {
        'seg': BceDiceLoss(),
    }
    if use_boundary:
        loss_modules['boundary'] = ScaleAwareBoundaryLoss(
            alpha=cfg.get('boundary_alpha', 1.0),
            gamma=cfg.get('boundary_gamma', 0.5),
            min_weight=cfg.get('boundary_min_weight', 1.0),
            max_weight=cfg.get('boundary_max_weight', 5.0),
        )
    if cfg.get('use_consistency_loss', False):
        loss_modules['consistency'] = PerturbationConsistencyLoss(loss_type='mse')

    # 5. Training Loop
    best_val_dice = 0.0
    history = []

    for epoch in range(1, epochs + 1):
        train_res = train_one_epoch(model, train_loader, optimizer, loss_modules, cfg, device)
        scheduler.step()

        # Validate every epoch (or every 5 epochs)
        val_res = validate(model, val_loader, device)

        curr_lr = optimizer.param_groups[0]['lr']
        val_dice = val_res['overall_dice']
        val_iou = val_res['overall_iou']
        small_dice = val_res['small_dice']

        print(
            f"Epoch [{epoch:03d}/{epochs:03d}] "
            f"Loss: {train_res['total_loss']:.4f} "
            f"(Seg: {train_res['seg_loss']:.4f}, Bnd: {train_res['boundary_loss']:.4f}, Cons: {train_res['consistency_loss']:.4f}) | "
            f"Val Dice: {val_dice:.4f} | Val IoU: {val_iou:.4f} | Small Dice: {small_dice:.4f} | LR: {curr_lr:.6f}"
        )

        record = {
            'epoch': epoch,
            'train_loss': train_res['total_loss'],
            'val_dice': val_dice,
            'val_iou': val_iou,
            'val_biou': val_res['overall_biou'],
            'val_hd95': val_res['overall_hd95'],
            'small_dice': small_dice,
            'large_dice': val_res['large_dice'],
        }
        history.append(record)

        # Save latest checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_dice': val_dice,
            'cfg': cfg,
        }, os.path.join(cfg['save_dir'], 'latest_model.pth'))

        # Save best checkpoint
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_dice': val_dice,
                'val_iou': val_iou,
                'cfg': cfg,
            }, os.path.join(cfg['save_dir'], 'best_model.pth'))
            print(f"  --> Saved new best model checkpoint! Best Val Dice: {best_val_dice:.4f}")

    with open(os.path.join(cfg['save_dir'], 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining completed! Best Val Dice achieved: {best_val_dice:.4f}")
    print(f"Checkpoints saved in: {cfg['save_dir']}")


if __name__ == '__main__':
    main()
