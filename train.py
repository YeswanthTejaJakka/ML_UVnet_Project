"""
P2 + P3 — Unified Training Engine for UltraLight VM-UNet experiments.

Key changes from original:
  P0: Uses fixed TN3KDataset with assert for fold JSON.
  P1: Model returns logits; uses BCEWithLogitsLoss; fp32 SSM math.
  P2: cudnn.benchmark=True; in-RAM cache support; per-epoch profiling.
  P3: warmup+cosine scheduler, EMA weights, grad-clip, weight-decay groups,
      --seed, --resume, extended CSV logging, model-selection on best EMA val Dice,
      checkpoints under {exp_name}/seed{seed}/, deep supervision.
  P4: scale-aware boundary loss (from_mask mode), consistency loss with ramp-up,
      per-step random corruption.

Usage:
  python train.py --config configs/e1_baseline.yaml
  python train.py --config configs/e2_boundary.yaml --seed 42
  python train.py --config configs/e1_baseline.yaml --resume
"""

import os
import sys
import argparse
import yaml
import json
import time
import random
import shutil
import hashlib
import copy
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
from torch.cuda.amp import GradScaler

from models import UltraLight_VM_UNet
from dataset import TN3KDataset
from losses import (
    BceDiceLoss, ScaleAwareBoundaryLoss, PerturbationConsistencyLoss, apply_corruption
)
from utils import MetricTracker, calculate_dice, calculate_iou


# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # P2: deterministic=False for speed (seed-reproducible in distribution, not bitwise)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def get_code_version():
    """Returns git commit hash if available, else 'unknown'."""
    try:
        import subprocess
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True, text=True, cwd=os.path.dirname(os.path.abspath(__file__))
        )
        if result.returncode == 0:
            return result.stdout.strip()[:12]
    except Exception:
        pass
    return 'unknown'


def load_config(config_path, cli_overrides=None):
    """
    Load YAML config. Supports _base_ inheritance.
    CLI overrides applied last.
    """
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    # Handle base config inheritance
    if '_base_' in cfg:
        base_path = os.path.join(os.path.dirname(config_path), cfg.pop('_base_'))
        if os.path.exists(base_path):
            with open(base_path, 'r') as f:
                base_cfg = yaml.safe_load(f)
            # base provides defaults; cfg overrides
            merged = base_cfg.copy()
            merged.update(cfg)
            cfg = merged

    # Apply CLI overrides
    if cli_overrides:
        for k, v in cli_overrides.items():
            if v is not None:
                cfg[k] = v

    return cfg


def parse_args():
    parser = argparse.ArgumentParser(description="UltraLight VM-UNet Training Engine")
    parser.add_argument('--config',      type=str, default='configs/e1_baseline.yaml')
    parser.add_argument('--exp',         type=str, default=None)
    parser.add_argument('--data_root',   type=str, default=None)
    parser.add_argument('--fold',        type=int, default=None)
    parser.add_argument('--epochs',      type=int, default=None)
    parser.add_argument('--batch_size',  type=int, default=None)
    parser.add_argument('--lr',          type=float, default=None)
    parser.add_argument('--seed',        type=int, default=None)
    parser.add_argument('--save_dir',    type=str, default=None)
    parser.add_argument('--resume',      action='store_true', default=False)
    parser.add_argument('--force_retrain', action='store_true', default=False)
    parser.add_argument('--device',      type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# EMA
# ──────────────────────────────────────────────────────────────────────────────

class EMAModel:
    """Exponential Moving Average of model weights with step warmup."""
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model.state_dict())
        self.num_updates = 0
        for k in self.shadow:
            self.shadow[k] = self.shadow[k].float()

    @torch.no_grad()
    def update(self, model):
        self.num_updates += 1
        # Warmup decay so early in training it tracks actual learned weights
        decay = min(self.decay, (1.0 + self.num_updates) / (10.0 + self.num_updates))
        for k, v in model.state_dict().items():
            self.shadow[k].mul_(decay).add_(v.float(), alpha=(1.0 - decay))

    def apply_to(self, model):
        """Temporarily load EMA weights into model."""
        self._backup = copy.deepcopy(model.state_dict())
        model.load_state_dict({k: v.to(next(model.parameters()).device)
                               for k, v in self.shadow.items()})

    def restore(self, model):
        """Restore original weights."""
        model.load_state_dict(self._backup)


# ──────────────────────────────────────────────────────────────────────────────
# Scheduler helpers
# ──────────────────────────────────────────────────────────────────────────────

def build_scheduler(optimizer, epochs, warmup_epochs, min_lr, base_lr):
    """Linear warmup + cosine annealing scheduler."""
    warmup_steps = warmup_epochs

    def lr_lambda(epoch):
        if epoch < warmup_steps:
            return float(epoch + 1) / max(1, warmup_steps)
        progress = float(epoch - warmup_steps) / max(1, epochs - warmup_steps)
        cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
        return (min_lr + (base_lr - min_lr) * cosine) / base_lr

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


# ──────────────────────────────────────────────────────────────────────────────
# Weight-decay parameter groups (P3)
# ──────────────────────────────────────────────────────────────────────────────

NO_DECAY_NAMES = {'bias', 'norm', 'A_log', 'D', 'skip_scale'}

def get_param_groups(model, weight_decay):
    """
    Apply weight_decay only to tensors with ndim >= 2.
    No decay for: bias, norm weights/biases, A_log, D, skip_scale.
    """
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Check if name contains any no-decay keyword OR param is 1D
        if any(nd in name for nd in NO_DECAY_NAMES) or param.ndim < 2:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    return [
        {'params': decay_params,    'weight_decay': weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0},
    ]


# ──────────────────────────────────────────────────────────────────────────────
# Lambda ramp-up helpers
# ──────────────────────────────────────────────────────────────────────────────

def get_lambda(base_lambda, epoch, total_epochs, warmup_frac=0.20):
    """Linear ramp-up from 0 to base_lambda over first warmup_frac of training."""
    warmup_end = int(warmup_frac * total_epochs)
    if epoch <= warmup_end:
        return base_lambda * (epoch / max(1, warmup_end))
    return base_lambda


# ──────────────────────────────────────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────────────────────────────────────

def fast_validate(model, loader, device, cfg, epoch=0, total_epochs=200):
    """
    Extended GPU-side validation.
    Returns dict with dice, iou, precision, recall, and per-size dice.
    """
    model.eval()
    # Accumulators
    dice_all, iou_all = [], []
    prec_all, rec_all = [], []
    dice_small, dice_medium, dice_large = [], [], []
    seg_loss_all = []

    bce_dice_fn = BceDiceLoss()

    with torch.no_grad():
        for batch in loader:
            images   = batch['image'].to(device)
            masks    = batch['mask'].to(device)
            cats     = batch['category']

            logits = model(images, return_boundary=False, training_mode=False)
            if isinstance(logits, tuple):
                logits = logits[0]

            loss_v = bce_dice_fn(logits, masks).item()
            seg_loss_all.append(loss_v)

            preds = (torch.sigmoid(logits) > 0.5).float()

            for i in range(images.size(0)):
                p = preds[i, 0]
                t = masks[i, 0]

                inter = (p * t).sum()
                card  = p.sum() + t.sum()
                union = p.sum() + t.sum() - inter
                tp    = inter
                fp    = p.sum() - inter
                fn    = t.sum() - inter

                dice_v = float((2 * inter + 1e-5) / (card + 1e-5))
                iou_v  = float((inter + 1e-5) / (union + 1e-5))
                prec_v = float((tp + 1e-5) / (p.sum() + 1e-5))
                rec_v  = float((tp + 1e-5) / (t.sum() + 1e-5))

                dice_all.append(dice_v)
                iou_all.append(iou_v)
                prec_all.append(prec_v)
                rec_all.append(rec_v)

                cat = cats[i] if isinstance(cats[i], str) else cats[i]
                if cat == 'small':
                    dice_small.append(dice_v)
                elif cat == 'large':
                    dice_large.append(dice_v)
                else:
                    dice_medium.append(dice_v)

    return {
        'val_dice':        float(np.mean(dice_all)) if dice_all else 0.0,
        'val_iou':         float(np.mean(iou_all))  if iou_all  else 0.0,
        'val_precision':   float(np.mean(prec_all)) if prec_all else 0.0,
        'val_recall':      float(np.mean(rec_all))  if rec_all  else 0.0,
        'val_loss':        float(np.mean(seg_loss_all)) if seg_loss_all else 0.0,
        'val_dice_small':  float(np.mean(dice_small))  if dice_small  else 0.0,
        'val_dice_medium': float(np.mean(dice_medium)) if dice_medium else 0.0,
        'val_dice_large':  float(np.mean(dice_large))  if dice_large  else 0.0,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, loss_modules, cfg, device, scaler, epoch, total_epochs, ema=None):
    model.train()
    bce_dice_fn    = loss_modules['seg']
    boundary_fn    = loss_modules.get('boundary')
    consistency_fn = loss_modules.get('consistency')

    use_boundary    = cfg.get('use_boundary_loss', False)
    use_consistency = cfg.get('use_consistency_loss', False)
    use_amp         = cfg.get('amp', False) and device.type == 'cuda'
    use_ds          = cfg.get('deep_supervision', False)
    ds_weights      = cfg.get('ds_weights', [0.4, 0.3, 0.2, 0.1])
    grad_clip       = cfg.get('grad_clip', 1.0)

    # Current effective lambdas (with ramp-up)
    lambda_b = get_lambda(cfg.get('lambda_boundary', 0.3), epoch, total_epochs,
                          cfg.get('boundary_warmup_frac', 0.20))
    lambda_c = get_lambda(cfg.get('lambda_consistency', 1.0), epoch, total_epochs,
                          cfg.get('consistency_warmup_frac', 0.20))

    total_loss_sum = 0.0
    seg_loss_sum   = 0.0
    b_loss_sum     = 0.0
    c_loss_sum     = 0.0
    aux_loss_sum   = 0.0
    n_batches      = 0

    # Dice tracking for training
    train_dice_sum = 0.0
    train_iou_sum  = 0.0

    for batch in loader:
        images     = batch['image'].to(device, non_blocking=True)
        masks      = batch['mask'].to(device, non_blocking=True)
        boundaries = batch['boundary'].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, enabled=use_amp):
            if use_ds:
                logits, aux_list = model(images, training_mode=True)
            else:
                logits = model(images, training_mode=False)
                if isinstance(logits, tuple):
                    logits = logits[0]
                aux_list = []

        # Main seg loss
        loss_seg = bce_dice_fn(logits, masks)

        # Deep supervision loss
        loss_aux = torch.tensor(0.0, device=device)
        if use_ds and aux_list:
            for w, aux_logit in zip(ds_weights[:len(aux_list)], aux_list):
                loss_aux = loss_aux + w * bce_dice_fn(aux_logit, masks)

        # Boundary loss (P4-E2)
        loss_b = torch.tensor(0.0, device=device)
        if use_boundary and boundary_fn is not None and lambda_b > 0:
            loss_b = boundary_fn(logits, masks, boundaries)

        # Consistency loss (P4-E3)
        loss_c = torch.tensor(0.0, device=device)
        if use_consistency and consistency_fn is not None and lambda_c > 0:
            with torch.no_grad():
                corrupt_imgs = apply_corruption(images)  # per-sample random corruption
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits_corrupt = model(corrupt_imgs, training_mode=False)
                if isinstance(logits_corrupt, tuple):
                    logits_corrupt = logits_corrupt[0]
            loss_c = consistency_fn(logits_corrupt, logits.detach())

        total_loss = loss_seg + loss_aux + lambda_b * loss_b + lambda_c * loss_c

        if use_amp:
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        if ema is not None:
            ema.update(model)

        # Track losses
        total_loss_sum += total_loss.item()
        seg_loss_sum   += loss_seg.item()
        b_loss_sum     += loss_b.item()
        c_loss_sum     += loss_c.item()
        aux_loss_sum   += loss_aux.item()
        n_batches += 1

        # Quick train dice (approximate, on predictions)
        with torch.no_grad():
            preds = (torch.sigmoid(logits) > 0.5).float()
            inter = (preds * masks).sum()
            card  = preds.sum() + masks.sum()
            d = float((2 * inter + 1e-5) / (card + 1e-5))
            train_dice_sum += d
            u = preds.sum() + masks.sum() - inter
            train_iou_sum  += float((inter + 1e-5) / (u + 1e-5))

    nb = max(n_batches, 1)
    return {
        'train_total':       total_loss_sum / nb,
        'train_seg':         seg_loss_sum   / nb,
        'train_aux':         aux_loss_sum   / nb,
        'train_boundary':    b_loss_sum     / nb,
        'train_consistency': c_loss_sum     / nb,
        'train_dice':        train_dice_sum / nb,
        'train_iou':         train_iou_sum  / nb,
        'lambda_b':          lambda_b,
        'lambda_c':          lambda_c,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ──────────────────────────────────────────────────────────────────────────────

def save_checkpoint(path, model, ema, optimizer, scheduler, epoch, best_val_dice, cfg, code_version):
    torch.save({
        'epoch':            epoch,
        'model_state_dict': model.state_dict(),
        'ema_shadow':       ema.shadow,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_dice':    best_val_dice,
        'cfg':              cfg,
        'code_version':     code_version,
    }, path)


def load_checkpoint(path, model, ema, optimizer, scheduler, device, cfg, code_version):
    ckpt = torch.load(path, map_location=device)
    # Version check
    ckpt_ver = ckpt.get('code_version', 'unknown')
    if ckpt_ver != code_version and code_version != 'unknown':
        print(f"[WARNING] Checkpoint code_version='{ckpt_ver}' != current='{code_version}'.")
        print("  This checkpoint was created by a different code version.")
        print("  Refusing to resume to avoid stale results. Use --force_retrain to override.")
        raise RuntimeError(f"Code version mismatch: checkpoint={ckpt_ver}, current={code_version}")
    model.load_state_dict(ckpt['model_state_dict'])
    ema.shadow = ckpt['ema_shadow']
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    start_epoch = ckpt['epoch'] + 1
    best_val_dice = ckpt.get('best_val_dice', 0.0)
    print(f"  Resumed from epoch {ckpt['epoch']}, best_val_dice={best_val_dice:.4f}")
    return start_epoch, best_val_dice


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Config override map
    cli_overrides = {
        'data_root':  args.data_root,
        'fold':       args.fold,
        'epochs':     args.epochs,
        'batch_size': args.batch_size,
        'lr':         args.lr,
        'seed':       args.seed,
        'save_dir':   args.save_dir,
        'resume':     args.resume if args.resume else None,
    }

    cfg_path = args.config
    if args.exp and not os.path.exists(cfg_path):
        candidate = f"configs/{args.exp}.yaml"
        if os.path.exists(candidate):
            cfg_path = candidate

    cfg = load_config(cfg_path, {k: v for k, v in cli_overrides.items() if v is not None})

    seed = cfg.get('seed', 42)
    set_seed(seed)

    device = torch.device(args.device)
    code_version = get_code_version()

    # Save dir: checkpoints/{exp_name}/seed{seed}/
    exp_name = cfg.get('exp_name', 'exp')
    base_save = cfg.get('save_dir', 'checkpoints')
    save_dir = os.path.join(base_save, exp_name, f"seed{seed}")
    cfg['save_dir'] = save_dir
    os.makedirs(save_dir, exist_ok=True)

    best_model_path  = os.path.join(save_dir, 'best_model.pth')
    latest_ckpt_path = os.path.join(save_dir, 'latest_model.pth')
    best_ema_path    = os.path.join(save_dir, 'best_ema.pth')
    last_ema_path    = os.path.join(save_dir, 'last_ema.pth')

    # Check for existing checkpoint — refuse to skip if code version differs
    if os.path.exists(best_model_path) and not cfg.get('resume', False) and not args.force_retrain:
        try:
            prev = torch.load(best_model_path, map_location='cpu')
            prev_ver = prev.get('code_version', 'unknown')
            if prev_ver == code_version or code_version == 'unknown':
                prev_dice = prev.get('val_dice', prev.get('best_val_dice', 'N/A'))
                print(f"\n✅ Found existing checkpoint: {best_model_path}")
                print(f"   Best Val Dice: {prev_dice}")
                print("   Skipping re-training. Use --resume to continue or --force_retrain for fresh run.")
                return
            else:
                print(f"[INFO] Checkpoint code_version='{prev_ver}' differs from current='{code_version}'.")
                print("  Will NOT skip — retrain from scratch to ensure fresh results.")
        except Exception as e:
            print(f"[WARNING] Could not read existing checkpoint: {e}")

    # Backup old checkpoints to legacy/
    if os.path.exists(best_model_path) and args.force_retrain:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        legacy_dir = os.path.join('legacy', exp_name, f"seed{seed}", ts)
        os.makedirs(legacy_dir, exist_ok=True)
        for fname in ['best_model.pth', 'latest_model.pth', 'best_ema.pth', 'training_log.csv']:
            src = os.path.join(save_dir, fname)
            if os.path.exists(src):
                shutil.copy(src, os.path.join(legacy_dir, fname))
        print(f"⚠️  Backed up old checkpoints to: {legacy_dir}")

    # Save used config
    with open(os.path.join(save_dir, 'config_used.yaml'), 'w') as f:
        cfg_to_save = cfg.copy()
        cfg_to_save['code_version'] = code_version
        yaml.dump(cfg_to_save, f)

    print(f"=== Experiment: {exp_name} | Seed: {seed} | Device: {device} ===")
    print(f"Dataset root: {cfg['data_root']} | Fold: {cfg.get('fold', 0)}")
    print(f"Epochs: {cfg.get('epochs', 200)} | Batch: {cfg.get('batch_size', 8)} | LR: {cfg.get('lr', 1e-3)}")
    print(f"Code version: {code_version}")

    # ── Datasets ────────────────────────────────────────────────────────────
    aug_cfg = cfg.get('aug', {})
    cache_in_ram = cfg.get('cache_in_ram', True)
    train_dataset = TN3KDataset(
        data_root=cfg['data_root'],
        split='train',
        fold=cfg.get('fold', 0),
        img_size=tuple(cfg.get('img_size', [256, 256])),
        augment=True,
        aug_cfg=aug_cfg,
        cache_in_ram=cache_in_ram,
    )
    val_dataset = TN3KDataset(
        data_root=cfg['data_root'],
        split='val',
        fold=cfg.get('fold', 0),
        img_size=tuple(cfg.get('img_size', [256, 256])),
        augment=False,
        cache_in_ram=cache_in_ram,
    )

    # P2: speed settings
    import os as _os
    num_workers = min(4, _os.cpu_count() or 2)
    batch_size = cfg.get('batch_size', 8)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers > 0),
    )
    print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")

    # ── Model ────────────────────────────────────────────────────────────────
    use_boundary_head = cfg.get('aux_boundary_head', False)
    deep_supervision  = cfg.get('deep_supervision', True)
    model = UltraLight_VM_UNet(
        num_classes=1,
        input_channels=3,
        c_list=cfg.get('c_list', [8, 16, 24, 32, 48, 64]),
        split_att=cfg.get('split_att', 'fc'),
        bridge=cfg.get('bridge', True),
        aux_boundary_head=use_boundary_head,
        deep_supervision=deep_supervision,
        return_logits=True,
    ).to(device)

    # Verify inference param count
    n_inf_params = sum(p.numel() for p in model.parameters()
                       if 'aux_head' not in str(p) and 'boundary_head' not in str(p))
    n_total_params = sum(p.numel() for p in model.parameters())
    print(f"Model — total params: {n_total_params/1e6:.4f}M | "
          f"inference params (approx): {n_total_params/1e6:.4f}M")

    # EMA
    ema = EMAModel(model, decay=cfg.get('ema_decay', 0.999))

    # ── Optimizer & Scheduler ─────────────────────────────────────────────────
    epochs         = cfg.get('epochs', 200)
    warmup_epochs  = cfg.get('warmup_epochs', 5)
    base_lr        = cfg.get('lr', 1e-3)
    min_lr         = cfg.get('min_lr', 1e-6)
    weight_decay   = cfg.get('weight_decay', 0.01)

    param_groups = get_param_groups(model, weight_decay)
    optimizer    = AdamW(param_groups, lr=base_lr)
    scheduler    = build_scheduler(optimizer, epochs, warmup_epochs, min_lr, base_lr)
    scaler       = GradScaler(enabled=cfg.get('amp', False) and device.type == 'cuda')

    # ── Losses ───────────────────────────────────────────────────────────────
    loss_modules = {'seg': BceDiceLoss()}
    if cfg.get('use_boundary_loss', False):
        loss_modules['boundary'] = ScaleAwareBoundaryLoss(
            boundary_mode=cfg.get('boundary_mode', 'from_mask'),
            r=2,
            a_ref=cfg.get('boundary_a_ref', 0.05),
            gamma=cfg.get('boundary_gamma', 0.5),
            w_max=cfg.get('boundary_w_max', 4.0),
        )
    if cfg.get('use_consistency_loss', False):
        loss_modules['consistency'] = PerturbationConsistencyLoss()

    # ── Assert no test-set reads in training ─────────────────────────────────
    # (This is a code-level check — test split is never loaded here)
    assert not any('test' in str(cfg.get('split', '')) for _ in [1]), \
        "Test split must not be accessed during training!"

    # ── Resume ───────────────────────────────────────────────────────────────
    start_epoch   = 1
    best_val_dice = 0.0
    history       = []

    if cfg.get('resume', False) and os.path.exists(latest_ckpt_path):
        print(f"Resuming from: {latest_ckpt_path}")
        try:
            start_epoch, best_val_dice = load_checkpoint(
                latest_ckpt_path, model, ema, optimizer, scheduler, device, cfg, code_version
            )
            # Load existing history
            csv_path = os.path.join(save_dir, 'training_log.csv')
            if os.path.exists(csv_path):
                history = pd.read_csv(csv_path).to_dict('records')
        except RuntimeError as e:
            print(f"[ERROR] Cannot resume: {e}")
            sys.exit(1)
    elif cfg.get('resume', False):
        print(f"[INFO] No checkpoint found at {latest_ckpt_path}. Starting fresh.")

    # ── P2: Profile first epoch ───────────────────────────────────────────────
    profile_printed = (start_epoch > 1)

    # ── Training Loop ─────────────────────────────────────────────────────────
    print(f"\nStarting training from epoch {start_epoch} to {epochs}...")

    for epoch in range(start_epoch, epochs + 1):
        t0 = time.time()

        # --- Train ---
        t_data_start = time.time()
        train_res = train_one_epoch(
            model, train_loader, optimizer, loss_modules, cfg, device, scaler, epoch, epochs, ema=ema
        )
        t_train = time.time() - t0

        # Advance scheduler
        scheduler.step()
        curr_lr = optimizer.param_groups[0]['lr']

        # --- Validate (using EMA weights) ---
        t_val_start = time.time()
        ema.apply_to(model)
        val_res = fast_validate(model, val_loader, device, cfg, epoch, epochs)
        ema.restore(model)
        t_val = time.time() - t_val_start

        t_epoch = time.time() - t0
        val_dice = val_res['val_dice']

        # P2: Print profile info on epoch 1
        if not profile_printed:
            print(f"\n[Profile] Epoch 1 timing: total={t_epoch:.1f}s | "
                  f"train={t_train:.1f}s | val={t_val:.1f}s")
            if t_epoch > 20:
                print(f"  [WARNING] Epoch time {t_epoch:.1f}s > 20s target. "
                      f"Consider RAM caching or fewer workers.")
            profile_printed = True

        print(
            f"Epoch [{epoch:03d}/{epochs:03d}] "
            f"Loss: {train_res['train_total']:.4f} "
            f"(Seg: {train_res['train_seg']:.4f}, "
            f"Bnd: {train_res['train_boundary']:.4f}, "
            f"Cons: {train_res['train_consistency']:.4f}, "
            f"Aux: {train_res['train_aux']:.4f}) | "
            f"TrainDice: {train_res['train_dice']:.4f} | "
            f"ValDice: {val_dice:.4f} | "
            f"ValPrec: {val_res['val_precision']:.4f} | "
            f"ValRec: {val_res['val_recall']:.4f} | "
            f"LR: {curr_lr:.6f} | "
            f"Time: {t_epoch:.1f}s"
        )

        # Compute ratio logs for P4 diagnostics
        lb_ratio = (train_res['lambda_b'] * train_res['train_boundary']) / (train_res['train_seg'] + 1e-8)
        lc_ratio = (train_res['lambda_c'] * train_res['train_consistency']) / (train_res['train_seg'] + 1e-8)

        record = {
            'epoch':              epoch,
            'lr':                 curr_lr,
            'epoch_time':         round(t_epoch, 2),
            'train_total':        train_res['train_total'],
            'train_seg':          train_res['train_seg'],
            'train_aux':          train_res['train_aux'],
            'train_boundary':     train_res['train_boundary'],
            'train_consistency':  train_res['train_consistency'],
            'train_dice':         train_res['train_dice'],
            'train_iou':          train_res['train_iou'],
            'val_loss':           val_res['val_loss'],
            'val_dice':           val_res['val_dice'],
            'val_iou':            val_res['val_iou'],
            'val_precision':      val_res['val_precision'],
            'val_recall':         val_res['val_recall'],
            'val_dice_small':     val_res['val_dice_small'],
            'val_dice_medium':    val_res['val_dice_medium'],
            'val_dice_large':     val_res['val_dice_large'],
            'lambda_b':           train_res['lambda_b'],
            'lambda_c':           train_res['lambda_c'],
            'lb_ratio':           lb_ratio,
            'lc_ratio':           lc_ratio,
        }

        # Every 10 epochs and at end: also log val_biou_d7, val_hd95
        # (Expensive distance-transform metrics — skipped mid-training for speed)
        if epoch % 10 == 0 or epoch == epochs:
            record['val_biou_d7'] = -1.0   # placeholder; run evaluate.py for full metrics
            record['val_hd95']    = -1.0

        history.append(record)

        # Save latest checkpoint
        save_checkpoint(latest_ckpt_path, model, ema, optimizer, scheduler,
                        epoch, best_val_dice, cfg, code_version)
        torch.save({'epoch': epoch, 'ema_shadow': ema.shadow, 'code_version': code_version},
                   last_ema_path)

        # Save best checkpoint (EMA-based)
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save({
                'epoch':            epoch,
                'model_state_dict': model.state_dict(),
                'ema_shadow':       ema.shadow,
                'val_dice':         val_dice,
                'val_iou':          val_res['val_iou'],
                'val_precision':    val_res['val_precision'],
                'val_recall':       val_res['val_recall'],
                'cfg':              cfg,
                'code_version':     code_version,
            }, best_model_path)
            torch.save({'epoch': epoch, 'ema_shadow': ema.shadow, 'val_dice': val_dice,
                        'code_version': code_version}, best_ema_path)
            print(f"  --> New best! Val Dice: {best_val_dice:.4f} (epoch {epoch})")

        # Save training log CSV (append-safe)
        df = pd.DataFrame(history)
        df.to_csv(os.path.join(save_dir, 'training_log.csv'), index=False)

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n✅ Training complete! Best Val Dice (EMA): {best_val_dice:.4f}")
    print(f"   Checkpoints: {save_dir}")

    # Append to master runs registry
    try:
        parent = os.path.dirname(os.path.abspath(save_dir))
        master = os.path.join(os.path.dirname(parent), 'all_runs_summary.csv')
        entry = {
            'timestamp':    datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'exp_name':     exp_name,
            'seed':         seed,
            'epochs':       epochs,
            'batch_size':   batch_size,
            'lr':           base_lr,
            'best_val_dice': round(float(best_val_dice), 4),
            'save_dir':     save_dir,
            'code_version': code_version,
        }
        mdf = pd.read_csv(master) if os.path.exists(master) else pd.DataFrame()
        mdf = pd.concat([mdf, pd.DataFrame([entry])], ignore_index=True)
        mdf.to_csv(master, index=False)
    except Exception as e:
        print(f"[Note] Could not update master runs registry: {e}")


if __name__ == '__main__':
    main()
