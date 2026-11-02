import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset.boundary_utils import compute_scale_weight


def safe_binary_cross_entropy(pred, target, reduction='mean'):
    """
    Computes binary cross-entropy safely in float32 with autocast disabled.
    Prevents PyTorch RuntimeError: 'torch.nn.functional.binary_cross_entropy and
    torch.nn.BCELoss are unsafe to autocast' while maintaining numerical stability.
    """
    p = pred.float().clamp(1e-7, 1.0 - 1e-7)
    t = target.float()
    
    device_type = 'cuda' if pred.is_cuda else 'cpu'
    
    try:
        with torch.amp.autocast(device_type, enabled=False):
            return F.binary_cross_entropy(p, t, reduction=reduction)
    except Exception:
        with torch.cuda.amp.autocast(enabled=False):
            return F.binary_cross_entropy(p, t, reduction=reduction)


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        """
        pred: [B, 1, H, W] in [0, 1]
        target: [B, 1, H, W] in {0, 1}
        """
        batch = pred.size(0)
        p = pred.float().view(batch, -1)
        t = target.float().view(batch, -1)
        intersection = (p * t).sum(dim=1)
        cardinality = p.sum(dim=1) + t.sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice.mean()


class BceDiceLoss(nn.Module):
    """Standard baseline segmentation loss used by UltraLight VM-UNet."""
    def __init__(self, wb=1.0, wd=1.0):
        super().__init__()
        self.wb = wb
        self.wd = wd
        self.dice = DiceLoss()

    def forward(self, pred, target):
        loss_bce = safe_binary_cross_entropy(pred, target, reduction='mean')
        loss_dice = self.dice(pred, target)
        return self.wb * loss_bce + self.wd * loss_dice


class ScaleAwareBoundaryLoss(nn.Module):
    """
    Supervises lesion boundary prediction with dynamic scale weights.
    Smaller lesions receive greater boundary loss emphasis to counteract
    fading gradients and edge degradation under downsampling.
    """
    def __init__(self, alpha=1.0, gamma=0.5, min_weight=1.0, max_weight=5.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.dice = DiceLoss()

    def forward(self, pred_boundary, gt_boundary, gt_mask):
        """
        pred_boundary: [B, 1, H, W] in [0, 1]
        gt_boundary:   [B, 1, H, W] in {0, 1}
        gt_mask:       [B, 1, H, W] in {0, 1} (used to compute lesion area)
        """
        scale_weights = compute_scale_weight(
            gt_mask,
            alpha=self.alpha,
            gamma=self.gamma,
            min_weight=self.min_weight,
            max_weight=self.max_weight
        ).squeeze(-1).squeeze(-1).float()  # [B, 1]

        # Sample-wise BCE for boundary computed safely
        bce_per_sample = safe_binary_cross_entropy(pred_boundary, gt_boundary, reduction='none')
        bce_sample_mean = bce_per_sample.mean(dim=(-2, -1))  # [B, 1]
        weighted_bce = (bce_sample_mean * scale_weights).mean()

        # Dice component
        loss_dice = self.dice(pred_boundary, gt_boundary)

        return weighted_bce + loss_dice


class PerturbationConsistencyLoss(nn.Module):
    """
    Penalizes divergence between predictions on clean vs. corrupted ultrasound inputs.
    Forces representations to remain stable under acoustic speckle noise and contrast loss.
    """
    def __init__(self, loss_type='mse'):
        super().__init__()
        self.loss_type = loss_type

    def forward(self, pred_corrupt, pred_clean):
        pc = pred_corrupt.float()
        pcl = pred_clean.detach().float()
        if self.loss_type == 'mse':
            return F.mse_loss(pc, pcl)
        elif self.loss_type == 'l1':
            return F.l1_loss(pc, pcl)
        else:
            return F.mse_loss(pc, pcl)
