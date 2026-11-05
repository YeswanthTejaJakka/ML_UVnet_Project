import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset.boundary_utils import extract_boundary_tensor


class DiceLoss(nn.Module):
    """
    Dice loss operating on predicted probabilities (or logits if from_logits=True).
    """
    def __init__(self, smooth=1e-5, from_logits=True):
        super().__init__()
        self.smooth = smooth
        self.from_logits = from_logits

    def forward(self, pred, target):
        """
        pred: [B, 1, H, W]
        target: [B, 1, H, W] in {0, 1}
        """
        if self.from_logits:
            p = torch.sigmoid(pred.float())
        else:
            p = pred.float()
        t = target.float()

        batch = p.size(0)
        p = p.view(batch, -1)
        t = t.view(batch, -1)

        intersection = (p * t).sum(dim=1)
        cardinality = p.sum(dim=1) + t.sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice.mean()


class SoftDiceLoss(nn.Module):
    """
    Soft Dice loss between two probability distributions in [0, 1].
    Used for consistency regularization (scale-invariant).
    """
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        """
        pred: [B, 1, H, W] in [0, 1]
        target: [B, 1, H, W] in [0, 1]
        """
        p = pred.float().view(pred.size(0), -1)
        t = target.float().view(target.size(0), -1)

        intersection = (p * t).sum(dim=1)
        cardinality = (p * p).sum(dim=1) + (t * t).sum(dim=1)
        soft_dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - soft_dice.mean()


class BceDiceLoss(nn.Module):
    """
    Standard segmentation loss: BCE with logits + Dice.
    P1 fix: Uses binary_cross_entropy_with_logits directly on raw logits.
    Supports optional sample-wise scale weights (for scale_aware_seg ablation).
    """
    def __init__(self, wb=1.0, wd=1.0):
        super().__init__()
        self.wb = wb
        self.wd = wd
        self.dice = DiceLoss(from_logits=True)

    def forward(self, pred_logits, target, sample_weights=None):
        """
        pred_logits: [B, 1, H, W]
        target: [B, 1, H, W]
        sample_weights: [B, 1, 1, 1] or None (normalized to batch mean = 1)
        """
        pred_f32 = pred_logits.float()
        target_f32 = target.float()

        if sample_weights is not None:
            # Per-sample weighted BCE
            bce_map = F.binary_cross_entropy_with_logits(pred_f32, target_f32, reduction='none')
            bce_per_sample = bce_map.mean(dim=(-2, -1), keepdim=True)  # [B, 1, 1, 1]
            loss_bce = (bce_per_sample * sample_weights).mean()

            # Per-sample weighted Dice
            p = torch.sigmoid(pred_f32).view(pred_f32.size(0), -1)
            t = target_f32.view(target_f32.size(0), -1)
            inter = (p * t).sum(dim=1)
            card = p.sum(dim=1) + t.sum(dim=1)
            dice_per_sample = 1.0 - (2.0 * inter + 1e-5) / (card + 1e-5)
            w_flat = sample_weights.view(pred_f32.size(0))
            loss_dice = (dice_per_sample * w_flat).mean()
        else:
            loss_bce = F.binary_cross_entropy_with_logits(pred_f32, target_f32, reduction='mean')
            loss_dice = self.dice(pred_f32, target_f32)

        return self.wb * loss_bce + self.wd * loss_dice


def compute_normalized_scale_weights(gt_mask, a_ref=0.05, gamma=0.5, w_max=4.0, eps=1e-4):
    """
    P4-E2: Per-sample scale weight:
    w_i = clip((A_ref / (rel_area_i + eps))^gamma, 1, w_max)
    Normalized to batch-mean 1 so gradient scale does not change.
    """
    B = gt_mask.shape[0]
    total_pixels = gt_mask.shape[-2] * gt_mask.shape[-1]
    area = (gt_mask > 0.5).float().sum(dim=(-2, -1), keepdim=True)  # [B, 1, 1, 1]
    rel_area = area / float(total_pixels)

    raw_w = torch.pow(a_ref / (rel_area + eps), gamma)
    w = raw_w.clamp(min=1.0, max=w_max)

    # Normalize to batch-mean 1
    batch_mean = w.mean(dim=0, keepdim=True)  # [1, 1, 1, 1]
    w_normalized = w / (batch_mean + 1e-7)
    return w_normalized


def extract_differentiable_boundary(prob, radius=2):
    """
    P4-E2: Differentiable morphological gradient directly from soft mask probability:
    dil = maxpool(p, k)
    ero = -maxpool(-p, k)
    b = dil - ero
    k = 2*r + 1 (5x5 for r=2 at 256x256)
    """
    k = 2 * radius + 1
    pad = radius
    dil = F.max_pool2d(prob, kernel_size=k, stride=1, padding=pad)
    ero = -F.max_pool2d(-prob, kernel_size=k, stride=1, padding=pad)
    b = (dil - ero).clamp(0.0, 1.0)
    return b


class ScaleAwareBoundaryLoss(nn.Module):
    """
    P4-E2: Lesion-scale-aware boundary supervision.
    Supports two modes:
      - 'from_mask' (default): derives predicted boundary directly from main mask probability
        via differentiable morphological gradient. Direct contour supervision, 0 extra params.
      - 'head': uses predictions from auxiliary boundary head (3x3 conv, training only).
    """
    def __init__(self, boundary_mode='from_mask', radius=2, a_ref=0.05, gamma=0.5, w_max=4.0):
        super().__init__()
        self.boundary_mode = boundary_mode
        self.radius = radius
        self.a_ref = a_ref
        self.gamma = gamma
        self.w_max = w_max
        self.dice = DiceLoss(from_logits=False)

    def forward(self, pred_main_logits, gt_mask, gt_boundary, pred_boundary_logits=None):
        """
        pred_main_logits: [B, 1, H, W]
        gt_mask: [B, 1, H, W]
        gt_boundary: [B, 1, H, W]
        pred_boundary_logits: [B, 1, H, W] or None
        """
        if self.boundary_mode == 'from_mask' or pred_boundary_logits is None:
            prob = torch.sigmoid(pred_main_logits.float())
            pred_b = extract_differentiable_boundary(prob, radius=self.radius)
        else:
            pred_b = torch.sigmoid(pred_boundary_logits.float())

        # Scale weights normalized to batch-mean 1
        w_i = compute_normalized_scale_weights(
            gt_mask, a_ref=self.a_ref, gamma=self.gamma, w_max=self.w_max
        )

        # BCE on boundary with scale weights
        bce_map = F.binary_cross_entropy(pred_b.clamp(1e-7, 1.0 - 1e-7), gt_boundary.float(), reduction='none')
        bce_per_sample = bce_map.mean(dim=(-2, -1), keepdim=True)
        weighted_bce = (bce_per_sample * w_i).mean()

        # Dice on boundary
        loss_dice = self.dice(pred_b, gt_boundary)

        return weighted_bce + loss_dice


class PerturbationConsistencyLoss(nn.Module):
    """
    P4-E3: Corruption-aware consistency loss on soft masks:
    L_c = (1 - SoftDice(p_corrupt, stopgrad(p_clean))) + MSE(p_corrupt, stopgrad(p_clean))
    Scale-invariant (SoftDice prevents domination by background/large objects).
    """
    def __init__(self):
        super().__init__()
        self.soft_dice = SoftDiceLoss()

    def forward(self, pred_corrupt_logits, pred_clean_logits):
        p_corrupt = torch.sigmoid(pred_corrupt_logits.float())
        p_clean = torch.sigmoid(pred_clean_logits.float()).detach()

        loss_dice = self.soft_dice(p_corrupt, p_clean)
        loss_mse = F.mse_loss(p_corrupt, p_clean)

        return loss_dice + loss_mse


def apply_corruption(images, speckle_sigma=None, contrast_factor=None, blur_sigma=None):
    """
    P4-E3: Per-sample random ultrasound corruption:
    - Multiplicative speckle: sigma ~ U(0.05, 0.30)
    - Contrast factor: factor ~ U(0.5, 1.0)
    - Optional Gaussian blur: sigma_blur ~ U(0, 1.0)
    """
    B, C, H, W = images.shape
    device = images.device
    corrupted = images.clone().float()

    # Multiplicative speckle: I + I * eta, eta ~ N(0, sigma^2)
    if speckle_sigma is None:
        sigma = torch.empty(B, 1, 1, 1, device=device).uniform_(0.05, 0.30)
    else:
        sigma = torch.full((B, 1, 1, 1), speckle_sigma, device=device)
    noise = torch.randn_like(corrupted) * sigma
    corrupted = corrupted + corrupted * noise

    # Dynamic contrast reduction: mean + factor * (I - mean)
    if contrast_factor is None:
        factor = torch.empty(B, 1, 1, 1, device=device).uniform_(0.5, 1.0)
    else:
        factor = torch.full((B, 1, 1, 1), contrast_factor, device=device)
    mean = corrupted.mean(dim=(-2, -1), keepdim=True)
    corrupted = mean + factor * (corrupted - mean)

    # Optional slight blur
    if blur_sigma is None:
        blur_s = random.uniform(0.0, 1.0)
    else:
        blur_s = blur_sigma

    if blur_s > 0.1:
        # 3x3 box blur approximation
        kernel = torch.ones(C, 1, 3, 3, device=device) / 9.0
        corrupted = F.conv2d(corrupted, kernel, padding=1, groups=C)

    return corrupted.clamp(0.0, 1.0)
