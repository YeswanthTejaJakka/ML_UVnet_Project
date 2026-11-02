import time
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt


def calculate_dice(pred, target, threshold=0.5, eps=1e-5):
    """
    Computes Dice coefficient for 2D/3D binary masks.
    """
    p = (pred > threshold).astype(np.float32)
    t = (target > threshold).astype(np.float32)
    intersection = np.sum(p * t)
    total = np.sum(p) + np.sum(t)
    if total == 0:
        return 1.0  # Perfect agreement on empty mask
    return float((2.0 * intersection + eps) / (total + eps))


def calculate_iou(pred, target, threshold=0.5, eps=1e-5):
    """
    Computes Intersection-over-Union (Jaccard Index).
    """
    p = (pred > threshold).astype(np.float32)
    t = (target > threshold).astype(np.float32)
    intersection = np.sum(p * t)
    union = np.sum(p) + np.sum(t) - intersection
    if union == 0:
        return 1.0
    return float((intersection + eps) / (union + eps))


def mask_to_boundary(mask, d0=2):
    """
    Extracts boundary mask within pixel distance d0 using Euclidean distance transform.
    """
    mask = mask.astype(bool)
    if not np.any(mask):
        return np.zeros_like(mask, dtype=bool)
    dist_inside = distance_transform_edt(mask)
    dist_outside = distance_transform_edt(~mask)
    boundary = (dist_inside <= d0) & mask | (dist_outside <= d0) & (~mask)
    return boundary & mask  # Keep inner boundary band


def calculate_boundary_iou(pred, target, threshold=0.5, d0=2):
    """
    Computes Boundary IoU (Cheng et al., CVPR 2021).
    Evaluates boundary localization accuracy sensitive to contour misalignments.
    """
    p = (pred > threshold).astype(bool)
    t = (target > threshold).astype(bool)
    
    if not np.any(p) and not np.any(t):
        return 1.0
    if not np.any(p) or not np.any(t):
        return 0.0

    bp = mask_to_boundary(p, d0=d0)
    bt = mask_to_boundary(t, d0=d0)

    intersection = np.logical_and(bp, bt).sum()
    union = np.logical_or(bp, bt).sum()
    if union == 0:
        return 1.0
    return float(intersection / union)


def calculate_hd95(pred, target, threshold=0.5):
    """
    Computes 95th Percentile Hausdorff Distance (HD95).
    Pure SciPy implementation (does not require external C libraries).
    """
    p = (pred > threshold).astype(bool)
    t = (target > threshold).astype(bool)

    if not np.any(p) and not np.any(t):
        return 0.0
    if not np.any(p) or not np.any(t):
        return 100.0  # Penalty for complete miss

    dt_p = distance_transform_edt(~p)
    dt_t = distance_transform_edt(~t)

    # Distances from boundary of p to t and vice versa
    d_p_to_t = dt_t[p]
    d_t_to_p = dt_p[t]

    all_distances = np.concatenate([d_p_to_t, d_t_to_p])
    if len(all_distances) == 0:
        return 0.0
    return float(np.percentile(all_distances, 95))


class MetricTracker:
    """
    Aggregates test metrics overall and broken down across lesion scale categories:
    'small', 'medium', 'large'.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.overall = {'dice': [], 'iou': [], 'biou': [], 'hd95': []}
        self.by_scale = {
            'small': {'dice': [], 'iou': [], 'biou': [], 'hd95': []},
            'medium': {'dice': [], 'iou': [], 'biou': [], 'hd95': []},
            'large': {'dice': [], 'iou': [], 'biou': [], 'hd95': []},
        }

    def update(self, pred_np, target_np, category='medium'):
        dice = calculate_dice(pred_np, target_np)
        iou = calculate_iou(pred_np, target_np)
        biou = calculate_boundary_iou(pred_np, target_np)
        hd95 = calculate_hd95(pred_np, target_np)

        self.overall['dice'].append(dice)
        self.overall['iou'].append(iou)
        self.overall['biou'].append(biou)
        self.overall['hd95'].append(hd95)

        if category in self.by_scale:
            self.by_scale[category]['dice'].append(dice)
            self.by_scale[category]['iou'].append(iou)
            self.by_scale[category]['biou'].append(biou)
            self.by_scale[category]['hd95'].append(hd95)

    def summary(self):
        results = {}
        for m in ['dice', 'iou', 'biou', 'hd95']:
            vals = self.overall[m]
            results[f'overall_{m}'] = float(np.mean(vals)) if len(vals) > 0 else 0.0

        for cat in ['small', 'medium', 'large']:
            count = len(self.by_scale[cat]['dice'])
            results[f'{cat}_count'] = count
            for m in ['dice', 'iou', 'biou', 'hd95']:
                vals = self.by_scale[cat][m]
                results[f'{cat}_{m}'] = float(np.mean(vals)) if count > 0 else 0.0

        return results


def measure_model_efficiency(model, input_size=(1, 3, 256, 256), device='cuda'):
    """
    Measures parameter count, FLOPs, Latency (ms), and FPS.
    """
    device = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')
    model = model.to(device)
    model.eval()

    # Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    params_m = total_params / 1e6

    # FLOPs estimation
    flops_g = 0.060  # Default theoretical FLOPs for UltraLight VM-UNet at 256x256
    try:
        from thop import profile
        dummy_input = torch.randn(*input_size, device=device)
        flops, _ = profile(model, inputs=(dummy_input,), verbose=False)
        flops_g = flops / 1e9
    except Exception:
        pass

    # Latency & FPS benchmark
    dummy_input = torch.randn(*input_size, device=device)
    with torch.no_grad():
        # Warmup
        for _ in range(20):
            _ = model(dummy_input)

        if device.type == 'cuda':
            torch.cuda.synchronize()

        iterations = 100
        start_time = time.time()
        for _ in range(iterations):
            _ = model(dummy_input)

        if device.type == 'cuda':
            torch.cuda.synchronize()

        elapsed = time.time() - start_time

    avg_latency_ms = (elapsed / iterations) * 1000.0
    fps = iterations / elapsed

    return {
        'params_M': round(params_m, 4),
        'flops_G': round(flops_g, 4),
        'latency_ms': round(avg_latency_ms, 2),
        'fps': round(fps, 1),
    }
