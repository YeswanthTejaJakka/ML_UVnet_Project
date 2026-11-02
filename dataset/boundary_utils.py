import torch
import torch.nn.functional as F
import numpy as np


def extract_boundary_tensor(mask, radius=1):
    """
    Extracts boundary band from binary mask tensor using morphological
    dilation and erosion via max pooling (100% PyTorch native, runs on CPU or GPU).
    
    Args:
        mask: Tensor of shape [B, 1, H, W] or [1, H, W] or [H, W] with values in {0, 1}
        radius: Half-width of boundary strip (kernel_size = 2 * radius + 1)
    Returns:
        boundary: Tensor of same shape as mask with 1s on the boundary and 0s elsewhere
    """
    orig_dim = mask.dim()
    if orig_dim == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif orig_dim == 3:
        mask = mask.unsqueeze(1) if mask.shape[0] != 1 else mask.unsqueeze(0)

    kernel_size = 2 * radius + 1
    padding = radius

    # Morphological Dilation = max_pool2d(mask)
    dilation = F.max_pool2d(mask.float(), kernel_size=kernel_size, stride=1, padding=padding)
    
    # Morphological Erosion = 1 - max_pool2d(1 - mask)
    erosion = 1.0 - F.max_pool2d(1.0 - mask.float(), kernel_size=kernel_size, stride=1, padding=padding)
    
    # Boundary strip
    boundary = (dilation - erosion).clamp(0.0, 1.0)

    if orig_dim == 2:
        return boundary.squeeze(0).squeeze(0)
    elif orig_dim == 3:
        return boundary.squeeze(1) if orig_dim == 3 and boundary.shape[0] == 1 else boundary
    return boundary


def compute_lesion_scale_category(mask_tensor_or_np, small_thresh=0.02, large_thresh=0.10):
    """
    Categorizes lesion into 'small', 'medium', or 'large' based on the
    fraction of pixels occupied by the nodule mask relative to total image area.
    
    Args:
        mask_tensor_or_np: Binary mask [H, W] or [1, H, W]
    Returns:
        category: str in {'empty', 'small', 'medium', 'large'}
        rel_area: float fraction of total image pixels
    """
    if isinstance(mask_tensor_or_np, torch.Tensor):
        area = (mask_tensor_or_np > 0.5).float().sum().item()
        total = mask_tensor_or_np.numel()
    else:
        area = np.sum(mask_tensor_or_np > 0.5)
        total = mask_tensor_or_np.size

    rel_area = float(area) / float(total + 1e-8)

    if rel_area < 1e-5:
        return 'empty', rel_area
    elif rel_area < small_thresh:
        return 'small', rel_area
    elif rel_area < large_thresh:
        return 'medium', rel_area
    else:
        return 'large', rel_area


def compute_scale_weight(mask_tensor, alpha=1.0, gamma=0.5, min_weight=1.0, max_weight=5.0):
    """
    Computes a sample-wise scale weight for boundary supervision.
    Small lesions receive a higher boundary weight to penalize fuzzy edges
    and counteract loss vanishing on small objects.
    
    w = clip(alpha * (1 / (rel_area + eps))^gamma, min_weight, max_weight)
    """
    area = (mask_tensor > 0.5).float().sum(dim=(-2, -1), keepdim=True)  # [B, 1, 1, 1]
    total_pixels = mask_tensor.shape[-2] * mask_tensor.shape[-1]
    rel_area = area / float(total_pixels)

    # Compute inverse-scale dynamic weight
    weight = alpha * torch.pow(1.0 / (rel_area + 1e-4), gamma)
    # If mask is empty, set weight to min_weight
    weight = torch.where(rel_area < 1e-5, torch.tensor(min_weight, device=mask_tensor.device), weight)
    weight = weight.clamp(min=min_weight, max=max_weight)
    return weight
