import torch
import torch.nn.functional as F


def add_speckle_noise(images, sigma=0.15):
    """
    Applies multiplicative ultrasound speckle noise:
    I_noisy = I + I * eta, where eta ~ N(0, sigma^2).
    
    Args:
        images: FloatTensor [B, C, H, W] in range [0, 1]
        sigma: Noise standard deviation
    Returns:
        noisy_images: FloatTensor [B, C, H, W] clamped to [0, 1]
    """
    if sigma <= 0.0:
        return images
    noise = torch.randn_like(images) * sigma
    noisy_images = images + images * noise
    return noisy_images.clamp(0.0, 1.0)


def reduce_ultrasound_contrast(images, factor=0.7):
    """
    Simulates ultrasound acoustic attenuation and loss of dynamic contrast.
    
    Args:
        images: FloatTensor [B, C, H, W] in [0, 1]
        factor: Contrast multiplier (< 1.0 reduces contrast)
    """
    mean = images.mean(dim=(-2, -1), keepdim=True)
    attenuated = mean + factor * (images - mean)
    return attenuated.clamp(0.0, 1.0)


def apply_ultrasound_corruption(images, speckle_sigma=0.15, contrast_factor=0.75):
    """
    Combines ultrasound speckle noise and contrast attenuation for
    corruption-aware consistency training.
    """
    corrupted = add_speckle_noise(images, sigma=speckle_sigma)
    if contrast_factor < 1.0:
        corrupted = reduce_ultrasound_contrast(corrupted, factor=contrast_factor)
    return corrupted
