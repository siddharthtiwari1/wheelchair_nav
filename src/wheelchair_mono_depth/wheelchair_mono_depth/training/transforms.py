"""DPT-compatible transforms for Depth Anything V2 fine-tuning.

ViT-based DPT requires input dimensions divisible by patch size (14 for DINOv2).
Standard ImageNet normalization is applied.

Training augmentations include:
- Random horizontal flip
- Color jitter (brightness, contrast, saturation)
- Gaussian noise (sigma 0-0.02)
- Random erasing (5-15% of image)
- Motion blur (kernel 3-7)
- Multi-scale resize (384-518, multiples of 14)
"""

import numpy as np
import cv2 as _cv2
import torch
import torchvision.transforms.functional as TF


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def ensure_multiple_of(x, multiple=14):
    """Round up to nearest multiple (ViT patch size)."""
    return ((x + multiple - 1) // multiple) * multiple


class TrainTransform:
    """Training transforms: resize, augment, normalize."""

    def __init__(self, size=518, max_depth=10.0, multi_scale=True):
        self.size = ensure_multiple_of(size)
        self.max_depth = max_depth
        self.multi_scale = multi_scale
        # Valid multi-scale sizes: 384, 392, 406, ..., 518 (multiples of 14)
        self.scale_sizes = [ensure_multiple_of(s) for s in range(384, size + 1, 14)]
        if self.size not in self.scale_sizes:
            self.scale_sizes.append(self.size)

    def __call__(self, rgb, depth):
        """
        Args:
            rgb: numpy HxWx3 uint8 BGR (from cv2)
            depth: numpy HxW float32 in meters
        Returns:
            rgb_tensor: 3xHxW float32 normalized
            depth_tensor: 1xHxW float32 in meters
            valid_mask: 1xHxW bool
        """
        # BGR -> RGB
        rgb = rgb[:, :, ::-1].copy()

        # Random horizontal flip
        if np.random.random() > 0.5:
            rgb = np.fliplr(rgb).copy()
            depth = np.fliplr(depth).copy()

        # Color jitter on RGB only
        rgb_tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        if np.random.random() > 0.5:
            rgb_tensor = TF.adjust_brightness(rgb_tensor, 0.8 + np.random.random() * 0.4)
        if np.random.random() > 0.5:
            rgb_tensor = TF.adjust_contrast(rgb_tensor, 0.8 + np.random.random() * 0.4)
        if np.random.random() > 0.5:
            rgb_tensor = TF.adjust_saturation(rgb_tensor, 0.8 + np.random.random() * 0.4)

        # Gaussian noise (sigma 0-0.02, applied 50% of the time)
        if np.random.random() > 0.5:
            sigma = np.random.uniform(0.0, 0.02)
            noise = torch.randn_like(rgb_tensor) * sigma
            rgb_tensor = (rgb_tensor + noise).clamp(0.0, 1.0)

        # Motion blur (kernel size 3-7, applied 30% of the time)
        if np.random.random() > 0.7:
            ksize = np.random.choice([3, 5, 7])
            # Horizontal or vertical motion blur
            if np.random.random() > 0.5:
                kernel = np.zeros((ksize, ksize), dtype=np.float32)
                kernel[ksize // 2, :] = 1.0 / ksize
            else:
                kernel = np.zeros((ksize, ksize), dtype=np.float32)
                kernel[:, ksize // 2] = 1.0 / ksize
            # Apply via numpy (faster than torch conv for small kernels)
            rgb_np = rgb_tensor.permute(1, 2, 0).numpy()
            rgb_np = _cv2.filter2D(rgb_np, -1, kernel)
            rgb_tensor = torch.from_numpy(rgb_np).permute(2, 0, 1).clamp(0.0, 1.0)

        # Multi-scale resize (random size from valid set) or fixed size
        if self.multi_scale and len(self.scale_sizes) > 1 and np.random.random() > 0.5:
            target_size = int(np.random.choice(self.scale_sizes))
        else:
            target_size = self.size

        rgb_tensor = TF.resize(rgb_tensor, [target_size, target_size], antialias=True)

        depth_tensor = torch.from_numpy(depth).unsqueeze(0).float()
        depth_tensor = TF.resize(depth_tensor, [target_size, target_size],
                                 interpolation=TF.InterpolationMode.NEAREST)

        # Valid mask before normalization
        valid_mask = (depth_tensor > 0) & (depth_tensor <= self.max_depth)

        # Random erasing (5-15% of image, applied 30% of the time)
        # Only erase RGB, leave depth/mask intact (simulates occlusion)
        if np.random.random() > 0.7:
            h, w = rgb_tensor.shape[1], rgb_tensor.shape[2]
            erase_ratio = np.random.uniform(0.05, 0.15)
            erase_h = int(h * np.sqrt(erase_ratio))
            erase_w = int(w * np.sqrt(erase_ratio))
            top = np.random.randint(0, max(1, h - erase_h))
            left = np.random.randint(0, max(1, w - erase_w))
            # Erase to mean color (gray)
            rgb_tensor[:, top:top + erase_h, left:left + erase_w] = 0.5

        # Normalize RGB
        rgb_tensor = TF.normalize(rgb_tensor, IMAGENET_MEAN, IMAGENET_STD)

        return rgb_tensor, depth_tensor, valid_mask


class ValTransform:
    """Validation transforms: resize + normalize only."""

    def __init__(self, size=518, max_depth=10.0):
        self.size = ensure_multiple_of(size)
        self.max_depth = max_depth

    def __call__(self, rgb, depth):
        rgb = rgb[:, :, ::-1].copy()

        rgb_tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        rgb_tensor = TF.resize(rgb_tensor, [self.size, self.size], antialias=True)

        depth_tensor = torch.from_numpy(depth).unsqueeze(0).float()
        depth_tensor = TF.resize(depth_tensor, [self.size, self.size],
                                 interpolation=TF.InterpolationMode.NEAREST)

        valid_mask = (depth_tensor > 0) & (depth_tensor <= self.max_depth)

        rgb_tensor = TF.normalize(rgb_tensor, IMAGENET_MEAN, IMAGENET_STD)

        return rgb_tensor, depth_tensor, valid_mask


def denormalize(tensor):
    """Reverse ImageNet normalization for visualization."""
    mean = torch.tensor(IMAGENET_MEAN, device=tensor.device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=tensor.device).view(3, 1, 1)
    return tensor * std + mean
