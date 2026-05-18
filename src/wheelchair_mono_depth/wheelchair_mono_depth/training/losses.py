"""Loss functions for monocular depth estimation fine-tuning.

SiLog (Scale-Invariant Logarithmic) loss is the standard for depth estimation,
used by Eigen et al., Depth Anything V2, and others. Combined with L1 for
sharper edges critical to wheelchair obstacle detection.

Edge-aware smoothness loss penalizes depth gradients except at RGB edges,
preserving object boundaries while smoothing flat regions.

Camera-aware confidence weighting accounts for RealSense depth noise:
D435i degrades beyond 2m (error proportional to z^2/(f*b)), so pixels beyond
2m are down-weighted. D455 is reliable throughout its range.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SiLogLoss(nn.Module):
    """Scale-Invariant Logarithmic loss.

    d_i = log(pred_i) - log(gt_i)
    L = sqrt(mean(d_i^2) - variance_weight * mean(d_i)^2)

    Only computed on valid pixels (where mask is True).
    """

    def __init__(self, variance_weight=0.5):
        super().__init__()
        self.variance_weight = variance_weight

    def forward(self, pred, gt, valid_mask, camera_weights=None):
        mask = valid_mask.squeeze(1) if valid_mask.dim() == 4 else valid_mask
        pred_m = pred.squeeze(1) if pred.dim() == 4 else pred
        gt_m = gt.squeeze(1) if gt.dim() == 4 else gt

        pred_valid = pred_m[mask]
        gt_valid = gt_m[mask]

        if pred_valid.numel() < 10:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        # Clamp to avoid log(0)
        pred_valid = pred_valid.clamp(min=1e-3)
        gt_valid = gt_valid.clamp(min=1e-3)

        log_diff = torch.log(pred_valid) - torch.log(gt_valid)

        if camera_weights is not None:
            w = camera_weights.squeeze(1) if camera_weights.dim() == 4 else camera_weights
            w_valid = w[mask]
            # Weighted mean
            w_sum = w_valid.sum().clamp(min=1e-6)
            weighted_sq = (w_valid * log_diff ** 2).sum() / w_sum
            weighted_mean = (w_valid * log_diff).sum() / w_sum
            silog = torch.sqrt(
                (weighted_sq - self.variance_weight * weighted_mean ** 2).clamp(min=1e-8)
            )
        else:
            silog = torch.sqrt(
                ((log_diff ** 2).mean()
                 - self.variance_weight * (log_diff.mean() ** 2)).clamp(min=1e-8)
            )
        return silog


class MaskedL1Loss(nn.Module):
    """L1 loss computed only on valid depth pixels."""

    def forward(self, pred, gt, valid_mask, camera_weights=None):
        mask = valid_mask.squeeze(1) if valid_mask.dim() == 4 else valid_mask
        pred_m = pred.squeeze(1) if pred.dim() == 4 else pred
        gt_m = gt.squeeze(1) if gt.dim() == 4 else gt

        pred_valid = pred_m[mask]
        gt_valid = gt_m[mask]

        if pred_valid.numel() < 10:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        l1 = torch.abs(pred_valid - gt_valid)

        if camera_weights is not None:
            w = camera_weights.squeeze(1) if camera_weights.dim() == 4 else camera_weights
            w_valid = w[mask]
            return (w_valid * l1).sum() / w_valid.sum().clamp(min=1e-6)

        return l1.mean()


class EdgeAwareSmoothnessLoss(nn.Module):
    """Edge-aware depth smoothness loss.

    Penalizes depth gradients except where RGB gradients are strong (object edges).
    This encourages smooth depth in textureless regions while preserving depth
    discontinuities at object boundaries.

    L = mean(|d_x depth| * exp(-|d_x rgb|) + |d_y depth| * exp(-|d_y rgb|))

    The RGB image must be denormalized or raw (0-1 range works best).
    """

    def forward(self, depth_pred, rgb, valid_mask):
        """
        Args:
            depth_pred: (B, 1, H, W) predicted depth
            rgb: (B, 3, H, W) RGB image (normalized is fine - gradients still work)
            valid_mask: (B, 1, H, W) boolean mask
        """
        pred = depth_pred if depth_pred.dim() == 4 else depth_pred.unsqueeze(1)
        mask = valid_mask if valid_mask.dim() == 4 else valid_mask.unsqueeze(1)

        # Depth gradients
        depth_dx = torch.abs(pred[:, :, :, :-1] - pred[:, :, :, 1:])
        depth_dy = torch.abs(pred[:, :, :-1, :] - pred[:, :, 1:, :])

        # RGB gradients (mean across channels)
        rgb_dx = torch.abs(rgb[:, :, :, :-1] - rgb[:, :, :, 1:]).mean(dim=1, keepdim=True)
        rgb_dy = torch.abs(rgb[:, :, :-1, :] - rgb[:, :, 1:, :]).mean(dim=1, keepdim=True)

        # Edge-aware weights: suppress smoothness penalty at RGB edges
        weight_x = torch.exp(-rgb_dx)
        weight_y = torch.exp(-rgb_dy)

        # Apply valid mask (crop to match gradient dimensions)
        mask_x = mask[:, :, :, :-1] & mask[:, :, :, 1:]
        mask_y = mask[:, :, :-1, :] & mask[:, :, 1:, :]

        smooth_x = depth_dx * weight_x
        smooth_y = depth_dy * weight_y

        n_x = mask_x.sum().clamp(min=1)
        n_y = mask_y.sum().clamp(min=1)

        loss = (smooth_x[mask_x].sum() / n_x + smooth_y[mask_y].sum() / n_y) / 2.0
        return loss


class DepthEstimationLoss(nn.Module):
    """Combined SiLog + L1 + edge-aware smoothness loss for metric depth fine-tuning.

    L = lambda_silog * SiLog(pred, gt) + lambda_l1 * L1(pred, gt)
        + lambda_smooth * EdgeSmooth(pred, rgb)
    """

    def __init__(self, lambda_silog=1.0, lambda_l1=0.1, lambda_smooth=0.0,
                 variance_weight=0.5):
        super().__init__()
        self.lambda_silog = lambda_silog
        self.lambda_l1 = lambda_l1
        self.lambda_smooth = lambda_smooth
        self.silog = SiLogLoss(variance_weight=variance_weight)
        self.l1 = MaskedL1Loss()
        self.smooth = EdgeAwareSmoothnessLoss() if lambda_smooth > 0 else None

    def forward(self, pred, gt, valid_mask, rgb=None, camera_weights=None):
        loss_silog = self.silog(pred, gt, valid_mask, camera_weights)
        loss_l1 = self.l1(pred, gt, valid_mask, camera_weights)
        total = self.lambda_silog * loss_silog + self.lambda_l1 * loss_l1

        if self.smooth is not None and rgb is not None:
            loss_smooth = self.smooth(pred, rgb, valid_mask)
            total = total + self.lambda_smooth * loss_smooth

        return total
