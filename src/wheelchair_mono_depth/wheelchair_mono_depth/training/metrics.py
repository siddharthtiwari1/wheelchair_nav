"""Evaluation metrics for monocular depth estimation.

Standard metrics matching the Waymo paper and depth estimation benchmarks:
- AbsRel: mean(|pred - gt| / gt)
- RMSE: sqrt(mean((pred - gt)^2))
- delta < 1.25: percentage where max(pred/gt, gt/pred) < 1.25
- SiLog: scale-invariant log error
"""

import torch
import numpy as np


def compute_depth_metrics(pred, gt, valid_mask, max_depth=10.0):
    """Compute standard monocular depth metrics.

    Args:
        pred: (B, 1, H, W) or (B, H, W) predicted depth in meters
        gt: (B, 1, H, W) or (B, H, W) ground truth depth in meters
        valid_mask: (B, 1, H, W) or (B, H, W) boolean mask
        max_depth: maximum depth to consider

    Returns:
        dict with abs_rel, rmse, delta_1, delta_2, delta_3, silog
    """
    pred = pred.squeeze(1) if pred.dim() == 4 else pred
    gt = gt.squeeze(1) if gt.dim() == 4 else gt
    mask = valid_mask.squeeze(1) if valid_mask.dim() == 4 else valid_mask

    # Additional range filtering
    mask = mask & (gt > 1e-3) & (gt <= max_depth) & (pred > 1e-3)

    pred_v = pred[mask]
    gt_v = gt[mask]

    if pred_v.numel() < 10:
        return {
            'abs_rel': 0.0, 'rmse': 0.0,
            'delta_1': 0.0, 'delta_2': 0.0, 'delta_3': 0.0,
            'silog': 0.0, 'n_valid': 0,
        }

    # AbsRel
    abs_rel = (torch.abs(pred_v - gt_v) / gt_v).mean().item()

    # RMSE
    rmse = torch.sqrt(((pred_v - gt_v) ** 2).mean()).item()

    # Delta thresholds
    ratio = torch.max(pred_v / gt_v, gt_v / pred_v)
    delta_1 = (ratio < 1.25).float().mean().item()
    delta_2 = (ratio < 1.25 ** 2).float().mean().item()
    delta_3 = (ratio < 1.25 ** 3).float().mean().item()

    # SiLog
    log_diff = torch.log(pred_v) - torch.log(gt_v)
    silog = torch.sqrt((log_diff ** 2).mean() - 0.5 * log_diff.mean() ** 2).item()

    return {
        'abs_rel': abs_rel,
        'rmse': rmse,
        'delta_1': delta_1,
        'delta_2': delta_2,
        'delta_3': delta_3,
        'silog': silog,
        'n_valid': pred_v.numel(),
    }


class MetricTracker:
    """Accumulates metrics across batches for epoch-level reporting."""

    def __init__(self):
        self.reset()

    def reset(self):
        self._sum = {}
        self._count = 0

    def update(self, metrics):
        n = metrics.get('n_valid', 1)
        if n == 0:
            return
        for k, v in metrics.items():
            if k == 'n_valid':
                continue
            self._sum[k] = self._sum.get(k, 0.0) + v
        self._count += 1

    def compute(self):
        if self._count == 0:
            return {}
        return {k: v / self._count for k, v in self._sum.items()}

    def summary_str(self):
        m = self.compute()
        if not m:
            return 'No metrics'
        return (f"AbsRel={m.get('abs_rel', 0):.4f} | "
                f"RMSE={m.get('rmse', 0):.3f} | "
                f"d<1.25={m.get('delta_1', 0):.3f} | "
                f"SiLog={m.get('silog', 0):.4f}")
