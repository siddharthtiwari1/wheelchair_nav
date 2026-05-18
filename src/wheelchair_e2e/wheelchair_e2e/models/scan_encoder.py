"""
E1: Static Scene Encoder — 1D CNN on polar scan.

Encodes single-frame /scan_fused (720 polar ranges) into static geometry features.

Why 1D CNN on polar scan (not BEV grid, not PointNet):
- /scan_fused is natively 720 polar ranges. Rasterizing to 200×200 then ResNet-18
  (11.2M) is 10,000x overparameterized for encoding obstacle geometry.
- 1D CNN preserves polar structure: angular neighborhoods are physical spatial
  neighborhoods. A doorway is a contiguous dip in range values.
- PointNet (Crowd-FM) needs (N,2) Cartesian — extra conversion, loses angular ordering.

~83K params (vs 170K for CrowdSurfer's Conv2d grid encoder, vs 11.2M for ResNet-18).
"""

import torch
import torch.nn as nn


class StaticSceneEncoder(nn.Module):
    """Encodes single-frame /scan_fused into 128-d static geometry features.

    Input:  (B, 1, scan_points) — polar ranges at time t
    Output: (B, 128)
    """

    def __init__(self, scan_points=720, output_dim=128):
        super().__init__()
        self.scan_points = scan_points
        self.output_dim = output_dim

        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, output_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )

    def forward(self, scan):
        """
        Args:
            scan: (B, 1, scan_points) or (B, scan_points) polar ranges

        Returns:
            features: (B, output_dim)
        """
        if scan.dim() == 2:
            scan = scan.unsqueeze(1)  # (B, 1, N)
        return self.cnn(scan)
