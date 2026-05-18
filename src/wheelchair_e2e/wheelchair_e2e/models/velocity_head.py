"""
Velocity Head — shared MLP for both E2E-1 and E2E-2.

Replaces waypoint/diffusion heads with direct velocity output.
Used by:
  - E2E-1 (NoMaD-Velocity): attached to ViNT transformer backbone
  - E2E-2 (BEV-Velocity): already integrated in BEVVelocityNet
"""

import torch
import torch.nn as nn


class VelocityHead(nn.Module):
    """
    MLP head: feature vector → (v, omega) in physical units.

    For E2E-1: input_dim=256 (ViNT transformer hidden)
    For standalone use with any backbone.
    """

    def __init__(self, input_dim=256, v_max=0.25, w_max=1.0):
        super().__init__()
        self.v_max = v_max
        self.w_max = w_max

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Tanh()  # output in [-1, 1]
        )

    def forward(self, features):
        """
        Args:
            features: (B, input_dim) feature vector from backbone
        Returns:
            velocity: (B, 2) = (v, omega)
                v in [0, v_max] m/s
                omega in [-w_max, w_max] rad/s
        """
        raw = self.mlp(features)  # shape: (B, 2)
        v = (raw[:, 0] + 1) / 2 * self.v_max   # [0, v_max]
        omega = raw[:, 1] * self.w_max           # [-w_max, w_max]
        return torch.stack([v, omega], dim=-1)
