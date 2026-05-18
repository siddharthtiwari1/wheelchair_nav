"""
E4: Velocity Context Encoder — MLP on odom history.

Encodes 10 timesteps of (v, ω, θ) into 64-d velocity context features.

Why this matters (Crowd-FM has NOTHING like this):
- CrowdSurfer: no ego-motion input at all. Robot is "amnesiac" about its own velocity.
- Crowd-FM: no ego-motion input. Same problem.
- Ours: 10 timesteps × (v, ω, θ) = 30-d input. Encodes velocity history, turning
  patterns, acceleration state. Enables smooth continuations (can't generate v=0.25
  if current v=0.02).

~5.8K params (unchanged from current KinoFlow odom encoder).
"""

import torch
import torch.nn as nn


class VelocityContextEncoder(nn.Module):
    """Encodes odom history into 64-d velocity context features.

    Input:  (B, 30) — flattened [v, ω, θ] × 10 steps
    Output: (B, 64)
    """

    def __init__(self, input_dim=30, output_dim=64):
        super().__init__()
        self.output_dim = output_dim

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, odom_history):
        """
        Args:
            odom_history: (B, 30) flattened [v, omega, theta] × 10 steps

        Returns:
            features: (B, output_dim)
        """
        return self.mlp(odom_history)
