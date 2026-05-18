"""
E3: Goal-to-Heading Encoder — MLP on polar goal representation.

Encodes relative goal as (distance, bearing, cos(bearing), sin(bearing)) → 64-d.

Why separate goal encoder (not BEV channel):
- Current: goal as 2D Gaussian blob in BEV channel 2, processed by ResNet-18.
  Uses 11.2M params to encode 2 DOF. Massively overparameterized.
- CrowdSurfer: scalar heading angle → Linear(1,4). Loses distance info entirely.
- Crowd-FM: 2D vector (dx, dy) → MLP. Better but no angle wrapping protection.
- Ours: (distance, bearing, cos, sin) → MLP → 64-d. Sin/cos prevents discontinuity
  at ±π. Distance preserved. 4→64 is richer than all baselines.

~4.4K params.
"""

import math
import torch
import torch.nn as nn


class GoalEncoder(nn.Module):
    """Encodes relative goal into 64-d features.

    Input:  (B, 4) = [normalized_distance, normalized_bearing, cos(bearing), sin(bearing)]
    Output: (B, 64)
    """

    def __init__(self, output_dim=64):
        super().__init__()
        self.output_dim = output_dim

        self.mlp = nn.Sequential(
            nn.Linear(4, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, goal_features):
        """
        Args:
            goal_features: (B, 4) = [norm_dist, norm_bearing, cos_b, sin_b]

        Returns:
            features: (B, output_dim)
        """
        return self.mlp(goal_features)


def compute_goal_features(goal_dx, goal_dy, max_dist=10.0):
    """Compute 4-d goal feature vector from relative goal position.

    Args:
        goal_dx: goal x in base_link frame (meters), scalar or (B,) tensor
        goal_dy: goal y in base_link frame (meters), scalar or (B,) tensor
        max_dist: normalization distance (meters)

    Returns:
        features: (4,) or (B, 4) = [norm_dist, norm_bearing, cos_b, sin_b]
    """
    if isinstance(goal_dx, (int, float)):
        distance = min(math.sqrt(goal_dx ** 2 + goal_dy ** 2), max_dist)
        bearing = math.atan2(goal_dy, goal_dx)
        return [
            distance / max_dist,        # [0, 1]
            bearing / math.pi,           # [-1, 1]
            math.cos(bearing),           # [-1, 1]
            math.sin(bearing),           # [-1, 1]
        ]
    else:
        # Batched tensor version
        distance = torch.sqrt(goal_dx ** 2 + goal_dy ** 2).clamp(max=max_dist)
        bearing = torch.atan2(goal_dy, goal_dx)
        return torch.stack([
            distance / max_dist,
            bearing / math.pi,
            torch.cos(bearing),
            torch.sin(bearing),
        ], dim=-1)
