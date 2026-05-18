"""
E2: Dynamic Obstacle Encoder — Temporal scan residuals + 1D CNN + Self-Attention.

Derives dynamic obstacle information directly from ego-compensated temporal scan
residuals, eliminating the need for an external tracker (unlike CrowdSurfer/Crowd-FM).

Pipeline:
    1. Ego-motion compensate scan[t-4:t] to current frame using odom deltas
    2. Compute pairwise temporal residuals: |scan[t] - warped_scan[t-k]| for k=1..4
    3. Stack residuals as multi-channel input (B, 4, 720)
    4. 1D CNN extracts motion features per angular bin
    5. Self-attention aggregates across full scan

Why temporal scan differencing (not external tracker):
- CrowdSurfer needs a separate obstacle tracker producing (x,y,vx,vy) per agent.
- Crowd-FM's Transformer over (N,4) agent states: same tracker dependency.
- We derive dynamics directly from scan_fused: ego-compensated temporal residuals
  reveal what moved, where, and how fast. No tracker needed.
- Works for ANY moving object (people, carts, doors) without detection/classification.

~200K params (CNN: ~83K + Attention: ~130K).
"""

import torch
import torch.nn as nn
import numpy as np


class DynamicObstacleEncoder(nn.Module):
    """Encodes T=5 ego-compensated scan frames into 128-d dynamic features.

    Input:  (B, temporal_frames-1, scan_points) — 4 temporal residual channels
    Output: (B, 128)
    """

    def __init__(self, scan_points=720, temporal_frames=5, output_dim=128,
                 n_heads=4, n_attn_layers=2):
        super().__init__()
        self.scan_points = scan_points
        self.temporal_frames = temporal_frames
        self.n_residuals = temporal_frames - 1  # 4 residual channels
        self.output_dim = output_dim

        # 1D CNN on stacked temporal residuals
        self.cnn = nn.Sequential(
            nn.Conv1d(self.n_residuals, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, output_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
        )
        # After CNN: (B, 128, ~90) spatial positions

        # Self-attention over spatial positions
        attn_layer = nn.TransformerEncoderLayer(
            d_model=output_dim,
            nhead=n_heads,
            dim_feedforward=output_dim * 4,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.self_attn = nn.TransformerEncoder(
            attn_layer, num_layers=n_attn_layers
        )

        # Pool + project
        self.output_proj = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, residuals):
        """
        Args:
            residuals: (B, n_residuals, scan_points) temporal residual channels
                       Each channel k = |scan[t] - warped_scan[t-k-1]|

        Returns:
            features: (B, output_dim)
        """
        # CNN: extract motion features per angular bin
        x = self.cnn(residuals)           # (B, 128, ~90)

        # Reshape for attention: (B, 128, L) -> (B, L, 128)
        x = x.permute(0, 2, 1)           # (B, L, 128)

        # Self-attention across spatial positions
        x = self.self_attn(x)            # (B, L, 128)

        # Mean pool across spatial positions
        x = x.mean(dim=1)               # (B, 128)

        return self.output_proj(x)       # (B, 128)


def compensate_scan(prev_ranges, prev_angles, odom_delta, current_angles=None):
    """Warp previous scan to current ego frame.

    Used to compute temporal residuals for dynamic obstacle detection.
    Reuses ego-motion compensation logic from bev_generator.py.

    Args:
        prev_ranges: (N,) range values from previous frame
        prev_angles: (N,) angle values from previous frame
        odom_delta: (3,) = (dx, dy, dtheta) from EKF between frames
        current_angles: (N,) angle values in current frame (for re-binning).
                        If None, uses prev_angles (approximate).

    Returns:
        warped_ranges: (N,) ranges warped to current frame
    """
    dx, dy, dtheta = odom_delta

    # Polar -> Cartesian in previous frame
    x = prev_ranges * np.cos(prev_angles)
    y = prev_ranges * np.sin(prev_angles)

    # Rotate + translate to current frame
    cos_dt = np.cos(-dtheta)
    sin_dt = np.sin(-dtheta)
    x_new = cos_dt * (x - dx) - sin_dt * (y - dy)
    y_new = sin_dt * (x - dx) + cos_dt * (y - dy)

    # Re-project to polar ranges in current frame
    warped_ranges = np.sqrt(x_new ** 2 + y_new ** 2)

    # Mask invalid (behind robot or too close)
    warped_ranges[warped_ranges < 0.05] = np.inf

    return warped_ranges.astype(np.float32)


def build_temporal_residuals(scan_buffer, odom_deltas, scan_angles):
    """Build temporal residual stack from scan buffer + odom deltas.

    Args:
        scan_buffer: list of T numpy arrays, each (N,) scan ranges.
                     scan_buffer[-1] is current frame.
        odom_deltas: list of T-1 numpy arrays, each (3,) = (dx, dy, dtheta).
                     odom_deltas[k] = odom from frame k to frame k+1.
        scan_angles: (N,) angle array for all scans (assumed constant).

    Returns:
        residuals: (T-1, N) temporal residual channels.
                   residuals[k] = |scan_current - warped_scan[T-2-k]|
    """
    T = len(scan_buffer)
    N = len(scan_buffer[-1])
    current_scan = scan_buffer[-1]
    residuals = np.zeros((T - 1, N), dtype=np.float32)

    for k in range(T - 1):
        # Frame index to warp from (most recent to oldest)
        frame_idx = T - 2 - k

        # Accumulate odom deltas from frame_idx to current (T-1)
        cumulative_dx, cumulative_dy, cumulative_dtheta = 0.0, 0.0, 0.0
        for j in range(frame_idx, T - 1):
            d = odom_deltas[j]
            # Compose transforms
            cos_t = np.cos(cumulative_dtheta)
            sin_t = np.sin(cumulative_dtheta)
            cumulative_dx += cos_t * d[0] - sin_t * d[1]
            cumulative_dy += sin_t * d[0] + cos_t * d[1]
            cumulative_dtheta += d[2]

        # Warp previous scan to current frame
        warped = compensate_scan(
            scan_buffer[frame_idx], scan_angles,
            (cumulative_dx, cumulative_dy, cumulative_dtheta)
        )

        # Temporal residual: large where something moved
        # Use finite-only comparison (inf-inf = nan)
        current_finite = np.where(np.isfinite(current_scan), current_scan, 0.0)
        warped_finite = np.where(np.isfinite(warped), warped, 0.0)
        residuals[k] = np.abs(current_finite - warped_finite)

        # Zero out where either scan was infinite (no obstacle to compare)
        invalid = ~np.isfinite(current_scan) | ~np.isfinite(warped)
        residuals[k][invalid] = 0.0

    return residuals
