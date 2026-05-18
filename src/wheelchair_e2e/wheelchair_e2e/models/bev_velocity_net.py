"""
E2E-2: BEV-to-Velocity CNN (Recommended for IROS)

Architecture: scan_depth_fusion → BEV grid → ResNet-18 + GRU → (v, ω) → /cmd_vel
Total: ~12M parameters
Inference: ~40ms on Jetson 8GB (TensorRT FP16), 15-25Hz

Inspired by ChauffeurNet (Waymo) BEV representation + imitation learning.
"""

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class BEVVelocityNet(nn.Module):
    """
    End-to-end: BEV grid → (v, omega)

    Input:
        bev: (B, 5, 200, 200) BEV occupancy grid
            Channel 0: LiDAR occupancy
            Channel 1: Depth camera occupancy (from fused scan)
            Channel 2: Goal direction (gaussian blob at goal position)
            Channel 3: Odometry trail (last 1s of ego-positions)
            Channel 4: Global route from planner (rendered path)
        odom_history: (B, 30) flattened odometry [v, ω, θ] × 10 steps
        hidden: (1, B, gru_hidden) GRU hidden state (None on first call)

    Output:
        velocity: (B, 2) = (v, omega) in physical units
            v ∈ [0, v_max] m/s (forward only for wheelchair)
            ω ∈ [-w_max, w_max] rad/s
        hidden: updated GRU hidden state
    """

    def __init__(self, bev_channels=5, v_max=0.25, w_max=0.35,
                 gru_hidden=256):
        super().__init__()
        self.v_max = v_max
        self.w_max = w_max

        # Visual encoder: ResNet-18 (11.2M params)
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        # Modify first conv for 4-channel BEV input
        backbone.conv1 = nn.Conv2d(
            bev_channels, 64, kernel_size=7,
            stride=2, padding=3, bias=False
        )
        # Remove final FC layer, keep up to avgpool
        self.encoder = nn.Sequential(
            *list(backbone.children())[:-2],  # up to layer4
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )  # output: 512-dim

        # Odom history encoder (v, omega, theta for last 10 steps)
        self.odom_encoder = nn.Sequential(
            nn.Linear(30, 64),  # 10 steps * 3 values
            nn.ReLU(),
            nn.Linear(64, 64)
        )

        # Temporal context: GRU (0.5M params)
        self.gru = nn.GRU(
            input_size=512 + 64,  # visual + odom
            hidden_size=gru_hidden,
            num_layers=1,
            batch_first=True
        )

        # Velocity head (0.05M params)
        self.velocity_head = nn.Sequential(
            nn.Linear(gru_hidden, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Tanh()  # output in [-1, 1]
        )

    def forward(self, bev, odom_history, hidden=None):
        """
        Args:
            bev: (B, 5, 200, 200) BEV grid
            odom_history: (B, 30) flattened odom
            hidden: (1, B, 256) GRU hidden state
        Returns:
            velocity: (B, 2) = (v, omega) in physical units
            hidden: updated GRU hidden state
        """
        # Encode BEV
        vis_feat = self.encoder(bev)            # (B, 512)
        odom_feat = self.odom_encoder(odom_history)  # (B, 64)

        # Combine and pass through GRU
        combined = torch.cat([vis_feat, odom_feat],
                             dim=-1)                 # (B, 576)
        combined = combined.unsqueeze(1)             # (B, 1, 576)
        gru_out, hidden = self.gru(combined, hidden)
        gru_out = gru_out.squeeze(1)                 # (B, 256)

        # Predict velocity
        raw = self.velocity_head(gru_out)        # (B, 2)
        v = (raw[:, 0] + 1) / 2 * self.v_max    # [0, v_max]
        omega = raw[:, 1] * self.w_max           # [-w_max, w_max]

        return torch.stack([v, omega], dim=-1), hidden

    def get_param_count(self):
        """Return parameter count breakdown."""
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        odom_params = sum(p.numel() for p in self.odom_encoder.parameters())
        gru_params = sum(p.numel() for p in self.gru.parameters())
        head_params = sum(p.numel() for p in self.velocity_head.parameters())
        total = encoder_params + odom_params + gru_params + head_params
        return {
            'encoder': encoder_params,
            'odom_encoder': odom_params,
            'gru': gru_params,
            'velocity_head': head_params,
            'total': total
        }
