"""
CFM-BEV: Conditional Flow Matching for Velocity Trajectory Generation.

Architecture:
    BEV (5ch) → ResNet-18 encoder → 512-d  ┐
                                             ├→ concat 576-d → GRU → conditioning
    Odom (30) → MLP encoder → 64-d ─────────┘                           │
                                                                         ↓
    Noise z ~ N(0,I) ──→ ConditionalVectorField ──→ [(v,ω)] × H steps
                              (1-5 Euler ODE steps)

The model generates H-step velocity trajectories. At inference, only
the first (v,ω) is executed (receding horizon). The full trajectory
gives the model foresight — it considers consequences of the first
action over the entire horizon.

Key difference from Nav2 DWB:
    Nav2: sample 100s of (v,ω) → simulate → score against costmap → pick best
    Ours: BEV → single forward pass → generate optimal velocity trajectory

~13M parameters total. 15-25Hz on Jetson 8GB with 1-step Euler.
"""

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

from wheelchair_e2e.models.flow_matching import (
    ConditionalVectorField, euler_sample
)


class CFMVelocityNet(nn.Module):
    """
    Conditional Flow Matching network for velocity trajectory generation.

    Input:
        bev: (B, 5, 200, 200) BEV grid
            Ch 0: LiDAR occupancy
            Ch 1: Depth camera occupancy
            Ch 2: Goal direction (gaussian)
            Ch 3: Ego-motion trail
            Ch 4: Global route from planner
        odom_history: (B, 30) flattened [v, ω, θ] × 10 steps
        hidden: (1, B, 256) GRU hidden state

    Output (training):
        conditioning: (B, 256) for flow matching loss
        hidden: updated GRU state

    Output (inference):
        velocity_traj: (B, H, 2) generated velocity trajectory
        hidden: updated GRU state
    """

    def __init__(self, bev_channels=5, v_max=0.25, w_max=1.0,
                 gru_hidden=256, horizon=10, cfm_hidden=512,
                 cfm_layers=4, n_euler_steps=3):
        super().__init__()
        self.v_max = v_max
        self.w_max = w_max
        self.horizon = horizon
        self.traj_dim = horizon * 2  # (v, ω) per step
        self.n_euler_steps = n_euler_steps

        # --- Visual encoder: ResNet-18 (11.2M params) ---
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        backbone.conv1 = nn.Conv2d(
            bev_channels, 64, kernel_size=7,
            stride=2, padding=3, bias=False
        )
        self.encoder = nn.Sequential(
            *list(backbone.children())[:-2],
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )  # output: 512-d

        # --- Odom encoder ---
        self.odom_encoder = nn.Sequential(
            nn.Linear(30, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )

        # --- GRU for temporal context ---
        self.gru = nn.GRU(
            input_size=512 + 64,
            hidden_size=gru_hidden,
            num_layers=1,
            batch_first=True
        )

        # --- Conditional Vector Field (CFM decoder) ---
        self.vector_field = ConditionalVectorField(
            traj_dim=self.traj_dim,
            cond_dim=gru_hidden,
            hidden_dim=cfm_hidden,
            n_layers=cfm_layers,
        )

    def encode(self, bev, odom_history, hidden=None):
        """
        Encode BEV + odom into conditioning features via GRU.

        Returns:
            cond: (B, 256) conditioning for CFM
            hidden: updated GRU state
        """
        vis_feat = self.encoder(bev)                      # (B, 512)
        odom_feat = self.odom_encoder(odom_history)       # (B, 64)
        combined = torch.cat([vis_feat, odom_feat], dim=-1)  # (B, 576)
        combined = combined.unsqueeze(1)                   # (B, 1, 576)
        gru_out, hidden = self.gru(combined, hidden)
        cond = gru_out.squeeze(1)                          # (B, 256)
        return cond, hidden

    def forward(self, bev, odom_history, hidden=None):
        """
        Training forward pass. Returns conditioning for flow matching loss.

        Use flow_matching_loss() externally with the returned conditioning
        and target trajectories.

        Returns:
            cond: (B, 256) conditioning features
            hidden: updated GRU hidden state
        """
        return self.encode(bev, odom_history, hidden)

    @torch.no_grad()
    def generate(self, bev, odom_history, hidden=None):
        """
        Inference: generate a velocity trajectory via Euler ODE.

        Returns:
            velocity_traj: (B, H, 2) velocity trajectory in physical units
                [:, :, 0] = v ∈ [0, v_max]
                [:, :, 1] = ω ∈ [-w_max, w_max]
            hidden: updated GRU state
        """
        cond, hidden = self.encode(bev, odom_history, hidden)

        # Generate trajectory via Euler integration of learned vector field
        raw_traj = euler_sample(
            self.vector_field, cond,
            traj_dim=self.traj_dim,
            n_steps=self.n_euler_steps,
            device=bev.device
        )  # (B, H*2)

        # Reshape to (B, H, 2) and scale to physical units
        raw_traj = raw_traj.view(-1, self.horizon, 2)

        # Apply tanh scaling to physical velocity limits
        v = (torch.tanh(raw_traj[:, :, 0]) + 1) / 2 * self.v_max
        omega = torch.tanh(raw_traj[:, :, 1]) * self.w_max

        velocity_traj = torch.stack([v, omega], dim=-1)  # (B, H, 2)
        return velocity_traj, hidden

    def get_param_count(self):
        """Return parameter count breakdown."""
        enc = sum(p.numel() for p in self.encoder.parameters())
        odom = sum(p.numel() for p in self.odom_encoder.parameters())
        gru = sum(p.numel() for p in self.gru.parameters())
        vf = sum(p.numel() for p in self.vector_field.parameters())
        total = enc + odom + gru + vf
        return {
            'encoder': enc,
            'odom_encoder': odom,
            'gru': gru,
            'vector_field': vf,
            'total': total,
        }
