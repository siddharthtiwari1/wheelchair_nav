"""
KinoFlow v2: Modular Kinodynamic Flow Matching for Wheelchair Navigation.

Two architectures in this file:
    1. KinoFlowNet (v1, LEGACY): ResNet-18 + MLP vector field. ~13M params.
    2. ModularKinoFlowNet (v2): 4 modular encoders + Transformer fusion +
       Trajectory Transformer vector field. ~1.3M params (10x smaller).

ModularKinoFlowNet beats both CrowdSurfer and Crowd-FM:
    - NH constraint by construction (velocity-space output, no PRIEST needed)
    - No external tracker (temporal scan residuals detect dynamics)
    - Temporal context (T=5 scan buffer + GRU across frames)
    - Ego-motion awareness (velocity context encoder)
    - Global temporal attention (Trajectory Transformer > MLP or 1D U-Net)

Architecture:
    E1: /scan_fused[t] → 1D CNN → 128-d (static scene)
    E2: temporal_residuals → 1D CNN + SelfAttn → 128-d (dynamic obstacles)
    E3: (dist, bearing, cos, sin) → MLP → 64-d (goal)
    E4: odom_history → MLP → 64-d (velocity context)
    Fusion: 4 tokens → Transformer(4h,2L) → pool → 256-d
    Temporal: GRU(256, 256) → 256-d
    Decoder: TrajectoryTransformerVectorField → (v, ω) × H steps

~1.3M params. ~15ms on Jetson 8GB with K=8 samples.
"""

import torch
import torch.nn as nn
import numpy as np
from torchvision.models import resnet18, ResNet18_Weights

from wheelchair_e2e.models.flow_matching import (
    ConditionalVectorField, euler_sample
)
from wheelchair_e2e.models.scan_encoder import StaticSceneEncoder
from wheelchair_e2e.models.dynamic_encoder import DynamicObstacleEncoder
from wheelchair_e2e.models.goal_encoder import GoalEncoder
from wheelchair_e2e.models.velocity_encoder import VelocityContextEncoder
from wheelchair_e2e.models.fusion import TransformerFusion
from wheelchair_e2e.models.trajectory_transformer import TrajectoryTransformerVectorField


class ModularKinoFlowNet(nn.Module):
    """
    KinoFlow v2: Modular encoder architecture with Trajectory Transformer.

    Keeps ALL method signatures from KinoFlowNet (encode, generate,
    generate_multi_sample, integrate_trajectory, scale_trajectory) for
    drop-in compatibility with training/inference code.

    New inputs (vs v1):
        - scan_current: (B, 720) current polar scan ranges
        - scan_residuals: (B, 4, 720) temporal residual channels
        - goal_features: (B, 4) [norm_dist, norm_bearing, cos_b, sin_b]
        - odom_history: (B, 30) same as v1

    The encode() method signature changes. Training/inference code must
    pass modular inputs instead of BEV.
    """

    def __init__(self, v_max=0.25, w_max=1.0, gru_hidden=256, horizon=10,
                 n_euler_steps=3, dt=0.1, n_samples=8,
                 # Encoder dims
                 scan_points=720, temporal_frames=5,
                 scene_dim=128, dynamic_dim=128, goal_dim=64, odom_dim=64,
                 # Transformer dims
                 d_model=128, cond_dim=256,
                 nhead=4, fusion_layers=2, decoder_layers=2,
                 # Ablation flags
                 no_dynamic=False, no_temporal=False,
                 no_goal=False, no_velocity=False,
                 mlp_vectorfield=False, concat_fusion=False,
                 monolithic=False,
                 # Legacy BEV support (for collision scoring)
                 bev_channels=5):
        super().__init__()
        self.v_max = v_max
        self.w_max = w_max
        self.horizon = horizon
        self.traj_dim = horizon * 2
        self.n_euler_steps = n_euler_steps
        self.dt = dt
        self.n_samples = n_samples
        self.scan_points = scan_points
        self.temporal_frames = temporal_frames
        self.cond_dim = cond_dim
        self.monolithic = monolithic

        # Ablation state
        self.no_dynamic = no_dynamic
        self.no_temporal = no_temporal  # T=1 (single frame, no residuals)
        self.no_goal = no_goal
        self.no_velocity = no_velocity

        if monolithic:
            # Fall back to v1 architecture (for ablation comparison)
            backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            backbone.conv1 = nn.Conv2d(
                bev_channels, 64, kernel_size=7,
                stride=2, padding=3, bias=False
            )
            self.encoder_legacy = nn.Sequential(
                *list(backbone.children())[:-2],
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten()
            )
            self.odom_encoder_legacy = nn.Sequential(
                nn.Linear(30, 64), nn.ReLU(), nn.Linear(64, 64)
            )
            self.gru = nn.GRU(
                input_size=512 + 64, hidden_size=gru_hidden,
                num_layers=1, batch_first=True
            )
            self.vector_field = ConditionalVectorField(
                traj_dim=self.traj_dim, cond_dim=gru_hidden,
                hidden_dim=512, n_layers=4,
            )
            return

        # --- E1: Static Scene Encoder ---
        self.scene_encoder = StaticSceneEncoder(
            scan_points=scan_points, output_dim=scene_dim
        )

        # --- E2: Dynamic Obstacle Encoder ---
        self.dynamic_encoder = DynamicObstacleEncoder(
            scan_points=scan_points, temporal_frames=temporal_frames,
            output_dim=dynamic_dim,
        )

        # --- E3: Goal Encoder ---
        self.goal_encoder = GoalEncoder(output_dim=goal_dim)

        # --- E4: Velocity Context Encoder ---
        self.velocity_encoder = VelocityContextEncoder(
            input_dim=30, output_dim=odom_dim
        )

        # --- Fusion ---
        self.fusion = TransformerFusion(
            scene_dim=scene_dim, dynamic_dim=dynamic_dim,
            goal_dim=goal_dim, odom_dim=odom_dim,
            d_model=d_model, cond_dim=cond_dim,
            nhead=nhead, num_layers=fusion_layers,
            no_dynamic=no_dynamic, no_goal=no_goal,
            no_velocity=no_velocity, concat_fusion=concat_fusion,
        )

        # --- GRU for temporal context ---
        self.gru = nn.GRU(
            input_size=cond_dim,
            hidden_size=gru_hidden,
            num_layers=1,
            batch_first=True,
        )

        # --- Decoder: Trajectory Transformer or MLP fallback ---
        if mlp_vectorfield:
            self.vector_field = ConditionalVectorField(
                traj_dim=self.traj_dim,
                cond_dim=gru_hidden,
                hidden_dim=512,
                n_layers=4,
            )
        else:
            self.vector_field = TrajectoryTransformerVectorField(
                horizon=horizon,
                cond_dim=gru_hidden,
                d_model=d_model,
                nhead=nhead,
                num_layers=decoder_layers,
            )

    def encode(self, scan_current, scan_residuals, goal_features,
               odom_history, hidden=None):
        """Encode modular inputs into conditioning features via fusion + GRU.

        Args:
            scan_current:   (B, 720) current polar scan ranges
            scan_residuals: (B, 4, 720) ego-compensated temporal residual channels
            goal_features:  (B, 4) [norm_dist, norm_bearing, cos_b, sin_b]
            odom_history:   (B, 30) flattened [v, ω, θ] × 10
            hidden:         GRU hidden state or None

        Returns:
            cond: (B, cond_dim) conditioning for vector field
            hidden: updated GRU state
        """
        # E1: static scene
        scene_feat = self.scene_encoder(scan_current)      # (B, 128)

        # E2: dynamic obstacles
        if self.no_dynamic or self.no_temporal:
            dynamic_feat = torch.zeros(
                scan_current.shape[0], self.dynamic_encoder.output_dim,
                device=scan_current.device
            )
        else:
            dynamic_feat = self.dynamic_encoder(scan_residuals)  # (B, 128)

        # E3: goal
        goal_feat = self.goal_encoder(goal_features)       # (B, 64)

        # E4: velocity context
        odom_feat = self.velocity_encoder(odom_history)    # (B, 64)

        # Fusion: cross-attention Transformer
        fused = self.fusion(scene_feat, dynamic_feat, goal_feat, odom_feat)
        # (B, cond_dim)

        # GRU temporal context
        fused = fused.unsqueeze(1)  # (B, 1, cond_dim)
        gru_out, hidden = self.gru(fused, hidden)
        cond = gru_out.squeeze(1)   # (B, cond_dim)

        return cond, hidden

    def encode_legacy(self, bev, odom_history, hidden=None):
        """Legacy v1 encode for monolithic ablation.

        Args:
            bev: (B, 5, 200, 200) BEV grid
            odom_history: (B, 30)
            hidden: GRU state

        Returns:
            cond: (B, 256) conditioning
            hidden: updated GRU state
        """
        vis_feat = self.encoder_legacy(bev)
        odom_feat = self.odom_encoder_legacy(odom_history)
        combined = torch.cat([vis_feat, odom_feat], dim=-1)
        combined = combined.unsqueeze(1)
        gru_out, hidden = self.gru(combined, hidden)
        cond = gru_out.squeeze(1)
        return cond, hidden

    def forward(self, scan_current, scan_residuals, goal_features,
                odom_history, hidden=None):
        """Training forward pass. Returns conditioning for flow matching loss.

        Returns:
            cond: (B, cond_dim) conditioning features
            hidden: updated GRU hidden state
        """
        if self.monolithic:
            # scan_current is actually BEV in monolithic mode
            return self.encode_legacy(scan_current, odom_history, hidden)
        return self.encode(scan_current, scan_residuals, goal_features,
                           odom_history, hidden)

    def scale_trajectory(self, raw_traj):
        """Scale raw CFM output to physical velocity units.

        Args:
            raw_traj: (B, H, 2) raw network output

        Returns:
            vel_traj: (B, H, 2) with v in [0, v_max], omega in [-w_max, w_max]
        """
        v = (torch.tanh(raw_traj[:, :, 0]) + 1) / 2 * self.v_max
        omega = torch.tanh(raw_traj[:, :, 1]) * self.w_max
        return torch.stack([v, omega], dim=-1)

    def integrate_trajectory(self, vel_traj):
        """Forward-integrate (v, omega) to get full kinodynamic poses.

        NH constraint satisfied by construction:
            x_{t+1} = x_t + v_t * cos(theta_t) * dt
            y_{t+1} = y_t + v_t * sin(theta_t) * dt
            theta_{t+1} = theta_t + omega_t * dt

        Args:
            vel_traj: (B, H, 2) velocity trajectory [v, omega]

        Returns:
            poses: (B, H, 3) pose trajectory [x, y, theta] in base_link frame
        """
        B, H, _ = vel_traj.shape
        device = vel_traj.device

        x = torch.zeros(B, device=device)
        y = torch.zeros(B, device=device)
        theta = torch.zeros(B, device=device)

        poses = []
        for t in range(H):
            v = vel_traj[:, t, 0]
            omega = vel_traj[:, t, 1]
            theta = theta + omega * self.dt
            x = x + v * torch.cos(theta) * self.dt
            y = y + v * torch.sin(theta) * self.dt
            poses.append(torch.stack([x, y, theta], dim=-1))

        return torch.stack(poses, dim=1)

    @torch.no_grad()
    def generate(self, scan_current, scan_residuals, goal_features,
                 odom_history, hidden=None, warm_start=None):
        """Generate a single velocity trajectory via Euler ODE.

        Args:
            scan_current:   (B, 720) current polar scan
            scan_residuals: (B, 4, 720) temporal residuals
            goal_features:  (B, 4) goal features
            odom_history:   (B, 30)
            hidden: GRU state
            warm_start: (B, H*2) optional warm-start noise

        Returns:
            vel_traj: (B, H, 2) in physical units
            poses: (B, H, 3) integrated poses [x, y, theta]
            hidden: updated GRU state
        """
        if self.monolithic:
            cond, hidden = self.encode_legacy(
                scan_current, odom_history, hidden)
        else:
            cond, hidden = self.encode(
                scan_current, scan_residuals, goal_features,
                odom_history, hidden)

        if warm_start is not None:
            raw_traj = _euler_sample_from(
                self.vector_field, cond, warm_start,
                n_steps=self.n_euler_steps, device=scan_current.device
            )
        else:
            raw_traj = euler_sample(
                self.vector_field, cond,
                traj_dim=self.traj_dim,
                n_steps=self.n_euler_steps,
                device=scan_current.device
            )

        raw_traj = raw_traj.view(-1, self.horizon, 2)
        vel_traj = self.scale_trajectory(raw_traj)
        poses = self.integrate_trajectory(vel_traj)

        return vel_traj, poses, hidden

    def set_learned_scorer(self, scorer):
        """Attach a trained DualSpaceScoringTransformer for inference."""
        self.learned_scorer = scorer
        self.learned_scorer.eval()

    @torch.no_grad()
    def generate_multi_sample(self, scan_current, scan_residuals,
                              goal_features, odom_history, hidden=None,
                              bev_occupancy=None, goal_dx=None, goal_dy=None,
                              warm_start=None):
        """Generate K trajectory samples and select the best one.

        Args:
            scan_current:   (1, 720) or (1, 5, 200, 200) if monolithic
            scan_residuals: (1, 4, 720) temporal residuals
            goal_features:  (1, 4) goal features
            odom_history:   (1, 30)
            hidden: GRU state
            bev_occupancy:  (1, 200, 200) obstacle map for collision scoring
            goal_dx: relative goal x (meters)
            goal_dy: relative goal y (meters)
            warm_start: (1, H*2) warm-start from previous best

        Returns:
            best_vel: (1, H, 2)
            best_poses: (1, H, 3)
            all_vel: (K, H, 2)
            all_poses: (K, H, 3)
            best_idx: int
            scores: (K,)
            hidden: updated GRU state
        """
        K = self.n_samples

        # Encode once (shared)
        if self.monolithic:
            cond, hidden = self.encode_legacy(
                scan_current, odom_history, hidden)
        else:
            cond, hidden = self.encode(
                scan_current, scan_residuals, goal_features,
                odom_history, hidden)

        # Replicate conditioning for K samples
        cond_k = cond.expand(K, -1)

        # Generate K noise samples
        z0 = torch.randn(K, self.traj_dim, device=cond.device)

        # Warm-start blending
        if warm_start is not None:
            ws = warm_start.view(1, self.horizon, 2)
            ws_shifted = torch.cat([ws[:, 1:, :], ws[:, -1:, :]], dim=1)
            ws_flat = ws_shifted.view(1, -1)
            alpha = 0.8
            z0[0] = alpha * ws_flat + (1 - alpha) * z0[0]

        # Euler ODE integration for K samples
        raw_trajs = _euler_sample_from(
            self.vector_field, cond_k, z0,
            n_steps=self.n_euler_steps, device=cond.device
        )

        raw_trajs = raw_trajs.view(K, self.horizon, 2)
        vel_trajs = self.scale_trajectory(raw_trajs)
        poses_trajs = self.integrate_trajectory(vel_trajs)

        # Score trajectories
        if hasattr(self, 'learned_scorer') and self.learned_scorer is not None:
            # For learned scorer, we need scene features
            # Use static scene encoder output as scene features
            if not self.monolithic:
                scene_features = self.scene_encoder(
                    scan_current).squeeze(0)  # (128,)
            else:
                scene_features = self.encoder_legacy(
                    scan_current).squeeze(0)  # (512,)
            bev_occ = bev_occupancy[0] if bev_occupancy is not None else None
            scores = self.learned_scorer.score_with_context(
                vel_trajs, poses_trajs, bev_occ,
                scene_features,
                goal_dx if goal_dx is not None else 1.0,
                goal_dy if goal_dy is not None else 0.0,
            )
        else:
            scores = self._score_trajectories(
                vel_trajs, poses_trajs, bev_occupancy, goal_dx, goal_dy
            )

        best_idx = scores.argmax().item()
        best_vel = vel_trajs[best_idx:best_idx+1]
        best_poses = poses_trajs[best_idx:best_idx+1]

        return (best_vel, best_poses, vel_trajs, poses_trajs,
                best_idx, scores, hidden)

    def _score_trajectories(self, vel_trajs, poses_trajs,
                            bev_occupancy=None, goal_dx=None, goal_dy=None):
        """Score K trajectory candidates. Higher = better.

        Components: goal progress (5.0), collision (-20.0),
        comfort (2.0), forward progress (1.0).
        """
        K = vel_trajs.shape[0]
        device = vel_trajs.device
        scores = torch.zeros(K, device=device)

        # 1. Goal progress
        if goal_dx is not None and goal_dy is not None:
            goal = torch.tensor([goal_dx, goal_dy], device=device)
            goal_dist_init = torch.norm(goal)
            endpoint_xy = poses_trajs[:, -1, :2]
            endpoint_dist = torch.norm(
                endpoint_xy - goal.unsqueeze(0), dim=-1)
            progress = (goal_dist_init - endpoint_dist) / max(
                goal_dist_init.item(), 0.1)
            scores += 5.0 * progress

        # 2. Collision penalty
        if bev_occupancy is not None:
            occ = bev_occupancy[0]
            grid_size = occ.shape[0]
            center = grid_size // 2
            resolution = 0.05
            safety_px = int(0.3 / resolution)

            for k in range(K):
                collision_cost = 0.0
                for t in range(poses_trajs.shape[1]):
                    px = int(poses_trajs[k, t, 0].item() / resolution + center)
                    py = int(poses_trajs[k, t, 1].item() / resolution + center)
                    px = max(0, min(grid_size - 1, px))
                    py = max(0, min(grid_size - 1, py))
                    x_lo = max(0, px - safety_px)
                    x_hi = min(grid_size, px + safety_px + 1)
                    y_lo = max(0, py - safety_px)
                    y_hi = min(grid_size, py + safety_px + 1)
                    patch = occ[y_lo:y_hi, x_lo:x_hi]
                    if patch.numel() > 0:
                        collision_cost += patch.max().item()
                scores[k] -= 20.0 * collision_cost / poses_trajs.shape[1]

        # 3. Comfort / smoothness
        if vel_trajs.shape[1] >= 2:
            accel = vel_trajs[:, 1:, :] - vel_trajs[:, :-1, :]
            jerk_score = (accel ** 2).sum(dim=(1, 2))
            max_jerk = jerk_score.max().item() + 1e-6
            scores += 2.0 * (1.0 - jerk_score / max_jerk)

        # 4. Forward progress
        avg_v = vel_trajs[:, :, 0].mean(dim=1)
        scores += 1.0 * avg_v / self.v_max

        return scores

    def get_param_count(self):
        """Return parameter count breakdown."""
        if self.monolithic:
            enc = sum(p.numel() for p in self.encoder_legacy.parameters())
            odom = sum(
                p.numel() for p in self.odom_encoder_legacy.parameters())
            gru = sum(p.numel() for p in self.gru.parameters())
            vf = sum(p.numel() for p in self.vector_field.parameters())
            total = enc + odom + gru + vf
            return {
                'encoder_legacy': enc, 'odom_encoder_legacy': odom,
                'gru': gru, 'vector_field': vf, 'total': total,
            }

        scene = sum(p.numel() for p in self.scene_encoder.parameters())
        dynamic = sum(p.numel() for p in self.dynamic_encoder.parameters())
        goal = sum(p.numel() for p in self.goal_encoder.parameters())
        velocity = sum(p.numel() for p in self.velocity_encoder.parameters())
        fusion = sum(p.numel() for p in self.fusion.parameters())
        gru = sum(p.numel() for p in self.gru.parameters())
        vf = sum(p.numel() for p in self.vector_field.parameters())
        total = scene + dynamic + goal + velocity + fusion + gru + vf
        return {
            'scene_encoder': scene,
            'dynamic_encoder': dynamic,
            'goal_encoder': goal,
            'velocity_encoder': velocity,
            'fusion': fusion,
            'gru': gru,
            'vector_field': vf,
            'total': total,
        }


# =========================================================================
# Legacy KinoFlowNet (v1) — kept for backward compatibility
# =========================================================================

class KinoFlowNet(nn.Module):
    """Legacy v1 KinoFlowNet. See ModularKinoFlowNet for v2.

    Kept for loading old checkpoints and running the --monolithic ablation.
    """

    def __init__(self, bev_channels=5, v_max=0.25, w_max=1.0,
                 gru_hidden=256, horizon=10, cfm_hidden=512,
                 cfm_layers=4, n_euler_steps=3, dt=0.1,
                 n_samples=8):
        super().__init__()
        self.v_max = v_max
        self.w_max = w_max
        self.horizon = horizon
        self.traj_dim = horizon * 2
        self.n_euler_steps = n_euler_steps
        self.dt = dt
        self.n_samples = n_samples

        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        backbone.conv1 = nn.Conv2d(
            bev_channels, 64, kernel_size=7,
            stride=2, padding=3, bias=False
        )
        self.encoder = nn.Sequential(
            *list(backbone.children())[:-2],
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )

        self.odom_encoder = nn.Sequential(
            nn.Linear(30, 64), nn.ReLU(), nn.Linear(64, 64)
        )

        self.gru = nn.GRU(
            input_size=512 + 64, hidden_size=gru_hidden,
            num_layers=1, batch_first=True
        )

        self.vector_field = ConditionalVectorField(
            traj_dim=self.traj_dim, cond_dim=gru_hidden,
            hidden_dim=cfm_hidden, n_layers=cfm_layers,
        )

    def encode(self, bev, odom_history, hidden=None):
        vis_feat = self.encoder(bev)
        odom_feat = self.odom_encoder(odom_history)
        combined = torch.cat([vis_feat, odom_feat], dim=-1)
        combined = combined.unsqueeze(1)
        gru_out, hidden = self.gru(combined, hidden)
        cond = gru_out.squeeze(1)
        return cond, hidden

    def forward(self, bev, odom_history, hidden=None):
        return self.encode(bev, odom_history, hidden)

    def scale_trajectory(self, raw_traj):
        if hasattr(self, 'use_bernstein') and self.use_bernstein:
            return self._bernstein_to_velocity(raw_traj)
        v = (torch.tanh(raw_traj[:, :, 0]) + 1) / 2 * self.v_max
        omega = torch.tanh(raw_traj[:, :, 1]) * self.w_max
        return torch.stack([v, omega], dim=-1)

    def _bernstein_to_velocity(self, coeffs):
        B, n_coeffs, _ = coeffs.shape
        degree = n_coeffs - 1
        device = coeffs.device
        t = torch.linspace(0, 1, self.horizon, device=device)
        basis = torch.zeros(self.horizon, n_coeffs, device=device)
        for i in range(n_coeffs):
            binom = 1.0
            for j in range(1, min(i, degree - i) + 1):
                binom = binom * (degree - j + 1) / j
            basis[:, i] = binom * (t ** i) * ((1 - t) ** (degree - i))
        scaled_coeffs = coeffs.clone()
        scaled_coeffs[:, :, 0] = (
            torch.tanh(coeffs[:, :, 0]) + 1) / 2 * self.v_max
        scaled_coeffs[:, :, 1] = torch.tanh(coeffs[:, :, 1]) * self.w_max
        vel_traj = torch.einsum('bnc,hn->bhc', scaled_coeffs, basis)
        return vel_traj

    def integrate_trajectory(self, vel_traj):
        B, H, _ = vel_traj.shape
        device = vel_traj.device
        x = torch.zeros(B, device=device)
        y = torch.zeros(B, device=device)
        theta = torch.zeros(B, device=device)
        poses = []
        for t in range(H):
            v = vel_traj[:, t, 0]
            omega = vel_traj[:, t, 1]
            theta = theta + omega * self.dt
            x = x + v * torch.cos(theta) * self.dt
            y = y + v * torch.sin(theta) * self.dt
            poses.append(torch.stack([x, y, theta], dim=-1))
        return torch.stack(poses, dim=1)

    @torch.no_grad()
    def generate(self, bev, odom_history, hidden=None, warm_start=None):
        cond, hidden = self.encode(bev, odom_history, hidden)
        if warm_start is not None:
            raw_traj = _euler_sample_from(
                self.vector_field, cond, warm_start,
                n_steps=self.n_euler_steps, device=bev.device)
        else:
            raw_traj = euler_sample(
                self.vector_field, cond, traj_dim=self.traj_dim,
                n_steps=self.n_euler_steps, device=bev.device)
        raw_traj = raw_traj.view(-1, self.horizon, 2)
        vel_traj = self.scale_trajectory(raw_traj)
        poses = self.integrate_trajectory(vel_traj)
        return vel_traj, poses, hidden

    def set_learned_scorer(self, scorer):
        self.learned_scorer = scorer
        self.learned_scorer.eval()

    @torch.no_grad()
    def generate_multi_sample(self, bev, odom_history, hidden=None,
                              bev_occupancy=None, goal_dx=None, goal_dy=None,
                              warm_start=None):
        K = self.n_samples
        cond, hidden = self.encode(bev, odom_history, hidden)
        cond_k = cond.expand(K, -1)
        z0 = torch.randn(K, self.traj_dim, device=bev.device)
        if warm_start is not None:
            ws = warm_start.view(1, self.horizon, 2)
            ws_shifted = torch.cat([ws[:, 1:, :], ws[:, -1:, :]], dim=1)
            ws_flat = ws_shifted.view(1, -1)
            z0[0] = 0.8 * ws_flat + 0.2 * z0[0]
        raw_trajs = _euler_sample_from(
            self.vector_field, cond_k, z0,
            n_steps=self.n_euler_steps, device=bev.device)
        raw_trajs = raw_trajs.view(K, self.horizon, 2)
        vel_trajs = self.scale_trajectory(raw_trajs)
        poses_trajs = self.integrate_trajectory(vel_trajs)

        if hasattr(self, 'learned_scorer') and self.learned_scorer is not None:
            scene_features = self.encoder(bev).squeeze(0)
            bev_occ = bev_occupancy[0] if bev_occupancy is not None else None
            scores = self.learned_scorer.score_with_context(
                vel_trajs, poses_trajs, bev_occ, scene_features,
                goal_dx if goal_dx is not None else 1.0,
                goal_dy if goal_dy is not None else 0.0)
        else:
            scores = self._score_trajectories(
                vel_trajs, poses_trajs, bev_occupancy, goal_dx, goal_dy)

        best_idx = scores.argmax().item()
        best_vel = vel_trajs[best_idx:best_idx+1]
        best_poses = poses_trajs[best_idx:best_idx+1]
        return (best_vel, best_poses, vel_trajs, poses_trajs,
                best_idx, scores, hidden)

    def _score_trajectories(self, vel_trajs, poses_trajs,
                            bev_occupancy=None, goal_dx=None, goal_dy=None):
        K = vel_trajs.shape[0]
        device = vel_trajs.device
        scores = torch.zeros(K, device=device)
        if goal_dx is not None and goal_dy is not None:
            goal = torch.tensor([goal_dx, goal_dy], device=device)
            goal_dist_init = torch.norm(goal)
            endpoint_xy = poses_trajs[:, -1, :2]
            endpoint_dist = torch.norm(
                endpoint_xy - goal.unsqueeze(0), dim=-1)
            progress = (goal_dist_init - endpoint_dist) / max(
                goal_dist_init.item(), 0.1)
            scores += 5.0 * progress
        if bev_occupancy is not None:
            occ = bev_occupancy[0]
            grid_size = occ.shape[0]
            center = grid_size // 2
            resolution = 0.05
            safety_px = int(0.3 / resolution)
            for k in range(K):
                cc = 0.0
                for t in range(poses_trajs.shape[1]):
                    px = int(
                        poses_trajs[k, t, 0].item() / resolution + center)
                    py = int(
                        poses_trajs[k, t, 1].item() / resolution + center)
                    px = max(0, min(grid_size - 1, px))
                    py = max(0, min(grid_size - 1, py))
                    x_lo, x_hi = max(0, px-safety_px), min(
                        grid_size, px+safety_px+1)
                    y_lo, y_hi = max(0, py-safety_px), min(
                        grid_size, py+safety_px+1)
                    patch = occ[y_lo:y_hi, x_lo:x_hi]
                    if patch.numel() > 0:
                        cc += patch.max().item()
                scores[k] -= 20.0 * cc / poses_trajs.shape[1]
        if vel_trajs.shape[1] >= 2:
            accel = vel_trajs[:, 1:, :] - vel_trajs[:, :-1, :]
            jerk_score = (accel ** 2).sum(dim=(1, 2))
            max_jerk = jerk_score.max().item() + 1e-6
            scores += 2.0 * (1.0 - jerk_score / max_jerk)
        avg_v = vel_trajs[:, :, 0].mean(dim=1)
        scores += 1.0 * avg_v / self.v_max
        return scores

    def get_param_count(self):
        enc = sum(p.numel() for p in self.encoder.parameters())
        odom = sum(p.numel() for p in self.odom_encoder.parameters())
        gru = sum(p.numel() for p in self.gru.parameters())
        vf = sum(p.numel() for p in self.vector_field.parameters())
        total = enc + odom + gru + vf
        return {
            'encoder': enc, 'odom_encoder': odom,
            'gru': gru, 'vector_field': vf, 'total': total,
        }


def _euler_sample_from(vector_field, cond, z0, n_steps=3, device='cpu'):
    """Euler ODE integration starting from a specific noise sample z0."""
    B = cond.shape[0]
    dt = 1.0 / n_steps
    x = z0.to(device)
    for i in range(n_steps):
        t = torch.full((B,), i * dt, device=device)
        v = vector_field(x, t, cond)
        x = x + v * dt
    return x
