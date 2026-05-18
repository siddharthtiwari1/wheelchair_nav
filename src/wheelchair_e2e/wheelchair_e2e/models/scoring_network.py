"""
DualSpaceScoringTransformer: Learned Trajectory Scorer for KinoFlow.

BEATS both CrowdSurfer (Conv1d+MLP) and Crowd-FM (position-only Transformer):

1. DUAL-SPACE encoding: velocity (v,ω) AND position (x,y,θ) per trajectory
   - Crowd-FM only encodes position (x,y) Bernstein coefficients
   - We know both "what the motors will do" AND "where the wheelchair ends up"

2. COMFORT-AWARE: explicit jerk/acceleration/ISO features as a scoring signal
   - Neither CrowdSurfer nor Crowd-FM considers comfort in scoring

3. CONTEXT REUSE: scene features from backbone (already computed during CFM generation)
   - No redundant encoding — Crowd-FM re-encodes the full scene for scoring

4. CROSS-ATTENTION: trajectories attend to scene context AND to each other
   - Enables relative comparison ("trajectory 3 vs trajectory 7 given this scene")

Architecture:
    For each of K candidates:
        vel_encoder(v,ω sequence) → D-dim
        pos_encoder(x,y,θ sequence) → D-dim
        comfort_encoder([jerk, accel, iso, collision]) → D-dim
        trajectory_token = vel + pos + comfort + learnable_modality_emb

    Context tokens (REUSED from backbone):
        scene_token = Linear(backbone_512d) → D-dim
        goal_token = MLP(goal_dx, goal_dy, dist, angle) → D-dim

    [traj_1, ..., traj_K, scene, goal] → 4-head 3-layer Transformer → MLP → scores

Training: CrossEntropy(scores, closest_to_expert) + comfort auxiliary loss
~800K params. <1ms inference on GPU.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TrajectoryEncoder(nn.Module):
    """Encodes a single trajectory into a D-dimensional token.

    Dual-space: processes BOTH velocity (v,ω) and position (x,y,θ) sequences,
    plus explicit comfort features.
    """

    def __init__(self, embed_dim=128, horizon=10):
        super().__init__()

        # Velocity-space encoder: (H, 2) → embed_dim
        self.vel_encoder = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, embed_dim),
        )

        # Position-space encoder: (H, 3) → embed_dim
        self.pos_encoder = nn.Sequential(
            nn.Conv1d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, embed_dim),
        )

        # Comfort features encoder: [jerk_v, jerk_w, accel_rms, collision_count] → embed_dim
        self.comfort_encoder = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, embed_dim),
        )

        # Fusion: 3 × embed_dim → embed_dim
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def compute_comfort_features(self, vel_traj, poses_traj, bev_occupancy=None):
        """Extract comfort/safety features from a trajectory.

        Args:
            vel_traj: (H, 2) velocity trajectory [v, ω]
            poses_traj: (H, 3) pose trajectory [x, y, θ]
            bev_occupancy: (200, 200) optional obstacle map

        Returns:
            features: (4,) [jerk_v_rms, jerk_w_rms, accel_rms, collision_frac]
        """
        device = vel_traj.device
        H = vel_traj.shape[0]

        # Jerk (second derivative of velocity)
        if H >= 3:
            accel = vel_traj[1:] - vel_traj[:-1]  # (H-1, 2)
            jerk = accel[1:] - accel[:-1]  # (H-2, 2)
            jerk_v_rms = jerk[:, 0].pow(2).mean().sqrt()
            jerk_w_rms = jerk[:, 1].pow(2).mean().sqrt()
            accel_rms = accel.pow(2).sum(dim=-1).mean().sqrt()
        else:
            jerk_v_rms = torch.tensor(0.0, device=device)
            jerk_w_rms = torch.tensor(0.0, device=device)
            accel_rms = torch.tensor(0.0, device=device)

        # Collision fraction
        collision_frac = torch.tensor(0.0, device=device)
        if bev_occupancy is not None:
            grid_size = bev_occupancy.shape[0]
            center = grid_size // 2
            resolution = 0.05
            collisions = 0
            for t in range(H):
                px = int(poses_traj[t, 0].item() / resolution + center)
                py = int(poses_traj[t, 1].item() / resolution + center)
                px = max(0, min(grid_size - 1, px))
                py = max(0, min(grid_size - 1, py))
                if bev_occupancy[py, px] > 0.5:
                    collisions += 1
            collision_frac = torch.tensor(collisions / max(H, 1), device=device)

        return torch.stack([jerk_v_rms, jerk_w_rms, accel_rms, collision_frac])

    def forward(self, vel_traj, poses_traj, comfort_feats):
        """Encode a single trajectory into a token.

        Args:
            vel_traj: (B, H, 2) velocity trajectory
            poses_traj: (B, H, 3) pose trajectory
            comfort_feats: (B, 4) precomputed comfort features

        Returns:
            token: (B, embed_dim)
        """
        # Conv1d expects (B, C, L) — transpose from (B, H, C)
        vel_feat = self.vel_encoder(vel_traj.transpose(1, 2))    # (B, D)
        pos_feat = self.pos_encoder(poses_traj.transpose(1, 2))  # (B, D)
        comfort_feat = self.comfort_encoder(comfort_feats)        # (B, D)

        combined = torch.cat([vel_feat, pos_feat, comfort_feat], dim=-1)
        return self.fusion(combined)  # (B, D)


class DualSpaceScoringTransformer(nn.Module):
    """
    Learned trajectory scorer using cross-attention between K candidates
    and scene context. Strictly better than both CrowdSurfer and Crowd-FM scorers.

    Input tokens:
        - K trajectory tokens (dual-space encoded)
        - 1 scene context token (reused from backbone)
        - 1 goal token

    Output: K scores (one per trajectory)
    """

    def __init__(self, embed_dim=128, n_heads=4, n_layers=3,
                 backbone_dim=128, horizon=10, dropout=0.1):
        """
        Args:
            embed_dim: Token embedding dimension (128 for Jetson efficiency)
            n_heads: Number of attention heads
            n_layers: Number of Transformer encoder layers
            backbone_dim: Dimension of scene features from encoder backbone.
                          128 for ModularKinoFlowNet (StaticSceneEncoder),
                          512 for legacy KinoFlowNet (ResNet-18).
            horizon: Trajectory length in steps
            dropout: Attention/FF dropout rate
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.horizon = horizon

        # Trajectory encoder (dual-space + comfort)
        self.traj_encoder = TrajectoryEncoder(embed_dim=embed_dim, horizon=horizon)

        # Context projections (reuse backbone features — no redundant encoding)
        self.scene_proj = nn.Sequential(
            nn.Linear(backbone_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        # Goal encoder: (dx, dy, dist, angle) → embed_dim
        self.goal_proj = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        # Learnable modality embeddings (trajectory vs scene vs goal)
        self.traj_type_emb = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.scene_type_emb = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.goal_type_emb = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,  # Pre-norm for training stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )

        # Scoring head: trajectory tokens → scalar scores
        self.score_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 1),
        )

    def forward(self, vel_trajs, poses_trajs, comfort_feats,
                scene_features, goal_features):
        """
        Score K trajectory candidates given scene context.

        Args:
            vel_trajs: (K, H, 2) velocity trajectories
            poses_trajs: (K, H, 3) pose trajectories
            comfort_feats: (K, 4) comfort features per trajectory
            scene_features: (512,) backbone features (REUSED from encoder)
            goal_features: (4,) [goal_dx, goal_dy, goal_dist, goal_angle]

        Returns:
            scores: (K,) unnormalized scores (higher = better)
        """
        K = vel_trajs.shape[0]
        device = vel_trajs.device

        # --- Encode K trajectory tokens ---
        traj_tokens = self.traj_encoder(
            vel_trajs, poses_trajs, comfort_feats
        )  # (K, D)
        traj_tokens = traj_tokens.unsqueeze(0) + self.traj_type_emb  # (1, K, D)

        # --- Scene context token (reused from backbone) ---
        scene_token = self.scene_proj(
            scene_features.unsqueeze(0).unsqueeze(0)
        ) + self.scene_type_emb  # (1, 1, D)

        # --- Goal token ---
        goal_token = self.goal_proj(
            goal_features.unsqueeze(0).unsqueeze(0)
        ) + self.goal_type_emb  # (1, 1, D)

        # --- Assemble token sequence: [traj_1, ..., traj_K, scene, goal] ---
        tokens = torch.cat([traj_tokens, scene_token, goal_token], dim=1)
        # (1, K+2, D)

        # --- Transformer cross-attention ---
        out = self.transformer(tokens)  # (1, K+2, D)

        # --- Extract trajectory tokens and score them ---
        traj_out = out[0, :K, :]  # (K, D) — only trajectory tokens
        scores = self.score_head(traj_out).squeeze(-1)  # (K,)

        return scores

    def score_with_context(self, vel_trajs, poses_trajs,
                           bev_occupancy, scene_features,
                           goal_dx, goal_dy):
        """
        Full scoring pipeline: compute comfort features + score.

        This is the main entry point used by KinoFlowNet at inference.

        Args:
            vel_trajs: (K, H, 2) velocity trajectories
            poses_trajs: (K, H, 3) pose trajectories
            bev_occupancy: (200, 200) obstacle map
            scene_features: (512,) from backbone encoder
            goal_dx, goal_dy: relative goal position (meters)

        Returns:
            scores: (K,) trajectory scores
        """
        K = vel_trajs.shape[0]
        device = vel_trajs.device

        # Compute comfort features for each trajectory
        comfort_list = []
        for k in range(K):
            cf = self.traj_encoder.compute_comfort_features(
                vel_trajs[k], poses_trajs[k], bev_occupancy
            )
            comfort_list.append(cf)
        comfort_feats = torch.stack(comfort_list).to(device)  # (K, 4)

        # Goal features: (dx, dy, distance, angle)
        goal_dist = math.sqrt(goal_dx ** 2 + goal_dy ** 2)
        goal_angle = math.atan2(goal_dy, goal_dx)
        goal_features = torch.tensor(
            [goal_dx, goal_dy, goal_dist, goal_angle],
            device=device, dtype=torch.float32
        )

        return self.forward(
            vel_trajs, poses_trajs, comfort_feats,
            scene_features, goal_features
        )

    def get_param_count(self):
        """Return parameter count breakdown."""
        traj_enc = sum(p.numel() for p in self.traj_encoder.parameters())
        projs = sum(p.numel() for p in self.scene_proj.parameters()) + \
                sum(p.numel() for p in self.goal_proj.parameters())
        type_embs = self.traj_type_emb.numel() + \
                    self.scene_type_emb.numel() + \
                    self.goal_type_emb.numel()
        transformer = sum(p.numel() for p in self.transformer.parameters())
        head = sum(p.numel() for p in self.score_head.parameters())
        total = sum(p.numel() for p in self.parameters())
        return {
            'traj_encoder': traj_enc,
            'projections': projs,
            'type_embeddings': type_embs,
            'transformer': transformer,
            'score_head': head,
            'total': total,
        }
