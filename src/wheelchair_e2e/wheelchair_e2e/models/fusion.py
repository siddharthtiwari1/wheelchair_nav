"""
Transformer Fusion — Cross-attention over 4 modality tokens.

Fuses outputs from E1 (static scene), E2 (dynamic obstacles), E3 (goal),
and E4 (velocity context) via a 4-token Transformer with learnable modality
embeddings.

Why Transformer fusion (not concat, not gated):
- CrowdSurfer: simple concat (32+32+4=68) → Linear → 32. No inter-modal reasoning.
- Crowd-FM: 8h 3L Transformer over 3 tokens. Works but expensive for 3 tokens.
- Current KinoFlow: concat (512+64=576). No inter-modal reasoning.
- Ours: Project all 4 encoders to 128-d → 4-token Transformer (4h, 2L).

Critical advantage: The goal token can attend to the dynamic obstacle token to
determine "is the goal direction blocked by a moving person?" The static scene token
can attend to the velocity token to determine "am I going fast enough to need to see
further ahead?" These cross-modal interactions are impossible with concatenation.

~260K params.
"""

import torch
import torch.nn as nn


class TransformerFusion(nn.Module):
    """Fuses 4 encoder outputs via cross-attention Transformer.

    Input:  scene(128), dynamic(128), goal(64), odom(64)
    Output: (B, cond_dim) — fused conditioning for GRU/decoder

    Supports ablation flags:
        no_dynamic: zeros out dynamic token
        no_goal: zeros out goal token
        no_velocity: zeros out velocity token
        concat_fusion: bypass Transformer, use simple concat + Linear
    """

    def __init__(self, scene_dim=128, dynamic_dim=128, goal_dim=64,
                 odom_dim=64, d_model=128, cond_dim=256,
                 nhead=4, num_layers=2,
                 # Ablation flags
                 no_dynamic=False, no_goal=False, no_velocity=False,
                 concat_fusion=False):
        super().__init__()
        self.d_model = d_model
        self.cond_dim = cond_dim
        self.no_dynamic = no_dynamic
        self.no_goal = no_goal
        self.no_velocity = no_velocity
        self.concat_fusion = concat_fusion

        # Project all modalities to common dimension
        self.proj_scene = nn.Sequential(
            nn.Linear(scene_dim, d_model),
            nn.LayerNorm(d_model),
        )
        self.proj_dynamic = nn.Sequential(
            nn.Linear(dynamic_dim, d_model),
            nn.LayerNorm(d_model),
        )
        self.proj_goal = nn.Sequential(
            nn.Linear(goal_dim, d_model),
            nn.LayerNorm(d_model),
        )
        self.proj_odom = nn.Sequential(
            nn.Linear(odom_dim, d_model),
            nn.LayerNorm(d_model),
        )

        # Learnable modality embeddings
        self.type_embeddings = nn.Parameter(
            torch.randn(4, 1, d_model) * 0.02
        )  # [scene, dynamic, goal, odom]

        if not concat_fusion:
            # Transformer: 4 tokens attend to each other
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * 4,
                activation='gelu',
                batch_first=True,
                norm_first=True,
            )
            self.transformer = nn.TransformerEncoder(
                encoder_layer, num_layers=num_layers
            )
        else:
            self.transformer = None

        # Output: pool 4 tokens → cond_dim
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, cond_dim),
            nn.LayerNorm(cond_dim),
        )

    def forward(self, scene_feat, dynamic_feat, goal_feat, odom_feat):
        """
        Args:
            scene_feat:   (B, scene_dim) from StaticSceneEncoder
            dynamic_feat: (B, dynamic_dim) from DynamicObstacleEncoder
            goal_feat:    (B, goal_dim) from GoalEncoder
            odom_feat:    (B, odom_dim) from VelocityContextEncoder

        Returns:
            cond: (B, cond_dim) fused conditioning vector
        """
        B = scene_feat.shape[0]

        # Project to common dimension
        s = self.proj_scene(scene_feat)      # (B, d_model)
        d = self.proj_dynamic(dynamic_feat)  # (B, d_model)
        g = self.proj_goal(goal_feat)        # (B, d_model)
        o = self.proj_odom(odom_feat)        # (B, d_model)

        # Ablation: zero out modalities
        if self.no_dynamic:
            d = torch.zeros_like(d)
        if self.no_goal:
            g = torch.zeros_like(g)
        if self.no_velocity:
            o = torch.zeros_like(o)

        # Add modality type embeddings
        s = s + self.type_embeddings[0].squeeze(0)  # broadcast to (B, d_model)
        d = d + self.type_embeddings[1].squeeze(0)
        g = g + self.type_embeddings[2].squeeze(0)
        o = o + self.type_embeddings[3].squeeze(0)

        # Stack as 4-token sequence
        tokens = torch.stack([s, d, g, o], dim=1)  # (B, 4, d_model)

        if not self.concat_fusion and self.transformer is not None:
            # Transformer cross-attention
            tokens = self.transformer(tokens)  # (B, 4, d_model)

        # Mean pool over 4 tokens → (B, d_model)
        pooled = tokens.mean(dim=1)

        # Project to conditioning dimension
        return self.output_proj(pooled)  # (B, cond_dim)
