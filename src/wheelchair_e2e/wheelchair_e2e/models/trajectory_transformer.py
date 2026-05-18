"""
Trajectory Transformer Vector Field — Attention-based denoiser for CFM.

Key architectural innovation: trajectory timesteps are TOKENS, not flat vectors.
Self-attention across steps → temporal coherence.
Cross-attention to conditioning → context-aware denoising.

vs MLP (current KinoFlow): 2.3M params, no temporal structure
vs 1D U-Net (Crowd-FM): skip connections but no global attention
vs THIS: ~570K params, explicit temporal + contextual attention

Drop-in replacement for ConditionalVectorField:
    forward(x_t, t, cond) → (B, H*2) predicted vector field

Same interface, same flow matching loss, same Euler sampling.
"""

import torch
import torch.nn as nn

from wheelchair_e2e.models.flow_matching import SinusoidalTimeEmbedding


class TrajectoryTransformerVectorField(nn.Module):
    """Attention-based vector field for trajectory generation via CFM.

    Each trajectory timestep is a token. Self-attention ensures temporal
    coherence across steps. Cross-attention injects scene context at every
    layer.

    Args:
        horizon: trajectory length (number of (v,ω) steps)
        cond_dim: conditioning dimension from encoder
        d_model: internal Transformer dimension
        nhead: number of attention heads
        num_layers: number of self-attn + cross-attn layer pairs
    """

    def __init__(self, horizon=10, cond_dim=256, d_model=128,
                 nhead=4, num_layers=2):
        super().__init__()
        self.horizon = horizon
        self.traj_dim = horizon * 2
        self.d_model = d_model

        # Token projection: (v, ω) per step → d_model
        self.token_proj = nn.Linear(2, d_model)

        # Positional embedding for H trajectory steps
        self.pos_embed = nn.Parameter(torch.randn(1, horizon, d_model) * 0.02)

        # Sinusoidal flow time embedding (same as ConditionalVectorField)
        self.time_embed = SinusoidalTimeEmbedding(d_model)
        self.time_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
        )

        # Conditioning projection
        self.cond_proj = nn.Linear(cond_dim, d_model)

        # Self-attention layers (temporal coherence across trajectory steps)
        self.self_attn_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * 4,
                activation='gelu',
                batch_first=True,
                norm_first=True,
            )
            for _ in range(num_layers)
        ])

        # Cross-attention layers: trajectory attends to conditioning
        self.cross_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(d_model, nhead, batch_first=True)
            for _ in range(num_layers)
        ])
        self.cross_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(num_layers)
        ])

        # Output: each token → 2-d (v, ω) direction
        self.output_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 2),
        )

    def forward(self, x_t, t, cond):
        """Predict vector field direction for denoising.

        Same interface as ConditionalVectorField — drop-in replacement.

        Args:
            x_t:  (B, H*2) noised trajectory at flow time t (flat)
            t:    (B,) flow time in [0, 1]
            cond: (B, cond_dim) conditioning from encoder

        Returns:
            v_field: (B, H*2) predicted vector field (flat)
        """
        B = x_t.shape[0]

        # Reshape flat trajectory → H tokens of 2-d each
        tokens = x_t.view(B, self.horizon, 2)      # (B, H, 2)
        tokens = self.token_proj(tokens)             # (B, H, d_model)
        tokens = tokens + self.pos_embed             # + positional embedding

        # Add flow time embedding to all tokens (broadcast)
        t_emb = self.time_proj(self.time_embed(t))   # (B, d_model)
        tokens = tokens + t_emb.unsqueeze(1)         # (B, H, d_model)

        # Conditioning as key/value for cross-attention
        cond_token = self.cond_proj(cond).unsqueeze(1)  # (B, 1, d_model)

        # Alternating self-attention + cross-attention
        for self_attn, cross_attn, cross_norm in zip(
            self.self_attn_layers,
            self.cross_attn_layers,
            self.cross_norms,
        ):
            # Self-attention: temporal coherence across trajectory steps
            tokens = self_attn(tokens)               # (B, H, d_model)

            # Cross-attention: each step attends to scene context
            residual = tokens
            tokens_normed = cross_norm(tokens)
            attn_out, _ = cross_attn(
                query=tokens_normed,
                key=cond_token,
                value=cond_token,
            )                                        # (B, H, d_model)
            tokens = residual + attn_out

        # Project each token to (v, ω) direction
        output = self.output_proj(tokens)            # (B, H, 2)
        return output.reshape(B, -1)                 # (B, H*2)
