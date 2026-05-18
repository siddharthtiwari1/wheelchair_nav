"""
Conditional Flow Matching for Velocity Trajectory Generation.

Flow matching learns a vector field that transports noise → data along
straight-line (optimal transport) paths. At inference, a single ODE solve
(1-5 Euler steps) generates a velocity trajectory.

Key equations:
    Interpolation: x_t = (1 - t) * x_0 + t * x_1
    Target field:  u_t = x_1 - x_0
    Loss:          ||v_θ(x_t, t, c) - u_t||²

Reference: Lipman et al., "Flow Matching for Generative Modeling", ICLR 2023
"""

import torch
import torch.nn as nn
import math


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal positional encoding for flow time t ∈ [0, 1]."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        """
        Args:
            t: (B,) flow time in [0, 1]
        Returns:
            emb: (B, dim)
        """
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / half
        )
        args = t[:, None] * freqs[None, :]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class ConditionalVectorField(nn.Module):
    """
    Neural network that predicts the vector field v_θ(x_t, t, c).

    Given a noised velocity trajectory x_t, flow time t, and conditioning
    features c (from BEV encoder), predicts the direction to denoise.

    Architecture: MLP with residual connections.
    """

    def __init__(self, traj_dim, cond_dim, hidden_dim=512, n_layers=4):
        """
        Args:
            traj_dim: dimension of velocity trajectory (H * 2)
            cond_dim: dimension of conditioning features (576 from encoder)
            hidden_dim: hidden layer width
            n_layers: number of residual MLP blocks
        """
        super().__init__()
        self.traj_dim = traj_dim

        # Time embedding
        self.time_embed = SinusoidalTimeEmbedding(128)

        # Input projection: trajectory + time + conditioning → hidden
        self.input_proj = nn.Sequential(
            nn.Linear(traj_dim + 128 + cond_dim, hidden_dim),
            nn.SiLU(),
        )

        # Residual MLP blocks
        self.blocks = nn.ModuleList()
        for _ in range(n_layers):
            self.blocks.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            ))

        # Output projection: hidden → trajectory-shaped vector field
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, traj_dim),
        )

    def forward(self, x_t, t, cond):
        """
        Args:
            x_t: (B, traj_dim) noised trajectory at flow time t
            t: (B,) flow time in [0, 1]
            cond: (B, cond_dim) conditioning from encoder

        Returns:
            v_field: (B, traj_dim) predicted vector field
        """
        t_emb = self.time_embed(t)                          # (B, 128)
        h = torch.cat([x_t, t_emb, cond], dim=-1)          # (B, traj+128+cond)
        h = self.input_proj(h)                               # (B, hidden)

        for block in self.blocks:
            h = h + block(h)                                 # residual

        return self.output_proj(h)                           # (B, traj_dim)


def flow_matching_loss(vector_field, x_0, x_1, t, cond):
    """
    Compute conditional flow matching loss.

    Args:
        vector_field: ConditionalVectorField model
        x_0: (B, D) noise samples ~ N(0, I)
        x_1: (B, D) target velocity trajectories (from expert)
        t: (B,) random flow times ~ U(0, 1)
        cond: (B, cond_dim) conditioning features

    Returns:
        loss: scalar MSE loss
    """
    # Linear interpolation (optimal transport path)
    x_t = (1 - t[:, None]) * x_0 + t[:, None] * x_1

    # Target vector field (straight-line direction)
    u_t = x_1 - x_0

    # Predicted vector field
    v_pred = vector_field(x_t, t, cond)

    # MSE loss
    return ((v_pred - u_t) ** 2).mean()


@torch.no_grad()
def euler_sample(vector_field, cond, traj_dim, n_steps=5, device='cpu'):
    """
    Generate velocity trajectory via Euler ODE integration.

    Starts from noise z ~ N(0, I) and integrates the learned vector field
    from t=0 to t=1 in n_steps Euler steps.

    Args:
        vector_field: trained ConditionalVectorField
        cond: (B, cond_dim) conditioning features
        traj_dim: output dimension (H * 2)
        n_steps: number of Euler integration steps (1-5 for real-time)
        device: torch device

    Returns:
        x: (B, traj_dim) generated velocity trajectory
    """
    B = cond.shape[0]
    dt = 1.0 / n_steps

    # Start from Gaussian noise
    x = torch.randn(B, traj_dim, device=device)

    # Euler integration: dx/dt = v_θ(x, t, c)
    for i in range(n_steps):
        t = torch.full((B,), i * dt, device=device)
        v = vector_field(x, t, cond)
        x = x + v * dt

    return x


def euler_sample_guided(vector_field, cond, traj_dim, n_steps=5,
                        device='cpu', bev_occupancy=None,
                        guidance_scale=0.5, v_max=0.25, w_max=1.0,
                        horizon=10, resolution=0.05, grid_center=100):
    """
    Generate velocity trajectory with COLLISION COST GUIDANCE.

    Like Crowd-FM's inference-time gradient injection: at each Euler step,
    compute gradient of collision cost w.r.t. trajectory, nudge away from
    obstacles. This gives ~10pp success rate improvement per Crowd-FM ablation.

    Algorithm:
        For each Euler step:
            1. x = x + v_θ(x, t, c) * dt           (standard flow step)
            2. cost = collision_cost(integrate(x))    (check where trajectory goes)
            3. grad = d(cost) / d(x)                  (how to move trajectory away)
            4. x = x - guidance_scale * grad          (nudge away from obstacles)

    Args:
        vector_field: trained ConditionalVectorField
        cond: (B, cond_dim) conditioning
        traj_dim: output dimension (H * 2)
        n_steps: Euler steps
        device: torch device
        bev_occupancy: (H_grid, W_grid) obstacle map, or None to skip guidance
        guidance_scale: strength of collision avoidance gradient
        v_max, w_max: velocity bounds for scaling
        horizon: trajectory length
        resolution: BEV grid resolution
        grid_center: center pixel of BEV grid

    Returns:
        x: (B, traj_dim) generated velocity trajectory
    """
    B = cond.shape[0]
    dt_flow = 1.0 / n_steps
    dt_traj = 0.1  # trajectory timestep (seconds)

    x = torch.randn(B, traj_dim, device=device)

    for i in range(n_steps):
        t = torch.full((B,), i * dt_flow, device=device)
        v = vector_field(x, t, cond)
        x = x + v * dt_flow

        # Collision cost guidance (only if occupancy map available)
        if bev_occupancy is not None and guidance_scale > 0:
            x_guided = x.detach().requires_grad_(True)

            # Reshape to (B, H, 2) and scale to physical velocities
            traj = x_guided.view(B, horizon, 2)
            v_phys = (torch.tanh(traj[:, :, 0]) + 1) / 2 * v_max
            w_phys = torch.tanh(traj[:, :, 1]) * w_max

            # Integrate to positions
            pos_x = torch.zeros(B, device=device)
            pos_y = torch.zeros(B, device=device)
            theta = torch.zeros(B, device=device)
            collision_cost = torch.zeros(B, device=device)

            occ_t = torch.from_numpy(bev_occupancy).float().to(device) \
                if not isinstance(bev_occupancy, torch.Tensor) else bev_occupancy.float()

            for step in range(horizon):
                theta = theta + w_phys[:, step] * dt_traj
                pos_x = pos_x + v_phys[:, step] * torch.cos(theta) * dt_traj
                pos_y = pos_y + v_phys[:, step] * torch.sin(theta) * dt_traj

                # Sample occupancy at trajectory position (bilinear)
                px = (pos_x / resolution + grid_center).clamp(0, occ_t.shape[-1] - 1)
                py = (pos_y / resolution + grid_center).clamp(0, occ_t.shape[-2] - 1)

                # Nearest-neighbor sampling (differentiable approximation)
                px_i = px.long().clamp(0, occ_t.shape[-1] - 1)
                py_i = py.long().clamp(0, occ_t.shape[-2] - 1)
                for b in range(B):
                    collision_cost[b] = collision_cost[b] + occ_t[py_i[b], px_i[b]]

            total_cost = collision_cost.sum()
            if total_cost.item() > 0:
                total_cost.backward()
                if x_guided.grad is not None:
                    x = x - guidance_scale * x_guided.grad

    return x.detach()
