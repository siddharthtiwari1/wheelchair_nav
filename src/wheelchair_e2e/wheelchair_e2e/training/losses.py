"""
Combined Loss Function for E2E Velocity Training.

Loss = m * [λ_v·MSE(v) + λ_ω·MSE(ω)] + λ_c·Collision(BEV, v, ω) + λ_j·Jerk

Where m ~ Bernoulli(0.5) is the imitation dropout mask.

Components:
    1. Imitation loss: MSE on (v, ω) predictions vs expert
    2. Collision loss: penalize predicted motion into occupied BEV cells
    3. Jerk loss: penalize velocity changes for smooth motion
    4. Imitation dropout: randomly drop imitation loss 50% of the time
       so environment losses (collision, jerk) dominate

Inspired by ChauffeurNet (Waymo): imitation dropout improved
obstacle avoidance from 55% → 90%.
"""

import torch
import torch.nn as nn


class E2EVelocityLoss(nn.Module):
    """
    Combined training loss for end-to-end velocity prediction.

    L = m · [λ_v·(v - v*)² + λ_ω·(ω - ω*)²]
        + λ_c · C(BEV, v, ω)
        + λ_j · ||a_t - a_{t-1}||²
    """

    def __init__(self, lambda_v=1.0, lambda_w=1.0,
                 lambda_collision=10.0, lambda_jerk=0.1,
                 imitation_dropout=0.5,
                 safety_radius=0.3, dt=0.1):
        """
        Args:
            lambda_v: weight for linear velocity MSE
            lambda_w: weight for angular velocity MSE
            lambda_collision: weight for collision penalty
            lambda_jerk: weight for jerk penalty
            imitation_dropout: probability of dropping imitation loss
            safety_radius: minimum safe distance to obstacles (meters)
            dt: time step for forward projection (seconds)
        """
        super().__init__()
        self.lambda_v = lambda_v
        self.lambda_w = lambda_w
        self.lambda_collision = lambda_collision
        self.lambda_jerk = lambda_jerk
        self.imitation_dropout = imitation_dropout
        self.safety_radius = safety_radius
        self.dt = dt

    def forward(self, pred_vel, target_vel, bev=None,
                prev_vel=None):
        """
        Compute combined loss.

        Args:
            pred_vel: (B, 2) predicted [v, omega]
            target_vel: (B, 2) expert [v, omega]
            bev: (B, 4, 200, 200) BEV grid (for collision loss)
            prev_vel: (B, 2) previous predicted velocity (for jerk loss)

        Returns:
            total_loss: scalar
            loss_dict: breakdown of individual losses
        """
        B = pred_vel.shape[0]

        # --- Imitation loss ---
        v_loss = ((pred_vel[:, 0] - target_vel[:, 0]) ** 2).mean()
        w_loss = ((pred_vel[:, 1] - target_vel[:, 1]) ** 2).mean()
        imitation_loss = self.lambda_v * v_loss + self.lambda_w * w_loss

        # Per-sample imitation dropout (ChauffeurNet §3.3):
        # each sample independently loses imitation signal with prob p,
        # forcing the model to learn from collision/jerk losses alone.
        if self.training and self.imitation_dropout > 0:
            mask = torch.bernoulli(
                torch.full((B,), 1.0 - self.imitation_dropout,
                           device=pred_vel.device))
            per_sample_imit = (
                self.lambda_v * (pred_vel[:, 0] - target_vel[:, 0]) ** 2
                + self.lambda_w * (pred_vel[:, 1] - target_vel[:, 1]) ** 2)
            imitation_loss = (per_sample_imit * mask).mean()

        # --- Collision loss ---
        collision_loss = torch.tensor(0.0, device=pred_vel.device)
        if bev is not None and self.lambda_collision > 0:
            collision_loss = self._collision_loss(pred_vel, bev)

        # --- Jerk loss ---
        jerk_loss = torch.tensor(0.0, device=pred_vel.device)
        if prev_vel is not None and self.lambda_jerk > 0:
            jerk_loss = ((pred_vel - prev_vel) ** 2).mean()

        # --- Total ---
        total = (imitation_loss
                 + self.lambda_collision * collision_loss
                 + self.lambda_jerk * jerk_loss)

        loss_dict = {
            'total': total.item(),
            'imitation': imitation_loss.item(),
            'v_mse': v_loss.item(),
            'w_mse': w_loss.item(),
            'collision': collision_loss.item(),
            'jerk': jerk_loss.item(),
        }

        return total, loss_dict

    def _collision_loss(self, pred_vel, bev):
        """
        Penalize predicted motion that brings wheelchair close to obstacles.

        Forward-project the wheelchair's position by dt seconds using
        predicted (v, ω), check if the resulting position is near
        occupied cells in the BEV.

        C(BEV, v, ω) = max(0, r_safety - min_d_i(BEV, v·dt, ω·dt))
        """
        B = pred_vel.shape[0]
        grid_size = bev.shape[-1]
        center = grid_size // 2
        resolution = 0.05  # m/pixel

        v = pred_vel[:, 0]      # (B,)
        omega = pred_vel[:, 1]  # (B,)

        # Differential drive arc projection:
        # For small dt, the wheelchair moves along an arc.
        # x' = v*dt*cos(omega*dt/2),  y' = v*dt*sin(omega*dt/2)
        # This accounts for turning — not just forward motion.
        half_dtheta = omega * self.dt * 0.5
        dx = v * self.dt * torch.cos(half_dtheta)  # (B,) forward
        dy = v * self.dt * torch.sin(half_dtheta)  # (B,) lateral

        # Convert to pixel coordinates (BEV is ego-centric)
        # BEV: x = forward = row index from center, y = left = col from center
        px = (dx / resolution + center).long()
        py = (dy / resolution + center).long()
        px = px.clamp(0, grid_size - 1)
        py = py.clamp(0, grid_size - 1)

        # Check occupancy in safety neighborhood around projected point
        # Use channels 0 and 1 (LiDAR + depth obstacles)
        occupancy = torch.max(bev[:, 0, :, :], bev[:, 1, :, :])  # (B,H,W)
        safety_px = int(self.safety_radius / resolution)

        loss = torch.tensor(0.0, device=pred_vel.device, requires_grad=False)
        for b in range(B):
            x_min = max(0, px[b].item() - safety_px)
            x_max = min(grid_size, px[b].item() + safety_px + 1)
            y_min = max(0, py[b].item() - safety_px)
            y_max = min(grid_size, py[b].item() + safety_px + 1)

            patch = occupancy[b, y_min:y_max, x_min:x_max]
            if patch.numel() > 0:
                loss = loss + patch.max()

        return loss / B
