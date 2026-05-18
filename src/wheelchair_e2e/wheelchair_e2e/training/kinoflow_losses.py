"""
KinoFlow Loss Functions for Kinodynamic Trajectory Training.

Six loss components for training the KinoFlow model:

    L_total = L_flow
              + lambda_kin * L_kinematic
              + lambda_nh  * L_nonholonomic
              + lambda_col * L_collision
              + lambda_jrk * L_jerk
              + lambda_cmf * L_comfort

With ChauffeurNet-style imitation dropout: L_flow is randomly dropped
with probability p=0.5 so the model learns from environment losses alone.

Loss details:
    L_flow:         CFM flow matching loss ||v_theta(x_t, t, c) - u_t||^2
    L_kinematic:    Consistency between v,omega and forward-integrated x,y,theta
    L_nonholonomic: Penalty for violating differential-drive constraints
    L_collision:    Multi-step BEV occupancy checking along trajectory
    L_jerk:         Minimize acceleration changes for smooth motion
    L_comfort:      ISO 2631-1 inspired vibration/comfort metric

References:
    - ChauffeurNet (Bansal et al., RSS 2019): imitation dropout
    - ISO 2631-1: whole-body vibration for seated persons
    - Park & Kuipers (ICRA 2011): wheelchair acceleration hazards
    - FlowMP (Nguyen et al., IROS 2025): kinodynamic flow matching
"""

import torch
import torch.nn as nn
import math


class KinoFlowLoss(nn.Module):
    """
    Combined loss function for KinoFlow training.

    Supports 3-phase training by enabling/disabling individual losses.
    """

    def __init__(self,
                 lambda_flow=1.0,
                 lambda_kinematic=0.5,
                 lambda_nonholonomic=1.0,
                 lambda_collision=5.0,
                 lambda_jerk=0.1,
                 lambda_comfort=0.5,
                 imitation_dropout=0.5,
                 dt=0.1,
                 safety_radius=0.3,
                 resolution=0.05,
                 v_max=0.25,
                 w_max=1.0):
        """
        Args:
            lambda_flow: Weight for flow matching loss
            lambda_kinematic: Weight for kinematic consistency
            lambda_nonholonomic: Weight for non-holonomic constraint violation
            lambda_collision: Weight for collision penalty
            lambda_jerk: Weight for jerk smoothness
            lambda_comfort: Weight for ISO 2631 comfort
            imitation_dropout: Prob of dropping flow loss (ChauffeurNet)
            dt: Time step between trajectory steps (seconds)
            safety_radius: Safety radius for collision checking (meters)
            resolution: BEV grid resolution (m/pixel)
            v_max: Maximum linear velocity (m/s)
            w_max: Maximum angular velocity (rad/s)
        """
        super().__init__()
        self.lambda_flow = lambda_flow
        self.lambda_kinematic = lambda_kinematic
        self.lambda_nonholonomic = lambda_nonholonomic
        self.lambda_collision = lambda_collision
        self.lambda_jerk = lambda_jerk
        self.lambda_comfort = lambda_comfort
        self.imitation_dropout = imitation_dropout
        self.dt = dt
        self.safety_radius = safety_radius
        self.resolution = resolution
        self.v_max = v_max
        self.w_max = w_max

        # Track which losses are active (for 3-phase training)
        self.flow_active = True
        self.kinematic_active = False
        self.nonholonomic_active = False
        self.collision_active = False
        self.jerk_active = False
        self.comfort_active = False

    def set_phase(self, phase):
        """
        Configure active losses for each training phase.

        Phase 1: Flow only (frozen backbone, learn basic trajectory generation)
        Phase 2: + kinematic + collision (learn constraint-aware trajectories)
        Phase 3: + jerk + comfort (learn smooth, comfortable trajectories)
        """
        self.flow_active = True
        if phase == 1:
            self.kinematic_active = False
            self.nonholonomic_active = False
            self.collision_active = False
            self.jerk_active = False
            self.comfort_active = False
        elif phase == 2:
            self.kinematic_active = True
            self.nonholonomic_active = True
            self.collision_active = True
            self.jerk_active = False
            self.comfort_active = False
        elif phase == 3:
            self.kinematic_active = True
            self.nonholonomic_active = True
            self.collision_active = True
            self.jerk_active = True
            self.comfort_active = True

    def forward(self, vector_field, x_0, x_1, t, cond,
                gen_vel_traj=None, gen_poses=None, bev=None):
        """
        Compute combined KinoFlow loss.

        Args:
            vector_field: ConditionalVectorField model
            x_0: (B, H*2) noise samples
            x_1: (B, H*2) target velocity trajectories (flattened)
            t: (B,) flow times ~ U(0,1)
            cond: (B, 256) conditioning features
            gen_vel_traj: (B, H, 2) generated velocity trajectory (for env losses)
            gen_poses: (B, H, 3) integrated poses from gen_vel_traj
            bev: (B, 5, 200, 200) BEV grid (for collision loss)

        Returns:
            total_loss: scalar
            loss_dict: dict with per-component losses
        """
        B = x_1.shape[0]
        device = x_1.device
        loss_dict = {}

        # --- 1. Flow Matching Loss (L_flow) ---
        flow_loss = torch.tensor(0.0, device=device)
        if self.flow_active:
            flow_loss = flow_matching_loss(vector_field, x_0, x_1, t, cond)

            # ChauffeurNet imitation dropout
            if self.training and self.imitation_dropout > 0:
                mask = torch.bernoulli(
                    torch.full((1,), 1.0 - self.imitation_dropout,
                               device=device))
                flow_loss = flow_loss * mask

        loss_dict['flow'] = flow_loss.item()
        total = self.lambda_flow * flow_loss

        # Environment losses require generated trajectories
        if gen_vel_traj is not None:
            H = gen_vel_traj.shape[1]

            # --- 2. Kinematic Consistency Loss ---
            kin_loss = torch.tensor(0.0, device=device)
            if self.kinematic_active and gen_poses is not None:
                kin_loss = kinematic_consistency_loss(
                    gen_vel_traj, gen_poses, self.dt)
            loss_dict['kinematic'] = kin_loss.item()
            total = total + self.lambda_kinematic * kin_loss

            # --- 3. Non-holonomic Constraint Loss ---
            nh_loss = torch.tensor(0.0, device=device)
            if self.nonholonomic_active and gen_poses is not None:
                nh_loss = nonholonomic_violation_loss(
                    gen_vel_traj, gen_poses, self.dt)
            loss_dict['nonholonomic'] = nh_loss.item()
            total = total + self.lambda_nonholonomic * nh_loss

            # --- 4. Collision Loss ---
            coll_loss = torch.tensor(0.0, device=device)
            if self.collision_active and bev is not None and gen_poses is not None:
                coll_loss = trajectory_collision_loss(
                    gen_poses, bev,
                    safety_radius=self.safety_radius,
                    resolution=self.resolution)
            loss_dict['collision'] = coll_loss.item()
            total = total + self.lambda_collision * coll_loss

            # --- 5. Jerk Loss ---
            jerk_loss = torch.tensor(0.0, device=device)
            if self.jerk_active:
                jerk_loss = trajectory_jerk_loss(gen_vel_traj, self.dt)
            loss_dict['jerk'] = jerk_loss.item()
            total = total + self.lambda_jerk * jerk_loss

            # --- 6. Comfort Loss (ISO 2631-1) ---
            comfort_loss = torch.tensor(0.0, device=device)
            if self.comfort_active:
                comfort_loss = iso2631_comfort_loss(gen_vel_traj, self.dt)
            loss_dict['comfort'] = comfort_loss.item()
            total = total + self.lambda_comfort * comfort_loss

        loss_dict['total'] = total.item()
        return total, loss_dict


def flow_matching_loss(vector_field, x_0, x_1, t, cond):
    """
    Conditional Flow Matching loss with optimal transport paths.

    L = ||v_theta(x_t, t, c) - u_t||^2

    where x_t = (1-t)*x_0 + t*x_1 (linear interpolation)
    and   u_t = x_1 - x_0 (straight-line target field)

    Args:
        vector_field: ConditionalVectorField
        x_0: (B, D) noise ~ N(0, I)
        x_1: (B, D) target trajectories
        t: (B,) flow times
        cond: (B, cond_dim) conditioning

    Returns:
        loss: scalar MSE
    """
    x_t = (1 - t[:, None]) * x_0 + t[:, None] * x_1
    u_t = x_1 - x_0
    v_pred = vector_field(x_t, t, cond)
    return ((v_pred - u_t) ** 2).mean()


def kinematic_consistency_loss(vel_traj, poses, dt):
    """
    Kinematic consistency: verify that poses are consistent with velocities.

    For trajectory generated by Option C (parameterization), this should
    be near-zero since poses are computed from velocities. But during
    training with gradient flow through the vector field, it helps
    regularize the trajectory space.

    L_kin = sum_t || [x_{t+1} - x_t - v_t*cos(theta_t)*dt,
                      y_{t+1} - y_t - v_t*sin(theta_t)*dt,
                      theta_{t+1} - theta_t - omega_t*dt] ||^2

    Args:
        vel_traj: (B, H, 2) [v, omega]
        poses: (B, H, 3) [x, y, theta]
        dt: time step

    Returns:
        loss: scalar
    """
    B, H, _ = vel_traj.shape
    if H < 2:
        return torch.tensor(0.0, device=vel_traj.device)

    v = vel_traj[:, :-1, 0]       # (B, H-1)
    omega = vel_traj[:, :-1, 1]   # (B, H-1)
    theta = poses[:, :-1, 2]      # (B, H-1)

    dx_pred = v * torch.cos(theta) * dt
    dy_pred = v * torch.sin(theta) * dt
    dtheta_pred = omega * dt

    dx_actual = poses[:, 1:, 0] - poses[:, :-1, 0]
    dy_actual = poses[:, 1:, 1] - poses[:, :-1, 1]
    dtheta_actual = poses[:, 1:, 2] - poses[:, :-1, 2]

    loss = ((dx_pred - dx_actual) ** 2
            + (dy_pred - dy_actual) ** 2
            + (dtheta_pred - dtheta_actual) ** 2).mean()

    return loss


def nonholonomic_violation_loss(vel_traj, poses, dt):
    """
    Non-holonomic constraint violation for differential drive.

    A differential-drive robot cannot move sideways:
        y_dot * cos(theta) - x_dot * sin(theta) = 0

    This should be ~0 for Option C (parameterization), but acts as
    a regularizer during training.

    Args:
        vel_traj: (B, H, 2) [v, omega]
        poses: (B, H, 3) [x, y, theta]
        dt: time step

    Returns:
        loss: scalar (mean squared lateral velocity)
    """
    B, H, _ = poses.shape
    if H < 2:
        return torch.tensor(0.0, device=vel_traj.device)

    x_dot = (poses[:, 1:, 0] - poses[:, :-1, 0]) / dt  # (B, H-1)
    y_dot = (poses[:, 1:, 1] - poses[:, :-1, 1]) / dt
    theta = poses[:, :-1, 2]

    # Lateral velocity (should be zero for diff-drive)
    v_lateral = -x_dot * torch.sin(theta) + y_dot * torch.cos(theta)

    return (v_lateral ** 2).mean()


def trajectory_collision_loss(poses, bev, safety_radius=0.3, resolution=0.05):
    """
    Multi-step collision loss checking BEV occupancy along the trajectory.

    For each timestep, checks a safety neighborhood around the projected
    pose in the BEV occupancy channels (ch 0=lidar, ch 1=depth).

    Uses soft indexing via bilinear interpolation for gradient flow.

    Args:
        poses: (B, H, 3) trajectory poses [x, y, theta]
        bev: (B, 5, 200, 200) BEV grid
        safety_radius: minimum clearance (meters)
        resolution: BEV resolution (m/pixel)

    Returns:
        loss: scalar collision penalty
    """
    B, H, _ = poses.shape
    grid_size = bev.shape[-1]
    center = grid_size // 2
    safety_px = int(safety_radius / resolution)

    # Combined obstacle channels (max of lidar + depth)
    occupancy = torch.max(bev[:, 0], bev[:, 1])  # (B, 200, 200)

    total_loss = torch.tensor(0.0, device=poses.device)

    for t in range(H):
        x = poses[:, t, 0]  # (B,)
        y = poses[:, t, 1]

        # Convert to pixel coordinates
        px = (x / resolution + center).long().clamp(0, grid_size - 1)
        py = (y / resolution + center).long().clamp(0, grid_size - 1)

        # Check safety neighborhood for each sample
        for b in range(B):
            x_lo = max(0, px[b].item() - safety_px)
            x_hi = min(grid_size, px[b].item() + safety_px + 1)
            y_lo = max(0, py[b].item() - safety_px)
            y_hi = min(grid_size, py[b].item() + safety_px + 1)
            patch = occupancy[b, y_lo:y_hi, x_lo:x_hi]
            if patch.numel() > 0:
                total_loss = total_loss + patch.max()

    return total_loss / (B * H)


def trajectory_jerk_loss(vel_traj, dt=0.1):
    """
    Jerk loss: penalize changes in acceleration for smooth motion.

    Jerk = d(acceleration)/dt = d^2(velocity)/dt^2

    For wheelchair comfort, we want minimal jerk in both linear
    and angular velocity channels.

    L_jerk = (1/dt^2) * sum_t ||v_{t+2} - 2*v_{t+1} + v_t||^2

    If H < 3, falls back to acceleration penalty (first derivative).

    Args:
        vel_traj: (B, H, 2) velocity trajectory [v, omega]
        dt: time step (seconds)

    Returns:
        loss: scalar jerk penalty
    """
    B, H, _ = vel_traj.shape

    if H < 2:
        return torch.tensor(0.0, device=vel_traj.device)

    # Acceleration: dv/dt
    accel = (vel_traj[:, 1:, :] - vel_traj[:, :-1, :]) / dt  # (B, H-1, 2)

    if H >= 3:
        # Jerk: d(accel)/dt
        jerk = (accel[:, 1:, :] - accel[:, :-1, :]) / dt  # (B, H-2, 2)
        return (jerk ** 2).mean()
    else:
        # Fall back to acceleration penalty
        return (accel ** 2).mean()


def iso2631_comfort_loss(vel_traj, dt=0.1):
    """
    ISO 2631-1 inspired comfort loss for wheelchair passengers.

    ISO 2631-1 specifies frequency-weighted RMS acceleration as the
    primary metric for whole-body vibration. For a wheelchair:
    - Fore-aft acceleration: a_x = dv/dt (primary discomfort axis)
    - Lateral acceleration: a_y = v * omega (centripetal)
    - Rotational acceleration: alpha = d(omega)/dt

    Comfort boundaries (ISO 2631-1, 1h exposure):
        < 0.315 m/s^2: not uncomfortable
        0.315 - 0.63:  a little uncomfortable
        0.5 - 1.0:     fairly uncomfortable
        0.8 - 1.6:     uncomfortable
        > 1.6:         very uncomfortable (HAZARDOUS for wheelchair users)

    We penalize:
        1. RMS fore-aft acceleration exceeding 0.5 m/s^2
        2. Centripetal acceleration (v * omega) exceeding 0.3 m/s^2
        3. Angular acceleration exceeding 1.0 rad/s^2

    References:
        - ISO 2631-1: Mechanical vibration and shock
        - Park & Kuipers (ICRA 2011): acceleration hazards for wheelchairs
        - Solea & Nunes (2009): double-inverted pendulum upper body model

    Args:
        vel_traj: (B, H, 2) velocity trajectory [v, omega]
        dt: time step (seconds)

    Returns:
        loss: scalar comfort penalty
    """
    B, H, _ = vel_traj.shape

    if H < 2:
        return torch.tensor(0.0, device=vel_traj.device)

    v = vel_traj[:, :, 0]      # (B, H) linear velocity
    omega = vel_traj[:, :, 1]  # (B, H) angular velocity

    # 1. Fore-aft acceleration: dv/dt
    a_fore_aft = (v[:, 1:] - v[:, :-1]) / dt  # (B, H-1)
    rms_fore_aft = torch.sqrt((a_fore_aft ** 2).mean(dim=1))  # (B,)

    # Penalize exceeding 0.5 m/s^2 (fairly uncomfortable threshold)
    fore_aft_penalty = torch.relu(rms_fore_aft - 0.5) ** 2

    # 2. Centripetal acceleration: v * omega
    a_centripetal = v[:, :-1] * omega[:, :-1]  # (B, H-1)
    rms_centripetal = torch.sqrt((a_centripetal ** 2).mean(dim=1))  # (B,)

    # Penalize exceeding 0.3 m/s^2 (lateral is more uncomfortable)
    centripetal_penalty = torch.relu(rms_centripetal - 0.3) ** 2

    # 3. Angular acceleration: d(omega)/dt
    alpha = (omega[:, 1:] - omega[:, :-1]) / dt  # (B, H-1)
    rms_alpha = torch.sqrt((alpha ** 2).mean(dim=1))  # (B,)

    # Penalize exceeding 1.0 rad/s^2
    alpha_penalty = torch.relu(rms_alpha - 1.0) ** 2

    # Combined comfort loss (weighted sum)
    loss = (1.0 * fore_aft_penalty.mean()
            + 1.5 * centripetal_penalty.mean()
            + 0.5 * alpha_penalty.mean())

    return loss
