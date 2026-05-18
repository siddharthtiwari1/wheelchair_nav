#!/usr/bin/env python3
"""
Generate synthetic training data for KinoFlow pipeline verification.

Creates random 2D environments with obstacles, simulates a differential-drive
wheelchair navigating toward goals using pure pursuit, and renders 5-channel
BEV grids with velocity labels in the exact format expected by
CFMTrajectoryDataset and train_kinoflow.py.

NO ROS dependency — pure numpy. Runs on any machine.

Output format (identical to bag_to_bev_velocity.py):
    data_dir/
        bev_000000.npy       (5, 200, 200) float32
        odom_000000.npy      (30,) float32
        labels.npy           (N, 2) [v, omega]
        traj_poses.npy       (N, H, 3) [x, y, theta]
        metadata.npz

Usage:
    python -m wheelchair_e2e.scripts.generate_synthetic_data \
        --output_dir /tmp/kinoflow_synthetic --n_episodes 100
"""

import os
import argparse
import numpy as np
from collections import deque


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate synthetic KinoFlow training data')
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--n_episodes', type=int, default=100,
                        help='Number of navigation episodes')
    parser.add_argument('--steps_per_episode', type=int, default=200,
                        help='Max steps per episode (at 10Hz = 20s)')
    parser.add_argument('--grid_size', type=int, default=200)
    parser.add_argument('--resolution', type=float, default=0.05,
                        help='Meters per pixel')
    parser.add_argument('--v_max', type=float, default=0.25)
    parser.add_argument('--w_max', type=float, default=1.0)
    parser.add_argument('--dt', type=float, default=0.1,
                        help='Time step (s)')
    parser.add_argument('--horizon', type=int, default=10)
    parser.add_argument('--rate', type=float, default=10.0)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


class Environment:
    """Simple 2D environment with circular and rectangular obstacles."""

    def __init__(self, world_size=10.0, n_circles=5, n_rects=3, rng=None):
        self.world_size = world_size
        self.rng = rng or np.random.default_rng()
        self.circles = []  # (cx, cy, radius)
        self.rects = []    # (x_min, y_min, x_max, y_max)

        # Add boundary walls
        w = world_size / 2
        wall_t = 0.1
        self.rects.extend([
            (-w, -w, w, -w + wall_t),      # bottom
            (-w, w - wall_t, w, w),          # top
            (-w, -w, -w + wall_t, w),        # left
            (w - wall_t, -w, w, w),          # right
        ])

        # Random circular obstacles
        for _ in range(n_circles):
            r = self.rng.uniform(0.2, 0.8)
            cx = self.rng.uniform(-w + 1.5, w - 1.5)
            cy = self.rng.uniform(-w + 1.5, w - 1.5)
            self.circles.append((cx, cy, r))

        # Random rectangular obstacles
        for _ in range(n_rects):
            w_obs = self.rng.uniform(0.3, 1.5)
            h_obs = self.rng.uniform(0.3, 1.5)
            cx = self.rng.uniform(-w + 2.0, w - 2.0)
            cy = self.rng.uniform(-w + 2.0, w - 2.0)
            self.rects.append((
                cx - w_obs / 2, cy - h_obs / 2,
                cx + w_obs / 2, cy + h_obs / 2))

    def is_occupied(self, x, y):
        """Check if a point is inside any obstacle."""
        for cx, cy, r in self.circles:
            if (x - cx)**2 + (y - cy)**2 < r**2:
                return True
        for x_min, y_min, x_max, y_max in self.rects:
            if x_min <= x <= x_max and y_min <= y <= y_max:
                return True
        return False

    def is_free(self, x, y, margin=0.3):
        """Check if a point is free with safety margin."""
        for cx, cy, r in self.circles:
            if (x - cx)**2 + (y - cy)**2 < (r + margin)**2:
                return False
        for x_min, y_min, x_max, y_max in self.rects:
            if (x_min - margin <= x <= x_max + margin and
                    y_min - margin <= y <= y_max + margin):
                return False
        return True

    def sample_free_point(self):
        """Sample a random collision-free point."""
        w = self.world_size / 2
        for _ in range(1000):
            x = self.rng.uniform(-w + 0.5, w - 0.5)
            y = self.rng.uniform(-w + 0.5, w - 0.5)
            if self.is_free(x, y, margin=0.4):
                return x, y
        return 0.0, 0.0

    def render_occupancy(self, robot_x, robot_y, robot_theta,
                         grid_size, resolution):
        """Render occupancy grid from robot's perspective (BEV Ch0)."""
        grid = np.zeros((grid_size, grid_size), dtype=np.float32)
        center = grid_size // 2
        cos_t = np.cos(-robot_theta)
        sin_t = np.sin(-robot_theta)

        # Simulate 360-degree laser scan
        n_rays = 720
        max_range = 5.0
        for i in range(n_rays):
            angle = robot_theta + (2 * np.pi * i / n_rays) - np.pi
            for d_idx in range(int(max_range / resolution)):
                d = d_idx * resolution
                wx = robot_x + d * np.cos(angle)
                wy = robot_y + d * np.sin(angle)

                if self.is_occupied(wx, wy):
                    # Transform to robot frame
                    dx = wx - robot_x
                    dy = wy - robot_y
                    rx = cos_t * dx - sin_t * dy
                    ry = sin_t * dx + cos_t * dy

                    gx = int(center + rx / resolution)
                    gy = int(center - ry / resolution)
                    if 0 <= gx < grid_size and 0 <= gy < grid_size:
                        grid[gy, gx] = 1.0
                    break

        return grid

    def min_obstacle_dist(self, x, y):
        """Minimum distance to any obstacle."""
        min_d = float('inf')
        for cx, cy, r in self.circles:
            d = np.sqrt((x - cx)**2 + (y - cy)**2) - r
            min_d = min(min_d, d)
        for x_min, y_min, x_max, y_max in self.rects:
            # Distance to rectangle
            dx = max(x_min - x, 0, x - x_max)
            dy = max(y_min - y, 0, y - y_max)
            d = np.sqrt(dx**2 + dy**2)
            if dx == 0 and dy == 0:
                d = -min(x - x_min, x_max - x, y - y_min, y_max - y)
            min_d = min(min_d, d)
        return min_d


def pure_pursuit(robot_x, robot_y, robot_theta, goal_x, goal_y,
                 v_max, w_max, lookahead=1.0):
    """Simple pure pursuit controller → (v, omega)."""
    dx = goal_x - robot_x
    dy = goal_y - robot_y
    dist = np.sqrt(dx**2 + dy**2)

    if dist < 0.1:
        return 0.0, 0.0

    # Goal angle in robot frame
    goal_angle = np.arctan2(dy, dx) - robot_theta
    goal_angle = (goal_angle + np.pi) % (2 * np.pi) - np.pi

    # Angular velocity proportional to heading error
    kp_w = 2.0
    omega = np.clip(kp_w * goal_angle, -w_max, w_max)

    # Linear velocity: slow down when turning or near goal
    v = v_max * max(0.0, 1.0 - abs(goal_angle) / np.pi)
    v = min(v, v_max * min(dist / lookahead, 1.0))

    # Add slight noise for diversity
    v += np.random.normal(0, 0.01)
    omega += np.random.normal(0, 0.02)

    return float(np.clip(v, 0.0, v_max)), float(np.clip(omega, -w_max, w_max))


def render_bev(env, robot_x, robot_y, robot_theta,
               goal_dx, goal_dy, odom_trail,
               grid_size, resolution, prev_occ=None):
    """Render a full 5-channel BEV grid."""
    center = grid_size // 2
    bev = np.zeros((5, grid_size, grid_size), dtype=np.float32)

    # Ch0: Occupancy from "laser scan"
    bev[0] = env.render_occupancy(robot_x, robot_y, robot_theta,
                                  grid_size, resolution)

    # Ch1: Temporal delta (moving obstacles — zero for static env)
    if prev_occ is not None:
        bev[1] = np.abs(bev[0] - prev_occ)
    # In static environments this will be mostly zero
    # (small artifacts from viewpoint change simulate noise)

    # Ch2: Goal direction (Gaussian blob)
    gx = int(center + goal_dx / resolution)
    gy = int(center - goal_dy / resolution)
    if 0 <= gx < grid_size and 0 <= gy < grid_size:
        sigma = 10  # pixels
        yy, xx = np.mgrid[0:grid_size, 0:grid_size]
        gauss = np.exp(-((xx - gx)**2 + (yy - gy)**2) / (2 * sigma**2))
        bev[2] = gauss.astype(np.float32)

    # Ch3: Ego-motion trail
    for tx, ty in odom_trail:
        px = int(center + tx / resolution)
        py = int(center - ty / resolution)
        if 0 <= px < grid_size and 0 <= py < grid_size:
            bev[3, py, px] = 1.0

    # Ch4: Route (straight line to goal for synthetic data)
    n_pts = 50
    for i in range(n_pts):
        frac = i / n_pts
        rx = frac * goal_dx
        ry = frac * goal_dy
        px = int(center + rx / resolution)
        py = int(center - ry / resolution)
        if 0 <= px < grid_size and 0 <= py < grid_size:
            bev[4, py, px] = 1.0

    return bev


def run_episode(env, start_x, start_y, start_theta, goal_x, goal_y,
                max_steps, dt, v_max, w_max, grid_size, resolution):
    """Simulate one episode, return samples."""
    x, y, theta = start_x, start_y, start_theta
    odom_history = deque(maxlen=10)
    odom_trail = deque(maxlen=10)
    prev_occ = None

    samples = []  # (bev, odom_flat, v, omega, x, y, theta)

    for step in range(max_steps):
        # Check if goal reached
        dist_to_goal = np.sqrt((goal_x - x)**2 + (goal_y - y)**2)
        if dist_to_goal < 0.2:
            break

        # Check collision
        if env.is_occupied(x, y):
            break

        # Compute control
        v, omega = pure_pursuit(x, y, theta, goal_x, goal_y, v_max, w_max)

        # Obstacle avoidance: slow down near obstacles
        obs_dist = env.min_obstacle_dist(x, y)
        if obs_dist < 0.5:
            v *= max(0.1, (obs_dist - 0.15) / 0.35)

        # Compute goal in robot frame
        dx_w = goal_x - x
        dy_w = goal_y - y
        cos_t = np.cos(-theta)
        sin_t = np.sin(-theta)
        goal_dx = cos_t * dx_w - sin_t * dy_w
        goal_dy = sin_t * dx_w + cos_t * dy_w

        # Update odom history
        odom_history.append([v, omega, theta])
        odom_list = list(odom_history)
        while len(odom_list) < 10:
            odom_list.insert(0, [0.0, 0.0, 0.0])
        odom_flat = np.array(odom_list, dtype=np.float32).flatten()

        # Render BEV
        bev = render_bev(env, x, y, theta, goal_dx, goal_dy,
                         list(odom_trail), grid_size, resolution, prev_occ)
        prev_occ = bev[0].copy()

        samples.append((bev, odom_flat, v, omega, x, y, theta))

        # Update ego trail (relative positions)
        new_trail = deque(maxlen=10)
        for tx, ty in odom_trail:
            # Shift by ego-motion
            dx_step = v * np.cos(theta) * dt
            dy_step = v * np.sin(theta) * dt
            new_trail.append((
                cos_t * (tx - dx_step) - sin_t * (ty - dy_step),
                sin_t * (tx - dx_step) + cos_t * (ty - dy_step)))
        new_trail.append((0.0, 0.0))
        odom_trail = new_trail

        # Step dynamics
        theta += omega * dt
        theta = (theta + np.pi) % (2 * np.pi) - np.pi
        x += v * np.cos(theta) * dt
        y += v * np.sin(theta) * dt

    return samples


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    sample_idx = 0
    all_labels = []
    all_segment_ids = []
    all_timestamps = []
    all_goals = []

    print(f"Generating {args.n_episodes} episodes...")

    for ep in range(args.n_episodes):
        # Random environment
        env = Environment(
            world_size=10.0,
            n_circles=rng.integers(3, 8),
            n_rects=rng.integers(2, 5),
            rng=rng)

        # Random start and goal
        sx, sy = env.sample_free_point()
        gx, gy = env.sample_free_point()

        # Ensure start and goal are far enough apart
        attempts = 0
        while np.sqrt((gx - sx)**2 + (gy - sy)**2) < 2.0 and attempts < 20:
            gx, gy = env.sample_free_point()
            attempts += 1

        stheta = rng.uniform(-np.pi, np.pi)

        # Run episode
        samples = run_episode(
            env, sx, sy, stheta, gx, gy,
            max_steps=args.steps_per_episode,
            dt=args.dt, v_max=args.v_max, w_max=args.w_max,
            grid_size=args.grid_size, resolution=args.resolution)

        if len(samples) < args.horizon + 1:
            continue

        # Save samples
        for bev, odom_flat, v, omega, x, y, theta in samples:
            np.save(os.path.join(args.output_dir,
                                 f'bev_{sample_idx:06d}.npy'), bev)
            np.save(os.path.join(args.output_dir,
                                 f'odom_{sample_idx:06d}.npy'), odom_flat)
            all_labels.append([v, omega])
            all_segment_ids.append(ep)
            all_timestamps.append(int(sample_idx * 1e8))
            all_goals.append([gx, gy])
            sample_idx += 1

        if (ep + 1) % 10 == 0:
            print(f"  Episode {ep + 1}/{args.n_episodes}: "
                  f"{sample_idx} total samples")

    if sample_idx == 0:
        print("ERROR: No samples generated!")
        return

    # Save labels
    labels = np.array(all_labels, dtype=np.float32)
    np.save(os.path.join(args.output_dir, 'labels.npy'), labels)

    # Compute trajectory pose labels
    H = args.horizon
    dt = args.dt
    segment_ids = np.array(all_segment_ids)
    n = len(labels)
    traj_poses = np.zeros((n, H, 3), dtype=np.float32)

    print(f"\nComputing trajectory poses (H={H})...")
    for i in range(n):
        if i + H > n:
            break
        seg_slice = segment_ids[i:i + H]
        if not np.all(seg_slice == seg_slice[0]):
            continue

        x, y, theta = 0.0, 0.0, 0.0
        for t in range(H):
            v = labels[i + t, 0]
            omega = labels[i + t, 1]
            theta += omega * dt
            x += v * np.cos(theta) * dt
            y += v * np.sin(theta) * dt
            traj_poses[i, t] = [x, y, theta]

    np.save(os.path.join(args.output_dir, 'traj_poses.npy'), traj_poses)

    # Save metadata
    np.savez(os.path.join(args.output_dir, 'metadata.npz'),
             timestamps=np.array(all_timestamps),
             goal_positions=np.array(all_goals, dtype=np.float32),
             segment_ids=segment_ids,
             n_segments=args.n_episodes,
             rate=args.rate,
             n_samples=sample_idx,
             horizon=H,
             label_source='synthetic_pure_pursuit')

    print(f"\nDone! {sample_idx} samples from {args.n_episodes} episodes")
    print(f"  v range:  [{labels[:, 0].min():.3f}, "
          f"{labels[:, 0].max():.3f}] m/s")
    print(f"  w range:  [{labels[:, 1].min():.3f}, "
          f"{labels[:, 1].max():.3f}] rad/s")

    valid_poses = traj_poses[traj_poses[:, -1, 0] != 0]
    if len(valid_poses) > 0:
        max_dist = np.sqrt(
            valid_poses[:, -1, 0]**2 + valid_poses[:, -1, 1]**2)
        print(f"  Traj endpoint dist: [{max_dist.min():.3f}, "
              f"{max_dist.max():.3f}] m")
    print(f"  Output: {args.output_dir}")


if __name__ == '__main__':
    main()
