#!/usr/bin/env python3
"""
Calibrated Noise Injection for Real-Data Phase Diagram.

Maps real sensor noise to calibrated noise levels and injects systematic
degradation into rosbag scan data, then evaluates all planners.

Real sensor noise baselines (from RealSense/RPLidar specs):
  RPLidar S3 clean:   eta ~ 0.01   (sigma ~ 0.01m at 3m)
  D455 at 3m:         eta ~ 0.012  (sigma ~ 0.036m)
  D435i at 3m:        eta ~ 0.023  (sigma ~ 0.068m)
  Sensor dropout 50%: eta ~ 0.5+

Noise types:
  1. Range Gaussian:  additive N(0, sigma) on range values
  2. Dropout:         random ranges set to inf (sensor failure)
  3. Speckle:         multiplicative noise (distance-dependent)
  4. Camera failure:  only degrade camera-contributed ranges (>180deg)

Output: real-data phase diagram (the paper's main figure).

Usage:
    python quantum_noise_injection.py \
        --session /home/sidd/wheelchair_nav/maps/session_20260226_124315/rosbag \
        --output_dir /home/sidd/wheelchair_nav/quantum_eval_results/noise
"""

import os
import sys
import argparse
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from wheelchair_e2e.quantum_nav.quantum_trajectory_planner import (
    QuantumTrajectoryPlanner
)
from wheelchair_e2e.quantum_nav.quantum_planner_v2 import (
    CostEncodedQuantumPlanner, PhaseInterferenceQuantumPlanner,
    TemporalEntanglementPlanner
)
from wheelchair_e2e.quantum_nav.baselines import (
    ClassicalBestCost, KNNMeanScore, MPPIPlanner
)

# Import the scan reader from our eval pipeline
from quantum_rosbag_eval import read_bag, scan_to_points, get_yaw


# ── Noise Injection Functions ───────────────────────────────────────

def inject_gaussian_noise(scan_ranges, sigma):
    """Add Gaussian noise to range values.

    sigma in meters. Matches RPLidar/RealSense noise models.
    """
    noisy = np.array(scan_ranges, dtype=np.float32)
    noise = np.random.normal(0, sigma, len(noisy))
    noisy += noise
    noisy[noisy < 0] = 0.0
    return noisy


def inject_dropout(scan_ranges, dropout_fraction):
    """Set random fraction of ranges to inf (sensor failure).

    Simulates partial occlusion or sensor dropout.
    """
    noisy = np.array(scan_ranges, dtype=np.float32)
    n_drop = int(len(noisy) * dropout_fraction)
    if n_drop > 0:
        drop_idx = np.random.choice(len(noisy), n_drop, replace=False)
        noisy[drop_idx] = float('inf')
    return noisy


def inject_speckle_noise(scan_ranges, sigma_mult):
    """Multiplicative noise: range *= (1 + N(0, sigma_mult)).

    Models distance-dependent noise (error proportional to range).
    """
    noisy = np.array(scan_ranges, dtype=np.float32)
    noise = np.random.normal(1.0, sigma_mult, len(noisy))
    noisy *= noise
    noisy[noisy < 0] = 0.0
    return noisy


def inject_camera_failure(scan_ranges, angle_min, angle_max):
    """Degrade only camera-contributed ranges.

    RPLidar covers 360deg. RealSense cameras contribute ranges in specific
    angular sectors. Camera failure = those sectors go to inf.

    Camera sectors (approximate, based on 3-camera setup):
      Front D455:  -30deg to +30deg (indices depend on scan resolution)
      Left D455:   +60deg to +120deg
      Right D435i: -120deg to -60deg
    """
    noisy = np.array(scan_ranges, dtype=np.float32)
    n = len(noisy)
    angles = np.linspace(angle_min, angle_max, n)

    # Camera sectors (radians)
    camera_sectors = [
        (-np.deg2rad(30), np.deg2rad(30)),    # front
        (np.deg2rad(60), np.deg2rad(120)),     # left
        (-np.deg2rad(120), -np.deg2rad(60)),   # right
    ]

    for a_min, a_max in camera_sectors:
        mask = (angles >= a_min) & (angles <= a_max)
        noisy[mask] = float('inf')

    return noisy


def estimate_eta_from_sigma(sigma_m, mean_range=3.0):
    """Convert noise sigma in meters to dimensionless eta.

    eta = sigma / mean_range (relative noise level)
    """
    return sigma_m / mean_range


# ── Evaluation Under Noise ──────────────────────────────────────────

def evaluate_under_noise(messages, output_dir, max_timesteps=500):
    """Evaluate all planners on real data with injected noise.

    Generates the real-data phase diagram.
    """
    # Extract clean scans and odom
    scan_topic = '/scan_fused'
    if not messages.get(scan_topic):
        scan_topic = '/scan_filtered'
    scans = messages[scan_topic]
    odoms = messages['/odometry/filtered']

    odom_times = np.array([t for t, _ in odoms])
    scan_times = np.array([t for t, _ in scans])

    t_start = max(odom_times[0], scan_times[0])
    t_end = min(odom_times[-1], scan_times[-1])
    dt_ns = int(1e9 / 10.0)  # 10Hz

    # Auto-detect goal
    final_odom = odoms[-1][1]
    goal_x = final_odom.pose.pose.position.x
    goal_y = final_odom.pose.pose.position.y

    # Build planners
    planners = {
        'quantum_sup_3q': QuantumTrajectoryPlanner(
            n_candidates=64, nn=3, use_superposition=True),
        'quantum_cost_encoded': CostEncodedQuantumPlanner(
            n_candidates=64, nn=3, temperature=1.0),
        'quantum_phase': PhaseInterferenceQuantumPlanner(
            n_candidates=64, nn=3),
        'quantum_temporal': TemporalEntanglementPlanner(
            n_candidates=64, nn=3, alpha=0.7),
        'knn_mean': KNNMeanScore(n_candidates=64, nn=3),
        'mppi': MPPIPlanner(n_candidates=64, lambda_=1.0),
        'classical_best': ClassicalBestCost(n_candidates=64),
    }

    # Noise configurations
    noise_configs = {
        'gaussian': {
            'sigmas': [0.01, 0.03, 0.05, 0.1, 0.2, 0.5, 1.0],
            'func': inject_gaussian_noise,
            'param_name': 'sigma (m)',
        },
        'dropout': {
            'sigmas': [0.0, 0.1, 0.2, 0.3, 0.5],
            'func': inject_dropout,
            'param_name': 'dropout fraction',
        },
        'speckle': {
            'sigmas': [0.01, 0.05, 0.1],
            'func': inject_speckle_noise,
            'param_name': 'sigma_mult',
        },
    }

    # Collect timesteps
    print(f"  Extracting clean timesteps...")
    clean_scans = []
    clean_odoms = []
    scan_idx = 0
    count = 0

    for t in range(int(t_start), int(t_end), dt_ns):
        while scan_idx < len(scans) - 1 and scans[scan_idx + 1][0] <= t:
            scan_idx += 1

        odom_idx = int(np.searchsorted(odom_times, t))
        odom_idx = min(odom_idx, len(odoms) - 1)
        odom_msg = odoms[odom_idx][1]

        v = odom_msg.twist.twist.linear.x
        w = odom_msg.twist.twist.angular.z
        if abs(v) < 0.01 and abs(w) < 0.01:
            continue

        clean_scans.append(scans[scan_idx][1])
        clean_odoms.append(odom_msg)
        count += 1
        if count >= max_timesteps:
            break

    n_ts = len(clean_scans)
    print(f"  {n_ts} clean timesteps extracted")

    # Phase diagram data: noise_type -> planner -> (n_levels, n_ts) phi values
    phase_data = {}

    for noise_type, cfg in noise_configs.items():
        print(f"\n  Noise type: {noise_type}")
        levels = cfg['sigmas']
        func = cfg['func']

        planner_phi = {name: np.zeros((len(levels), n_ts))
                       for name in planners}

        for li, level in enumerate(levels):
            print(f"    Level {level} ({cfg['param_name']})...", end='', flush=True)
            t0 = time.time()

            for ti in range(n_ts):
                scan_msg = clean_scans[ti]
                odom_msg = clean_odoms[ti]

                # Inject noise
                if noise_type == 'camera_failure' and level > 0:
                    noisy_ranges = inject_camera_failure(
                        scan_msg.ranges, scan_msg.angle_min, scan_msg.angle_max)
                elif noise_type == 'dropout':
                    noisy_ranges = func(scan_msg.ranges, level)
                else:
                    noisy_ranges = func(scan_msg.ranges, level)

                # Convert to obstacle points
                n = len(noisy_ranges)
                angles = np.linspace(scan_msg.angle_min, scan_msg.angle_max, n)
                valid = ((noisy_ranges > 0.1) & (noisy_ranges < 5.0)
                         & np.isfinite(noisy_ranges))
                if np.any(valid):
                    x = noisy_ranges[valid] * np.cos(angles[valid])
                    y = noisy_ranges[valid] * np.sin(angles[valid])
                    obs_pts = np.column_stack([x, y])
                else:
                    obs_pts = np.zeros((0, 2))

                # Robot-frame goal
                rx = odom_msg.pose.pose.position.x
                ry = odom_msg.pose.pose.position.y
                rtheta = get_yaw(odom_msg.pose.pose.orientation)
                dx = goal_x - rx
                dy = goal_y - ry
                cos_t = np.cos(-rtheta)
                sin_t = np.sin(-rtheta)
                gx = cos_t * dx - sin_t * dy
                gy = sin_t * dx + cos_t * dy

                # Run each planner
                for name, planner in planners.items():
                    if isinstance(planner, QuantumTrajectoryPlanner):
                        eta = estimate_eta_from_sigma(
                            level if noise_type != 'dropout' else level * 0.5)
                        decision = planner.select_trajectory(
                            0, 0, 0, gx, gy, obs_pts, sensor_noise=max(eta, 0.01))
                        planner_phi[name][li, ti] = decision.confidence
                    else:
                        result = planner.plan(0, 0, 0, gx, gy, obs_pts)
                        planner_phi[name][li, ti] = result['confidence']

            elapsed = time.time() - t0
            print(f" {elapsed:.1f}s")

        phase_data[noise_type] = {
            'levels': levels,
            'param_name': cfg['param_name'],
            'phi': {name: planner_phi[name] for name in planners},
        }

    # Save phase diagram data
    for noise_type, data in phase_data.items():
        np.savez(
            os.path.join(output_dir, f'phase_data_{noise_type}.npz'),
            levels=np.array(data['levels']),
            param_name=data['param_name'],
            planner_names=list(data['phi'].keys()),
            **{f'phi_{name}': arr for name, arr in data['phi'].items()},
        )

    # Print phase diagram table
    print(f"\n{'='*70}")
    print("  REAL-DATA PHASE DIAGRAM (Gaussian noise)")
    print(f"{'='*70}")

    if 'gaussian' in phase_data:
        gd = phase_data['gaussian']
        levels = gd['levels']
        print(f"\n  {'Planner':<30}", end="")
        for sigma in levels:
            print(f" {sigma:>6.2f}m", end="")
        print()
        print(f"  {'-'*30}", end="")
        for _ in levels:
            print(f" {'-'*7}", end="")
        print()

        for name in planners:
            phi_mean = np.mean(gd['phi'][name], axis=1)
            print(f"  {name:<30}", end="")
            for pm in phi_mean:
                print(f" {pm:>6.3f} ", end="")
            print()

    # Compute eta mapping
    print(f"\n  Noise level mapping (sigma -> eta):")
    for sigma in [0.01, 0.03, 0.05, 0.1, 0.2, 0.5, 1.0]:
        eta = estimate_eta_from_sigma(sigma)
        sensor = "RPLidar" if sigma < 0.02 else (
            "D455@3m" if sigma < 0.05 else (
                "D435i@3m" if sigma < 0.1 else "degraded"))
        print(f"    sigma={sigma:.3f}m -> eta={eta:.4f} (~{sensor})")

    print(f"\nResults saved to {output_dir}/")
    return phase_data


def main():
    parser = argparse.ArgumentParser(
        description='Noise injection experiments for quantum nav')
    parser.add_argument('--session', type=str,
                        default='/home/sidd/wheelchair_nav/maps/session_20260226_124315/rosbag')
    parser.add_argument('--output_dir', type=str,
                        default='/home/sidd/wheelchair_nav/quantum_eval_results/noise')
    parser.add_argument('--max_timesteps', type=int, default=500,
                        help='Max timesteps to process (for speed)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Reading bag: {args.session}")
    messages = read_bag(args.session)

    evaluate_under_noise(messages, args.output_dir, args.max_timesteps)


if __name__ == '__main__':
    main()
