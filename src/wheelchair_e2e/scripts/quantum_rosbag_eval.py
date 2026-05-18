#!/usr/bin/env python3
"""
Offline Rosbag Evaluation for Quantum-Inspired Trajectory Planner.

Reads real sensor data from rosbag sessions and runs all planners
(quantum + baselines) on each timestep, recording scores and metrics
for the GO/NO-GO ablation decision.

Reuses:
  - read_bag() from bag_to_bev_velocity.py (MCAP format handling)
  - LaserScan->Cartesian from bev_generator.py

Pipeline:
  1. Read rosbag -> extract (obstacle_points, robot_pose, goal) at 10Hz
  2. For each timestep, run all planners
  3. Record (v, omega, phi, mode, clearance, goal_progress, time_ms, scores)
  4. Save results for ablation analysis

Usage:
    python quantum_rosbag_eval.py \
        --session /home/sidd/wheelchair_nav/maps/session_20260226_124315/rosbag \
        --output_dir /home/sidd/wheelchair_nav/quantum_eval_results

    # Process all sessions:
    python quantum_rosbag_eval.py --all_sessions --output_dir quantum_eval_results
"""

import os
import sys
import argparse
import time
import glob
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Optional

# ROS message types (available even without running ROS)
try:
    from sensor_msgs.msg import LaserScan
    from nav_msgs.msg import Odometry
    from geometry_msgs.msg import PoseStamped, Twist
except ImportError:
    print("ERROR: ROS2 message types not found. Source /opt/ros/jazzy/setup.bash")
    sys.exit(1)

# Add parent paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from wheelchair_e2e.quantum_nav.quantum_trajectory_planner import (
    QuantumTrajectoryPlanner, AdaptiveQuantumPlanner, QuantumDecision
)
from wheelchair_e2e.quantum_nav.baselines import (
    ClassicalBestCost, KNNMeanScore, WeightedKNN,
    MPPIPlanner, IdentityScoring, RPPGroundTruth, BasePlanner
)
from wheelchair_e2e.quantum_nav.quantum_planner_v2 import (
    CostEncodedQuantumPlanner, PhaseInterferenceQuantumPlanner,
    TemporalEntanglementPlanner
)
from wheelchair_e2e.quantum_nav.metrics import (
    compute_all_metrics, spearman_rank_correlation
)


# ── Rosbag Reading (adapted from bag_to_bev_velocity.py) ────────────

def read_bag(bag_path):
    """Read scan, odom, goal, and cmd_vel from a ROS2 bag.

    Supports both rosbag2 directories (with metadata.yaml) and raw MCAP.
    Returns dict of topic -> [(timestamp_ns, message), ...]
    """
    topic_types = {
        '/scan_fused': LaserScan,
        '/scan_filtered': LaserScan,
        '/odometry/filtered': Odometry,
        '/goal_pose': PoseStamped,
        '/cmd_vel': Twist,
    }
    messages = {t: [] for t in topic_types}

    metadata_path = os.path.join(bag_path, 'metadata.yaml')
    has_metadata = os.path.isfile(metadata_path)

    if has_metadata:
        from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
        from rclpy.serialization import deserialize_message

        mcap_files = glob.glob(os.path.join(bag_path, '*.mcap'))
        storage_id = 'mcap' if mcap_files else 'sqlite3'
        print(f"  Using rosbag2 reader (storage: {storage_id})")

        reader = SequentialReader()
        storage_options = StorageOptions(uri=bag_path, storage_id=storage_id)
        converter_options = ConverterOptions(
            input_serialization_format='cdr',
            output_serialization_format='cdr')
        reader.open(storage_options, converter_options)

        while reader.has_next():
            topic, data, timestamp = reader.read_next()
            if topic in topic_types:
                msg = deserialize_message(data, topic_types[topic])
                messages[topic].append((timestamp, msg))
    else:
        from mcap.reader import make_reader
        from mcap_ros2.decoder import DecoderFactory

        mcap_files = sorted(glob.glob(os.path.join(bag_path, '*.mcap')))
        if not mcap_files:
            raise RuntimeError(f"No .mcap files found in {bag_path}")

        print(f"  Using direct MCAP reader ({len(mcap_files)} files)")
        decoder = DecoderFactory()

        for mcap_file in mcap_files:
            fname = os.path.basename(mcap_file)
            try:
                with open(mcap_file, 'rb') as f:
                    reader = make_reader(f, decoder_factories=[decoder])
                    count = 0
                    for schema, channel, message, decoded_msg in \
                            reader.iter_decoded_messages(
                                topics=list(topic_types.keys())):
                        if channel.topic in topic_types:
                            messages[channel.topic].append(
                                (message.log_time, decoded_msg))
                            count += 1
                    print(f"    {fname}: {count} msgs")
            except Exception as e:
                print(f"    {fname}: SKIPPED ({e})")

        for topic in messages:
            messages[topic].sort(key=lambda x: x[0])

    return messages


# ── LaserScan -> Obstacle Points (from bev_generator.py logic) ──────

def scan_to_points(scan_msg, range_min=0.1, range_max=5.0):
    """Convert LaserScan to (M, 2) obstacle point cloud in robot frame.

    Adapted from bev_generator.py:51-79 but returns raw Cartesian points
    instead of a BEV grid.
    """
    ranges = np.array(scan_msg.ranges, dtype=np.float32)
    n = len(ranges)
    angles = np.linspace(scan_msg.angle_min, scan_msg.angle_max, n)

    valid = (ranges > range_min) & (ranges < range_max) & np.isfinite(ranges)

    if not np.any(valid):
        return np.zeros((0, 2))

    x = ranges[valid] * np.cos(angles[valid])
    y = ranges[valid] * np.sin(angles[valid])

    return np.column_stack([x, y])


def get_yaw(orientation):
    """Extract yaw from quaternion."""
    q = orientation
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return np.arctan2(siny, cosy)


# ── Extract timesteps from rosbag ───────────────────────────────────

@dataclass
class Timestep:
    """One evaluation timestep extracted from rosbag data."""
    t_ns: int
    obstacle_points: np.ndarray   # (M, 2) in robot frame
    robot_x: float
    robot_y: float
    robot_theta: float
    goal_x: float
    goal_y: float
    v_actual: float               # from EKF odom (ground truth executed)
    omega_actual: float
    cmd_v: float = 0.0            # from /cmd_vel (Nav2 commanded)
    cmd_omega: float = 0.0
    clearance_actual: float = 0.0 # min obstacle distance


def extract_timesteps(messages, rate_hz=10.0, goal_x=None, goal_y=None):
    """Extract evaluation timesteps at fixed rate from rosbag data.

    Returns list of Timestep objects.
    """
    # Pick scan topic
    scan_topic = '/scan_fused'
    if not messages.get(scan_topic):
        scan_topic = '/scan_filtered'
    if not messages.get(scan_topic):
        raise RuntimeError("No scan topic found in bag")

    scans = messages[scan_topic]
    odoms = messages['/odometry/filtered']
    goals = messages.get('/goal_pose', [])
    cmd_vels = messages.get('/cmd_vel', [])

    if not scans or not odoms:
        raise RuntimeError(f"Missing data: {len(scans)} scans, {len(odoms)} odoms")

    print(f"  Scans ({scan_topic}): {len(scans)}")
    print(f"  Odom: {len(odoms)}")
    print(f"  Goals: {len(goals)}")
    print(f"  CmdVel: {len(cmd_vels)}")

    # Time indices
    odom_times = np.array([t for t, _ in odoms])
    scan_times = np.array([t for t, _ in scans])
    cmd_times = np.array([t for t, _ in cmd_vels]) if cmd_vels else np.array([])

    # Use goal_pose messages or manual goal
    current_goal_x = goal_x if goal_x is not None else 3.0
    current_goal_y = goal_y if goal_y is not None else 0.0

    if goals:
        current_goal_x = goals[-1][1].pose.position.x
        current_goal_y = goals[-1][1].pose.position.y
        print(f"  Goal from bag: ({current_goal_x:.2f}, {current_goal_y:.2f})")
    else:
        # Auto-detect goal as final odom position
        final_odom = odoms[-1][1]
        current_goal_x = final_odom.pose.pose.position.x
        current_goal_y = final_odom.pose.pose.position.y
        print(f"  Goal (auto=final pos): ({current_goal_x:.2f}, {current_goal_y:.2f})")

    # Sample at fixed rate
    dt_ns = int(1e9 / rate_hz)
    t_start = max(odom_times[0], scan_times[0])
    t_end = min(odom_times[-1], scan_times[-1])

    timesteps = []
    scan_idx = 0

    for t in range(int(t_start), int(t_end), dt_ns):
        # Find nearest scan
        while scan_idx < len(scans) - 1 and scans[scan_idx + 1][0] <= t:
            scan_idx += 1
        scan_msg = scans[scan_idx][1]

        # Find nearest odom
        odom_idx = int(np.searchsorted(odom_times, t))
        odom_idx = min(odom_idx, len(odoms) - 1)
        odom_msg = odoms[odom_idx][1]

        # Extract robot pose
        rx = odom_msg.pose.pose.position.x
        ry = odom_msg.pose.pose.position.y
        rtheta = get_yaw(odom_msg.pose.pose.orientation)

        # Executed velocity from EKF
        v_actual = odom_msg.twist.twist.linear.x
        omega_actual = odom_msg.twist.twist.angular.z

        # Skip completely stationary
        if abs(v_actual) < 0.01 and abs(omega_actual) < 0.01:
            continue

        # Obstacle points in robot frame
        obs_pts = scan_to_points(scan_msg)

        # Min clearance
        if len(obs_pts) > 0:
            dists = np.linalg.norm(obs_pts, axis=1)
            clearance = float(np.min(dists))
        else:
            clearance = float('inf')

        # Commanded velocity (from Nav2)
        cmd_v, cmd_omega = 0.0, 0.0
        if len(cmd_times) > 0:
            ci = int(np.searchsorted(cmd_times, t))
            ci = min(ci, len(cmd_vels) - 1)
            cmd_msg = cmd_vels[ci][1]
            cmd_v = cmd_msg.linear.x
            cmd_omega = cmd_msg.angular.z

        # Update goal if there's a newer /goal_pose
        if goals:
            for gt, gp in goals:
                if gt <= t:
                    current_goal_x = gp.pose.position.x
                    current_goal_y = gp.pose.position.y

        # Transform goal to robot frame for planners
        dx = current_goal_x - rx
        dy = current_goal_y - ry
        cos_t = np.cos(-rtheta)
        sin_t = np.sin(-rtheta)
        goal_rx = cos_t * dx - sin_t * dy
        goal_ry = sin_t * dx + cos_t * dy

        ts = Timestep(
            t_ns=t,
            obstacle_points=obs_pts,
            robot_x=0.0,    # planners work in robot frame
            robot_y=0.0,
            robot_theta=0.0,
            goal_x=goal_rx,
            goal_y=goal_ry,
            v_actual=v_actual,
            omega_actual=omega_actual,
            cmd_v=cmd_v,
            cmd_omega=cmd_omega,
            clearance_actual=clearance,
        )
        timesteps.append(ts)

    return timesteps


# ── Planner Registry ────────────────────────────────────────────────

@dataclass
class PlannerResult:
    """Result from running one planner on one timestep."""
    planner_name: str
    v: float
    omega: float
    confidence: float
    mode: str
    time_ms: float
    scores: np.ndarray
    clearance: float = 0.0
    goal_progress: float = 0.0


def build_planners():
    """Create all planners for the evaluation battery."""
    planners = {}

    # Quantum planners (current formulation)
    planners['quantum_sup_3q'] = QuantumTrajectoryPlanner(
        n_candidates=64, nn=3, use_superposition=True)
    planners['quantum_sup_2q'] = QuantumTrajectoryPlanner(
        n_candidates=64, nn=2, use_superposition=True)
    planners['quantum_random_3q'] = QuantumTrajectoryPlanner(
        n_candidates=64, nn=3, use_superposition=False)
    planners['quantum_adaptive'] = AdaptiveQuantumPlanner(
        n_candidates=64, nn=3, use_superposition=True)

    # Modified quantum planners (v2 — genuinely non-trivial)
    planners['quantum_cost_encoded'] = CostEncodedQuantumPlanner(
        n_candidates=64, nn=3, temperature=1.0)
    planners['quantum_phase_interference'] = PhaseInterferenceQuantumPlanner(
        n_candidates=64, nn=3)
    planners['quantum_temporal'] = TemporalEntanglementPlanner(
        n_candidates=64, nn=3, alpha=0.7)

    # Classical baselines
    planners['classical_best_cost'] = ClassicalBestCost(n_candidates=64)
    planners['knn_mean'] = KNNMeanScore(n_candidates=64, nn=3)
    planners['knn_weighted'] = WeightedKNN(n_candidates=64, nn=3)
    planners['mppi'] = MPPIPlanner(n_candidates=64, lambda_=1.0)
    planners['identity_scoring'] = IdentityScoring(n_candidates=64)

    return planners


def run_planner(planner, name, timestep):
    """Run a single planner on a single timestep, return PlannerResult."""
    t0 = time.time()

    if isinstance(planner, QuantumTrajectoryPlanner):
        # All quantum variants use the same interface
        if isinstance(planner, AdaptiveQuantumPlanner):
            decision = planner.select_trajectory_adaptive(
                timestep.robot_x, timestep.robot_y, timestep.robot_theta,
                timestep.goal_x, timestep.goal_y,
                timestep.obstacle_points)
        else:
            decision = planner.select_trajectory(
                timestep.robot_x, timestep.robot_y, timestep.robot_theta,
                timestep.goal_x, timestep.goal_y,
                timestep.obstacle_points,
                sensor_noise=0.1)
        elapsed = (time.time() - t0) * 1000
        return PlannerResult(
            planner_name=name,
            v=decision.v, omega=decision.omega,
            confidence=decision.confidence,
            mode=decision.mode,
            time_ms=elapsed,
            scores=decision.quantum_scores,
        )
    elif isinstance(planner, BasePlanner):
        decision = planner.plan(
            timestep.robot_x, timestep.robot_y, timestep.robot_theta,
            timestep.goal_x, timestep.goal_y,
            timestep.obstacle_points)
        elapsed = (time.time() - t0) * 1000
        return PlannerResult(
            planner_name=name,
            v=decision['v'], omega=decision['omega'],
            confidence=decision['confidence'],
            mode=decision['mode'],
            time_ms=elapsed,
            scores=decision['scores'],
        )
    else:
        raise ValueError(f"Unknown planner type: {type(planner)}")


# ── Main Evaluation ─────────────────────────────────────────────────

def evaluate_session(bag_path, planners, output_dir, max_timesteps=None):
    """Evaluate all planners on a single rosbag session."""
    session_name = os.path.basename(os.path.dirname(bag_path))
    print(f"\n{'='*60}")
    print(f"Session: {session_name}")
    print(f"Bag: {bag_path}")
    print(f"{'='*60}")

    # Read bag
    messages = read_bag(bag_path)

    # Extract timesteps
    timesteps = extract_timesteps(messages, rate_hz=10.0)
    print(f"  Extracted {len(timesteps)} timesteps")

    if max_timesteps:
        timesteps = timesteps[:max_timesteps]
        print(f"  Limited to {len(timesteps)} timesteps")

    if not timesteps:
        print("  WARNING: No timesteps extracted, skipping session")
        return None

    # Prepare result storage
    n_ts = len(timesteps)
    planner_names = list(planners.keys())
    n_planners = len(planner_names)

    results = {
        'session': session_name,
        'n_timesteps': n_ts,
        'planner_names': planner_names,
        # Per-timestep ground truth
        'gt_v': np.zeros(n_ts),
        'gt_omega': np.zeros(n_ts),
        'gt_clearance': np.zeros(n_ts),
        'timestamps_ns': np.zeros(n_ts, dtype=np.int64),
        'goal_dist': np.zeros(n_ts),
        # Per-planner per-timestep results
        'v': np.zeros((n_planners, n_ts)),
        'omega': np.zeros((n_planners, n_ts)),
        'confidence': np.zeros((n_planners, n_ts)),
        'mode': np.empty((n_planners, n_ts), dtype='U8'),
        'time_ms': np.zeros((n_planners, n_ts)),
    }

    # Score arrays — variable length per planner, store as list
    all_scores = {name: [] for name in planner_names}

    # Run evaluation
    print(f"\n  Running {n_planners} planners on {n_ts} timesteps...")
    t_eval_start = time.time()

    for ti, ts in enumerate(timesteps):
        # Ground truth
        results['gt_v'][ti] = ts.v_actual
        results['gt_omega'][ti] = ts.omega_actual
        results['gt_clearance'][ti] = ts.clearance_actual
        results['timestamps_ns'][ti] = ts.t_ns
        results['goal_dist'][ti] = np.sqrt(ts.goal_x**2 + ts.goal_y**2)

        for pi, name in enumerate(planner_names):
            try:
                pr = run_planner(planners[name], name, ts)
                results['v'][pi, ti] = pr.v
                results['omega'][pi, ti] = pr.omega
                results['confidence'][pi, ti] = pr.confidence
                results['mode'][pi, ti] = pr.mode
                results['time_ms'][pi, ti] = pr.time_ms
                all_scores[name].append(pr.scores)
            except Exception as e:
                if ti == 0:
                    print(f"    WARNING: {name} failed: {e}")
                results['v'][pi, ti] = 0.0
                results['omega'][pi, ti] = 0.0
                results['confidence'][pi, ti] = 0.0
                results['mode'][pi, ti] = 'error'
                results['time_ms'][pi, ti] = 0.0
                all_scores[name].append(np.zeros(64))

        if (ti + 1) % 200 == 0:
            elapsed = time.time() - t_eval_start
            rate = (ti + 1) / elapsed
            eta = (n_ts - ti - 1) / rate
            print(f"    {ti+1}/{n_ts} ({rate:.1f} ts/s, ETA {eta:.0f}s)")

    total_time = time.time() - t_eval_start
    print(f"  Done in {total_time:.1f}s "
          f"({n_ts * n_planners / total_time:.0f} planner-evals/s)")

    # Save results
    session_dir = os.path.join(output_dir, session_name)
    os.makedirs(session_dir, exist_ok=True)

    np.savez(os.path.join(session_dir, 'eval_results.npz'),
             **{k: v for k, v in results.items()
                if isinstance(v, np.ndarray)},
             planner_names=np.array(planner_names),
             session=session_name)

    # Save scores separately (variable length)
    for name in planner_names:
        scores_arr = np.array(all_scores[name])
        np.save(os.path.join(session_dir, f'scores_{name}.npy'), scores_arr)

    print(f"  Results saved to {session_dir}/")

    # Print summary
    print(f"\n  {'Planner':<30} {'Mean phi':>9} {'Mean ms':>9} {'Exploit%':>9}")
    print(f"  {'-'*60}")
    for pi, name in enumerate(planner_names):
        mean_phi = np.mean(results['confidence'][pi])
        mean_ms = np.mean(results['time_ms'][pi])
        exploit_pct = np.mean(results['mode'][pi] == 'exploit') * 100
        print(f"  {name:<30} {mean_phi:9.4f} {mean_ms:8.2f}ms {exploit_pct:8.1f}%")

    return results


def find_sessions(base_dir='/home/sidd/wheelchair_nav/maps'):
    """Find all rosbag session directories."""
    sessions = sorted(glob.glob(os.path.join(base_dir, 'session_*/rosbag')))
    return sessions


def main():
    parser = argparse.ArgumentParser(
        description='Offline quantum nav evaluation on rosbag data')
    parser.add_argument('--session', type=str, default=None,
                        help='Path to specific rosbag directory')
    parser.add_argument('--all_sessions', action='store_true',
                        help='Process all available sessions')
    parser.add_argument('--output_dir', type=str,
                        default='/home/sidd/wheelchair_nav/quantum_eval_results',
                        help='Output directory for results')
    parser.add_argument('--max_timesteps', type=int, default=None,
                        help='Max timesteps per session (for testing)')
    parser.add_argument('--rate', type=float, default=10.0,
                        help='Evaluation rate in Hz')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Build all planners
    print("Building planners...")
    planners = build_planners()
    print(f"  {len(planners)} planners ready: {list(planners.keys())}")

    # Find sessions
    if args.session:
        sessions = [args.session]
    elif args.all_sessions:
        sessions = find_sessions()
        print(f"\nFound {len(sessions)} sessions")
    else:
        # Default: largest session
        sessions = [
            '/home/sidd/wheelchair_nav/maps/session_20260226_124315/rosbag'
        ]

    # Evaluate
    all_results = []
    for bag_path in sessions:
        if not os.path.isdir(bag_path):
            print(f"WARNING: {bag_path} not found, skipping")
            continue
        result = evaluate_session(
            bag_path, planners, args.output_dir,
            max_timesteps=args.max_timesteps)
        if result is not None:
            all_results.append(result)

    # Cross-session summary
    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print(f"Cross-Session Summary ({len(all_results)} sessions)")
        print(f"{'='*60}")

        planner_names = all_results[0]['planner_names']
        for pi, name in enumerate(planner_names):
            phis = [r['confidence'][pi].mean() for r in all_results]
            times = [r['time_ms'][pi].mean() for r in all_results]
            print(f"  {name:<30} phi={np.mean(phis):.4f}+/-{np.std(phis):.4f}  "
                  f"time={np.mean(times):.2f}ms")

    # Quick ablation: quantum vs k-NN correlation
    if all_results:
        print(f"\n{'='*60}")
        print("Quick Ablation: Quantum vs k-NN Rank Correlation")
        print(f"{'='*60}")

        for result in all_results:
            session = result['session']
            pnames = list(result['planner_names'])

            # Find quantum and knn indices
            q_idx = pnames.index('quantum_sup_3q') if 'quantum_sup_3q' in pnames else None
            k_idx = pnames.index('knn_mean') if 'knn_mean' in pnames else None

            if q_idx is not None and k_idx is not None:
                q_conf = result['confidence'][q_idx]
                k_conf = result['confidence'][k_idx]
                rho = spearman_rank_correlation(q_conf, k_conf)
                print(f"  {session}: Spearman r = {rho:.4f}")
                if rho > 0.95:
                    print(f"    >>> QUANTUM ~ kNN (cosmetic formalism)")
                elif rho < 0.9:
                    print(f"    >>> GENUINE DIFFERENCE detected!")
                else:
                    print(f"    >>> Marginal difference — needs more data")

    print(f"\nResults in: {args.output_dir}")


if __name__ == '__main__':
    main()
