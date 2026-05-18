#!/usr/bin/env python3
"""
Convert ROS2 bag recordings to BEV-Velocity training data.

Reads a rosbag containing:
    /scan_fused          (LaserScan)  - fused 360 scan
    /odometry/filtered   (Odometry)   - EKF fused odometry
    /goal_pose           (PoseStamped) - optional: explicit goal positions
    /plan                (Path)        - optional: Nav2 global plan for route

Velocity labels come from odom TWIST (actual executed motion from
wheel encoders + IMU filtered by EKF), NOT from /cmd_vel.

Goals: from /goal_pose if available, otherwise auto-detected from
5-second stops (|v| < 0.02 m/s marks end of driving segment).

Outputs:
    data_dir/
        bev_000000.npy   (5, 200, 200)  - 5-channel BEV grid
        odom_000000.npy  (30,)          - odom history [v,w,theta] x 10
        labels.npy       (N, 2)         - [v, omega] from odom twist
        traj_poses.npy   (N, H, 3)     - forward-integrated [x,y,theta]
        metadata.npz                    - timestamps, goals, segments

Usage:
    # Auto-detect goals from 5-second stops:
    python bag_to_bev_velocity.py --bag_path /path/to/bag --output_dir /path/to/data

    # Override with manual goal (useful for single-destination bags):
    python bag_to_bev_velocity.py --bag_path /path/to/bag --output_dir /path/to/data \
        --goal_x 5.0 --goal_y 0.0
"""

import os
import argparse
import numpy as np
from collections import deque

try:
    from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
    from rclpy.serialization import deserialize_message
except ImportError:
    SequentialReader = None  # Will use direct MCAP reader
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from wheelchair_e2e.bev_generator import BEVGenerator


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert rosbag to BEV-velocity training data')
    parser.add_argument('--bag_path', type=str, required=True,
                        help='Path to ROS2 bag directory')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for training data')
    parser.add_argument('--goal_x', type=float, default=None,
                        help='Manual goal x (meters, odom frame). '
                             'If not set, auto-detect from 5s stops.')
    parser.add_argument('--goal_y', type=float, default=None,
                        help='Manual goal y (meters, odom frame)')
    parser.add_argument('--rate', type=float, default=10.0,
                        help='Output sample rate (Hz)')
    parser.add_argument('--min_velocity', type=float, default=0.02,
                        help='Skip samples below this speed (m/s)')
    parser.add_argument('--stop_duration', type=float, default=2.0,
                        help='Seconds of near-zero velocity to detect goal')
    parser.add_argument('--stop_threshold', type=float, default=0.02,
                        help='Velocity threshold for stop detection (m/s)')
    parser.add_argument('--horizon', type=int, default=10,
                        help='Trajectory horizon for pose labels (steps)')
    return parser.parse_args()


def read_bag(bag_path):
    """Read scan, odom, goal, and plan messages from a ROS2 bag.

    Supports both rosbag2 directories (with metadata.yaml) and raw MCAP
    directories (without metadata.yaml, e.g. interrupted recordings).
    """
    topic_types = {
        '/scan_fused': LaserScan,
        '/scan_filtered': LaserScan,
        '/odometry/filtered': Odometry,
        '/goal_pose': PoseStamped,
        '/plan': Path,
    }
    messages = {t: [] for t in topic_types}

    # Check if metadata.yaml exists (required for rosbag2_py)
    metadata_path = os.path.join(bag_path, 'metadata.yaml')
    has_metadata = os.path.isfile(metadata_path)

    if has_metadata:
        # Standard rosbag2 reader
        import glob
        mcap_files = glob.glob(os.path.join(bag_path, '*.mcap'))
        storage_id = 'mcap' if mcap_files else 'sqlite3'
        print(f"Using rosbag2 reader (storage: {storage_id})")
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
        # Direct MCAP reader (no metadata.yaml)
        import glob
        from mcap.reader import make_reader
        from mcap_ros2.decoder import DecoderFactory

        mcap_files = sorted(glob.glob(os.path.join(bag_path, '*.mcap')))
        if not mcap_files:
            raise RuntimeError(f"No .mcap files found in {bag_path}")

        print(f"Using direct MCAP reader ({len(mcap_files)} files, "
              f"no metadata.yaml)")
        decoder = DecoderFactory()

        for mcap_file in mcap_files:
            fname = os.path.basename(mcap_file)
            try:
                with open(mcap_file, 'rb') as f:
                    reader = make_reader(f, decoder_factories=[decoder])
                    count = 0
                    for schema, channel, message, decoded_msg in \
                            reader.iter_decoded_messages(topics=list(topic_types.keys())):
                        if channel.topic in topic_types:
                            messages[channel.topic].append(
                                (message.log_time, decoded_msg))
                            count += 1
                    print(f"  {fname}: {count} msgs")
            except Exception as e:
                print(f"  {fname}: SKIPPED ({e})")

        # Sort all messages by timestamp (files may overlap slightly)
        for topic in messages:
            messages[topic].sort(key=lambda x: x[0])

    return messages


def get_yaw(orientation):
    """Extract yaw from quaternion."""
    q = orientation
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return np.arctan2(siny, cosy)


def detect_segments(odoms, stop_duration_s, stop_threshold):
    """Detect driving segments separated by 5-second stops.

    Returns list of (start_idx, end_idx, goal_x, goal_y) tuples.
    The goal for each segment is the odom position at the stop point.
    """
    if not odoms:
        return []

    segments = []
    segment_start = 0
    stop_start = None

    for i, (timestamp, msg) in enumerate(odoms):
        v = abs(msg.twist.twist.linear.x)
        w = abs(msg.twist.twist.angular.z)
        speed = v + abs(w) * 0.1  # small weight on angular

        if speed < stop_threshold:
            if stop_start is None:
                stop_start = i
            # Check if we've been stopped long enough
            stop_time_ns = timestamp - odoms[stop_start][0]
            if stop_time_ns >= stop_duration_s * 1e9:
                # This is a goal! End of segment.
                # Goal position = where we stopped
                goal_msg = odoms[stop_start][1]
                goal_x = goal_msg.pose.pose.position.x
                goal_y = goal_msg.pose.pose.position.y

                if stop_start > segment_start:
                    segments.append((
                        segment_start, stop_start, goal_x, goal_y))

                # Next segment starts after the stop
                segment_start = i + 1
                stop_start = None
        else:
            stop_start = None

    # Last segment: goal is the final odom position
    if segment_start < len(odoms) - 1:
        final_msg = odoms[-1][1]
        goal_x = final_msg.pose.pose.position.x
        goal_y = final_msg.pose.pose.position.y
        segments.append((segment_start, len(odoms) - 1, goal_x, goal_y))

    return segments


def synthesize_route(odoms, odom_times, current_idx, goal_x, goal_y,
                     seg_end_idx, subsample_step=5):
    """Synthesize a route from future odometry trajectory toward the goal.

    Uses the actual driven path (future odom positions from current_idx
    to seg_end) as a "plan". This is what the robot actually drove,
    subsampled to ~2Hz for a clean route line on the BEV.

    Args:
        odoms: list of (timestamp, Odometry) tuples
        odom_times: numpy array of timestamps
        current_idx: current odom index
        goal_x, goal_y: goal position in odom frame
        seg_end_idx: end of current segment
        subsample_step: take every Nth odom for route (default 5 = ~10Hz)

    Returns:
        list of (x, y) in base_link frame, or None if too few points
    """
    # Get current pose for frame transform
    curr_msg = odoms[current_idx][1]
    curr_x = curr_msg.pose.pose.position.x
    curr_y = curr_msg.pose.pose.position.y
    curr_yaw = get_yaw(curr_msg.pose.pose.orientation)
    cos_yaw = np.cos(-curr_yaw)
    sin_yaw = np.sin(-curr_yaw)

    route = []
    # Sample future odom positions up to segment end (max ~5m ahead)
    max_ahead = min(seg_end_idx, current_idx + 500)  # ~10s at 50Hz
    for i in range(current_idx, max_ahead, subsample_step):
        ox = odoms[i][1].pose.pose.position.x - curr_x
        oy = odoms[i][1].pose.pose.position.y - curr_y
        # Transform to base_link frame
        rx = cos_yaw * ox - sin_yaw * oy
        ry = sin_yaw * ox + cos_yaw * oy
        route.append((rx, ry))
        # Stop if route extends beyond BEV grid (5m from center)
        if rx * rx + ry * ry > 25.0:
            break

    # Append goal as final point
    gx = goal_x - curr_x
    gy = goal_y - curr_y
    goal_rx = cos_yaw * gx - sin_yaw * gy
    goal_ry = sin_yaw * gx + cos_yaw * gy
    if goal_rx * goal_rx + goal_ry * goal_ry <= 25.0:
        route.append((goal_rx, goal_ry))

    return route if len(route) >= 2 else None


def compute_goal_relative(odom_msg, goal_x, goal_y):
    """Compute goal position relative to current wheelchair pose."""
    x = odom_msg.pose.pose.position.x
    y = odom_msg.pose.pose.position.y
    yaw = get_yaw(odom_msg.pose.pose.orientation)

    dx_world = goal_x - x
    dy_world = goal_y - y

    cos_yaw = np.cos(-yaw)
    sin_yaw = np.sin(-yaw)
    dx_base = cos_yaw * dx_world - sin_yaw * dy_world
    dy_base = sin_yaw * dx_world + cos_yaw * dy_world

    return dx_base, dy_base


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Reading bag: {args.bag_path}")
    messages = read_bag(args.bag_path)

    # Find available scan topic
    scan_topic = '/scan_fused'
    if not messages.get(scan_topic):
        scan_topic = '/scan_filtered'
        if not messages.get(scan_topic):
            print("ERROR: No /scan_fused or /scan_filtered in bag")
            return

    scans = messages[scan_topic]
    odoms = messages['/odometry/filtered']

    print(f"Scans ({scan_topic}): {len(scans)}")
    print(f"Odometry: {len(odoms)}")

    if not scans or not odoms:
        print("ERROR: Missing required topics")
        return

    # Read optional topics
    goal_poses = messages.get('/goal_pose', [])
    plans = messages.get('/plan', [])
    print(f"Goal poses (/goal_pose): {len(goal_poses)}")
    print(f"Plans (/plan): {len(plans)}")

    # Detect driving segments
    manual_goal = (args.goal_x is not None and args.goal_y is not None)

    if manual_goal:
        # Single segment with manual goal
        segments = [(0, len(odoms) - 1, args.goal_x, args.goal_y)]
        print(f"Using manual goal: ({args.goal_x}, {args.goal_y})")
    elif goal_poses:
        # Use /goal_pose messages to define segments
        segments = []
        goal_times = [t for t, _ in goal_poses]
        for gi, (gt, gp) in enumerate(goal_poses):
            gx = gp.pose.position.x
            gy = gp.pose.position.y
            # Segment: from this goal publish to next (or end)
            seg_start_idx = np.searchsorted(odom_times, gt)
            if gi + 1 < len(goal_poses):
                seg_end_idx = np.searchsorted(odom_times, goal_times[gi + 1])
            else:
                seg_end_idx = len(odoms) - 1
            seg_start_idx = min(seg_start_idx, len(odoms) - 1)
            seg_end_idx = min(seg_end_idx, len(odoms) - 1)
            if seg_end_idx > seg_start_idx:
                segments.append((seg_start_idx, seg_end_idx, gx, gy))
        print(f"Goal-pose segments: {len(segments)}")
        for i, (s, e, gx, gy) in enumerate(segments):
            t_dur = (odoms[e][0] - odoms[s][0]) / 1e9
            print(f"  Seg {i}: odom[{s}:{e}] = {t_dur:.1f}s, "
                  f"goal=({gx:.2f}, {gy:.2f})")
    else:
        segments = detect_segments(
            odoms, args.stop_duration, args.stop_threshold)
        print(f"Auto-detected {len(segments)} driving segments:")
        for i, (s, e, gx, gy) in enumerate(segments):
            t_dur = (odoms[e][0] - odoms[s][0]) / 1e9
            print(f"  Seg {i}: odom[{s}:{e}] = {t_dur:.1f}s, "
                  f"goal=({gx:.2f}, {gy:.2f})")

    if not segments:
        print("ERROR: No driving segments found (all stationary?)")
        return

    # Build BEV generator
    bev_gen = BEVGenerator(grid_size=200, resolution=0.05)

    # Index times
    odom_times = np.array([t for t, _ in odoms])
    dt_ns = int(1e9 / args.rate)

    sample_idx = 0
    all_labels = []
    all_timestamps = []
    all_goals = []
    all_segment_ids = []
    # Additional data for ModularKinoFlowNet v2
    all_scan_ranges = []
    all_scan_odom = []
    all_goal_relative = []

    # Index plan times for route lookup
    plan_times = np.array([t for t, _ in plans]) if plans else np.array([])

    for seg_id, (seg_start, seg_end, goal_x, goal_y) in enumerate(segments):
        t_start = odoms[seg_start][0]
        t_end = odoms[seg_end][0]
        odom_history = deque(maxlen=10)

        print(f"\nProcessing segment {seg_id} "
              f"({(t_end - t_start) / 1e9:.1f}s)...")

        scan_idx = 0
        for t in range(t_start, t_end, dt_ns):
            # Find nearest scan
            while (scan_idx < len(scans) - 1
                   and scans[scan_idx + 1][0] <= t):
                scan_idx += 1
            scan_msg = scans[scan_idx][1]

            # Find nearest odom
            odom_idx = np.searchsorted(odom_times, t)
            odom_idx = min(odom_idx, len(odoms) - 1)
            odom_msg = odoms[odom_idx][1]

            # Velocity labels from odom twist (actual executed motion)
            v = odom_msg.twist.twist.linear.x
            omega = odom_msg.twist.twist.angular.z

            # Skip stationary samples
            if abs(v) < args.min_velocity and abs(omega) < args.min_velocity:
                continue

            # Relative goal from odom pose
            goal_dx, goal_dy = compute_goal_relative(
                odom_msg, goal_x, goal_y)

            # Odom history
            yaw = get_yaw(odom_msg.pose.pose.orientation)
            odom_history.append([v, omega, yaw])

            # Pad with zeros if fewer than 10 steps
            odom_list = list(odom_history)
            while len(odom_list) < 10:
                odom_list.insert(0, [0.0, 0.0, 0.0])
            odom_flat = np.array(odom_list, dtype=np.float32).flatten()

            # Ego-motion trail for BEV channel 3
            odom_trail = []
            curr_x = odom_msg.pose.pose.position.x
            curr_y = odom_msg.pose.pose.position.y
            cos_yaw = np.cos(-yaw)
            sin_yaw = np.sin(-yaw)
            for oi in range(max(0, odom_idx - 10), odom_idx):
                ox = odoms[oi][1].pose.pose.position.x - curr_x
                oy = odoms[oi][1].pose.pose.position.y - curr_y
                odom_trail.append((
                    cos_yaw * ox - sin_yaw * oy,
                    sin_yaw * ox + cos_yaw * oy))

            # Route points for BEV channel 4
            route_points = None
            if len(plan_times) > 0:
                # Use actual /plan messages if available
                pi = np.searchsorted(plan_times, t)
                pi = min(pi, len(plans) - 1)
                plan_msg = plans[pi][1]
                route_points = []
                for pose_s in plan_msg.poses:
                    px = pose_s.pose.position.x - curr_x
                    py = pose_s.pose.position.y - curr_y
                    route_points.append((
                        cos_yaw * px - sin_yaw * py,
                        sin_yaw * px + cos_yaw * py))
            else:
                # Synthesize route from future odometry trajectory
                route_points = synthesize_route(
                    odoms, odom_times, odom_idx,
                    goal_x, goal_y, seg_end)

            # Ego odom for temporal delta (channel 1)
            ego_odom = (curr_x, curr_y, yaw)

            # Generate BEV grid
            bev = bev_gen.scan_msg_to_bev(
                scan_msg, goal_dx, goal_dy, odom_trail,
                route_points=route_points, ego_odom=ego_odom)

            # Save sample
            np.save(os.path.join(args.output_dir,
                                 f'bev_{sample_idx:06d}.npy'), bev)
            np.save(os.path.join(args.output_dir,
                                 f'odom_{sample_idx:06d}.npy'), odom_flat)

            # Save scan ranges + odom pose for v2 modular pipeline
            scan_ranges_arr = np.array(scan_msg.ranges, dtype=np.float32)
            # Pad/truncate to 720 points
            if len(scan_ranges_arr) < 720:
                scan_ranges_arr = np.pad(
                    scan_ranges_arr, (0, 720 - len(scan_ranges_arr)),
                    constant_values=np.inf)
            elif len(scan_ranges_arr) > 720:
                scan_ranges_arr = scan_ranges_arr[:720]
            all_scan_ranges.append(scan_ranges_arr)
            all_scan_odom.append([curr_x, curr_y, yaw])

            # Goal features: (norm_dist, norm_bearing, cos_b, sin_b)
            import math
            gdist = min(math.sqrt(goal_dx**2 + goal_dy**2), 10.0) / 10.0
            gbearing = math.atan2(goal_dy, goal_dx)
            all_goal_relative.append([
                gdist, gbearing / math.pi,
                math.cos(gbearing), math.sin(gbearing),
            ])

            all_labels.append([v, omega])
            all_timestamps.append(t)
            all_goals.append([goal_x, goal_y])
            all_segment_ids.append(seg_id)
            sample_idx += 1

            if sample_idx % 500 == 0:
                print(f"  {sample_idx} samples...")

    # Save labels and metadata
    labels = np.array(all_labels, dtype=np.float32)
    np.save(os.path.join(args.output_dir, 'labels.npy'), labels)

    # Compute trajectory pose labels by forward-integrating (v, omega)
    # For each sample i, integrate labels[i:i+H] to get (x, y, theta) poses
    H = args.horizon
    dt = 1.0 / args.rate
    segment_ids = np.array(all_segment_ids)
    n = len(labels)
    traj_poses = np.zeros((n, H, 3), dtype=np.float32)  # (x, y, theta)

    print(f"\nComputing trajectory poses (H={H})...")
    for i in range(n):
        # Check if H future steps are within the same segment
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

    # Save v2 modular pipeline data
    if all_scan_ranges:
        np.save(os.path.join(args.output_dir, 'scan_ranges.npy'),
                np.array(all_scan_ranges, dtype=np.float32))
        np.save(os.path.join(args.output_dir, 'scan_odom.npy'),
                np.array(all_scan_odom, dtype=np.float32))
        np.save(os.path.join(args.output_dir, 'goal_relative.npy'),
                np.array(all_goal_relative, dtype=np.float32))
        print(f"  Saved v2 data: scan_ranges.npy ({n},{720}), "
              f"scan_odom.npy ({n},3), goal_relative.npy ({n},4)")

    np.savez(os.path.join(args.output_dir, 'metadata.npz'),
             timestamps=np.array(all_timestamps),
             goal_positions=np.array(all_goals, dtype=np.float32),
             segment_ids=segment_ids,
             n_segments=len(segments),
             rate=args.rate,
             n_samples=sample_idx,
             horizon=H,
             label_source='odometry/filtered twist (executed velocity)')

    print(f"\nDone! {sample_idx} samples from {len(segments)} segments")
    if sample_idx > 0:
        print(f"  v range:  [{labels[:, 0].min():.3f}, "
              f"{labels[:, 0].max():.3f}] m/s")
        print(f"  w range:  [{labels[:, 1].min():.3f}, "
              f"{labels[:, 1].max():.3f}] rad/s")
        # Trajectory stats
        valid_poses = traj_poses[traj_poses[:, -1, 0] != 0]
        if len(valid_poses) > 0:
            max_dist = np.sqrt(
                valid_poses[:, -1, 0]**2 + valid_poses[:, -1, 1]**2)
            print(f"  Traj endpoint dist: [{max_dist.min():.3f}, "
                  f"{max_dist.max():.3f}] m (mean={max_dist.mean():.3f})")
    print(f"  Output: {args.output_dir}")


if __name__ == '__main__':
    main()
