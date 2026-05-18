#!/usr/bin/env python3
"""
Bag Extract Synced — Offline synchronized extraction from MCAP rosbag.

Reads an MCAP rosbag recorded by dataset_recorder.py and extracts
camera-rate synchronized frames for ML training.

Strategy:
  - Master clock: front camera RGB (~6Hz) drives the sampling rate
  - For each RGB frame, find the NEAREST message on every other topic
    within a configurable tolerance window (default 100ms)
  - Discard frames where any required stream is missing
  - Save structured output ready for BEV grid generation

Output structure:
  output_dir/
    front_camera/
      rgb/000000.jpg, 000001.jpg, ...
      depth/000000.png, 000001.png, ...    (uint16 millimeters)
    left_camera/
      rgb/000000.jpg, ...
      depth/000000.png, ...
    right_camera/
      rgb/000000.jpg, ...
      depth/000000.png, ...
    odometry/000000.json, 000001.json, ...
    scans/000000.npz, 000001.npz, ...       (scan_filtered + scan_fused)
    imu/000000.json, 000001.json, ...
    timestamps.csv                           (master index)
    metadata.json                            (session info, topic rates)

Dependencies:
  pip install rosbags numpy opencv-python

Usage:
  python3 bag_extract_synced.py /path/to/recording_YYYYMMDD_HHMMSS/rosbag
  python3 bag_extract_synced.py /path/to/rosbag -o /path/to/output --tolerance 0.08
  python3 bag_extract_synced.py /path/to/rosbag --cameras front  # front only
"""

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np


def extract_synced(bag_path, output_dir, tolerance_s=0.1, cameras=None):
    """Extract synchronized frames from MCAP rosbag."""
    try:
        from rosbags.highlevel import AnyReader
        from rosbags.typesys import Stores, get_typestore
    except ImportError:
        print('ERROR: rosbags library not installed.')
        print('Install with: pip install rosbags')
        sys.exit(1)

    if cameras is None:
        cameras = ['front', 'left', 'right']

    # Topic mapping
    CAMERA_TOPICS = {
        'front': {
            'rgb': '/camera/color/image_raw',
            'depth': '/camera/aligned_depth_to_color/image_raw',
            'info': '/camera/color/camera_info',
        },
        'left': {
            'rgb': '/mapping_camera/color/image_raw',
            'depth': '/mapping_camera/aligned_depth_to_color/image_raw',
            'info': '/mapping_camera/color/camera_info',
        },
        'right': {
            'rgb': '/right_camera/color/image_raw',
            'depth': '/right_camera/aligned_depth_to_color/image_raw',
            'info': '/right_camera/color/camera_info',
        },
    }

    SENSOR_TOPICS = {
        'odom': '/odometry/filtered',
        'odom_raw': '/wc_control/odom',
        'scan_filtered': '/scan_filtered',
        'scan_fused': '/scan_fused',
        'imu': '/imu',
        'cmd_vel': '/cmd_vel',
    }

    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    for cam in cameras:
        os.makedirs(os.path.join(output_dir, f'{cam}_camera', 'rgb'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, f'{cam}_camera', 'depth'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'odometry'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'scans'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'imu'), exist_ok=True)

    typestore = get_typestore(Stores.ROS2_JAZZY)
    from pathlib import Path
    bag_path = Path(bag_path)

    print(f'Reading bag: {bag_path}')
    print(f'Output: {output_dir}')
    print(f'Cameras: {cameras}')
    print(f'Tolerance: {tolerance_s * 1000:.0f} ms')
    print()

    # ====================================================================
    # PASS 1: Index all messages by topic and timestamp
    # ====================================================================
    print('Pass 1: Indexing all messages...')
    t0 = time.time()

    # Store (timestamp_ns, raw_data, msgtype) per topic
    topic_msgs = {}

    with AnyReader([bag_path], default_typestore=typestore) as reader:
        # Collect all relevant topic connections
        wanted_topics = set()
        for cam in cameras:
            wanted_topics.update(CAMERA_TOPICS[cam].values())
        wanted_topics.update(SENSOR_TOPICS.values())

        conns = [c for c in reader.connections if c.topic in wanted_topics]
        topic_names = {c.topic for c in conns}
        print(f'  Found {len(topic_names)}/{len(wanted_topics)} topics in bag')
        missing = wanted_topics - topic_names
        if missing:
            print(f'  Missing: {missing}')

        for conn, timestamp, rawdata in reader.messages(connections=conns):
            topic = conn.topic
            if topic not in topic_msgs:
                topic_msgs[topic] = []
            topic_msgs[topic].append((timestamp, rawdata, conn.msgtype))

    # Sort each topic by timestamp
    for topic in topic_msgs:
        topic_msgs[topic].sort(key=lambda x: x[0])

    t1 = time.time()
    print(f'  Indexed {sum(len(v) for v in topic_msgs.values())} messages '
          f'in {t1 - t0:.1f}s')
    print()

    # Print topic counts
    print('  Topic message counts:')
    for topic in sorted(topic_msgs.keys()):
        msgs = topic_msgs[topic]
        duration_s = (msgs[-1][0] - msgs[0][0]) / 1e9 if len(msgs) > 1 else 0
        hz = len(msgs) / duration_s if duration_s > 0 else 0
        print(f'    {topic:<55} {len(msgs):>8} msgs  ({hz:.1f} Hz)')
    print()

    # ====================================================================
    # PASS 2: Synchronize to front camera RGB as master clock
    # ====================================================================
    master_topic = CAMERA_TOPICS['front']['rgb']
    if master_topic not in topic_msgs:
        print(f'ERROR: Master topic {master_topic} not found in bag!')
        sys.exit(1)

    master_msgs = topic_msgs[master_topic]
    tolerance_ns = int(tolerance_s * 1e9)

    print(f'Pass 2: Synchronizing {len(master_msgs)} master frames '
          f'(tolerance {tolerance_s * 1000:.0f}ms)...')

    # Helper: find nearest message within tolerance
    def find_nearest(topic_list, target_ns):
        """Binary search for nearest timestamp within tolerance."""
        if not topic_list:
            return None
        # Binary search
        lo, hi = 0, len(topic_list) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if topic_list[mid][0] < target_ns:
                lo = mid + 1
            else:
                hi = mid
        # Check lo and lo-1
        best_idx = lo
        if lo > 0:
            if abs(topic_list[lo - 1][0] - target_ns) < abs(topic_list[lo][0] - target_ns):
                best_idx = lo - 1
        if abs(topic_list[best_idx][0] - target_ns) <= tolerance_ns:
            return topic_list[best_idx]
        return None

    # Deserialize helpers
    def decode_image(raw, msgtype, typestore_ref):
        """Decode ROS Image message to numpy array."""
        msg = typestore_ref.deserialize_cdr(raw, msgtype)
        h, w = msg.height, msg.width
        encoding = msg.encoding

        if encoding in ('rgb8', 'bgr8'):
            img = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w, 3)
            if encoding == 'rgb8':
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            return img
        elif encoding == '16UC1':
            return np.frombuffer(msg.data, dtype=np.uint16).reshape(h, w)
        elif encoding == '32FC1':
            depth_f = np.frombuffer(msg.data, dtype=np.float32).reshape(h, w)
            return (depth_f * 1000).astype(np.uint16)  # meters -> mm
        elif encoding == 'mono8':
            return np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w)
        else:
            print(f'  WARNING: Unknown encoding {encoding}, treating as uint8')
            return np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w, -1)

    def decode_odom(raw, msgtype, typestore_ref):
        """Decode Odometry message to dict."""
        msg = typestore_ref.deserialize_cdr(raw, msgtype)
        p = msg.pose.pose.position
        o = msg.pose.pose.orientation
        lv = msg.twist.twist.linear
        av = msg.twist.twist.angular
        return {
            'position': {'x': float(p.x), 'y': float(p.y), 'z': float(p.z)},
            'orientation': {
                'x': float(o.x), 'y': float(o.y),
                'z': float(o.z), 'w': float(o.w),
            },
            'linear_velocity': {
                'x': float(lv.x), 'y': float(lv.y), 'z': float(lv.z),
            },
            'angular_velocity': {
                'x': float(av.x), 'y': float(av.y), 'z': float(av.z),
            },
        }

    def decode_scan(raw, msgtype, typestore_ref):
        """Decode LaserScan message to dict with ranges array."""
        msg = typestore_ref.deserialize_cdr(raw, msgtype)
        return {
            'angle_min': float(msg.angle_min),
            'angle_max': float(msg.angle_max),
            'angle_increment': float(msg.angle_increment),
            'range_min': float(msg.range_min),
            'range_max': float(msg.range_max),
            'ranges': np.array(msg.ranges, dtype=np.float32),
        }

    def decode_imu(raw, msgtype, typestore_ref):
        """Decode IMU message to dict."""
        msg = typestore_ref.deserialize_cdr(raw, msgtype)
        o = msg.orientation
        av = msg.angular_velocity
        la = msg.linear_acceleration
        return {
            'orientation': {
                'x': float(o.x), 'y': float(o.y),
                'z': float(o.z), 'w': float(o.w),
            },
            'angular_velocity': {
                'x': float(av.x), 'y': float(av.y), 'z': float(av.z),
            },
            'linear_acceleration': {
                'x': float(la.x), 'y': float(la.y), 'z': float(la.z),
            },
        }

    # ====================================================================
    # PASS 2 continued: Extract synchronized frames
    # ====================================================================
    csv_rows = []
    saved = 0
    skipped = 0

    with AnyReader([bag_path], default_typestore=typestore) as reader:
        ts_ref = typestore

        for idx, (master_ts, master_raw, master_msgtype) in enumerate(master_msgs):
            if idx % 100 == 0:
                print(f'  Processing frame {idx}/{len(master_msgs)}...')

            frame_data = {}
            frame_ok = True

            # Find nearest for each camera stream
            for cam in cameras:
                for stream in ('rgb', 'depth'):
                    topic = CAMERA_TOPICS[cam][stream]
                    if topic == master_topic and stream == 'rgb':
                        # This IS the master — use directly
                        frame_data[(cam, stream)] = (master_ts, master_raw, master_msgtype)
                        continue
                    if topic not in topic_msgs:
                        frame_ok = False
                        break
                    match = find_nearest(topic_msgs[topic], master_ts)
                    if match is None:
                        frame_ok = False
                        break
                    frame_data[(cam, stream)] = match
                if not frame_ok:
                    break

            # Find nearest sensor data (not required — save None if missing)
            sensor_data = {}
            for sensor_name, topic in SENSOR_TOPICS.items():
                if topic in topic_msgs:
                    match = find_nearest(topic_msgs[topic], master_ts)
                    sensor_data[sensor_name] = match
                else:
                    sensor_data[sensor_name] = None

            if not frame_ok:
                skipped += 1
                continue

            # Decode and save images
            master_ts_sec = master_ts / 1e9
            csv_row = {'index': saved, 'timestamp_ns': master_ts, 'timestamp_s': master_ts_sec}

            for cam in cameras:
                # RGB
                ts_r, raw_r, mt_r = frame_data[(cam, 'rgb')]
                rgb_img = decode_image(raw_r, mt_r, ts_ref)
                rgb_path = os.path.join(output_dir, f'{cam}_camera', 'rgb', f'{saved:06d}.jpg')
                cv2.imwrite(rgb_path, rgb_img)
                csv_row[f'{cam}_rgb_dt_ms'] = abs(ts_r - master_ts) / 1e6

                # Depth
                ts_d, raw_d, mt_d = frame_data[(cam, 'depth')]
                depth_img = decode_image(raw_d, mt_d, ts_ref)
                depth_path = os.path.join(output_dir, f'{cam}_camera', 'depth', f'{saved:06d}.png')
                cv2.imwrite(depth_path, depth_img)
                csv_row[f'{cam}_depth_dt_ms'] = abs(ts_d - master_ts) / 1e6

            # Save odometry
            odom_match = sensor_data.get('odom')
            if odom_match:
                odom_dict = decode_odom(odom_match[1], odom_match[2], ts_ref)
                odom_dict['timestamp_ns'] = int(odom_match[0])
                odom_dict['dt_ms'] = abs(odom_match[0] - master_ts) / 1e6
                odom_path = os.path.join(output_dir, 'odometry', f'{saved:06d}.json')
                with open(odom_path, 'w') as f:
                    json.dump(odom_dict, f, indent=2)
                csv_row['odom_dt_ms'] = odom_dict['dt_ms']

            # Save scans
            scan_dict = {}
            for scan_name in ('scan_filtered', 'scan_fused'):
                sm = sensor_data.get(scan_name)
                if sm:
                    sd = decode_scan(sm[1], sm[2], ts_ref)
                    scan_dict[f'{scan_name}_ranges'] = sd['ranges']
                    scan_dict[f'{scan_name}_angle_min'] = sd['angle_min']
                    scan_dict[f'{scan_name}_angle_max'] = sd['angle_max']
                    scan_dict[f'{scan_name}_angle_increment'] = sd['angle_increment']
                    csv_row[f'{scan_name}_dt_ms'] = abs(sm[0] - master_ts) / 1e6
            if scan_dict:
                scan_path = os.path.join(output_dir, 'scans', f'{saved:06d}.npz')
                np.savez_compressed(scan_path, **scan_dict)

            # Save IMU
            imu_match = sensor_data.get('imu')
            if imu_match:
                imu_dict = decode_imu(imu_match[1], imu_match[2], ts_ref)
                imu_dict['timestamp_ns'] = int(imu_match[0])
                imu_dict['dt_ms'] = abs(imu_match[0] - master_ts) / 1e6
                imu_path = os.path.join(output_dir, 'imu', f'{saved:06d}.json')
                with open(imu_path, 'w') as f:
                    json.dump(imu_dict, f, indent=2)
                csv_row['imu_dt_ms'] = imu_dict['dt_ms']

            csv_rows.append(csv_row)
            saved += 1

    # Write master timestamp CSV
    if csv_rows:
        csv_path = os.path.join(output_dir, 'timestamps.csv')
        keys = list(csv_rows[0].keys())
        with open(csv_path, 'w') as f:
            f.write(','.join(keys) + '\n')
            for row in csv_rows:
                f.write(','.join(str(row.get(k, '')) for k in keys) + '\n')

    # Write metadata
    metadata = {
        'bag_path': str(bag_path),
        'cameras': cameras,
        'tolerance_s': tolerance_s,
        'total_master_frames': len(master_msgs),
        'saved_synced_frames': saved,
        'skipped_frames': skipped,
        'extraction_time': datetime.now().isoformat(),
    }
    from datetime import datetime
    metadata['extraction_time'] = datetime.now().isoformat()
    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    t2 = time.time()
    print()
    print('=' * 55)
    print('  EXTRACTION COMPLETE')
    print('=' * 55)
    print(f'  Saved:   {saved} synced frames')
    print(f'  Skipped: {skipped} (missing data within tolerance)')
    print(f'  Time:    {t2 - t0:.1f}s')
    print(f'  Output:  {output_dir}')
    print('=' * 55)


def main():
    parser = argparse.ArgumentParser(
        description='Extract synchronized multi-sensor frames from MCAP rosbag.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('bag_path', help='Path to rosbag directory (MCAP)')
    parser.add_argument('-o', '--output', default=None,
                        help='Output directory (default: bag_path/../synced_TIMESTAMP)')
    parser.add_argument('--tolerance', type=float, default=0.1,
                        help='Max time offset for sync (seconds, default 0.1)')
    parser.add_argument('--cameras', nargs='+', default=['front', 'left', 'right'],
                        choices=['front', 'left', 'right'],
                        help='Which cameras to extract (default: all 3)')

    args = parser.parse_args()

    if args.output is None:
        from datetime import datetime
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        parent = os.path.dirname(os.path.abspath(args.bag_path))
        args.output = os.path.join(parent, f'synced_{ts}')

    extract_synced(
        bag_path=args.bag_path,
        output_dir=args.output,
        tolerance_s=args.tolerance,
        cameras=args.cameras,
    )


if __name__ == '__main__':
    main()
