#!/usr/bin/env python3
"""
ROBUST NAVIGATION DATA LOGGER v2.0
===================================
Efficient, comprehensive logging for debugging wheelchair navigation.

NEW IN v2.0:
  - Goal pose tracking (where robot is trying to go)
  - Path info (length, validity, num poses)
  - Distance to goal (computed)
  - Controller feedback
  - Buffered writing for efficiency
  - Summary stats on shutdown

COLUMNS (90 total):
  1-1:   wall_time
  2-12:  IMU (stamp, orientation, angular_vel, linear_accel)
  13-19: Raw IMU (stamp, angular_vel, linear_accel)
  20-33: Raw odometry (stamp, position, orientation, velocities)
  34-47: EKF odometry (stamp, position, orientation, velocities)
  48-54: AMCL (stamp, x, y, yaw, covariances)
  55-61: TF map->base_link (stamp, x, y, z, yaw, valid, error)
  62-65: cmd_vel (stamp, vx, vy, wz)
  66-68: Scan (stamp, points, age_ms)
  69-71: Scan filtered (stamp, points, age_ms)
  72-74: Nav status (goal_active, status, status_text)
  75-76: Costmap ages (local, global)
  77-80: Goal pose (x, y, yaw, active)
  81-84: Path info (length, num_poses, valid, first_heading) - NEW: first_heading
  85-86: Local plan (num_poses, valid) - NEW for stuck rotation debug
  87-89: Computed (dist_to_goal, heading_error, robot_yaw) - CRITICAL for debug

Usage:
  ros2 run scripts topic_data_logger

Output: /home/sidd/wheelchair_nav/src/data_logs/wheelchair_fusion_log_YYYYMMDD_HHMMSS.csv

Author: Navigation DevOps Engineer
Date: 2026-01-13
"""

import csv
import math
import os
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Deque

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSPresetProfiles, QoSProfile, ReliabilityPolicy, DurabilityPolicy
from nav_msgs.msg import Odometry, OccupancyGrid, Path as NavPath
from sensor_msgs.msg import Imu, LaserScan
from geometry_msgs.msg import PoseWithCovarianceStamped, Twist, TwistStamped, PoseStamped
from action_msgs.msg import GoalStatusArray

import tf2_ros
from tf2_ros import TransformException


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def stamp_to_float(msg_stamp) -> float:
    """Convert ROS2 Time to seconds as float."""
    return msg_stamp.sec + msg_stamp.nanosec * 1e-9


def quat_to_yaw(q) -> float:
    """Extract yaw from quaternion."""
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


# ============================================================================
# DATA SNAPSHOTS
# ============================================================================

@dataclass
class ImuData:
    stamp: float = 0.0
    qx: float = 0.0
    qy: float = 0.0
    qz: float = 0.0
    qw: float = 1.0
    wx: float = 0.0
    wy: float = 0.0
    wz: float = 0.0
    ax: float = 0.0
    ay: float = 0.0
    az: float = 9.81


@dataclass
class OdomData:
    stamp: float = 0.0
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    qx: float = 0.0
    qy: float = 0.0
    qz: float = 0.0
    qw: float = 1.0
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    wx: float = 0.0
    wy: float = 0.0
    wz: float = 0.0


@dataclass
class AmclData:
    stamp: float = 0.0
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0
    cov_xx: float = 0.0
    cov_yy: float = 0.0
    cov_yaw: float = 0.0


@dataclass
class TfData:
    stamp: float = 0.0
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    yaw: float = 0.0
    valid: bool = False
    error: str = ""


@dataclass
class CmdVelData:
    stamp: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    wz: float = 0.0


@dataclass
class ScanData:
    stamp: float = 0.0
    num_points: int = 0
    age_ms: float = 9999.0


@dataclass
class NavStatusData:
    goal_active: bool = False
    status: int = 0
    status_text: str = "UNKNOWN"


@dataclass
class CostmapData:
    age_ms: float = 9999.0


@dataclass
class GoalData:
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0
    active: bool = False


@dataclass
class PathData:
    length: float = 0.0
    num_poses: int = 0
    valid: bool = False
    first_heading: float = 0.0  # Heading of first path segment (for heading error)


@dataclass
class LocalPlanData:
    num_poses: int = 0
    valid: bool = False


# ============================================================================
# MAIN LOGGER NODE
# ============================================================================

class RobustDataLogger(Node):
    """Efficient, comprehensive navigation data logger."""

    # CSV Header
    HEADER = [
        'wall_time',
        # IMU (11 fields)
        'imu_stamp', 'imu_qx', 'imu_qy', 'imu_qz', 'imu_qw',
        'imu_wx', 'imu_wy', 'imu_wz', 'imu_ax', 'imu_ay', 'imu_az',
        # Raw IMU (7 fields)
        'raw_imu_stamp', 'raw_imu_wx', 'raw_imu_wy', 'raw_imu_wz',
        'raw_imu_ax', 'raw_imu_ay', 'raw_imu_az',
        # Raw Odom (14 fields)
        'raw_odom_stamp', 'raw_x', 'raw_y', 'raw_z',
        'raw_qx', 'raw_qy', 'raw_qz', 'raw_qw',
        'raw_vx', 'raw_vy', 'raw_vz', 'raw_wx', 'raw_wy', 'raw_wz',
        # EKF Odom (14 fields)
        'ekf_stamp', 'ekf_x', 'ekf_y', 'ekf_z',
        'ekf_qx', 'ekf_qy', 'ekf_qz', 'ekf_qw',
        'ekf_vx', 'ekf_vy', 'ekf_vz', 'ekf_wx', 'ekf_wy', 'ekf_wz',
        # AMCL (7 fields)
        'amcl_stamp', 'amcl_x', 'amcl_y', 'amcl_yaw',
        'amcl_cov_xx', 'amcl_cov_yy', 'amcl_cov_yaw',
        # TF (7 fields)
        'tf_stamp', 'tf_x', 'tf_y', 'tf_z', 'tf_yaw', 'tf_valid', 'tf_error',
        # cmd_vel (4 fields)
        'cmd_stamp', 'cmd_vx', 'cmd_vy', 'cmd_wz',
        # Scan (3 fields)
        'scan_stamp', 'scan_points', 'scan_age_ms',
        # Scan filtered (3 fields)
        'scan_filt_stamp', 'scan_filt_points', 'scan_filt_age_ms',
        # Nav status (3 fields)
        'nav_goal_active', 'nav_status', 'nav_status_text',
        # Costmap (2 fields)
        'local_costmap_age_ms', 'global_costmap_age_ms',
        # Goal (4 fields) - NEW
        'goal_x', 'goal_y', 'goal_yaw', 'goal_active',
        # Path (4 fields) - NEW
        'path_length', 'path_num_poses', 'path_valid', 'path_heading',
        # Local plan (2 fields) - NEW for stuck rotation debug
        'local_plan_poses', 'local_plan_valid',
        # Computed (3 fields) - CRITICAL for stuck rotation debug
        'dist_to_goal', 'heading_error', 'robot_yaw',
    ]

    def __init__(self):
        super().__init__('robust_data_logger')

        # Parameters
        self.declare_parameter('log_frequency_hz', 10.0)
        # Default output to ~/wheelchair_nav_logs (portable across systems)
        default_log_dir = os.path.join(os.path.expanduser('~'), 'wheelchair_nav_logs')
        self.declare_parameter('output_dir', default_log_dir)
        self.declare_parameter('buffer_size', 100)  # Write every N rows

        log_freq = self.get_parameter('log_frequency_hz').value
        output_dir = self.get_parameter('output_dir').value
        self._buffer_size = self.get_parameter('buffer_size').value

        # Create output file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_path = Path(output_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        self._csv_path = log_path / f'wheelchair_fusion_log_{timestamp}.csv'

        # TF buffer
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        # QoS profiles
        sensor_qos = QoSPresetProfiles.SENSOR_DATA.value
        reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=10
        )

        # Data storage (initialized with defaults)
        self._imu = ImuData()
        self._raw_imu = ImuData()
        self._raw_odom = OdomData()
        self._ekf_odom = OdomData()
        self._amcl = AmclData()
        self._cmd_vel = CmdVelData()
        self._scan = ScanData()
        self._scan_filt = ScanData()
        self._nav_status = NavStatusData()
        self._local_costmap = CostmapData()
        self._global_costmap = CostmapData()
        self._goal = GoalData()
        self._path = PathData()
        self._local_plan = LocalPlanData()

        # Statistics
        self._rows_written = 0
        self._start_time = time.time()
        self._max_cmd_vx = 0.0
        self._max_cmd_wz = 0.0
        self._tf_errors = 0
        self._nav_aborts = 0

        # Write buffer for efficiency
        self._buffer: Deque[list] = deque(maxlen=self._buffer_size * 2)

        # Subscribers
        self.create_subscription(Imu, '/imu', self._imu_cb, sensor_qos)
        self.create_subscription(Imu, '/camera/imu', self._raw_imu_cb, sensor_qos)
        self.create_subscription(Odometry, '/wc_control/odom', self._raw_odom_cb, reliable_qos)
        self.create_subscription(Odometry, '/odometry/filtered', self._ekf_odom_cb, reliable_qos)
        self.create_subscription(PoseWithCovarianceStamped, '/amcl_pose', self._amcl_cb, 10)
        self.create_subscription(Twist, '/cmd_vel', self._cmd_vel_cb, 10)
        self.create_subscription(LaserScan, '/scan', self._scan_cb, sensor_qos)
        self.create_subscription(LaserScan, '/scan_filtered', self._scan_filt_cb, sensor_qos)
        self.create_subscription(GoalStatusArray, '/navigate_to_pose/_action/status', self._nav_status_cb, 10)
        self.create_subscription(OccupancyGrid, '/local_costmap/costmap', self._local_costmap_cb, reliable_qos)
        self.create_subscription(OccupancyGrid, '/global_costmap/costmap', self._global_costmap_cb, reliable_qos)
        self.create_subscription(PoseStamped, '/goal_pose', self._goal_cb, 10)
        self.create_subscription(NavPath, '/plan', self._path_cb, 10)
        self.create_subscription(NavPath, '/local_plan', self._local_plan_cb, 10)

        # Open CSV and write header
        self._csv_file = open(self._csv_path, 'w', newline='', buffering=1)
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow(self.HEADER)

        self.get_logger().info(f'📊 Logging to: {self._csv_path}')
        self.get_logger().info(f'   Frequency: {log_freq} Hz, Buffer: {self._buffer_size} rows')

        # Timer
        self._timer = self.create_timer(1.0 / log_freq, self._log_row)

    # ========================================================================
    # CALLBACKS
    # ========================================================================

    def _imu_cb(self, msg: Imu):
        self._imu = ImuData(
            stamp=stamp_to_float(msg.header.stamp),
            qx=msg.orientation.x, qy=msg.orientation.y,
            qz=msg.orientation.z, qw=msg.orientation.w,
            wx=msg.angular_velocity.x, wy=msg.angular_velocity.y, wz=msg.angular_velocity.z,
            ax=msg.linear_acceleration.x, ay=msg.linear_acceleration.y, az=msg.linear_acceleration.z,
        )

    def _raw_imu_cb(self, msg: Imu):
        self._raw_imu = ImuData(
            stamp=stamp_to_float(msg.header.stamp),
            wx=msg.angular_velocity.x, wy=msg.angular_velocity.y, wz=msg.angular_velocity.z,
            ax=msg.linear_acceleration.x, ay=msg.linear_acceleration.y, az=msg.linear_acceleration.z,
        )

    def _raw_odom_cb(self, msg: Odometry):
        self._raw_odom = self._extract_odom(msg)

    def _ekf_odom_cb(self, msg: Odometry):
        self._ekf_odom = self._extract_odom(msg)

    def _amcl_cb(self, msg: PoseWithCovarianceStamped):
        pose = msg.pose.pose
        cov = msg.pose.covariance
        self._amcl = AmclData(
            stamp=stamp_to_float(msg.header.stamp),
            x=pose.position.x, y=pose.position.y,
            yaw=quat_to_yaw(pose.orientation),
            cov_xx=cov[0], cov_yy=cov[7], cov_yaw=cov[35],
        )

    def _cmd_vel_cb(self, msg: Twist):
        self._cmd_vel = CmdVelData(
            stamp=time.time(),
            vx=msg.linear.x, vy=msg.linear.y, wz=msg.angular.z,
        )
        # Track max velocities
        self._max_cmd_vx = max(self._max_cmd_vx, abs(msg.linear.x))
        self._max_cmd_wz = max(self._max_cmd_wz, abs(msg.angular.z))

    def _scan_cb(self, msg: LaserScan):
        now = time.time()
        scan_time = stamp_to_float(msg.header.stamp)
        valid = sum(1 for r in msg.ranges if msg.range_min < r < msg.range_max)
        self._scan = ScanData(stamp=scan_time, num_points=valid, age_ms=(now - scan_time) * 1000)

    def _scan_filt_cb(self, msg: LaserScan):
        now = time.time()
        scan_time = stamp_to_float(msg.header.stamp)
        valid = sum(1 for r in msg.ranges if msg.range_min < r < msg.range_max)
        self._scan_filt = ScanData(stamp=scan_time, num_points=valid, age_ms=(now - scan_time) * 1000)

    def _nav_status_cb(self, msg: GoalStatusArray):
        if msg.status_list:
            latest = msg.status_list[-1]
            status_map = {0: 'UNKNOWN', 1: 'ACCEPTED', 2: 'EXECUTING',
                         3: 'CANCELING', 4: 'SUCCEEDED', 5: 'CANCELED', 6: 'ABORTED'}
            text = status_map.get(latest.status, 'UNKNOWN')
            self._nav_status = NavStatusData(
                goal_active=latest.status in [1, 2, 3],
                status=latest.status,
                status_text=text,
            )
            if latest.status == 6:  # ABORTED
                self._nav_aborts += 1
        else:
            self._nav_status = NavStatusData()

    def _local_costmap_cb(self, msg: OccupancyGrid):
        now = time.time()
        stamp = stamp_to_float(msg.header.stamp)
        self._local_costmap = CostmapData(age_ms=(now - stamp) * 1000)

    def _global_costmap_cb(self, msg: OccupancyGrid):
        now = time.time()
        stamp = stamp_to_float(msg.header.stamp)
        self._global_costmap = CostmapData(age_ms=(now - stamp) * 1000)

    def _goal_cb(self, msg: PoseStamped):
        self._goal = GoalData(
            x=msg.pose.position.x,
            y=msg.pose.position.y,
            yaw=quat_to_yaw(msg.pose.orientation),
            active=True,
        )

    def _path_cb(self, msg: NavPath):
        if len(msg.poses) < 2:
            self._path = PathData(length=0.0, num_poses=0, valid=False, first_heading=0.0)
            return
        # Calculate path length and first segment heading
        length = 0.0
        for i in range(1, len(msg.poses)):
            dx = msg.poses[i].pose.position.x - msg.poses[i-1].pose.position.x
            dy = msg.poses[i].pose.position.y - msg.poses[i-1].pose.position.y
            length += math.sqrt(dx*dx + dy*dy)
        # First segment heading (direction robot should face)
        dx = msg.poses[1].pose.position.x - msg.poses[0].pose.position.x
        dy = msg.poses[1].pose.position.y - msg.poses[0].pose.position.y
        first_heading = math.atan2(dy, dx)
        self._path = PathData(length=length, num_poses=len(msg.poses), valid=True, first_heading=first_heading)

    def _local_plan_cb(self, msg: NavPath):
        self._local_plan = LocalPlanData(num_poses=len(msg.poses), valid=len(msg.poses) > 0)

    @staticmethod
    def _extract_odom(msg: Odometry) -> OdomData:
        p = msg.pose.pose
        t = msg.twist.twist
        return OdomData(
            stamp=stamp_to_float(msg.header.stamp),
            x=p.position.x, y=p.position.y, z=p.position.z,
            qx=p.orientation.x, qy=p.orientation.y, qz=p.orientation.z, qw=p.orientation.w,
            vx=t.linear.x, vy=t.linear.y, vz=t.linear.z,
            wx=t.angular.x, wy=t.angular.y, wz=t.angular.z,
        )

    def _get_tf(self) -> TfData:
        """Get map->base_link transform."""
        try:
            trans = self._tf_buffer.lookup_transform('map', 'base_link', rclpy.time.Time())
            t = trans.transform.translation
            q = trans.transform.rotation
            yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                            1.0 - 2.0 * (q.y * q.y + q.z * q.z))
            return TfData(
                stamp=stamp_to_float(trans.header.stamp),
                x=t.x, y=t.y, z=t.z, yaw=yaw, valid=True, error=""
            )
        except TransformException as e:
            self._tf_errors += 1
            return TfData(valid=False, error=str(e)[:50])

    # ========================================================================
    # LOGGING
    # ========================================================================

    def _log_row(self):
        """Build and buffer one row of data."""
        now = time.time()
        tf = self._get_tf()

        # Compute distance to goal
        if self._goal.active and tf.valid:
            dist = math.sqrt((self._goal.x - tf.x)**2 + (self._goal.y - tf.y)**2)
        else:
            dist = -1.0

        # Compute heading error (CRITICAL for stuck rotation debug)
        # Heading error = angle between robot heading and path direction
        robot_yaw = tf.yaw if tf.valid else 0.0
        if self._path.valid and tf.valid:
            # Normalize angle difference to [-pi, pi]
            heading_error = self._path.first_heading - robot_yaw
            while heading_error > math.pi:
                heading_error -= 2 * math.pi
            while heading_error < -math.pi:
                heading_error += 2 * math.pi
        else:
            heading_error = 0.0

        row = [
            now,
            # IMU
            self._imu.stamp, self._imu.qx, self._imu.qy, self._imu.qz, self._imu.qw,
            self._imu.wx, self._imu.wy, self._imu.wz,
            self._imu.ax, self._imu.ay, self._imu.az,
            # Raw IMU
            self._raw_imu.stamp, self._raw_imu.wx, self._raw_imu.wy, self._raw_imu.wz,
            self._raw_imu.ax, self._raw_imu.ay, self._raw_imu.az,
            # Raw Odom
            self._raw_odom.stamp, self._raw_odom.x, self._raw_odom.y, self._raw_odom.z,
            self._raw_odom.qx, self._raw_odom.qy, self._raw_odom.qz, self._raw_odom.qw,
            self._raw_odom.vx, self._raw_odom.vy, self._raw_odom.vz,
            self._raw_odom.wx, self._raw_odom.wy, self._raw_odom.wz,
            # EKF Odom
            self._ekf_odom.stamp, self._ekf_odom.x, self._ekf_odom.y, self._ekf_odom.z,
            self._ekf_odom.qx, self._ekf_odom.qy, self._ekf_odom.qz, self._ekf_odom.qw,
            self._ekf_odom.vx, self._ekf_odom.vy, self._ekf_odom.vz,
            self._ekf_odom.wx, self._ekf_odom.wy, self._ekf_odom.wz,
            # AMCL
            self._amcl.stamp, self._amcl.x, self._amcl.y, self._amcl.yaw,
            self._amcl.cov_xx, self._amcl.cov_yy, self._amcl.cov_yaw,
            # TF
            tf.stamp, tf.x, tf.y, tf.z, tf.yaw, 1.0 if tf.valid else 0.0, tf.error,
            # cmd_vel
            self._cmd_vel.stamp, self._cmd_vel.vx, self._cmd_vel.vy, self._cmd_vel.wz,
            # Scan
            self._scan.stamp, self._scan.num_points, self._scan.age_ms,
            # Scan filtered
            self._scan_filt.stamp, self._scan_filt.num_points, self._scan_filt.age_ms,
            # Nav status
            1.0 if self._nav_status.goal_active else 0.0,
            self._nav_status.status, self._nav_status.status_text,
            # Costmap
            self._local_costmap.age_ms, self._global_costmap.age_ms,
            # Goal
            self._goal.x, self._goal.y, self._goal.yaw, 1.0 if self._goal.active else 0.0,
            # Path (4 fields)
            self._path.length, self._path.num_poses, 1.0 if self._path.valid else 0.0, self._path.first_heading,
            # Local plan (2 fields)
            self._local_plan.num_poses, 1.0 if self._local_plan.valid else 0.0,
            # Computed (3 fields) - CRITICAL
            dist, heading_error, robot_yaw,
        ]

        self._buffer.append(row)
        self._rows_written += 1

        # Flush buffer periodically
        if len(self._buffer) >= self._buffer_size:
            self._flush_buffer()

    def _flush_buffer(self):
        """Write buffered rows to CSV."""
        if self._buffer:
            self._csv_writer.writerows(self._buffer)
            self._csv_file.flush()
            self._buffer.clear()

    def destroy_node(self):
        """Cleanup and print summary."""
        self._flush_buffer()
        self._csv_file.close()

        duration = time.time() - self._start_time
        self.get_logger().info(f'\n{"="*60}')
        self.get_logger().info(f'📊 DATA LOGGER SUMMARY')
        self.get_logger().info(f'{"="*60}')
        self.get_logger().info(f'  File: {self._csv_path}')
        self.get_logger().info(f'  Duration: {duration:.1f}s')
        self.get_logger().info(f'  Rows: {self._rows_written}')
        self.get_logger().info(f'  Rate: {self._rows_written/max(duration,1):.1f} Hz')
        self.get_logger().info(f'  Max cmd_vx: {self._max_cmd_vx:.3f} m/s')
        self.get_logger().info(f'  Max cmd_wz: {self._max_cmd_wz:.3f} rad/s')
        self.get_logger().info(f'  TF errors: {self._tf_errors}')
        self.get_logger().info(f'  Nav aborts: {self._nav_aborts}')
        self.get_logger().info(f'{"="*60}\n')

        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = RobustDataLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
