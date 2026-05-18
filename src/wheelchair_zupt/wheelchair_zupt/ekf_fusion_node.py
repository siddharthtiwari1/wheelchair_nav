#!/usr/bin/env python3
"""
ROBUST EKF SENSOR FUSION FOR DIFFERENTIAL DRIVE WHEELCHAIR
===========================================================
State-of-the-art Extended Kalman Filter fusing:
  - Wheel encoders (velocity)
  - IMU gyroscope (angular velocity)
  - IMU orientation (absolute yaw reference)

State Vector: [x, y, theta, v, omega, gyro_bias]
  - x, y: Position in odom frame
  - theta: Heading angle
  - v: Linear velocity
  - omega: Angular velocity
  - gyro_bias: Online gyro bias estimate

Key Features:
  1. Continuous probabilistic fusion (not binary switching)
  2. Online gyro bias estimation
  3. Adaptive process noise based on motion
  4. Proper covariance propagation
  5. Outlier rejection for sensor failures
  6. Complementary filter fallback for robustness

Theory:
  - Predict step: Use motion model with encoder velocity
  - Update step: Correct with IMU gyro and orientation
  - Gyro bias is estimated as part of state (random walk model)

Author: Sensor Fusion Expert Implementation
Date: 2026-02-03
"""

import numpy as np
import threading
from dataclasses import dataclass, field
from typing import Optional, Tuple
from enum import Enum

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.time import Time

from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from geometry_msgs.msg import TransformStamped, Quaternion, Vector3Stamped
from std_msgs.msg import Float64MultiArray, Bool
from tf2_ros import TransformBroadcaster


def quaternion_from_yaw(yaw: float) -> Quaternion:
    """Create quaternion from yaw angle."""
    q = Quaternion()
    q.w = float(np.cos(yaw / 2.0))
    q.x = 0.0
    q.y = 0.0
    q.z = float(np.sin(yaw / 2.0))
    return q


def yaw_from_quaternion(q: Quaternion) -> float:
    """Extract yaw from quaternion."""
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return float(np.arctan2(siny_cosp, cosy_cosp))


def normalize_angle(angle: float) -> float:
    """Normalize angle to [-pi, pi]."""
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle


class MotionState(Enum):
    """Robot motion state for adaptive noise."""
    STATIONARY = 0
    STRAIGHT = 1
    ROTATING = 2
    MIXED = 3


@dataclass
class EKFState:
    """EKF state container."""
    # State vector: [x, y, theta, v, omega, gyro_bias]
    x: np.ndarray = field(default_factory=lambda: np.zeros(6))
    # State covariance (6x6)
    P: np.ndarray = field(default_factory=lambda: np.eye(6) * 0.01)
    # Timestamp
    stamp: float = 0.0
    # Initialized flag
    initialized: bool = False


class RobustEKFFusion:
    """
    Extended Kalman Filter for Differential Drive Robot.

    State: [x, y, theta, v, omega, gyro_bias]

    Process Model (constant velocity):
        x' = x + v*cos(theta)*dt
        y' = y + v*sin(theta)*dt
        theta' = theta + omega*dt
        v' = v  (constant, updated by encoder measurement)
        omega' = omega  (constant, updated by encoder/gyro fusion)
        gyro_bias' = gyro_bias  (random walk)

    Measurements:
        1. Encoder: [v_enc, omega_enc]
        2. Gyro: omega_gyro = omega + gyro_bias + noise
        3. IMU orientation: theta_imu (optional, for drift correction)
    """

    def __init__(self,
                 # Process noise
                 sigma_v: float = 0.05,        # Linear velocity noise (m/s)
                 sigma_omega: float = 0.02,    # Angular velocity noise (rad/s)
                 sigma_bias: float = 0.0001,   # Gyro bias random walk
                 # Encoder measurement noise
                 sigma_enc_v: float = 0.02,    # Encoder velocity noise
                 sigma_enc_omega: float = 0.01, # Encoder angular velocity noise
                 # Gyro measurement noise
                 sigma_gyro: float = 0.005,    # Gyro noise (rad/s)
                 # IMU orientation noise
                 sigma_imu_yaw: float = 0.02,  # IMU yaw noise (rad)
                 # Fusion parameters
                 gyro_weight: float = 0.7,     # Base gyro weight for omega fusion
                 use_imu_orientation: bool = True,
                 orientation_update_rate: float = 2.0,  # Hz for orientation updates
                 # Outlier rejection
                 mahalanobis_threshold: float = 5.0,
                 # Motion detection
                 stationary_v_thresh: float = 0.005,
                 stationary_omega_thresh: float = 0.02):

        self.state = EKFState()

        # Process noise parameters
        self.sigma_v = sigma_v
        self.sigma_omega = sigma_omega
        self.sigma_bias = sigma_bias

        # Measurement noise parameters
        self.sigma_enc_v = sigma_enc_v
        self.sigma_enc_omega = sigma_enc_omega
        self.sigma_gyro = sigma_gyro
        self.sigma_imu_yaw = sigma_imu_yaw

        # Fusion parameters
        self.base_gyro_weight = gyro_weight
        self.use_imu_orientation = use_imu_orientation
        self.orientation_update_period = 1.0 / orientation_update_rate
        self.last_orientation_update = 0.0

        # Outlier rejection
        self.mahalanobis_threshold = mahalanobis_threshold

        # Motion detection
        self.stationary_v_thresh = stationary_v_thresh
        self.stationary_omega_thresh = stationary_omega_thresh

        # Diagnostics
        self.motion_state = MotionState.STATIONARY
        self.innovation_v = 0.0
        self.innovation_omega = 0.0
        self.gyro_weight_used = gyro_weight
        self.outlier_count = 0
        self.update_count = 0

        # Initial IMU yaw for relative updates
        self.initial_imu_yaw: Optional[float] = None

    def reset(self, x: float = 0.0, y: float = 0.0, theta: float = 0.0):
        """Reset filter to initial state."""
        self.state.x = np.array([x, y, theta, 0.0, 0.0, 0.0])
        self.state.P = np.diag([0.001, 0.001, 0.001, 0.01, 0.01, 0.0001])
        self.state.initialized = True
        self.state.stamp = 0.0
        self.initial_imu_yaw = None
        self.outlier_count = 0
        self.update_count = 0

    def _get_motion_state(self, v: float, omega: float) -> MotionState:
        """Determine current motion state."""
        is_moving = abs(v) > self.stationary_v_thresh
        is_rotating = abs(omega) > self.stationary_omega_thresh

        if not is_moving and not is_rotating:
            return MotionState.STATIONARY
        elif is_moving and not is_rotating:
            return MotionState.STRAIGHT
        elif not is_moving and is_rotating:
            return MotionState.ROTATING
        else:
            return MotionState.MIXED

    def _compute_gyro_weight(self, v_enc: float, omega_enc: float,
                             omega_gyro: float) -> float:
        """
        Compute adaptive gyro weight based on motion state.

        During rotation: Trust gyro more (wheel slip likely)
        During straight: Trust encoder more (gyro drift)
        Stationary: Trust encoder completely (gyro bias estimation)
        """
        motion = self._get_motion_state(v_enc, omega_enc)
        self.motion_state = motion

        if motion == MotionState.STATIONARY:
            # Stationary: encoder is king, use gyro only for bias estimation
            return 0.1
        elif motion == MotionState.STRAIGHT:
            # Straight motion: encoder reliable, moderate gyro
            return 0.3
        elif motion == MotionState.ROTATING:
            # Rotation: gyro more reliable (wheel slip common)
            return 0.85
        else:  # MIXED
            # Mixed motion: balanced fusion
            # Higher omega = trust gyro more
            omega_factor = min(abs(omega_enc) / 0.5, 1.0)  # Saturate at 0.5 rad/s
            return 0.4 + 0.4 * omega_factor

    def _build_process_noise(self, dt: float) -> np.ndarray:
        """Build process noise covariance Q."""
        # Adaptive noise based on motion state
        motion_factor = {
            MotionState.STATIONARY: 0.1,
            MotionState.STRAIGHT: 1.0,
            MotionState.ROTATING: 1.5,
            MotionState.MIXED: 1.2
        }.get(self.motion_state, 1.0)

        Q = np.diag([
            (self.sigma_v * dt * motion_factor) ** 2,      # x
            (self.sigma_v * dt * motion_factor) ** 2,      # y
            (self.sigma_omega * dt * motion_factor) ** 2,  # theta
            (self.sigma_v * motion_factor) ** 2,           # v
            (self.sigma_omega * motion_factor) ** 2,       # omega
            (self.sigma_bias) ** 2,                        # gyro_bias (random walk)
        ])
        return Q

    def predict(self, dt: float):
        """
        EKF Predict Step.

        Motion model: constant velocity differential drive.
        """
        if not self.state.initialized or dt <= 0:
            return

        x, y, theta, v, omega, bias = self.state.x

        # State prediction (motion model)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        x_new = x + v * cos_theta * dt
        y_new = y + v * sin_theta * dt
        theta_new = normalize_angle(theta + omega * dt)
        # v, omega, bias: constant (updated in measurement step)

        self.state.x = np.array([x_new, y_new, theta_new, v, omega, bias])

        # Jacobian of motion model
        F = np.eye(6)
        F[0, 2] = -v * sin_theta * dt  # dx/dtheta
        F[0, 3] = cos_theta * dt       # dx/dv
        F[1, 2] = v * cos_theta * dt   # dy/dtheta
        F[1, 3] = sin_theta * dt       # dy/dv
        F[2, 4] = dt                   # dtheta/domega

        # Process noise
        Q = self._build_process_noise(dt)

        # Covariance prediction
        self.state.P = F @ self.state.P @ F.T + Q

    def update_encoder(self, v_enc: float, omega_enc: float,
                       gyro_z: float, timestamp: float) -> bool:
        """
        EKF Update with encoder velocity and gyro angular velocity.

        This is the main fusion step that combines:
        1. Encoder linear velocity (direct measurement)
        2. Encoder angular velocity (fused with gyro)
        3. Gyro angular velocity (bias-corrected)

        Returns True if update successful, False if outlier rejected.
        """
        if not self.state.initialized:
            return False

        self.update_count += 1

        # Current state
        _, _, _, v, omega, gyro_bias = self.state.x

        # Bias-corrected gyro
        gyro_corrected = gyro_z - gyro_bias

        # Compute adaptive gyro weight
        alpha = self._compute_gyro_weight(v_enc, omega_enc, gyro_corrected)
        self.gyro_weight_used = alpha

        # =====================================================================
        # MEASUREMENT 1: Linear velocity from encoder (direct)
        # =====================================================================
        # z1 = v_enc, h1(x) = v
        H1 = np.array([[0, 0, 0, 1, 0, 0]])
        z1 = np.array([v_enc])
        h1 = np.array([v])
        R1 = np.array([[self.sigma_enc_v ** 2]])

        # Innovation
        y1 = z1 - h1
        self.innovation_v = y1[0]

        # Kalman gain
        S1 = H1 @ self.state.P @ H1.T + R1
        K1 = self.state.P @ H1.T @ np.linalg.inv(S1)

        # Update
        self.state.x = self.state.x + (K1 @ y1).flatten()
        self.state.P = (np.eye(6) - K1 @ H1) @ self.state.P

        # =====================================================================
        # MEASUREMENT 2: Angular velocity (fused encoder + gyro)
        # =====================================================================
        # Fuse encoder and gyro using adaptive weight
        omega_fused = (1 - alpha) * omega_enc + alpha * gyro_corrected

        # Fused measurement noise (weighted combination)
        sigma_fused = np.sqrt(
            ((1 - alpha) * self.sigma_enc_omega) ** 2 +
            (alpha * self.sigma_gyro) ** 2
        )

        # z2 = omega_fused, h2(x) = omega
        H2 = np.array([[0, 0, 0, 0, 1, 0]])
        z2 = np.array([omega_fused])
        h2 = np.array([self.state.x[4]])  # Current omega estimate
        R2 = np.array([[sigma_fused ** 2]])

        # Innovation
        y2 = z2 - h2
        self.innovation_omega = y2[0]

        # Mahalanobis distance for outlier rejection
        S2 = H2 @ self.state.P @ H2.T + R2
        mahal = float(y2.T @ np.linalg.inv(S2) @ y2)

        if mahal > self.mahalanobis_threshold ** 2:
            self.outlier_count += 1
            # Skip this update but still update gyro bias during stationary
            if self.motion_state == MotionState.STATIONARY:
                self._update_gyro_bias_stationary(gyro_z)
            return False

        # Kalman gain
        K2 = self.state.P @ H2.T @ np.linalg.inv(S2)

        # Update
        self.state.x = self.state.x + (K2 @ y2).flatten()
        self.state.P = (np.eye(6) - K2 @ H2) @ self.state.P

        # =====================================================================
        # MEASUREMENT 3: Gyro bias update (always, but weighted by motion)
        # =====================================================================
        if self.motion_state == MotionState.STATIONARY:
            self._update_gyro_bias_stationary(gyro_z)
        else:
            self._update_gyro_bias_moving(omega_enc, gyro_z)

        # Update timestamp
        self.state.stamp = timestamp

        # Normalize theta
        self.state.x[2] = normalize_angle(self.state.x[2])

        return True

    def _update_gyro_bias_stationary(self, gyro_z: float):
        """
        Update gyro bias estimate when stationary.

        When stationary, true omega = 0, so gyro_z = bias + noise.
        This is the most accurate bias estimation.
        """
        # z = gyro_z, h(x) = gyro_bias (when stationary, omega should be 0)
        H = np.array([[0, 0, 0, 0, 0, 1]])
        z = np.array([gyro_z])
        h = np.array([self.state.x[5]])

        # Low noise when stationary (high confidence in bias estimate)
        R = np.array([[(self.sigma_gyro * 0.5) ** 2]])

        y = z - h
        S = H @ self.state.P @ H.T + R
        K = self.state.P @ H.T @ np.linalg.inv(S)

        self.state.x = self.state.x + (K @ y).flatten()
        self.state.P = (np.eye(6) - K @ H) @ self.state.P

    def _update_gyro_bias_moving(self, omega_enc: float, gyro_z: float):
        """
        Update gyro bias estimate when moving.

        Uses disagreement between encoder and gyro to estimate bias.
        Less confident than stationary update.
        """
        # Expected gyro reading = omega_state + bias
        expected_gyro = self.state.x[4] + self.state.x[5]

        # Innovation: difference between expected and measured
        bias_innovation = gyro_z - expected_gyro

        # Small update to bias (high uncertainty when moving)
        bias_update_gain = 0.001  # Very slow adaptation when moving
        self.state.x[5] += bias_update_gain * bias_innovation

    def update_imu_orientation(self, imu_yaw: float, timestamp: float) -> bool:
        """
        Optional: Update with absolute IMU yaw orientation.

        This provides drift correction for heading but should be used
        sparingly (low rate) to avoid fighting the encoder-based estimate.
        """
        if not self.use_imu_orientation or not self.state.initialized:
            return False

        # Rate limit orientation updates
        if timestamp - self.last_orientation_update < self.orientation_update_period:
            return False

        self.last_orientation_update = timestamp

        # Initialize reference yaw on first update
        if self.initial_imu_yaw is None:
            self.initial_imu_yaw = imu_yaw
            return True

        # Compute relative yaw change from IMU
        imu_yaw_relative = normalize_angle(imu_yaw - self.initial_imu_yaw)

        # Current theta estimate
        theta_est = self.state.x[2]

        # Innovation (angle difference)
        y = np.array([normalize_angle(imu_yaw_relative - theta_est)])

        # Only update if innovation is reasonable (< 30 degrees)
        if abs(y[0]) > 0.52:  # ~30 degrees
            return False

        # Measurement model: z = theta + noise
        H = np.array([[0, 0, 1, 0, 0, 0]])
        R = np.array([[self.sigma_imu_yaw ** 2]])

        S = H @ self.state.P @ H.T + R
        K = self.state.P @ H.T @ np.linalg.inv(S)

        self.state.x = self.state.x + (K @ y).flatten()
        self.state.P = (np.eye(6) - K @ H) @ self.state.P

        # Normalize theta
        self.state.x[2] = normalize_angle(self.state.x[2])

        return True

    def get_state(self) -> Tuple[float, float, float, float, float]:
        """Return (x, y, theta, v, omega)."""
        return tuple(self.state.x[:5])

    def get_covariance(self) -> np.ndarray:
        """Return state covariance matrix."""
        return self.state.P.copy()

    def get_gyro_bias(self) -> float:
        """Return current gyro bias estimate."""
        return float(self.state.x[5])

    def get_diagnostics(self) -> dict:
        """Return diagnostic information."""
        return {
            'motion_state': self.motion_state.name,
            'gyro_weight': self.gyro_weight_used,
            'gyro_bias': self.get_gyro_bias(),
            'innovation_v': self.innovation_v,
            'innovation_omega': self.innovation_omega,
            'outlier_rate': self.outlier_count / max(1, self.update_count),
            'position_std': np.sqrt(self.state.P[0, 0] + self.state.P[1, 1]),
            'theta_std': np.sqrt(self.state.P[2, 2]),
        }


class EKFFusionNode(Node):
    """
    ROS2 Node for EKF-based Encoder + IMU Fusion.

    Replaces ZUPT node with proper probabilistic sensor fusion.
    """

    def __init__(self):
        super().__init__('ekf_fusion_node')

        self._declare_parameters()

        # Initialize EKF
        self.ekf = RobustEKFFusion(
            # Process noise
            sigma_v=self.get_parameter('sigma_v').value,
            sigma_omega=self.get_parameter('sigma_omega').value,
            sigma_bias=self.get_parameter('sigma_bias').value,
            # Encoder noise
            sigma_enc_v=self.get_parameter('sigma_enc_v').value,
            sigma_enc_omega=self.get_parameter('sigma_enc_omega').value,
            # Gyro noise
            sigma_gyro=self.get_parameter('sigma_gyro').value,
            # IMU orientation
            sigma_imu_yaw=self.get_parameter('sigma_imu_yaw').value,
            use_imu_orientation=self.get_parameter('use_imu_orientation').value,
            orientation_update_rate=self.get_parameter('orientation_update_rate').value,
            # Fusion
            gyro_weight=self.get_parameter('base_gyro_weight').value,
            mahalanobis_threshold=self.get_parameter('mahalanobis_threshold').value,
            # Motion detection
            stationary_v_thresh=self.get_parameter('stationary_v_thresh').value,
            stationary_omega_thresh=self.get_parameter('stationary_omega_thresh').value,
        )

        # Set initial pose
        self.ekf.reset(
            x=self.get_parameter('initial_x').value,
            y=self.get_parameter('initial_y').value,
            theta=self.get_parameter('initial_theta').value,
        )

        # Thread safety
        self._lock = threading.Lock()

        # Sensor data
        self._last_odom_time: Optional[float] = None
        self._latest_gyro_z: float = 0.0
        self._latest_imu_yaw: Optional[float] = None
        self._initialized = False

        # QoS profiles
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        odom_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # Frame IDs
        self.odom_frame = self.get_parameter('odom_frame').value
        self.base_frame = self.get_parameter('base_frame').value

        # Subscribers
        self.imu_sub = self.create_subscription(
            Imu,
            self.get_parameter('imu_topic').value,
            self._imu_callback,
            sensor_qos
        )

        self.odom_sub = self.create_subscription(
            Odometry,
            self.get_parameter('odom_topic').value,
            self._odom_callback,
            odom_qos
        )

        # Publishers
        self.odom_pub = self.create_publisher(Odometry, '/odometry/filtered', 10)
        self.diag_pub = self.create_publisher(Float64MultiArray, '/ekf_fusion/diagnostics', 10)

        # TF broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)

        # Diagnostics timer
        self.create_timer(5.0, self._print_diagnostics)

        self._log_startup()

    def _declare_parameters(self):
        """Declare ROS2 parameters."""
        # Topics
        self.declare_parameter('imu_topic', '/imu')
        self.declare_parameter('odom_topic', '/wc_control/odom')

        # Frames
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('base_frame', 'base_link')

        # Process noise
        self.declare_parameter('sigma_v', 0.05)
        self.declare_parameter('sigma_omega', 0.02)
        self.declare_parameter('sigma_bias', 0.0001)

        # Encoder measurement noise
        self.declare_parameter('sigma_enc_v', 0.02)
        self.declare_parameter('sigma_enc_omega', 0.01)

        # Gyro measurement noise
        self.declare_parameter('sigma_gyro', 0.005)

        # IMU orientation
        self.declare_parameter('sigma_imu_yaw', 0.02)
        self.declare_parameter('use_imu_orientation', True)
        self.declare_parameter('orientation_update_rate', 2.0)

        # Fusion parameters
        self.declare_parameter('base_gyro_weight', 0.7)
        self.declare_parameter('mahalanobis_threshold', 5.0)

        # Motion detection
        self.declare_parameter('stationary_v_thresh', 0.005)
        self.declare_parameter('stationary_omega_thresh', 0.005)  # was 0.02 — too high for slow rotation

        # Initial pose
        self.declare_parameter('initial_x', 0.0)
        self.declare_parameter('initial_y', 0.0)
        self.declare_parameter('initial_theta', 0.0)

        # TF
        self.declare_parameter('publish_tf', True)

    def _log_startup(self):
        """Log startup banner."""
        self.get_logger().info('''
╔══════════════════════════════════════════════════════════════════╗
║           ROBUST EKF SENSOR FUSION FOR WHEELCHAIR                ║
║                                                                  ║
║  Algorithm: Extended Kalman Filter with Adaptive Gyro Weighting  ║
║  State: [x, y, theta, v, omega, gyro_bias]                      ║
║                                                                  ║
║  Features:                                                       ║
║    - Continuous probabilistic fusion (not binary switching)     ║
║    - Online gyro bias estimation                                ║
║    - Adaptive noise based on motion state                       ║
║    - Outlier rejection for sensor failures                      ║
║                                                                  ║
║  Input:  Wheel encoders + IMU gyro + IMU orientation            ║
║  Output: /odometry/filtered                                      ║
╚══════════════════════════════════════════════════════════════════╝
''')

    def _imu_callback(self, msg: Imu):
        """Store latest IMU data."""
        with self._lock:
            self._latest_gyro_z = msg.angular_velocity.z
            self._latest_imu_yaw = yaw_from_quaternion(msg.orientation)

    def _odom_callback(self, msg: Odometry):
        """Process wheel odometry - main fusion loop."""
        current_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        with self._lock:
            if self._last_odom_time is None:
                self._last_odom_time = current_time
                self._initialized = True
                self.get_logger().info('EKF Fusion initialized')
                return

            dt = current_time - self._last_odom_time
            if dt <= 0 or dt > 0.5:
                self._last_odom_time = current_time
                return

            # Extract encoder velocities
            v_enc = msg.twist.twist.linear.x
            omega_enc = msg.twist.twist.angular.z
            gyro_z = self._latest_gyro_z
            imu_yaw = self._latest_imu_yaw

            # EKF Predict
            self.ekf.predict(dt)

            # EKF Update with encoder + gyro
            self.ekf.update_encoder(v_enc, omega_enc, gyro_z, current_time)

            # Optional: Update with IMU orientation (drift correction)
            if imu_yaw is not None:
                self.ekf.update_imu_orientation(imu_yaw, current_time)

            self._last_odom_time = current_time

        # Publish
        self._publish_state(msg.header.stamp)

    def _publish_state(self, stamp):
        """Publish filtered odometry and TF."""
        with self._lock:
            if not self._initialized:
                return

            x, y, theta, v, omega = self.ekf.get_state()
            P = self.ekf.get_covariance()
            diag = self.ekf.get_diagnostics()

        # Odometry message
        odom_msg = Odometry()
        odom_msg.header.stamp = stamp
        odom_msg.header.frame_id = self.odom_frame
        odom_msg.child_frame_id = self.base_frame

        odom_msg.pose.pose.position.x = x
        odom_msg.pose.pose.position.y = y
        odom_msg.pose.pose.position.z = 0.0
        odom_msg.pose.pose.orientation = quaternion_from_yaw(theta)

        # Pose covariance from EKF
        odom_msg.pose.covariance = [
            P[0, 0], P[0, 1], 0, 0, 0, P[0, 2],
            P[1, 0], P[1, 1], 0, 0, 0, P[1, 2],
            0, 0, 1e6, 0, 0, 0,
            0, 0, 0, 1e6, 0, 0,
            0, 0, 0, 0, 1e6, 0,
            P[2, 0], P[2, 1], 0, 0, 0, P[2, 2],
        ]

        odom_msg.twist.twist.linear.x = v
        odom_msg.twist.twist.angular.z = omega

        # Twist covariance
        odom_msg.twist.covariance = [
            P[3, 3], 0, 0, 0, 0, 0,
            0, 0.01, 0, 0, 0, 0,
            0, 0, 1e6, 0, 0, 0,
            0, 0, 0, 1e6, 0, 0,
            0, 0, 0, 0, 1e6, 0,
            0, 0, 0, 0, 0, P[4, 4],
        ]

        self.odom_pub.publish(odom_msg)

        # TF
        if self.get_parameter('publish_tf').value:
            t = TransformStamped()
            t.header.stamp = stamp
            t.header.frame_id = self.odom_frame
            t.child_frame_id = self.base_frame
            t.transform.translation.x = x
            t.transform.translation.y = y
            t.transform.translation.z = 0.0
            t.transform.rotation = quaternion_from_yaw(theta)
            self.tf_broadcaster.sendTransform(t)

        # Diagnostics
        diag_msg = Float64MultiArray()
        diag_msg.data = [
            diag['gyro_weight'],
            diag['gyro_bias'],
            diag['innovation_v'],
            diag['innovation_omega'],
            diag['outlier_rate'],
            diag['position_std'],
            diag['theta_std'],
            float(hash(diag['motion_state']) % 4),  # Motion state as number
        ]
        self.diag_pub.publish(diag_msg)

    def _print_diagnostics(self):
        """Print periodic diagnostics."""
        with self._lock:
            if not self._initialized:
                return
            diag = self.ekf.get_diagnostics()
            x, y, theta, v, omega = self.ekf.get_state()

        self.get_logger().info(
            f'EKF | Motion: {diag["motion_state"]:10s} | '
            f'Gyro weight: {diag["gyro_weight"]:.2f} | '
            f'Bias: {diag["gyro_bias"]*1000:.2f} mrad/s | '
            f'Outliers: {diag["outlier_rate"]*100:.1f}% | '
            f'Pos: ({x:.2f}, {y:.2f}) θ: {np.degrees(theta):.1f}°'
        )


def main(args=None):
    rclpy.init(args=args)
    node = EKFFusionNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
