#!/usr/bin/env python3
"""
IMPROVED EKF SENSOR FUSION FOR WHEELCHAIR
==========================================
Re-enables ZUPT, gyro bias calibration, and adds heading constraints
to reduce odometry drift from ~15% to <5% using only encoders + gyro.

Fixes over robust_ekf_zupt_node.py:
  1. Stationary detection with gyro cross-check (prevents false ZUPT during slow turns)
  2. Safe gyro bias calibration with stability guards
  3. Heading constraint from trajectory during straight-line motion (NEW)
  4. Zero-omega kinematic constraint during straight-line motion (NEW)
  5. Velocity-scaled process noise model
  6. Gyro health monitoring

State Vector: [x, y, theta, v, omega, gyro_bias]

All features individually toggleable via parameters for A/B testing.

Usage:
    ros2 run wheelchair_zupt improved_ekf_node --ros-args \
        -p enable_zupt:=true -p enable_heading_constraint:=true
"""

import numpy as np
import threading
from dataclasses import dataclass
from typing import Optional, Tuple
from collections import deque
from enum import Enum

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from geometry_msgs.msg import TransformStamped, Quaternion, Vector3Stamped
from std_msgs.msg import Float64MultiArray, Bool
from tf2_ros import TransformBroadcaster


# =============================================================================
# UTILITIES
# =============================================================================

def quaternion_from_yaw(yaw: float) -> Quaternion:
    q = Quaternion()
    q.w = float(np.cos(yaw / 2.0))
    q.z = float(np.sin(yaw / 2.0))
    return q


def yaw_from_quaternion(q: Quaternion) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return float(np.arctan2(siny_cosp, cosy_cosp))


def normalize_angle(angle: float) -> float:
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle


class MotionState(Enum):
    STATIONARY = 0
    MOVING_STRAIGHT = 1
    ROTATING = 2
    MIXED = 3


@dataclass
class StationaryResult:
    is_stationary: bool
    confidence: float           # 0.0 to 1.0
    is_calibration_ready: bool  # safe for bias calibration (stricter)


# =============================================================================
# IMPROVED STATIONARY DETECTOR — Fix 1: gyro cross-check
# =============================================================================

class ImprovedStationaryDetector:
    """
    Multi-sensor stationary detection: encoder + accelerometer + gyro.

    Key fix: gyro cross-check prevents false stationary during slow turns.
    The original detector only used encoder + accel, which falsely triggered
    at 0.01-0.05 rad/s turns (below encoder threshold but real rotation).
    """

    GRAVITY = 9.80665

    def __init__(self,
                 encoder_v_thresh: float = 0.005,
                 encoder_omega_thresh: float = 0.005,
                 gyro_stationary_thresh: float = 0.01,
                 accel_gravity_tolerance: float = 0.3,
                 accel_xy_thresh: float = 0.15,
                 min_stationary_samples: int = 15,
                 min_calibration_samples: int = 30,
                 hysteresis_samples: int = 5,
                 gyro_override_thresh: float = 0.08,
                 use_accel_check: bool = True):
        self.encoder_v_thresh = encoder_v_thresh
        self.encoder_omega_thresh = encoder_omega_thresh
        self.gyro_stationary_thresh = gyro_stationary_thresh
        self.accel_gravity_tolerance = accel_gravity_tolerance
        self.accel_xy_thresh = accel_xy_thresh
        self.min_stationary_samples = min_stationary_samples
        self.min_calibration_samples = min_calibration_samples
        self.hysteresis_samples = hysteresis_samples
        self.gyro_override_thresh = gyro_override_thresh
        self.use_accel_check = use_accel_check

        self.is_stationary = False
        self.stationary_count = 0
        self.moving_count = 0
        self.consecutive_stationary = 0

    def update(self, v_enc: float, omega_enc: float,
               gyro_z_corrected: float,
               accel_x: float = 0.0, accel_y: float = 0.0,
               accel_z: float = 9.81) -> StationaryResult:
        # Encoder check
        enc_ok = abs(v_enc) < self.encoder_v_thresh and \
                 abs(omega_enc) < self.encoder_omega_thresh

        # Gyro cross-check (THE KEY FIX)
        gyro_ok = abs(gyro_z_corrected) < self.gyro_stationary_thresh

        # Accelerometer check (optional)
        if self.use_accel_check:
            mag = np.sqrt(accel_x**2 + accel_y**2 + accel_z**2)
            accel_ok = abs(mag - self.GRAVITY) < self.accel_gravity_tolerance and \
                       np.sqrt(accel_x**2 + accel_y**2) < self.accel_xy_thresh
        else:
            accel_ok = True

        all_agree = enc_ok and gyro_ok and accel_ok

        # Gyro override: instant exit if gyro detects significant rotation
        if self.is_stationary and abs(gyro_z_corrected) > self.gyro_override_thresh:
            self.is_stationary = False
            self.moving_count = self.hysteresis_samples
            self.stationary_count = 0
            self.consecutive_stationary = 0
            return StationaryResult(False, 0.0, False)

        # Hysteresis state machine
        if all_agree:
            self.stationary_count += 1
            self.moving_count = 0
            self.consecutive_stationary += 1
            if not self.is_stationary and \
               self.stationary_count >= self.min_stationary_samples:
                self.is_stationary = True
        else:
            self.moving_count += 1
            self.stationary_count = 0
            self.consecutive_stationary = 0
            if self.is_stationary and \
               self.moving_count >= self.hysteresis_samples:
                self.is_stationary = False

        confidence = min(1.0, self.consecutive_stationary / self.min_calibration_samples) \
            if self.is_stationary else 0.0
        cal_ready = self.is_stationary and \
            self.consecutive_stationary >= self.min_calibration_samples

        return StationaryResult(self.is_stationary, confidence, cal_ready)


# =============================================================================
# SAFE GYRO BIAS CALIBRATOR — Fix 2: guarded calibration
# =============================================================================

class SafeGyroBiasCalibrator:
    """
    Runtime gyro bias calibration with safety guards.

    Guards against the original failure mode (absorbing real rotation as bias):
    1. Only calibrates when is_calibration_ready (30+ samples all-sensors-agree)
    2. Slow alpha (0.005 vs 0.02)
    3. Stability check: std of recent samples must be low
    4. Max bias magnitude clamp
    5. Max change per session clamp
    """

    def __init__(self,
                 initial_bias: float = 0.0,
                 alpha: float = 0.005,
                 max_bias_magnitude: float = 0.05,
                 max_bias_change: float = 0.002,
                 stability_window: int = 50,
                 max_gyro_std: float = 0.008):
        self.bias = initial_bias
        self.alpha = alpha
        self.max_bias_magnitude = max_bias_magnitude
        self.max_bias_change = max_bias_change
        self.max_gyro_std = max_gyro_std
        self.gyro_buffer = deque(maxlen=stability_window)
        self.session_start_bias = initial_bias
        self.in_session = False

    def update(self, gyro_z: float, is_calibration_ready: bool) -> Tuple[float, bool]:
        """Returns (current_bias, should_publish)."""
        if not is_calibration_ready:
            self.in_session = False
            return self.bias, False

        # Start new calibration session
        if not self.in_session:
            self.in_session = True
            self.session_start_bias = self.bias
            self.gyro_buffer.clear()

        self.gyro_buffer.append(gyro_z)

        # Need enough samples for stability check
        if len(self.gyro_buffer) < 20:
            return self.bias, False

        # Guard: check stability (low variance = truly stationary)
        std = np.std(list(self.gyro_buffer))
        if std > self.max_gyro_std:
            return self.bias, False

        # EMA update
        new_bias = self.alpha * gyro_z + (1 - self.alpha) * self.bias

        # Guard: clamp magnitude
        new_bias = np.clip(new_bias, -self.max_bias_magnitude,
                           self.max_bias_magnitude)

        # Guard: clamp change per session
        delta = new_bias - self.session_start_bias
        if abs(delta) > self.max_bias_change:
            new_bias = self.session_start_bias + np.sign(delta) * self.max_bias_change

        old_bias = self.bias
        self.bias = new_bias

        should_publish = abs(self.bias - old_bias) > 0.0005
        return self.bias, should_publish


# =============================================================================
# GYRO HEALTH MONITOR — Fix 6
# =============================================================================

class GyroHealthMonitor:
    """Detect frozen or noisy gyro sensor."""

    def __init__(self, window_size: int = 100):
        self.buffer = deque(maxlen=window_size)
        self.trust_factor = 1.0
        self.status = 'unknown'

    def add_sample(self, gyro_z: float):
        self.buffer.append(gyro_z)

    def check(self) -> Tuple[str, float]:
        if len(self.buffer) < 50:
            self.status = 'unknown'
            self.trust_factor = 1.0
            return self.status, self.trust_factor

        var = np.var(list(self.buffer))
        if var < 1e-10:
            self.status = 'frozen'
            self.trust_factor = 5.0
        elif var > 0.01:
            self.status = 'noisy'
            self.trust_factor = 3.0
        else:
            self.status = 'ok'
            self.trust_factor = 1.0

        return self.status, self.trust_factor


# =============================================================================
# IMPROVED EKF — Fixes 3, 4, 5 + re-enabled ZUPT
# =============================================================================

class ImprovedEKF:
    """
    6-state EKF with heading constraints and re-enabled ZUPT.

    State: [x, y, theta, v, omega, gyro_bias]
    New: heading constraint from trajectory, zero-omega constraint,
         velocity-scaled process noise.
    """

    def __init__(self,
                 sigma_v: float = 0.05,
                 sigma_omega: float = 0.02,
                 sigma_bias: float = 0.0001,
                 sigma_enc_v: float = 0.02,
                 sigma_enc_omega: float = 0.015,
                 sigma_gyro: float = 0.005,
                 sigma_zupt_v: float = 0.001,
                 sigma_zupt_omega: float = 0.001,
                 mahalanobis_threshold: float = 5.0,
                 # Heading constraint params (Fix 3)
                 heading_sigma: float = 0.15,
                 heading_min_v: float = 0.05,
                 heading_max_omega: float = 0.02,
                 heading_min_disp: float = 0.01,
                 heading_warmup: int = 5,
                 heading_cooldown: int = 10,
                 # Straight-line constraint params (Fix 4)
                 straight_omega_sigma: float = 0.008,
                 straight_min_v: float = 0.05,
                 straight_max_omega_enc: float = 0.005):
        # Store params
        self.sigma_v = sigma_v
        self.sigma_omega = sigma_omega
        self.sigma_bias = sigma_bias
        self.sigma_enc_v = sigma_enc_v
        self.sigma_enc_omega = sigma_enc_omega
        self.sigma_gyro = sigma_gyro
        self.sigma_zupt_v = sigma_zupt_v
        self.sigma_zupt_omega = sigma_zupt_omega
        self.mahalanobis_threshold = mahalanobis_threshold

        # Heading constraint
        self.heading_sigma = heading_sigma
        self.heading_min_v = heading_min_v
        self.heading_max_omega = heading_max_omega
        self.heading_min_disp = heading_min_disp
        self.heading_warmup = heading_warmup
        self.heading_cooldown = heading_cooldown

        # Straight-line constraint
        self.straight_omega_sigma = straight_omega_sigma
        self.straight_min_v = straight_min_v
        self.straight_max_omega_enc = straight_max_omega_enc

        # State
        self.x = np.zeros(6)  # [x, y, theta, v, omega, gyro_bias]
        self.P = np.diag([0.001, 0.001, 0.001, 0.01, 0.01, 0.0001])
        self.initialized = False
        self.motion_state = MotionState.STATIONARY
        self.gyro_weight = 0.5

        # Heading constraint state
        self._prev_x = 0.0
        self._prev_y = 0.0
        self._straight_count = 0
        self._post_turn_cooldown = 0

    def initialize(self, x: float = 0.0, y: float = 0.0, theta: float = 0.0):
        self.x = np.array([x, y, theta, 0.0, 0.0, 0.0])
        self.P = np.diag([0.001, 0.001, 0.001, 0.01, 0.01, 0.0001])
        self.initialized = True
        self._prev_x = x
        self._prev_y = y

    def set_gyro_bias(self, bias: float):
        self.x[5] = bias

    # --- Process noise (Fix 5) ---
    def _build_Q(self, dt: float) -> np.ndarray:
        v = abs(self.x[3])
        omega = abs(self.x[4])

        if self.motion_state == MotionState.STATIONARY:
            v_factor = 0.1
            theta_factor = 0.1
        elif self.motion_state == MotionState.ROTATING:
            v_factor = 1.0
            theta_factor = 1.0 + 4.0 * min(omega / 0.15, 1.0)  # up to 5x
        elif self.motion_state == MotionState.MOVING_STRAIGHT:
            v_factor = max(0.3, v / 0.25)
            theta_factor = 0.5  # heading should be stable
        else:  # MIXED
            v_factor = max(0.5, v / 0.25)
            theta_factor = 1.0 + 2.0 * min(omega / 0.15, 1.0)  # up to 3x

        return np.diag([
            (self.sigma_v * dt * v_factor) ** 2,
            (self.sigma_v * dt * v_factor) ** 2,
            (self.sigma_omega * dt * theta_factor) ** 2,
            (self.sigma_v * v_factor) ** 2,
            (self.sigma_omega * theta_factor) ** 2,
            self.sigma_bias ** 2,
        ])

    # --- Predict ---
    def predict(self, dt: float):
        if not self.initialized:
            return
        x, y, theta, v, omega, bias = self.x
        self.x[0] = x + v * np.cos(theta) * dt
        self.x[1] = y + v * np.sin(theta) * dt
        self.x[2] = normalize_angle(theta + omega * dt)

        F = np.eye(6)
        F[0, 2] = -v * np.sin(theta) * dt
        F[0, 3] = np.cos(theta) * dt
        F[1, 2] = v * np.cos(theta) * dt
        F[1, 3] = np.sin(theta) * dt
        F[2, 4] = dt

        Q = self._build_Q(dt)
        self.P = F @ self.P @ F.T + Q

    # --- Motion state ---
    def _get_motion_state(self, v_enc: float, omega_enc: float,
                          gyro_corrected: float) -> MotionState:
        is_moving = abs(v_enc) > 0.01
        effective_omega = max(abs(omega_enc), abs(gyro_corrected))
        is_rotating = effective_omega > 0.03

        if not is_moving and not is_rotating:
            return MotionState.STATIONARY
        elif is_moving and not is_rotating:
            return MotionState.MOVING_STRAIGHT
        elif not is_moving and is_rotating:
            return MotionState.ROTATING
        else:
            return MotionState.MIXED

    def _compute_gyro_weight(self, motion: MotionState, omega_enc: float,
                             gyro_corrected: float) -> float:
        if motion == MotionState.STATIONARY:
            return 0.1
        elif motion == MotionState.ROTATING:
            return 0.85
        elif motion == MotionState.MOVING_STRAIGHT:
            gyro_disagree = abs(gyro_corrected) > 0.05 and abs(omega_enc) < 0.03
            return 0.65 if gyro_disagree else 0.3
        else:  # MIXED
            effective = max(abs(omega_enc), abs(gyro_corrected))
            return 0.4 + 0.4 * min(effective / 0.5, 1.0)

    # --- EKF update helper ---
    def _ekf_update(self, H: np.ndarray, z: np.ndarray, R: np.ndarray,
                    check_mahalanobis: bool = False) -> bool:
        y = z - H @ self.x
        # Normalize angle innovations
        if H.shape[0] == 1 and H[0, 2] == 1:
            y[0] = normalize_angle(y[0])

        S = H @ self.P @ H.T + R
        if check_mahalanobis:
            mahal = float(y.T @ np.linalg.solve(S, y))
            if mahal > self.mahalanobis_threshold ** 2:
                return False

        K = self.P @ H.T @ np.linalg.solve(S, np.eye(S.shape[0]))
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ H) @ self.P
        self.P = 0.5 * (self.P + self.P.T)
        self.x[2] = normalize_angle(self.x[2])
        return True

    # --- ZUPT update (re-enabled) ---
    def update_zupt(self):
        H = np.array([[0, 0, 0, 1, 0, 0],
                       [0, 0, 0, 0, 1, 0]], dtype=float)
        z = np.array([0.0, 0.0])
        R = np.diag([self.sigma_zupt_v ** 2, self.sigma_zupt_omega ** 2])
        self._ekf_update(H, z, R)
        self.x[3] = 0.0
        self.x[4] = 0.0

    # --- Encoder + gyro update ---
    def update_encoder_gyro(self, v_enc: float, omega_enc: float,
                            gyro_z: float, gyro_bias: float,
                            gyro_trust_factor: float = 1.0):
        gyro_corrected = gyro_z - gyro_bias

        self.motion_state = self._get_motion_state(v_enc, omega_enc, gyro_corrected)
        alpha = self._compute_gyro_weight(self.motion_state, omega_enc, gyro_corrected)
        self.gyro_weight = alpha

        # Linear velocity update
        H_v = np.array([[0, 0, 0, 1, 0, 0]], dtype=float)
        z_v = np.array([v_enc])
        R_v = np.array([[self.sigma_enc_v ** 2]])
        self._ekf_update(H_v, z_v, R_v)

        # Fused angular velocity update
        omega_fused = (1 - alpha) * omega_enc + alpha * gyro_corrected
        sigma_fused = np.sqrt(
            ((1 - alpha) * self.sigma_enc_omega) ** 2 +
            (alpha * self.sigma_gyro * gyro_trust_factor) ** 2
        )
        H_w = np.array([[0, 0, 0, 0, 1, 0]], dtype=float)
        z_w = np.array([omega_fused])
        R_w = np.array([[sigma_fused ** 2]])
        self._ekf_update(H_w, z_w, R_w, check_mahalanobis=True)

    # --- Fix 3: Heading constraint from trajectory ---
    def update_heading_constraint(self, v_enc: float, omega_enc: float) -> bool:
        """Soft heading correction during straight-line motion."""
        is_straight = abs(v_enc) > self.heading_min_v and \
                      abs(omega_enc) < self.heading_max_omega

        # Cooldown after turns
        if not is_straight:
            self._straight_count = 0
            if self._post_turn_cooldown < self.heading_cooldown:
                self._post_turn_cooldown += 1
            # Save current position for next straight segment
            self._prev_x = self.x[0]
            self._prev_y = self.x[1]
            return False

        if self._post_turn_cooldown < self.heading_cooldown:
            self._post_turn_cooldown += 1
            self._prev_x = self.x[0]
            self._prev_y = self.x[1]
            return False

        self._straight_count += 1

        if self._straight_count < self.heading_warmup:
            self._prev_x = self.x[0]
            self._prev_y = self.x[1]
            return False

        # Compute heading from position delta
        dx = self.x[0] - self._prev_x
        dy = self.x[1] - self._prev_y
        disp = np.sqrt(dx ** 2 + dy ** 2)

        if disp < self.heading_min_disp:
            return False

        theta_traj = np.arctan2(dy, dx)

        # Account for reverse motion
        if v_enc < 0:
            theta_traj = normalize_angle(theta_traj + np.pi)

        # Inject as soft measurement
        H = np.array([[0, 0, 1, 0, 0, 0]], dtype=float)
        z = np.array([theta_traj])
        R = np.array([[self.heading_sigma ** 2]])

        # Clamp innovation to prevent large jumps
        innovation = normalize_angle(theta_traj - self.x[2])
        if abs(innovation) > 0.05:  # ~3 degrees max per cycle
            return False

        result = self._ekf_update(H, z, R)

        # Update reference position
        self._prev_x = self.x[0]
        self._prev_y = self.x[1]

        return result

    # --- Fix 4: Zero-omega constraint during straight line ---
    def update_straight_constraint(self, v_enc: float, omega_enc: float) -> bool:
        """Differential drive constraint: equal wheel speed → omega = 0."""
        if abs(v_enc) < self.straight_min_v or \
           abs(omega_enc) > self.straight_max_omega_enc:
            return False

        H = np.array([[0, 0, 0, 0, 1, 0]], dtype=float)
        z = np.array([0.0])
        R = np.array([[self.straight_omega_sigma ** 2]])
        return self._ekf_update(H, z, R)

    # --- Getters ---
    def get_state(self) -> Tuple[float, float, float, float, float]:
        return self.x[0], self.x[1], self.x[2], self.x[3], self.x[4]

    def get_covariance(self) -> np.ndarray:
        return self.P.copy()

    def get_diagnostics(self) -> dict:
        return {
            'motion_state': self.motion_state.value,
            'gyro_weight': self.gyro_weight,
            'gyro_bias': self.x[5],
            'P_theta': self.P[2, 2],
            'straight_count': self._straight_count,
        }


# =============================================================================
# ROS2 NODE
# =============================================================================

class ImprovedEKFNode(Node):
    def __init__(self):
        super().__init__('improved_ekf_node')
        self._declare_parameters()

        # Feature toggles
        self.enable_zupt = self.get_parameter('enable_zupt').value
        self.enable_bias_cal = self.get_parameter('enable_bias_calibration').value
        self.enable_heading = self.get_parameter('enable_heading_constraint').value
        self.enable_straight = self.get_parameter('enable_straight_omega_constraint').value
        self.enable_health = self.get_parameter('enable_gyro_health_monitor').value

        # Stationary detector
        self.stationary = ImprovedStationaryDetector(
            encoder_v_thresh=self.get_parameter('stationary_v_thresh').value,
            encoder_omega_thresh=self.get_parameter('stationary_omega_thresh').value,
            gyro_stationary_thresh=self.get_parameter('gyro_stationary_thresh').value,
            accel_gravity_tolerance=self.get_parameter('accel_gravity_tolerance').value,
            accel_xy_thresh=self.get_parameter('accel_xy_thresh').value,
            min_stationary_samples=self.get_parameter('min_stationary_samples').value,
            min_calibration_samples=self.get_parameter('min_calibration_samples').value,
            hysteresis_samples=self.get_parameter('hysteresis_samples').value,
            gyro_override_thresh=self.get_parameter('gyro_override_thresh').value,
        )

        # Bias calibrator
        self.bias_cal = SafeGyroBiasCalibrator(
            initial_bias=self.get_parameter('initial_gyro_bias').value,
            alpha=self.get_parameter('bias_alpha').value,
            max_bias_magnitude=self.get_parameter('bias_max_magnitude').value,
            max_bias_change=self.get_parameter('bias_max_change').value,
            stability_window=self.get_parameter('bias_stability_window').value,
            max_gyro_std=self.get_parameter('bias_max_gyro_std').value,
        )

        # Gyro health
        self.gyro_health = GyroHealthMonitor(
            window_size=self.get_parameter('gyro_health_window').value,
        )

        # EKF
        self.ekf = ImprovedEKF(
            sigma_v=self.get_parameter('sigma_v').value,
            sigma_omega=self.get_parameter('sigma_omega').value,
            sigma_bias=self.get_parameter('sigma_bias').value,
            sigma_enc_v=self.get_parameter('sigma_enc_v').value,
            sigma_enc_omega=self.get_parameter('sigma_enc_omega').value,
            sigma_gyro=self.get_parameter('sigma_gyro').value,
            sigma_zupt_v=self.get_parameter('sigma_zupt_v').value,
            sigma_zupt_omega=self.get_parameter('sigma_zupt_omega').value,
            mahalanobis_threshold=self.get_parameter('mahalanobis_threshold').value,
            heading_sigma=self.get_parameter('heading_constraint_sigma').value,
            heading_min_v=self.get_parameter('heading_constraint_min_v').value,
            heading_max_omega=self.get_parameter('heading_constraint_max_omega').value,
            heading_min_disp=self.get_parameter('heading_constraint_min_disp').value,
            heading_warmup=self.get_parameter('heading_constraint_warmup').value,
            heading_cooldown=self.get_parameter('heading_constraint_cooldown').value,
            straight_omega_sigma=self.get_parameter('straight_omega_sigma').value,
            straight_min_v=self.get_parameter('straight_min_v').value,
            straight_max_omega_enc=self.get_parameter('straight_max_omega_enc').value,
        )
        self.ekf.initialize(
            x=self.get_parameter('initial_x').value,
            y=self.get_parameter('initial_y').value,
            theta=self.get_parameter('initial_theta').value,
        )

        # Thread safety
        self._lock = threading.Lock()
        self._latest_gyro_z: float = 0.0
        self._latest_accel = np.array([0.0, 0.0, 9.81])
        self._last_odom_time: Optional[float] = None
        self._initialized = False

        # QoS
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST, depth=10)
        latched_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST, depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL)

        # Subscribers
        self.create_subscription(
            Odometry, self.get_parameter('odom_topic').value,
            self._odom_cb, sensor_qos)
        self.create_subscription(
            Imu, self.get_parameter('imu_topic').value,
            self._imu_cb, sensor_qos)
        self.create_subscription(
            Vector3Stamped, '/imu/calibrated_bias',
            self._bias_cb, latched_qos)

        # Publishers
        self.odom_pub = self.create_publisher(Odometry, '/odometry/filtered', 10)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.stat_pub = self.create_publisher(Bool, '/ekf_improved/stationary', 10)
        self.diag_pub = self.create_publisher(
            Float64MultiArray, '/ekf_improved/diagnostics', 10)
        self.bias_pub = self.create_publisher(
            Vector3Stamped, '/imu/calibrated_bias', latched_qos)

        # Gyro health timer
        if self.enable_health:
            interval = self.get_parameter('gyro_health_check_interval').value
            self.create_timer(interval, self._health_check_cb)

        # Diagnostic timer
        self.create_timer(2.0, self._diag_timer_cb)

        # Feature summary
        features = []
        if self.enable_zupt: features.append('ZUPT')
        if self.enable_bias_cal: features.append('BiasCal')
        if self.enable_heading: features.append('HeadingConstraint')
        if self.enable_straight: features.append('StraightConstraint')
        if self.enable_health: features.append('GyroHealth')

        self.get_logger().info(
            f'╔══════════════════════════════════════════════════╗')
        self.get_logger().info(
            f'║    IMPROVED EKF SENSOR FUSION                   ║')
        self.get_logger().info(
            f'║    Features: {", ".join(features):<36}║')
        self.get_logger().info(
            f'╚══════════════════════════════════════════════════╝')

    def _declare_parameters(self):
        # Topics
        self.declare_parameter('imu_topic', '/imu')
        self.declare_parameter('odom_topic', '/wc_control/odom')
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('publish_tf', True)

        # Stationary detection
        self.declare_parameter('stationary_v_thresh', 0.005)
        self.declare_parameter('stationary_omega_thresh', 0.005)
        self.declare_parameter('gyro_stationary_thresh', 0.01)
        self.declare_parameter('accel_gravity_tolerance', 0.3)
        self.declare_parameter('accel_xy_thresh', 0.15)
        self.declare_parameter('min_stationary_samples', 15)
        self.declare_parameter('min_calibration_samples', 30)
        self.declare_parameter('hysteresis_samples', 5)
        self.declare_parameter('gyro_override_thresh', 0.08)

        # Bias calibration
        self.declare_parameter('initial_gyro_bias', 0.0)
        self.declare_parameter('bias_alpha', 0.005)
        self.declare_parameter('bias_max_magnitude', 0.05)
        self.declare_parameter('bias_max_change', 0.002)
        self.declare_parameter('bias_stability_window', 50)
        self.declare_parameter('bias_max_gyro_std', 0.008)

        # EKF noise
        self.declare_parameter('sigma_v', 0.05)
        self.declare_parameter('sigma_omega', 0.02)
        self.declare_parameter('sigma_bias', 0.0001)
        self.declare_parameter('sigma_enc_v', 0.02)
        self.declare_parameter('sigma_enc_omega', 0.015)
        self.declare_parameter('sigma_gyro', 0.005)
        self.declare_parameter('sigma_zupt_v', 0.001)
        self.declare_parameter('sigma_zupt_omega', 0.001)
        self.declare_parameter('mahalanobis_threshold', 5.0)

        # Heading constraint (Fix 3)
        self.declare_parameter('heading_constraint_sigma', 0.15)
        self.declare_parameter('heading_constraint_min_v', 0.05)
        self.declare_parameter('heading_constraint_max_omega', 0.02)
        self.declare_parameter('heading_constraint_min_disp', 0.01)
        self.declare_parameter('heading_constraint_warmup', 5)
        self.declare_parameter('heading_constraint_cooldown', 10)

        # Straight-line constraint (Fix 4)
        self.declare_parameter('straight_omega_sigma', 0.008)
        self.declare_parameter('straight_min_v', 0.05)
        self.declare_parameter('straight_max_omega_enc', 0.005)

        # Gyro health (Fix 6)
        self.declare_parameter('gyro_health_window', 100)
        self.declare_parameter('gyro_health_check_interval', 5.0)

        # Feature toggles
        self.declare_parameter('enable_zupt', True)
        self.declare_parameter('enable_bias_calibration', True)
        self.declare_parameter('enable_heading_constraint', True)
        self.declare_parameter('enable_straight_omega_constraint', True)
        self.declare_parameter('enable_gyro_health_monitor', True)

        # Initial pose
        self.declare_parameter('initial_x', 0.0)
        self.declare_parameter('initial_y', 0.0)
        self.declare_parameter('initial_theta', 0.0)

    # --- Callbacks ---

    def _imu_cb(self, msg: Imu):
        with self._lock:
            self._latest_gyro_z = msg.angular_velocity.z
            self._latest_accel = np.array([
                msg.linear_acceleration.x,
                msg.linear_acceleration.y,
                msg.linear_acceleration.z,
            ])

    def _bias_cb(self, msg: Vector3Stamped):
        external_bias = msg.vector.z
        with self._lock:
            if abs(external_bias) < 0.1:
                self.bias_cal.bias = external_bias
                self.ekf.set_gyro_bias(external_bias)
                self.get_logger().info(
                    f'External bias received: {external_bias*1000:.3f} mrad/s')

    def _odom_cb(self, msg: Odometry):
        stamp = msg.header.stamp
        current_time = stamp.sec + stamp.nanosec * 1e-9

        if not self._initialized:
            self._last_odom_time = current_time
            self._initialized = True
            return

        dt = current_time - self._last_odom_time
        if dt <= 0 or dt > 0.5:
            self._last_odom_time = current_time
            return

        v_enc = msg.twist.twist.linear.x
        omega_enc = msg.twist.twist.angular.z

        with self._lock:
            gyro_z = self._latest_gyro_z
            accel = self._latest_accel.copy()

            # Gyro health monitoring
            if self.enable_health:
                self.gyro_health.add_sample(gyro_z)

            # Get bias
            gyro_bias = self.bias_cal.bias
            gyro_corrected = gyro_z - gyro_bias

            # Stationary detection (with gyro cross-check)
            stat = self.stationary.update(
                v_enc, omega_enc, gyro_corrected,
                accel[0], accel[1], accel[2])

            # EKF predict
            self.ekf.predict(dt)

            # EKF update
            if stat.is_stationary and self.enable_zupt:
                self.ekf.update_zupt()
            else:
                self.ekf.update_encoder_gyro(
                    v_enc, omega_enc, gyro_z, gyro_bias,
                    self.gyro_health.trust_factor if self.enable_health else 1.0)

                if self.enable_straight:
                    self.ekf.update_straight_constraint(v_enc, omega_enc)

                if self.enable_heading:
                    self.ekf.update_heading_constraint(v_enc, omega_enc)

            # Bias calibration
            if self.enable_bias_cal and stat.is_calibration_ready:
                bias, should_pub = self.bias_cal.update(gyro_z, True)
                self.ekf.set_gyro_bias(bias)
                if should_pub:
                    self._publish_bias(bias, stamp)
            elif self.enable_bias_cal:
                self.bias_cal.update(gyro_z, False)

            self._last_odom_time = current_time

            # Get state
            x, y, theta, v, omega = self.ekf.get_state()
            P = self.ekf.get_covariance()

        # NaN guard
        if np.isnan(x) or np.isnan(y) or np.isnan(theta):
            self.get_logger().error('NaN in EKF state! Reinitializing.')
            with self._lock:
                self.ekf.initialize()
            return

        # Publish
        self._publish_odom(stamp, x, y, theta, v, omega, P, stat.is_stationary)

    def _publish_odom(self, stamp, x, y, theta, v, omega, P, is_stationary):
        odom_frame = self.get_parameter('odom_frame').value
        base_frame = self.get_parameter('base_frame').value

        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = odom_frame
        odom.child_frame_id = base_frame
        odom.pose.pose.position.x = x
        odom.pose.pose.position.y = y
        odom.pose.pose.orientation = quaternion_from_yaw(theta)

        odom.pose.covariance[0] = P[0, 0]
        odom.pose.covariance[7] = P[1, 1]
        odom.pose.covariance[35] = P[2, 2]

        odom.twist.twist.linear.x = v
        odom.twist.twist.angular.z = omega
        odom.twist.covariance[0] = P[3, 3]
        odom.twist.covariance[35] = P[4, 4]

        self.odom_pub.publish(odom)

        # TF
        if self.get_parameter('publish_tf').value:
            t = TransformStamped()
            t.header.stamp = stamp
            t.header.frame_id = odom_frame
            t.child_frame_id = base_frame
            t.transform.translation.x = x
            t.transform.translation.y = y
            q = quaternion_from_yaw(theta)
            t.transform.rotation = q
            self.tf_broadcaster.sendTransform(t)

        # Stationary status
        stat_msg = Bool()
        stat_msg.data = is_stationary
        self.stat_pub.publish(stat_msg)

    def _publish_bias(self, bias: float, stamp):
        msg = Vector3Stamped()
        msg.header.stamp = stamp
        msg.header.frame_id = 'base_link'
        msg.vector.z = bias
        self.bias_pub.publish(msg)
        self.get_logger().info(f'Published bias: {bias*1000:.3f} mrad/s')

    def _health_check_cb(self):
        status, factor = self.gyro_health.check()
        if status != 'ok':
            self.get_logger().warn(
                f'Gyro health: {status} (trust_factor={factor:.1f})')

    def _diag_timer_cb(self):
        with self._lock:
            diag = self.ekf.get_diagnostics()
            stat_conf = self.stationary.consecutive_stationary

        msg = Float64MultiArray()
        msg.data = [
            float(diag['motion_state']),
            diag['gyro_weight'],
            diag['gyro_bias'] * 1000,  # mrad/s
            diag['P_theta'],
            float(stat_conf),
            float(diag['straight_count']),
            self.gyro_health.trust_factor,
            self.bias_cal.bias * 1000,  # mrad/s
        ]
        self.diag_pub.publish(msg)

        x, y, theta, v, omega = self.ekf.get_state()
        self.get_logger().info(
            f'pos=({x:.2f},{y:.2f}) θ={np.degrees(theta):.1f}° '
            f'v={v:.3f} ω={omega:.3f} '
            f'bias={self.bias_cal.bias*1000:.2f}mrad/s '
            f'α={diag["gyro_weight"]:.2f} '
            f'stat={stat_conf} '
            f'gyro={self.gyro_health.status}')


def main(args=None):
    rclpy.init(args=args)
    node = ImprovedEKFNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
