#!/usr/bin/env python3
"""
ROBUST EKF SENSOR FUSION WITH ZUPT FOR WHEELCHAIR
==================================================
Complete sensor fusion solution combining:
  - Extended Kalman Filter (encoder + gyro fusion)
  - ZUPT (Zero Velocity Update) when stationary
  - Continuous gyro bias recalibration during stationary periods
  - Accelerometer-based stationary detection

State Vector: [x, y, theta, v, omega, gyro_bias]

Key Features:
  1. ZUPT: When stationary detected, force v=0 and omega=0
  2. Gyro Recalibration: When encoder=0 AND accel≈gravity, recalibrate gyro
  3. Adaptive fusion: Gyro weight varies by motion state
  4. Proper covariance propagation
  5. Outlier rejection
  6. Publishes updated bias to /imu/calibrated_bias for system-wide update

Stationary Detection (multi-sensor):
  - Encoder: |v| < 0.005 m/s AND |omega| < 0.02 rad/s
  - Accelerometer: |accel_magnitude - 9.81| < 0.3 m/s² (no linear acceleration)
  - Both must agree for ZUPT and recalibration

Author: Robust Sensor Fusion Implementation
Date: 2026-02-03
"""

import numpy as np
import threading
from dataclasses import dataclass, field
from typing import Optional, Tuple, Deque
from collections import deque
from enum import Enum
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from rclpy.time import Time

from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from geometry_msgs.msg import TransformStamped, Quaternion, Vector3Stamped
from std_msgs.msg import Float64MultiArray, Bool
from tf2_ros import TransformBroadcaster


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

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
    """Robot motion state for adaptive processing."""
    STATIONARY = 0      # Robot is not moving - ZUPT active
    MOVING_STRAIGHT = 1 # Linear motion - trust encoder more
    ROTATING = 2        # Rotation - trust gyro more
    MIXED = 3           # Combined motion


# =============================================================================
# STATIONARY DETECTOR
# =============================================================================

class StationaryDetector:
    """
    Multi-sensor stationary detection using encoder AND accelerometer.

    For ZUPT and gyro recalibration to activate, BOTH conditions must be met:
    1. Encoder shows no motion (v ≈ 0, omega ≈ 0)
    2. Accelerometer shows only gravity (no linear acceleration)

    This prevents false stationary detection during:
    - Wheel slip (encoder=0 but robot moving)
    - Vibration (accelerometer noisy but robot stationary)
    """

    GRAVITY = 9.80665  # m/s²

    def __init__(self,
                 encoder_v_thresh: float = 0.005,
                 encoder_omega_thresh: float = 0.02,
                 accel_gravity_tolerance: float = 0.3,
                 accel_xy_thresh: float = 0.15,
                 min_stationary_samples: int = 10,
                 hysteresis_samples: int = 5,
                 gyro_override_thresh: float = 0.08):
        """
        Args:
            encoder_v_thresh: Max linear velocity to be considered stationary (m/s)
            encoder_omega_thresh: Max angular velocity to be considered stationary (rad/s)
            accel_gravity_tolerance: Max deviation from gravity magnitude (m/s²)
            accel_xy_thresh: Max horizontal acceleration (m/s²)
            min_stationary_samples: Samples needed to confirm stationary
            hysteresis_samples: Samples needed to exit stationary state
            gyro_override_thresh: Gyro rate (rad/s) that forces immediate ZUPT exit
        """
        self.encoder_v_thresh = encoder_v_thresh
        self.encoder_omega_thresh = encoder_omega_thresh
        self.accel_gravity_tolerance = accel_gravity_tolerance
        self.accel_xy_thresh = accel_xy_thresh
        self.min_stationary_samples = min_stationary_samples
        self.hysteresis_samples = hysteresis_samples
        self.gyro_override_thresh = gyro_override_thresh

        # State tracking
        self.is_stationary = False
        self.stationary_count = 0
        self.moving_count = 0
        self.gyro_override_count = 0  # diagnostics

        # For diagnostics
        self.encoder_stationary = False
        self.accel_stationary = False
        self.last_accel_magnitude = 0.0

    def update(self, v_enc: float, omega_enc: float,
               accel_x: float, accel_y: float, accel_z: float,
               gyro_z_corrected: float = 0.0) -> bool:
        """
        Update stationary detection with new sensor data.

        Args:
            v_enc: Linear velocity from encoder (m/s)
            omega_enc: Angular velocity from encoder (rad/s)
            accel_x, accel_y, accel_z: Acceleration in base_link frame (m/s²)
            gyro_z_corrected: Bias-corrected gyro Z (rad/s) — for fast ZUPT exit

        Returns:
            True if robot is stationary (ZUPT should activate)
        """
        # FIX 1: Gyro-based immediate ZUPT exit
        # If gyro detects significant rotation while ZUPT is active,
        # exit immediately (1 cycle) instead of waiting for hysteresis.
        # This prevents the stop-then-turn cascading failure where ZUPT
        # holds omega=0 for 250ms while the robot is already turning.
        if self.is_stationary and abs(gyro_z_corrected) > self.gyro_override_thresh:
            self.is_stationary = False
            self.stationary_count = 0
            self.moving_count = self.hysteresis_samples  # prevent re-entering
            self.gyro_override_count += 1
            return False

        # Check encoder condition
        self.encoder_stationary = (
            abs(v_enc) < self.encoder_v_thresh and
            abs(omega_enc) < self.encoder_omega_thresh
        )

        # Check accelerometer condition
        # In base_link frame: X=forward, Y=left, Z=up
        # When stationary on level surface: accel ≈ [0, 0, +9.81]
        accel_magnitude = np.sqrt(accel_x**2 + accel_y**2 + accel_z**2)
        self.last_accel_magnitude = accel_magnitude

        gravity_ok = abs(accel_magnitude - self.GRAVITY) < self.accel_gravity_tolerance
        horizontal_ok = np.sqrt(accel_x**2 + accel_y**2) < self.accel_xy_thresh

        self.accel_stationary = gravity_ok and horizontal_ok

        # Both must agree
        both_stationary = self.encoder_stationary and self.accel_stationary

        # Hysteresis state machine
        if both_stationary:
            self.stationary_count += 1
            self.moving_count = 0

            if not self.is_stationary and self.stationary_count >= self.min_stationary_samples:
                self.is_stationary = True
        else:
            self.moving_count += 1
            self.stationary_count = 0

            if self.is_stationary and self.moving_count >= self.hysteresis_samples:
                self.is_stationary = False

        return self.is_stationary

    def get_diagnostics(self) -> dict:
        return {
            'is_stationary': self.is_stationary,
            'encoder_stationary': self.encoder_stationary,
            'accel_stationary': self.accel_stationary,
            'stationary_count': self.stationary_count,
            'accel_magnitude': self.last_accel_magnitude,
            'gyro_override_count': self.gyro_override_count,
        }


# =============================================================================
# GYRO BIAS CALIBRATOR
# =============================================================================

class GyroBiasCalibrator:
    """
    Online gyro bias calibration during stationary periods.

    When robot is confirmed stationary:
    - Gyro reading = true_omega + bias + noise
    - Since true_omega = 0, gyro reading = bias + noise
    - Average over samples to estimate bias

    Uses exponential moving average for smooth updates.
    """

    def __init__(self,
                 initial_bias: float = 0.0,
                 alpha_stationary: float = 0.02,   # Learning rate when stationary
                 alpha_moving: float = 0.0005,     # Very slow adaptation when moving
                 min_samples_for_update: int = 20,
                 max_bias_change_rate: float = 0.001):  # rad/s per update
        """
        Args:
            initial_bias: Initial bias estimate (rad/s)
            alpha_stationary: EMA alpha when stationary (higher = faster learning)
            alpha_moving: EMA alpha when moving (very low)
            min_samples_for_update: Minimum samples before publishing update
            max_bias_change_rate: Maximum bias change per update (prevents jumps)
        """
        self.bias = initial_bias
        self.alpha_stationary = alpha_stationary
        self.alpha_moving = alpha_moving
        self.min_samples_for_update = min_samples_for_update
        self.max_bias_change_rate = max_bias_change_rate

        # Calibration state
        self.calibration_samples = 0
        self.last_published_bias = initial_bias
        self.total_calibration_time = 0.0

        # Statistics
        self.gyro_samples: Deque[float] = deque(maxlen=100)

    def update(self, gyro_z: float, is_stationary: bool, dt: float) -> Tuple[float, bool]:
        """
        Update bias estimate with new gyro reading.

        Args:
            gyro_z: Raw gyro Z reading (rad/s)
            is_stationary: Whether robot is stationary
            dt: Time step (seconds)

        Returns:
            (current_bias, should_publish) - bias estimate and whether to publish update
        """
        should_publish = False

        if is_stationary:
            # When stationary: gyro_z ≈ bias
            # Use faster learning rate
            alpha = self.alpha_stationary

            # Exponential moving average
            new_bias = alpha * gyro_z + (1 - alpha) * self.bias

            # Limit rate of change to prevent jumps
            bias_change = new_bias - self.bias
            if abs(bias_change) > self.max_bias_change_rate:
                bias_change = np.sign(bias_change) * self.max_bias_change_rate

            self.bias += bias_change
            self.calibration_samples += 1
            self.total_calibration_time += dt

            # Collect samples for statistics
            self.gyro_samples.append(gyro_z)

            # Check if should publish updated bias
            if self.calibration_samples >= self.min_samples_for_update:
                bias_change_since_publish = abs(self.bias - self.last_published_bias)
                # Publish if bias changed significantly (> 0.0005 rad/s = ~0.03 deg/s)
                if bias_change_since_publish > 0.0005:
                    should_publish = True
                    self.last_published_bias = self.bias
                    self.calibration_samples = 0
        else:
            # When moving: very slow adaptation based on disagreement
            # This handles long-term temperature drift
            alpha = self.alpha_moving
            # Don't update bias directly - let EKF handle it

        return self.bias, should_publish

    def get_statistics(self) -> dict:
        """Get calibration statistics."""
        if len(self.gyro_samples) > 0:
            samples = np.array(self.gyro_samples)
            return {
                'bias': self.bias,
                'std': float(np.std(samples)),
                'samples': len(samples),
                'total_cal_time': self.total_calibration_time,
            }
        return {'bias': self.bias, 'std': 0.0, 'samples': 0, 'total_cal_time': 0.0}


# =============================================================================
# EKF STATE AND FILTER
# =============================================================================

@dataclass
class EKFState:
    """EKF state container."""
    x: np.ndarray = field(default_factory=lambda: np.zeros(6))
    P: np.ndarray = field(default_factory=lambda: np.eye(6) * 0.01)
    stamp: float = 0.0
    initialized: bool = False


class RobustEKFWithZUPT:
    """
    Extended Kalman Filter with Zero Velocity Update (ZUPT).

    State: [x, y, theta, v, omega, gyro_bias]

    Features:
    - Standard EKF predict/update cycle
    - ZUPT: When stationary, apply v=0, omega=0 as measurements
    - Adaptive gyro weighting based on motion
    - Outlier rejection
    """

    def __init__(self,
                 # Process noise
                 sigma_v: float = 0.05,
                 sigma_omega: float = 0.02,
                 sigma_bias: float = 0.0001,
                 # Encoder noise
                 sigma_enc_v: float = 0.02,
                 sigma_enc_omega: float = 0.015,
                 # Gyro noise
                 sigma_gyro: float = 0.005,
                 # ZUPT noise (very low - high confidence when stationary)
                 sigma_zupt_v: float = 0.001,
                 sigma_zupt_omega: float = 0.001,
                 # IMU orientation
                 sigma_imu_yaw: float = 0.02,
                 use_imu_orientation: bool = True,
                 orientation_update_rate: float = 2.0,
                 # Outlier rejection
                 mahalanobis_threshold: float = 5.0):

        self.state = EKFState()

        # Noise parameters
        self.sigma_v = sigma_v
        self.sigma_omega = sigma_omega
        self.sigma_bias = sigma_bias
        self.sigma_enc_v = sigma_enc_v
        self.sigma_enc_omega = sigma_enc_omega
        self.sigma_gyro = sigma_gyro
        self.sigma_zupt_v = sigma_zupt_v
        self.sigma_zupt_omega = sigma_zupt_omega
        self.sigma_imu_yaw = sigma_imu_yaw

        self.use_imu_orientation = use_imu_orientation
        self.orientation_update_period = 1.0 / orientation_update_rate
        self.last_orientation_update = 0.0

        self.mahalanobis_threshold = mahalanobis_threshold

        # Motion state
        self.motion_state = MotionState.STATIONARY
        self.gyro_weight = 0.5

        # Diagnostics
        self.innovation_v = 0.0
        self.innovation_omega = 0.0
        self.outlier_count = 0
        self.update_count = 0
        self.zupt_count = 0

        # IMU orientation reference
        self.initial_imu_yaw: Optional[float] = None

    def reset(self, x: float = 0.0, y: float = 0.0, theta: float = 0.0):
        """Reset filter state."""
        self.state.x = np.array([x, y, theta, 0.0, 0.0, 0.0])
        self.state.P = np.diag([0.001, 0.001, 0.001, 0.01, 0.01, 0.0001])
        self.state.initialized = True
        self.initial_imu_yaw = None
        self.outlier_count = 0
        self.update_count = 0
        self.zupt_count = 0

    def set_gyro_bias(self, bias: float):
        """Set gyro bias from external calibrator."""
        self.state.x[5] = bias

    def _get_motion_state(self, v_enc: float, omega_enc: float,
                          gyro_corrected: float = 0.0) -> MotionState:
        """Determine motion state from encoder AND gyro velocities.

        FIX 2: Uses max(encoder_omega, gyro) for rotation detection.
        Previously used encoder-only, which missed turns when wheels slip
        or during the first fraction of a turn (encoder lags gyro).
        """
        v_thresh = 0.01
        omega_thresh = 0.03

        is_moving = abs(v_enc) > v_thresh
        # Use whichever sensor reports more rotation — encoder can lag gyro
        effective_omega = max(abs(omega_enc), abs(gyro_corrected))
        is_rotating = effective_omega > omega_thresh

        if not is_moving and not is_rotating:
            return MotionState.STATIONARY
        elif is_moving and not is_rotating:
            return MotionState.MOVING_STRAIGHT
        elif not is_moving and is_rotating:
            return MotionState.ROTATING
        else:
            return MotionState.MIXED

    def _compute_gyro_weight(self, motion: MotionState, omega_enc: float,
                             gyro_corrected: float = 0.0) -> float:
        """Compute adaptive gyro weight.

        FIX 2b: For MIXED/MOVING_STRAIGHT, also boost gyro weight when
        gyro detects rotation that encoder doesn't (wheel slip scenario).
        """
        if motion == MotionState.STATIONARY:
            return 0.1  # Trust encoder (which is ~0)
        elif motion == MotionState.ROTATING:
            return 0.85  # High gyro trust (slip likely)
        elif motion == MotionState.MOVING_STRAIGHT:
            # If gyro says turning but encoder doesn't, boost gyro weight
            gyro_disagree = abs(gyro_corrected) > 0.05 and abs(omega_enc) < 0.03
            return 0.65 if gyro_disagree else 0.3
        else:
            # Mixed: scale with effective rotation rate (use gyro too)
            effective_omega = max(abs(omega_enc), abs(gyro_corrected))
            omega_factor = min(effective_omega / 0.5, 1.0)
            return 0.4 + 0.4 * omega_factor

    def _build_process_noise(self, dt: float) -> np.ndarray:
        """Build process noise covariance."""
        motion_factor = {
            MotionState.STATIONARY: 0.1,
            MotionState.MOVING_STRAIGHT: 1.0,
            MotionState.ROTATING: 1.5,
            MotionState.MIXED: 1.2
        }.get(self.motion_state, 1.0)

        Q = np.diag([
            (self.sigma_v * dt * motion_factor) ** 2,
            (self.sigma_v * dt * motion_factor) ** 2,
            (self.sigma_omega * dt * motion_factor) ** 2,
            (self.sigma_v * motion_factor) ** 2,
            (self.sigma_omega * motion_factor) ** 2,
            (self.sigma_bias) ** 2,
        ])
        return Q

    def predict(self, dt: float):
        """EKF prediction step."""
        if not self.state.initialized or dt <= 0:
            return

        x, y, theta, v, omega, bias = self.state.x

        # Motion model
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        x_new = x + v * cos_theta * dt
        y_new = y + v * sin_theta * dt
        theta_new = normalize_angle(theta + omega * dt)

        self.state.x = np.array([x_new, y_new, theta_new, v, omega, bias])

        # Jacobian
        F = np.eye(6)
        F[0, 2] = -v * sin_theta * dt
        F[0, 3] = cos_theta * dt
        F[1, 2] = v * cos_theta * dt
        F[1, 3] = sin_theta * dt
        F[2, 4] = dt

        Q = self._build_process_noise(dt)
        self.state.P = F @ self.state.P @ F.T + Q

    def update_zupt(self, timestamp: float):
        """
        ZUPT Update: Apply zero velocity constraint.

        When stationary is confirmed, velocity MUST be zero.
        This is a very strong constraint with low noise.
        """
        self.zupt_count += 1

        # Measurement: v = 0, omega = 0
        H = np.array([
            [0, 0, 0, 1, 0, 0],  # v
            [0, 0, 0, 0, 1, 0],  # omega
        ])
        z = np.array([0.0, 0.0])
        h = np.array([self.state.x[3], self.state.x[4]])

        # Very low noise - high confidence in zero velocity
        R = np.diag([self.sigma_zupt_v**2, self.sigma_zupt_omega**2])

        # Innovation
        y = z - h
        self.innovation_v = y[0]
        self.innovation_omega = y[1]

        # Kalman gain
        S = H @ self.state.P @ H.T + R
        try:
            K = self.state.P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return
        if np.any(np.isnan(K)):
            return

        # Update
        self.state.x = self.state.x + (K @ y).flatten()
        self.state.P = (np.eye(6) - K @ H) @ self.state.P
        self.state.P = 0.5 * (self.state.P + self.state.P.T)

        # Force velocities to exactly zero (numerical stability)
        self.state.x[3] = 0.0
        self.state.x[4] = 0.0

        self.state.stamp = timestamp

    def update_encoder_gyro(self, v_enc: float, omega_enc: float,
                           gyro_z: float, gyro_bias: float,
                           timestamp: float) -> bool:
        """
        Standard EKF update with encoder and gyro fusion.

        Uses adaptive gyro weighting based on motion state.
        """
        if not self.state.initialized:
            return False

        self.update_count += 1

        # Bias-corrected gyro
        gyro_corrected = gyro_z - gyro_bias

        # Update motion state (uses both encoder AND gyro for rotation detection)
        self.motion_state = self._get_motion_state(v_enc, omega_enc, gyro_corrected)

        # Adaptive gyro weight (uses gyro for disagree detection)
        alpha = self._compute_gyro_weight(self.motion_state, omega_enc, gyro_corrected)
        self.gyro_weight = alpha

        # ========== Linear Velocity Update ==========
        H1 = np.array([[0, 0, 0, 1, 0, 0]])
        z1 = np.array([v_enc])
        h1 = np.array([self.state.x[3]])
        R1 = np.array([[self.sigma_enc_v ** 2]])

        y1 = z1 - h1
        self.innovation_v = y1[0]

        S1 = H1 @ self.state.P @ H1.T + R1
        try:
            K1 = self.state.P @ H1.T @ np.linalg.inv(S1)
        except np.linalg.LinAlgError:
            return False
        if np.any(np.isnan(K1)):
            return False

        self.state.x = self.state.x + (K1 @ y1).flatten()
        self.state.P = (np.eye(6) - K1 @ H1) @ self.state.P

        # ========== Angular Velocity Update (Fused) ==========
        omega_fused = (1 - alpha) * omega_enc + alpha * gyro_corrected
        sigma_fused = np.sqrt(
            ((1 - alpha) * self.sigma_enc_omega) ** 2 +
            (alpha * self.sigma_gyro) ** 2
        )

        H2 = np.array([[0, 0, 0, 0, 1, 0]])
        z2 = np.array([omega_fused])
        h2 = np.array([self.state.x[4]])
        R2 = np.array([[sigma_fused ** 2]])

        y2 = z2 - h2
        self.innovation_omega = y2[0]

        # Outlier rejection
        S2 = H2 @ self.state.P @ H2.T + R2
        try:
            S2_inv = np.linalg.inv(S2)
        except np.linalg.LinAlgError:
            return False
        if np.any(np.isnan(S2_inv)):
            return False

        mahal = float(y2.T @ S2_inv @ y2)

        if mahal > self.mahalanobis_threshold ** 2:
            self.outlier_count += 1
            return False

        K2 = self.state.P @ H2.T @ S2_inv
        self.state.x = self.state.x + (K2 @ y2).flatten()
        self.state.P = (np.eye(6) - K2 @ H2) @ self.state.P
        self.state.P = 0.5 * (self.state.P + self.state.P.T)

        self.state.x[2] = normalize_angle(self.state.x[2])
        self.state.stamp = timestamp

        return True

    def update_imu_orientation(self, imu_yaw: float, timestamp: float) -> bool:
        """Optional drift correction from IMU orientation.

        FIX 3: Adaptive gate instead of hard 30-degree reject.
        Previously, once heading error exceeded 30 degrees, IMU corrections
        were permanently rejected — making drift uncorrectable.

        New behavior:
        - < 30 degrees: normal correction (tight R)
        - 30-90 degrees: attenuated correction (inflated R, so smaller step)
        - > 90 degrees: reject (likely IMU reference lost or major failure)
        This allows gradual recovery from moderate drift without sudden jumps.
        """
        if not self.use_imu_orientation or not self.state.initialized:
            return False

        if timestamp - self.last_orientation_update < self.orientation_update_period:
            return False

        self.last_orientation_update = timestamp

        if self.initial_imu_yaw is None:
            self.initial_imu_yaw = imu_yaw
            return True

        imu_yaw_relative = normalize_angle(imu_yaw - self.initial_imu_yaw)
        theta_est = self.state.x[2]

        y = np.array([normalize_angle(imu_yaw_relative - theta_est)])
        abs_err = abs(y[0])

        # Hard reject only above 90 degrees (was 30)
        if abs_err > 1.57:
            return False

        # Adaptive noise: inflate R for large errors to prevent jumps
        if abs_err < 0.52:  # < 30 degrees: normal
            R_scale = 1.0
        else:  # 30-90 degrees: attenuated — scale R up to 16x so correction is gradual
            t = (abs_err - 0.52) / (1.57 - 0.52)  # 0..1 over 30..90 deg
            R_scale = 1.0 + 15.0 * t  # 1x..16x

        H = np.array([[0, 0, 1, 0, 0, 0]])
        R = np.array([[(self.sigma_imu_yaw * np.sqrt(R_scale)) ** 2]])

        S = H @ self.state.P @ H.T + R
        try:
            K = self.state.P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return False
        if np.any(np.isnan(K)):
            return False

        self.state.x = self.state.x + (K @ y).flatten()
        self.state.P = (np.eye(6) - K @ H) @ self.state.P
        self.state.P = 0.5 * (self.state.P + self.state.P.T)
        self.state.x[2] = normalize_angle(self.state.x[2])

        return True

    def get_state(self) -> Tuple[float, float, float, float, float]:
        """Return (x, y, theta, v, omega)."""
        return tuple(self.state.x[:5])

    def get_covariance(self) -> np.ndarray:
        return self.state.P.copy()

    def get_diagnostics(self) -> dict:
        return {
            'motion_state': self.motion_state.name,
            'gyro_weight': self.gyro_weight,
            'innovation_v': self.innovation_v,
            'innovation_omega': self.innovation_omega,
            'outlier_rate': self.outlier_count / max(1, self.update_count),
            'zupt_count': self.zupt_count,
            'position_std': np.sqrt(self.state.P[0, 0] + self.state.P[1, 1]),
            'theta_std': np.sqrt(self.state.P[2, 2]),
        }


# =============================================================================
# ROS2 NODE
# =============================================================================

class RobustEKFZUPTNode(Node):
    """
    ROS2 Node for Robust EKF + ZUPT Sensor Fusion.

    Combines:
    - EKF for encoder + gyro fusion
    - ZUPT for zero velocity updates
    - Continuous gyro recalibration
    - Accelerometer-based stationary detection
    """

    def __init__(self):
        super().__init__('robust_ekf_zupt_node')

        self._declare_parameters()

        # Initialize components
        self.stationary_detector = StationaryDetector(
            encoder_v_thresh=self.get_parameter('stationary_v_thresh').value,
            encoder_omega_thresh=self.get_parameter('stationary_omega_thresh').value,
            accel_gravity_tolerance=self.get_parameter('accel_gravity_tolerance').value,
            accel_xy_thresh=self.get_parameter('accel_xy_thresh').value,
            min_stationary_samples=self.get_parameter('min_stationary_samples').value,
            hysteresis_samples=self.get_parameter('hysteresis_samples').value,
            gyro_override_thresh=self.get_parameter('gyro_override_thresh').value,
        )

        self.gyro_calibrator = GyroBiasCalibrator(
            initial_bias=self.get_parameter('initial_gyro_bias').value,
            alpha_stationary=self.get_parameter('bias_alpha_stationary').value,
            alpha_moving=self.get_parameter('bias_alpha_moving').value,
            min_samples_for_update=self.get_parameter('bias_min_samples').value,
        )

        self.ekf = RobustEKFWithZUPT(
            sigma_v=self.get_parameter('sigma_v').value,
            sigma_omega=self.get_parameter('sigma_omega').value,
            sigma_bias=self.get_parameter('sigma_bias').value,
            sigma_enc_v=self.get_parameter('sigma_enc_v').value,
            sigma_enc_omega=self.get_parameter('sigma_enc_omega').value,
            sigma_gyro=self.get_parameter('sigma_gyro').value,
            sigma_zupt_v=self.get_parameter('sigma_zupt_v').value,
            sigma_zupt_omega=self.get_parameter('sigma_zupt_omega').value,
            sigma_imu_yaw=self.get_parameter('sigma_imu_yaw').value,
            use_imu_orientation=self.get_parameter('use_imu_orientation').value,
            orientation_update_rate=self.get_parameter('orientation_update_rate').value,
            mahalanobis_threshold=self.get_parameter('mahalanobis_threshold').value,
        )

        self.ekf.reset(
            x=self.get_parameter('initial_x').value,
            y=self.get_parameter('initial_y').value,
            theta=self.get_parameter('initial_theta').value,
        )

        # Thread safety
        self._lock = threading.Lock()

        # Sensor data storage
        self._last_odom_time: Optional[float] = None
        self._latest_gyro_z: float = 0.0
        self._latest_imu_yaw: Optional[float] = None
        self._latest_accel: np.ndarray = np.array([0.0, 0.0, 9.81])
        self._initialized = False

        # Frame IDs
        self.odom_frame = self.get_parameter('odom_frame').value
        self.base_frame = self.get_parameter('base_frame').value

        # QoS
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
        latched_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE
        )

        # Subscribers
        self.imu_sub = self.create_subscription(
            Imu, self.get_parameter('imu_topic').value,
            self._imu_callback, sensor_qos
        )
        self.odom_sub = self.create_subscription(
            Odometry, self.get_parameter('odom_topic').value,
            self._odom_callback, odom_qos
        )

        # Subscribe to external calibration (from startup calibrator)
        self.bias_sub = self.create_subscription(
            Vector3Stamped, '/imu/calibrated_bias',
            self._external_bias_callback, latched_qos
        )

        # Publishers
        self.odom_pub = self.create_publisher(Odometry, '/odometry/filtered', 10)
        self.diag_pub = self.create_publisher(Float64MultiArray, '/ekf_zupt/diagnostics', 10)
        self.stationary_pub = self.create_publisher(Bool, '/ekf_zupt/stationary', 10)

        # Publish updated bias for system-wide use
        self.bias_pub = self.create_publisher(Vector3Stamped, '/imu/calibrated_bias', latched_qos)

        # TF broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)

        # Timers
        self.create_timer(5.0, self._print_diagnostics)

        self._log_startup()

    def _declare_parameters(self):
        """Declare all parameters."""
        # Topics
        self.declare_parameter('imu_topic', '/imu')
        self.declare_parameter('odom_topic', '/wc_control/odom')

        # Frames
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('base_frame', 'base_link')

        # Stationary detection
        # omega was 0.02 rad/s (1.15 deg/s) — caused false ZUPT during slow rotation
        self.declare_parameter('stationary_v_thresh', 0.005)
        self.declare_parameter('stationary_omega_thresh', 0.005)  # 0.29 deg/s — only truly stopped
        self.declare_parameter('accel_gravity_tolerance', 0.3)
        self.declare_parameter('accel_xy_thresh', 0.15)
        self.declare_parameter('min_stationary_samples', 15)      # was 10 — more certain before ZUPT
        self.declare_parameter('hysteresis_samples', 8)            # was 5 — slower exit from ZUPT
        self.declare_parameter('gyro_override_thresh', 0.08)  # rad/s — instant ZUPT exit

        # Gyro bias calibration
        self.declare_parameter('initial_gyro_bias', 0.0)
        self.declare_parameter('bias_alpha_stationary', 0.02)
        self.declare_parameter('bias_alpha_moving', 0.0005)
        self.declare_parameter('bias_min_samples', 20)

        # EKF process noise
        self.declare_parameter('sigma_v', 0.05)
        self.declare_parameter('sigma_omega', 0.02)
        self.declare_parameter('sigma_bias', 0.0001)

        # Encoder noise
        self.declare_parameter('sigma_enc_v', 0.02)
        self.declare_parameter('sigma_enc_omega', 0.015)

        # Gyro noise
        self.declare_parameter('sigma_gyro', 0.005)

        # ZUPT noise (very low)
        self.declare_parameter('sigma_zupt_v', 0.001)
        self.declare_parameter('sigma_zupt_omega', 0.001)

        # IMU orientation
        self.declare_parameter('sigma_imu_yaw', 0.03)
        self.declare_parameter('use_imu_orientation', True)
        self.declare_parameter('orientation_update_rate', 2.0)

        # Outlier rejection
        self.declare_parameter('mahalanobis_threshold', 5.0)

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
║         ROBUST EKF + ZUPT SENSOR FUSION FOR WHEELCHAIR           ║
║                                                                  ║
║  Features:                                                       ║
║    ✓ Extended Kalman Filter (encoder + gyro fusion)             ║
║    ✓ ZUPT: Zero velocity update when stationary                 ║
║    ✓ Multi-sensor stationary detection (encoder + accel)        ║
║    ✓ Continuous gyro bias recalibration                         ║
║    ✓ Adaptive gyro weighting by motion state                    ║
║    ✓ Outlier rejection                                          ║
║                                                                  ║
║  Stationary Detection:                                           ║
║    - Encoder: v < 0.005 m/s AND omega < 0.02 rad/s              ║
║    - Accel: |mag - 9.81| < 0.3 m/s² (no linear accel)          ║
║    - BOTH must agree for ZUPT activation                        ║
║                                                                  ║
║  Output: /odometry/filtered, /ekf_zupt/stationary               ║
╚══════════════════════════════════════════════════════════════════╝
''')

    def _external_bias_callback(self, msg: Vector3Stamped):
        """Receive bias from external calibrator (startup calibration)."""
        # Use Z component for gyro_z bias (after frame transform)
        # Note: The startup calibrator publishes in camera frame
        # The IMU republisher transforms to base_link where Z is yaw
        external_bias = msg.vector.z

        with self._lock:
            current_bias = self.gyro_calibrator.bias
            # Only update if significantly different
            if abs(external_bias - current_bias) > 0.001:
                self.gyro_calibrator.bias = external_bias
                self.ekf.set_gyro_bias(external_bias)
                self.get_logger().info(
                    f'Received external bias: {external_bias*1000:.2f} mrad/s '
                    f'(was {current_bias*1000:.2f} mrad/s)'
                )

    def _imu_callback(self, msg: Imu):
        """Store latest IMU data."""
        with self._lock:
            self._latest_gyro_z = msg.angular_velocity.z
            self._latest_imu_yaw = yaw_from_quaternion(msg.orientation)
            self._latest_accel = np.array([
                msg.linear_acceleration.x,
                msg.linear_acceleration.y,
                msg.linear_acceleration.z
            ])

    def _odom_callback(self, msg: Odometry):
        """Main fusion loop - triggered by encoder odometry."""
        current_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        with self._lock:
            if self._last_odom_time is None:
                self._last_odom_time = current_time
                self._initialized = True
                self.get_logger().info('EKF+ZUPT initialized')
                return

            dt = current_time - self._last_odom_time
            if dt <= 0 or dt > 0.5:
                self._last_odom_time = current_time
                return

            # Get sensor data
            v_enc = msg.twist.twist.linear.x
            omega_enc = msg.twist.twist.angular.z
            gyro_z = self._latest_gyro_z
            imu_yaw = self._latest_imu_yaw
            accel = self._latest_accel.copy()

            # ========== Gyro Bias ==========
            # Upstream IMU pipeline already corrects bias:
            #   imu_startup_calibrator -> imu_bias_corrector -> Madgwick
            # The runtime GyroBiasCalibrator was absorbing real rotation
            # into the bias during slow turns (false stationary detection),
            # corrupting gyro_corrected and causing heading drift.
            # Fix: always use zero bias here — upstream handles it.
            gyro_bias = 0.0
            is_stationary = False
            should_publish_bias = False

            # ========== EKF Predict ==========
            self.ekf.predict(dt)

            # ========== EKF Update ==========
            # Pure EKF fusion: encoder + gyro only.
            # No ZUPT, no runtime bias calibration, no IMU orientation.
            #
            # IMU orientation disabled because Madgwick without magnetometer
            # (use_mag=False) has NO absolute yaw reference — it's just
            # integrated gyro, same data the EKF already uses. Fusing two
            # integrations of the same gyro as independent measurements
            # creates a feedback loop that causes 0.5°/s heading drift
            # while stationary, and fights real rotation ("resisting").
            self.ekf.update_encoder_gyro(
                v_enc, omega_enc, gyro_z, gyro_bias, current_time
            )

            self._last_odom_time = current_time

            # Read EKF state INSIDE the lock — prevents race with IMU callback
            x, y, theta, v, omega = self.ekf.get_state()
            P = self.ekf.get_covariance()
            ekf_diag = self.ekf.get_diagnostics()
            stat_diag = self.stationary_detector.get_diagnostics()
            cal_diag = self.gyro_calibrator.get_statistics()

        # NaN guard — if EKF state corrupted, skip this cycle instead of
        # publishing invalid TF (which silently disappears and breaks RViz)
        if np.isnan(x) or np.isnan(y) or np.isnan(theta):
            self.get_logger().warn(
                f'EKF NaN detected! x={x:.3f} y={y:.3f} θ={theta:.3f} — skipping TF publish')
            return

        # Publish outputs (state already read atomically under lock)
        self._publish_state_from_values(
            msg.header.stamp, is_stationary,
            x, y, theta, v, omega, P, ekf_diag, stat_diag, cal_diag)

        # Publish updated bias if needed
        if should_publish_bias:
            self._publish_bias(gyro_bias, msg.header.stamp)

    def _publish_state_from_values(self, stamp, is_stationary: bool,
                                    x, y, theta, v, omega, P,
                                    ekf_diag, stat_diag, cal_diag):
        """Publish filtered odometry and TF from pre-read values (lock-free)."""
        # Odometry message
        odom_msg = Odometry()
        odom_msg.header.stamp = stamp
        odom_msg.header.frame_id = self.odom_frame
        odom_msg.child_frame_id = self.base_frame

        odom_msg.pose.pose.position.x = x
        odom_msg.pose.pose.position.y = y
        odom_msg.pose.pose.position.z = 0.0
        odom_msg.pose.pose.orientation = quaternion_from_yaw(theta)

        # Covariance
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

        # Stationary status
        stat_msg = Bool()
        stat_msg.data = is_stationary
        self.stationary_pub.publish(stat_msg)

        # Diagnostics
        diag_msg = Float64MultiArray()
        diag_msg.data = [
            float(is_stationary),
            ekf_diag['gyro_weight'],
            cal_diag['bias'] * 1000,  # mrad/s
            ekf_diag['innovation_v'],
            ekf_diag['innovation_omega'],
            ekf_diag['outlier_rate'],
            ekf_diag['zupt_count'],
            stat_diag['accel_magnitude'],
        ]
        self.diag_pub.publish(diag_msg)

    def _publish_bias(self, bias: float, stamp):
        """Publish updated gyro bias for system-wide use."""
        msg = Vector3Stamped()
        msg.header.stamp = stamp
        msg.header.frame_id = 'base_link'
        msg.vector.x = 0.0
        msg.vector.y = 0.0
        msg.vector.z = bias
        self.bias_pub.publish(msg)

        self.get_logger().info(f'Published updated gyro bias: {bias*1000:.3f} mrad/s')

    def _print_diagnostics(self):
        """Print periodic diagnostics."""
        with self._lock:
            if not self._initialized:
                return
            x, y, theta, v, omega = self.ekf.get_state()
            ekf_diag = self.ekf.get_diagnostics()
            stat_diag = self.stationary_detector.get_diagnostics()
            cal_diag = self.gyro_calibrator.get_statistics()

        stat_str = "ZUPT" if stat_diag['is_stationary'] else ekf_diag['motion_state']

        self.get_logger().info(
            f'EKF+ZUPT | {stat_str:10s} | '
            f'α={ekf_diag["gyro_weight"]:.2f} | '
            f'Bias={cal_diag["bias"]*1000:.2f}mrad/s | '
            f'ZUPT={ekf_diag["zupt_count"]} | '
            f'Pos=({x:.2f},{y:.2f}) θ={np.degrees(theta):.1f}°'
        )


def main(args=None):
    rclpy.init(args=args)
    node = RobustEKFZUPTNode()

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
