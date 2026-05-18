#!/usr/bin/env python3
"""
IMU DIAGNOSTIC TOOL - Static Test Analysis
==========================================
Based on IMU Pre-integration and Gravity Compensation theory.

This script performs a comprehensive IMU health check during a static test:
1. Gravity alignment verification
2. Gyro bias estimation and drift analysis
3. Madgwick filter tracking verification
4. Noise characterization (Allan variance proxy)
5. EKF fusion quality assessment

Usage:
    # Terminal 1: Launch the wheelchair system
    ros2 launch wheelchair_bringup wheelchair_fusion_nav.launch.py \
        map_name:=/home/sidd/wheelchair_nav/maps/my_map_final_cleaned.yaml

    # Terminal 2: Run this diagnostic (keep robot STATIONARY for 30 sec)
    ros2 run scripts imu_diagnostic

Author: IMU Pre-integration Analysis Tool
"""

import os
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSPresetProfiles
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Vector3Stamped
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from datetime import datetime
from pathlib import Path
import csv
import time


@dataclass
class IMUReading:
    """Single IMU reading with all components."""
    timestamp: float
    orientation: np.ndarray  # quaternion [x, y, z, w]
    angular_velocity: np.ndarray  # [wx, wy, wz] rad/s
    linear_acceleration: np.ndarray  # [ax, ay, az] m/s²


@dataclass
class DiagnosticResults:
    """Results from IMU diagnostic analysis."""
    # Gravity
    gravity_magnitude: float = 0.0
    gravity_error_percent: float = 0.0
    gravity_vector_body: np.ndarray = field(default_factory=lambda: np.zeros(3))
    tilt_from_level: float = 0.0  # degrees

    # Gyro Bias
    gyro_bias_mean: np.ndarray = field(default_factory=lambda: np.zeros(3))
    gyro_bias_std: np.ndarray = field(default_factory=lambda: np.zeros(3))
    gyro_drift_deg_per_min: np.ndarray = field(default_factory=lambda: np.zeros(3))

    # Accel Bias
    accel_bias_mean: np.ndarray = field(default_factory=lambda: np.zeros(3))  # After gravity removal
    accel_bias_std: np.ndarray = field(default_factory=lambda: np.zeros(3))

    # Madgwick Analysis
    quaternion_yaw_change: float = 0.0  # degrees
    integrated_gyro_yaw: float = 0.0  # degrees
    madgwick_tracking_ratio: float = 0.0
    madgwick_healthy: bool = False

    # Noise Characteristics
    gyro_noise_density: float = 0.0  # rad/s/√Hz
    accel_noise_density: float = 0.0  # m/s²/√Hz

    # Position Drift (from EKF)
    position_drift_mm: float = 0.0
    velocity_drift_mm_s: float = 0.0

    # Overall Health
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


def quat_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Convert quaternion [x,y,z,w] to 3x3 rotation matrix."""
    x, y, z, w = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
    ])


def quat_to_yaw(q: np.ndarray) -> float:
    """Extract yaw (Z rotation) from quaternion [x,y,z,w]."""
    x, y, z, w = q
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return np.arctan2(siny_cosp, cosy_cosp)


def normalize_angle(angle: float) -> float:
    """Normalize angle to [-pi, pi]."""
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle


class IMUDiagnosticNode(Node):
    """
    ROS2 node for comprehensive IMU diagnostic during static test.
    """

    def __init__(self):
        super().__init__('imu_diagnostic')

        # Parameters
        self.declare_parameter('test_duration', 30.0)  # seconds
        self.declare_parameter('gravity_magnitude', 9.81)  # m/s²
        # Default output to ~/wheelchair_nav_logs (portable across systems)
        default_log_dir = os.path.join(os.path.expanduser('~'), 'wheelchair_nav_logs')
        self.declare_parameter('output_dir', default_log_dir)

        self.test_duration = self.get_parameter('test_duration').value
        self.expected_gravity = self.get_parameter('gravity_magnitude').value
        self.output_dir = Path(self.get_parameter('output_dir').value)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Data storage
        self.imu_data: List[IMUReading] = []  # /imu (after Madgwick + republisher)
        self.raw_cam_imu_data: List[IMUReading] = []  # /camera/imu (raw)
        self.raw_odom_data: List[Tuple[float, np.ndarray, np.ndarray]] = []  # (t, pos, vel)
        self.ekf_odom_data: List[Tuple[float, np.ndarray, np.ndarray]] = []  # (t, pos, vel)

        # Timing
        self.start_time: Optional[float] = None
        self.test_complete = False

        # QoS for sensor data
        qos = QoSPresetProfiles.SENSOR_DATA.value

        # Subscribers
        self.create_subscription(Imu, '/imu', self._imu_callback, qos)
        self.create_subscription(Imu, '/camera/imu', self._raw_cam_imu_callback, qos)
        self.create_subscription(Odometry, '/wc_control/odom', self._raw_odom_callback, qos)
        self.create_subscription(Odometry, '/odometry/filtered', self._ekf_odom_callback, qos)

        # Timer for progress updates
        self.progress_timer = self.create_timer(5.0, self._progress_callback)

        self.get_logger().info("=" * 70)
        self.get_logger().info("IMU DIAGNOSTIC TOOL - STATIC TEST")
        self.get_logger().info("=" * 70)
        self.get_logger().info(f"Test duration: {self.test_duration} seconds")
        self.get_logger().info("IMPORTANT: Keep the robot COMPLETELY STATIONARY!")
        self.get_logger().info("=" * 70)
        self.get_logger().info("Waiting for IMU data...")

    def _get_time(self) -> float:
        """Get current ROS time as float."""
        return self.get_clock().now().nanoseconds / 1e9

    def _imu_callback(self, msg: Imu):
        """Process /imu messages (after Madgwick + republisher)."""
        if self.test_complete:
            return

        now = self._get_time()

        if self.start_time is None:
            self.start_time = now
            self.get_logger().info(f"Test started at {now:.2f}")

        reading = IMUReading(
            timestamp=now,
            orientation=np.array([msg.orientation.x, msg.orientation.y,
                                  msg.orientation.z, msg.orientation.w]),
            angular_velocity=np.array([msg.angular_velocity.x, msg.angular_velocity.y,
                                       msg.angular_velocity.z]),
            linear_acceleration=np.array([msg.linear_acceleration.x, msg.linear_acceleration.y,
                                          msg.linear_acceleration.z])
        )
        self.imu_data.append(reading)

        # Check if test is complete
        if now - self.start_time >= self.test_duration:
            self._complete_test()

    def _raw_cam_imu_callback(self, msg: Imu):
        """Process /camera/imu messages (raw from RealSense)."""
        if self.test_complete or self.start_time is None:
            return

        now = self._get_time()
        reading = IMUReading(
            timestamp=now,
            orientation=np.array([0, 0, 0, 1]),  # Raw doesn't have orientation
            angular_velocity=np.array([msg.angular_velocity.x, msg.angular_velocity.y,
                                       msg.angular_velocity.z]),
            linear_acceleration=np.array([msg.linear_acceleration.x, msg.linear_acceleration.y,
                                          msg.linear_acceleration.z])
        )
        self.raw_cam_imu_data.append(reading)

    def _raw_odom_callback(self, msg: Odometry):
        """Process /wc_control/odom messages."""
        if self.test_complete or self.start_time is None:
            return

        now = self._get_time()
        pos = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y,
                        msg.pose.pose.position.z])
        vel = np.array([msg.twist.twist.linear.x, msg.twist.twist.linear.y,
                        msg.twist.twist.linear.z])
        self.raw_odom_data.append((now, pos, vel))

    def _ekf_odom_callback(self, msg: Odometry):
        """Process /odometry/filtered messages."""
        if self.test_complete or self.start_time is None:
            return

        now = self._get_time()
        pos = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y,
                        msg.pose.pose.position.z])
        vel = np.array([msg.twist.twist.linear.x, msg.twist.twist.linear.y,
                        msg.twist.twist.linear.z])
        self.ekf_odom_data.append((now, pos, vel))

    def _progress_callback(self):
        """Print progress during test."""
        if self.start_time is None or self.test_complete:
            return

        elapsed = self._get_time() - self.start_time
        remaining = self.test_duration - elapsed

        if remaining > 0:
            self.get_logger().info(
                f"Progress: {elapsed:.0f}s / {self.test_duration:.0f}s | "
                f"IMU samples: {len(self.imu_data)} | "
                f"Raw CAM IMU: {len(self.raw_cam_imu_data)}"
            )

    def _complete_test(self):
        """Finalize test and run analysis."""
        self.test_complete = True
        self.progress_timer.cancel()

        self.get_logger().info("=" * 70)
        self.get_logger().info("TEST COMPLETE - ANALYZING DATA...")
        self.get_logger().info("=" * 70)

        # Run analysis
        results = self._analyze_data()

        # Save raw data
        self._save_raw_data()

        # Print results
        self._print_results(results)

        # Generate fix recommendations
        self._generate_fixes(results)

        self.get_logger().info("=" * 70)
        self.get_logger().info("DIAGNOSTIC COMPLETE")
        self.get_logger().info("=" * 70)

        # Shutdown
        rclpy.shutdown()

    def _analyze_data(self) -> DiagnosticResults:
        """Perform comprehensive analysis on collected data."""
        results = DiagnosticResults()

        if len(self.imu_data) < 100:
            results.issues.append("CRITICAL: Not enough IMU data collected!")
            return results

        # Convert to numpy arrays for analysis
        timestamps = np.array([r.timestamp for r in self.imu_data])
        orientations = np.array([r.orientation for r in self.imu_data])
        gyros = np.array([r.angular_velocity for r in self.imu_data])
        accels = np.array([r.linear_acceleration for r in self.imu_data])

        # Time deltas
        dt = np.diff(timestamps)
        avg_dt = np.mean(dt)
        imu_rate = 1.0 / avg_dt if avg_dt > 0 else 0

        self.get_logger().info(f"IMU Rate: {imu_rate:.1f} Hz, Samples: {len(self.imu_data)}")

        # ====================================================================
        # 1. GRAVITY ANALYSIS
        # ====================================================================
        results.gravity_vector_body = np.mean(accels, axis=0)
        results.gravity_magnitude = np.linalg.norm(results.gravity_vector_body)
        results.gravity_error_percent = abs(results.gravity_magnitude - self.expected_gravity) / self.expected_gravity * 100

        # Tilt from level (gravity should be pure Z in ENU when level)
        # In base_link frame, if level, accel should be [0, 0, +g] or [0, 0, -g]
        horizontal_g = np.sqrt(results.gravity_vector_body[0]**2 + results.gravity_vector_body[1]**2)
        results.tilt_from_level = np.degrees(np.arctan2(horizontal_g, abs(results.gravity_vector_body[2])))

        if results.gravity_error_percent > 5:
            results.issues.append(f"Gravity magnitude error: {results.gravity_error_percent:.1f}% (expected ~9.81 m/s²)")

        if results.tilt_from_level > 3:
            results.warnings.append(f"Robot tilted {results.tilt_from_level:.1f}° from level")

        # ====================================================================
        # 2. GYRO BIAS ANALYSIS
        # ====================================================================
        results.gyro_bias_mean = np.mean(gyros, axis=0)
        results.gyro_bias_std = np.std(gyros, axis=0)

        # Drift rate (integrate gyro over time)
        gyro_integral = np.cumsum(gyros, axis=0) * avg_dt
        total_time = timestamps[-1] - timestamps[0]
        results.gyro_drift_deg_per_min = np.degrees(gyro_integral[-1]) / (total_time / 60)

        # Check for excessive bias
        bias_threshold = 0.01  # rad/s = 0.57 deg/s
        for i, axis in enumerate(['X', 'Y', 'Z']):
            if abs(results.gyro_bias_mean[i]) > bias_threshold:
                results.warnings.append(
                    f"Gyro {axis} bias: {np.degrees(results.gyro_bias_mean[i]):.3f} °/s "
                    f"(drift: {results.gyro_drift_deg_per_min[i]:.1f} °/min)"
                )

        # ====================================================================
        # 3. ACCELEROMETER BIAS (after gravity removal)
        # ====================================================================
        # Remove gravity to get residual bias
        gravity_direction = results.gravity_vector_body / results.gravity_magnitude
        accel_bias = accels - gravity_direction * self.expected_gravity
        results.accel_bias_mean = np.mean(accel_bias, axis=0)
        results.accel_bias_std = np.std(accel_bias, axis=0)

        # ====================================================================
        # 4. MADGWICK FILTER ANALYSIS (THE CRITICAL CHECK!)
        # ====================================================================
        # Compare quaternion yaw change vs integrated gyro
        yaws = np.array([quat_to_yaw(q) for q in orientations])

        # Unwrap yaw for proper integration comparison
        yaws_unwrapped = np.unwrap(yaws)
        results.quaternion_yaw_change = np.degrees(yaws_unwrapped[-1] - yaws_unwrapped[0])

        # Integrate gyro Z
        gyro_z = gyros[:, 2]
        results.integrated_gyro_yaw = np.degrees(np.sum(gyro_z[:-1] * dt))

        # Calculate tracking ratio
        if abs(results.quaternion_yaw_change) > 0.1:
            results.madgwick_tracking_ratio = abs(results.integrated_gyro_yaw / results.quaternion_yaw_change)
        else:
            # Both near zero is OK for static test
            results.madgwick_tracking_ratio = 1.0 if abs(results.integrated_gyro_yaw) < 1.0 else float('inf')

        # Healthy if ratio is close to 1 (gyro matches quaternion)
        results.madgwick_healthy = 0.5 < results.madgwick_tracking_ratio < 2.0

        if not results.madgwick_healthy:
            results.issues.append(
                f"MADGWICK NOT TRACKING GYRO! Ratio: {results.madgwick_tracking_ratio:.1f}x "
                f"(Quat: {results.quaternion_yaw_change:.1f}°, Gyro: {results.integrated_gyro_yaw:.1f}°)"
            )

        # ====================================================================
        # 5. NOISE CHARACTERIZATION
        # ====================================================================
        # Estimate noise density from variance and sample rate
        # noise_density ≈ std * sqrt(dt)
        results.gyro_noise_density = np.mean(results.gyro_bias_std) * np.sqrt(avg_dt)
        results.accel_noise_density = np.mean(results.accel_bias_std) * np.sqrt(avg_dt)

        # ====================================================================
        # 6. EKF/ODOM DRIFT ANALYSIS
        # ====================================================================
        if len(self.ekf_odom_data) > 10:
            ekf_positions = np.array([p for _, p, _ in self.ekf_odom_data])
            ekf_velocities = np.array([v for _, _, v in self.ekf_odom_data])

            # Position drift (should be ~0 when static)
            pos_drift = ekf_positions[-1] - ekf_positions[0]
            results.position_drift_mm = np.linalg.norm(pos_drift) * 1000

            # Velocity (should be ~0 when static)
            results.velocity_drift_mm_s = np.linalg.norm(np.mean(ekf_velocities, axis=0)) * 1000

            if results.position_drift_mm > 50:
                results.warnings.append(f"EKF position drift: {results.position_drift_mm:.1f}mm during static test")

            if results.velocity_drift_mm_s > 10:
                results.warnings.append(f"EKF velocity non-zero: {results.velocity_drift_mm_s:.1f}mm/s during static test")

        # ====================================================================
        # 7. RAW CAMERA IMU ANALYSIS
        # ====================================================================
        if len(self.raw_cam_imu_data) > 100:
            raw_gyros = np.array([r.angular_velocity for r in self.raw_cam_imu_data])
            raw_accels = np.array([r.linear_acceleration for r in self.raw_cam_imu_data])

            raw_gyro_bias = np.mean(raw_gyros, axis=0)
            raw_accel_mag = np.linalg.norm(np.mean(raw_accels, axis=0))

            self.get_logger().info(f"Raw Camera IMU Gyro Bias: X={np.degrees(raw_gyro_bias[0]):.4f}°/s, "
                                   f"Y={np.degrees(raw_gyro_bias[1]):.4f}°/s, Z={np.degrees(raw_gyro_bias[2]):.4f}°/s")
            self.get_logger().info(f"Raw Camera IMU Accel Magnitude: {raw_accel_mag:.3f} m/s²")

        return results

    def _save_raw_data(self):
        """Save raw data to CSV for further analysis."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Save IMU data
        imu_file = self.output_dir / f'imu_diagnostic_{timestamp}.csv'
        with open(imu_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'qx', 'qy', 'qz', 'qw',
                'gyro_x', 'gyro_y', 'gyro_z',
                'accel_x', 'accel_y', 'accel_z'
            ])
            for r in self.imu_data:
                writer.writerow([
                    r.timestamp, *r.orientation, *r.angular_velocity, *r.linear_acceleration
                ])

        self.get_logger().info(f"Saved IMU data to: {imu_file}")

        # Save raw camera IMU data
        if self.raw_cam_imu_data:
            raw_file = self.output_dir / f'raw_cam_imu_diagnostic_{timestamp}.csv'
            with open(raw_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'gyro_x', 'gyro_y', 'gyro_z',
                    'accel_x', 'accel_y', 'accel_z'
                ])
                for r in self.raw_cam_imu_data:
                    writer.writerow([
                        r.timestamp, *r.angular_velocity, *r.linear_acceleration
                    ])
            self.get_logger().info(f"Saved raw camera IMU data to: {raw_file}")

    def _print_results(self, results: DiagnosticResults):
        """Print comprehensive results."""
        self.get_logger().info("")
        self.get_logger().info("=" * 70)
        self.get_logger().info("DIAGNOSTIC RESULTS")
        self.get_logger().info("=" * 70)

        # Gravity
        self.get_logger().info("")
        self.get_logger().info("--- GRAVITY ANALYSIS ---")
        self.get_logger().info(f"  Measured magnitude: {results.gravity_magnitude:.4f} m/s² "
                               f"(expected: {self.expected_gravity:.2f}, error: {results.gravity_error_percent:.2f}%)")
        self.get_logger().info(f"  Body frame vector: [{results.gravity_vector_body[0]:.4f}, "
                               f"{results.gravity_vector_body[1]:.4f}, {results.gravity_vector_body[2]:.4f}]")
        self.get_logger().info(f"  Tilt from level: {results.tilt_from_level:.2f}°")

        # Gyro
        self.get_logger().info("")
        self.get_logger().info("--- GYROSCOPE BIAS ---")
        self.get_logger().info(f"  Mean bias (°/s): X={np.degrees(results.gyro_bias_mean[0]):.4f}, "
                               f"Y={np.degrees(results.gyro_bias_mean[1]):.4f}, Z={np.degrees(results.gyro_bias_mean[2]):.4f}")
        self.get_logger().info(f"  Std dev (°/s):   X={np.degrees(results.gyro_bias_std[0]):.4f}, "
                               f"Y={np.degrees(results.gyro_bias_std[1]):.4f}, Z={np.degrees(results.gyro_bias_std[2]):.4f}")
        self.get_logger().info(f"  Drift (°/min):   X={results.gyro_drift_deg_per_min[0]:.2f}, "
                               f"Y={results.gyro_drift_deg_per_min[1]:.2f}, Z={results.gyro_drift_deg_per_min[2]:.2f}")

        # Accel
        self.get_logger().info("")
        self.get_logger().info("--- ACCELEROMETER (after gravity removal) ---")
        self.get_logger().info(f"  Residual bias (m/s²): X={results.accel_bias_mean[0]:.4f}, "
                               f"Y={results.accel_bias_mean[1]:.4f}, Z={results.accel_bias_mean[2]:.4f}")
        self.get_logger().info(f"  Noise std (m/s²):     X={results.accel_bias_std[0]:.4f}, "
                               f"Y={results.accel_bias_std[1]:.4f}, Z={results.accel_bias_std[2]:.4f}")

        # Madgwick - THE CRITICAL CHECK
        self.get_logger().info("")
        self.get_logger().info("--- MADGWICK FILTER CHECK (CRITICAL!) ---")
        status = "HEALTHY" if results.madgwick_healthy else "BROKEN!"
        self.get_logger().info(f"  Status: {status}")
        self.get_logger().info(f"  Quaternion yaw change: {results.quaternion_yaw_change:.2f}°")
        self.get_logger().info(f"  Integrated gyro yaw:   {results.integrated_gyro_yaw:.2f}°")
        self.get_logger().info(f"  Tracking ratio:        {results.madgwick_tracking_ratio:.1f}x "
                               f"(should be ~1.0)")

        # Noise
        self.get_logger().info("")
        self.get_logger().info("--- NOISE CHARACTERISTICS ---")
        self.get_logger().info(f"  Gyro noise density:  {results.gyro_noise_density:.6f} rad/s/√Hz "
                               f"({np.degrees(results.gyro_noise_density):.4f} °/s/√Hz)")
        self.get_logger().info(f"  Accel noise density: {results.accel_noise_density:.6f} m/s²/√Hz")

        # EKF drift
        self.get_logger().info("")
        self.get_logger().info("--- EKF DRIFT (during static test) ---")
        self.get_logger().info(f"  Position drift: {results.position_drift_mm:.1f} mm")
        self.get_logger().info(f"  Velocity (should be 0): {results.velocity_drift_mm_s:.1f} mm/s")

        # Issues and Warnings
        if results.issues:
            self.get_logger().info("")
            self.get_logger().info("=" * 70)
            self.get_logger().error("CRITICAL ISSUES FOUND:")
            for issue in results.issues:
                self.get_logger().error(f"  ❌ {issue}")

        if results.warnings:
            self.get_logger().info("")
            self.get_logger().warn("WARNINGS:")
            for warning in results.warnings:
                self.get_logger().warn(f"  ⚠️  {warning}")

    def _generate_fixes(self, results: DiagnosticResults):
        """Generate specific fix recommendations."""
        self.get_logger().info("")
        self.get_logger().info("=" * 70)
        self.get_logger().info("RECOMMENDED FIXES")
        self.get_logger().info("=" * 70)

        fix_num = 1

        # Madgwick fix (most critical)
        if not results.madgwick_healthy:
            self.get_logger().info("")
            self.get_logger().info(f"FIX #{fix_num}: MADGWICK FILTER NOT TRACKING GYRO")
            self.get_logger().info("-" * 50)
            self.get_logger().info("The Madgwick filter is not integrating gyroscope data properly.")
            self.get_logger().info("This is why IMU yaw barely changes despite gyro measuring rotation.")
            self.get_logger().info("")
            self.get_logger().info("Option A: Increase Madgwick gain (let gyro dominate)")
            self.get_logger().info("  Edit wheelchair_sensors.launch.py, add to imu_filter params:")
            self.get_logger().info("    'gain': 0.5,  # Default is 0.1, try 0.3-0.5")
            self.get_logger().info("")
            self.get_logger().info("Option B: Bypass Madgwick entirely for yaw")
            self.get_logger().info("  Modify EKF to use gyro angular velocity directly:")
            self.get_logger().info("  In ekf.yaml, set:")
            self.get_logger().info("    imu0_config: [..., false, false, false,  # NO yaw from quat")
            self.get_logger().info("                 ..., false, false, true,   # YES yaw_vel from gyro")
            self.get_logger().info("")
            self.get_logger().info("Option C: Use complementary filter instead")
            self.get_logger().info("  Replace Madgwick with simpler yaw = alpha*gyro + (1-alpha)*accel")
            fix_num += 1

        # Gyro bias fix
        gyro_z_bias_deg = np.degrees(results.gyro_bias_mean[2])
        if abs(gyro_z_bias_deg) > 0.1:  # > 0.1 deg/s
            self.get_logger().info("")
            self.get_logger().info(f"FIX #{fix_num}: GYRO Z BIAS CORRECTION")
            self.get_logger().info("-" * 50)
            self.get_logger().info(f"Measured gyro Z bias: {gyro_z_bias_deg:.4f} °/s = {results.gyro_bias_mean[2]:.6f} rad/s")
            self.get_logger().info("")
            self.get_logger().info("Update imu_bias_corrector.py parameters:")
            self.get_logger().info(f"  'gyro_x_bias': {results.gyro_bias_mean[0]:.6f},")
            self.get_logger().info(f"  'gyro_y_bias': {results.gyro_bias_mean[1]:.6f},")
            self.get_logger().info(f"  'gyro_z_bias': {results.gyro_bias_mean[2]:.6f},")
            fix_num += 1

        # Gravity alignment fix
        if results.tilt_from_level > 2:
            self.get_logger().info("")
            self.get_logger().info(f"FIX #{fix_num}: LEVEL THE ROBOT OR CALIBRATE GRAVITY")
            self.get_logger().info("-" * 50)
            self.get_logger().info(f"Robot is tilted {results.tilt_from_level:.1f}° from level.")
            self.get_logger().info("Either:")
            self.get_logger().info("  1. Physically level the robot during calibration")
            self.get_logger().info("  2. Initialize gravity from current accelerometer reading:")
            self.get_logger().info(f"     g_body = [{results.gravity_vector_body[0]:.4f}, "
                                   f"{results.gravity_vector_body[1]:.4f}, {results.gravity_vector_body[2]:.4f}]")
            fix_num += 1

        # EKF config recommendation
        self.get_logger().info("")
        self.get_logger().info(f"FIX #{fix_num}: RECOMMENDED EKF CONFIG")
        self.get_logger().info("-" * 50)
        self.get_logger().info("Based on analysis, use this EKF configuration:")
        self.get_logger().info("")
        self.get_logger().info("# ekf.yaml - IMU config")
        self.get_logger().info("# Order: x, y, z, roll, pitch, yaw, vx, vy, vz, vroll, vpitch, vyaw, ax, ay, az")
        self.get_logger().info("imu0_config: [false, false, false,    # No position from IMU")
        self.get_logger().info("              false, false, false,    # No orientation (Madgwick broken)")
        self.get_logger().info("              false, false, false,    # No velocity")
        self.get_logger().info("              false, false, true,     # YES: yaw_vel from gyro")
        self.get_logger().info("              false, false, false]    # No acceleration")
        self.get_logger().info("")
        self.get_logger().info("# Wheel odom config - use yaw from wheels")
        self.get_logger().info("odom0_config: [true,  true,  false,   # x, y position")
        self.get_logger().info("              false, false, true,     # yaw from wheel encoders")
        self.get_logger().info("              true,  false, false,    # x_vel")
        self.get_logger().info("              false, false, false,    # No angular vel (use IMU)")
        self.get_logger().info("              false, false, false]    # No acceleration")

        # Noise parameters
        self.get_logger().info("")
        self.get_logger().info(f"FIX #{fix_num + 1}: IMU NOISE PARAMETERS FOR EKF")
        self.get_logger().info("-" * 50)
        self.get_logger().info("Measured noise characteristics:")
        self.get_logger().info(f"  gyro_noise_density: {results.gyro_noise_density:.6f}  # rad/s/√Hz")
        self.get_logger().info(f"  accel_noise_density: {results.accel_noise_density:.6f}  # m/s²/√Hz")
        self.get_logger().info("")
        self.get_logger().info("For robot_localization EKF, set in ekf.yaml:")
        self.get_logger().info("  # Note: robot_localization expects variance, not density")
        gyro_var = results.gyro_noise_density ** 2 * 100  # Approx for 100 Hz
        accel_var = results.accel_noise_density ** 2 * 100
        self.get_logger().info(f"  # Approximate covariance at ~100 Hz IMU rate:")
        self.get_logger().info(f"  # gyro_variance: {gyro_var:.8f}")
        self.get_logger().info(f"  # accel_variance: {accel_var:.8f}")


def main(args=None):
    rclpy.init(args=args)
    node = IMUDiagnosticNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Interrupted by user")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
