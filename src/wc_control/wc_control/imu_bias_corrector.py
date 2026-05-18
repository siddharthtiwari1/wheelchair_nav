#!/usr/bin/env python3
"""
IMU Bias Corrector Node
========================
Applies gyro bias correction to raw IMU data BEFORE the Madgwick filter.

This fixes yaw drift at the source, so the Madgwick filter orientation
output will be drift-free.

Pipeline position:
  /camera/imu (raw, camera frame)
    → [THIS NODE subtracts bias in CAMERA FRAME]
    → /camera/imu_corrected
    → Madgwick filter
    → /imu/data
    → republisher
    → /imu (base_link)
    → EKF

Calibration values measured from static tests:
- gyro_x_bias: -0.004302 rad/s (default, overridden by startup calibrator)
- gyro_y_bias: 0.000787 rad/s
- gyro_z_bias: 0.000948 rad/s

Accelerometer bias calibration (2026-01-08, camera frame):
- accel_x_bias: 0.001458 m/s²
- accel_y_bias: 0.012577 m/s²
- accel_z_bias: 0.144767 m/s²

Dynamic Calibration:
- Subscribes to /imu/calibrated_bias from imu_startup_calibrator
- If startup calibration succeeds, uses fresh bias values
- If not available, uses default static calibration values

Author: Siddharth Tiwari
Date: 2025-12-05
Updated: 2025-12-06 - Added dynamic bias update from startup calibrator
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSPresetProfiles, QoSProfile, DurabilityPolicy, ReliabilityPolicy
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Vector3Stamped


class ImuBiasCorrector(Node):
    """Apply gyro bias correction to raw IMU data with dynamic calibration support."""

    def __init__(self):
        super().__init__('imu_bias_corrector')

        # Declare parameters with calibrated default values
        self.declare_parameter('input_topic', '/camera/imu')
        self.declare_parameter('output_topic', '/camera/imu_corrected')
        self.declare_parameter('calibrated_bias_topic', '/imu/calibrated_bias')

        # Gyro bias values (static calibration 2025-12-06 17:38 from full_system_20251206_173829.csv)
        # DIRECT raw camera IMU measurements from 1044-second static test
        # These are applied to /camera/imu BEFORE Madgwick filter
        # NOTE: These are CAMERA FRAME biases (camera_imu_optical_frame)
        #       camera_x bias → affects base_link YAW (after transform)
        self.declare_parameter('gyro_x_bias', -0.004302)   # rad/s - measured raw camera gyro X mean
        self.declare_parameter('gyro_y_bias', 0.000787)    # rad/s - measured raw camera gyro Y mean
        self.declare_parameter('gyro_z_bias', 0.000948)    # rad/s - measured raw camera gyro Z mean

        # Accelerometer bias values (CALIBRATED 2026-01-08)
        # Calibrated from 2959 samples during 15s static test on level surface
        # These are in CAMERA FRAME (camera_imu_optical_frame)
        # Expected: [0, -9.80665, 0], Measured: [0.00146, -9.79407, 0.14477]
        self.declare_parameter('accel_x_bias', 0.001458)   # m/s² - measured 2026-01-08
        self.declare_parameter('accel_y_bias', 0.012577)   # m/s² - (measured - expected gravity)
        self.declare_parameter('accel_z_bias', 0.144767)   # m/s² - measured 2026-01-08
        self.declare_parameter('apply_accel_bias', True)   # ENABLED with calibrated values

        # Get parameters
        self._input_topic = self.get_parameter('input_topic').get_parameter_value().string_value
        self._output_topic = self.get_parameter('output_topic').get_parameter_value().string_value
        self._calibrated_bias_topic = self.get_parameter('calibrated_bias_topic').get_parameter_value().string_value

        self._gyro_x_bias = self.get_parameter('gyro_x_bias').get_parameter_value().double_value
        self._gyro_y_bias = self.get_parameter('gyro_y_bias').get_parameter_value().double_value
        self._gyro_z_bias = self.get_parameter('gyro_z_bias').get_parameter_value().double_value

        self._accel_x_bias = self.get_parameter('accel_x_bias').get_parameter_value().double_value
        self._accel_y_bias = self.get_parameter('accel_y_bias').get_parameter_value().double_value
        self._accel_z_bias = self.get_parameter('accel_z_bias').get_parameter_value().double_value
        self._apply_accel_bias = self.get_parameter('apply_accel_bias').get_parameter_value().bool_value

        # Store default bias for comparison logging
        self._default_bias = (self._gyro_x_bias, self._gyro_y_bias, self._gyro_z_bias)
        self._using_calibrated_bias = False

        # Create IMU publisher and subscriber
        qos = QoSPresetProfiles.SENSOR_DATA.value
        self._publisher = self.create_publisher(Imu, self._output_topic, qos)
        self._subscription = self.create_subscription(
            Imu, self._input_topic, self._imu_callback, qos
        )

        # Subscribe to calibrated bias (latched/transient local QoS)
        latched_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE
        )
        self._bias_subscription = self.create_subscription(
            Vector3Stamped, self._calibrated_bias_topic, self._bias_callback, latched_qos
        )

        accel_status = "ENABLED" if self._apply_accel_bias else "DISABLED"
        self.get_logger().info(
            f'IMU Bias Corrector started: {self._input_topic} -> {self._output_topic}\n'
            f'Default gyro bias (rad/s): X={self._gyro_x_bias:.6f}, Y={self._gyro_y_bias:.6f}, Z={self._gyro_z_bias:.6f}\n'
            f'Accel bias correction: {accel_status}\n'
            f'  Accel bias (m/s²): X={self._accel_x_bias:.6f}, Y={self._accel_y_bias:.6f}, Z={self._accel_z_bias:.6f}\n'
            f'Waiting for calibrated bias on: {self._calibrated_bias_topic}'
        )

    def _bias_callback(self, msg: Vector3Stamped):
        """Update bias from startup calibrator."""
        old_bias = (self._gyro_x_bias, self._gyro_y_bias, self._gyro_z_bias)

        self._gyro_x_bias = msg.vector.x
        self._gyro_y_bias = msg.vector.y
        self._gyro_z_bias = msg.vector.z
        self._using_calibrated_bias = True

        # Calculate drift difference in deg/min
        diff_x = (self._gyro_x_bias - self._default_bias[0]) * 57.2958 * 60  # rad/s to deg/min
        diff_y = (self._gyro_y_bias - self._default_bias[1]) * 57.2958 * 60
        diff_z = (self._gyro_z_bias - self._default_bias[2]) * 57.2958 * 60

        self.get_logger().info(
            f'Received calibrated bias from startup calibrator!\n'
            f'  New bias (rad/s): X={self._gyro_x_bias:.6f}, Y={self._gyro_y_bias:.6f}, Z={self._gyro_z_bias:.6f}\n'
            f'  Old bias (rad/s): X={old_bias[0]:.6f}, Y={old_bias[1]:.6f}, Z={old_bias[2]:.6f}\n'
            f'  Drift adjustment (deg/min): X={diff_x:+.2f}, Y={diff_y:+.2f}, Z={diff_z:+.2f}'
        )

    def _imu_callback(self, msg: Imu):
        """Apply bias correction and republish."""
        corrected = Imu()
        corrected.header = msg.header

        # Copy orientation (unchanged)
        corrected.orientation = msg.orientation
        corrected.orientation_covariance = msg.orientation_covariance

        # Apply gyro bias correction
        corrected.angular_velocity.x = msg.angular_velocity.x - self._gyro_x_bias
        corrected.angular_velocity.y = msg.angular_velocity.y - self._gyro_y_bias
        corrected.angular_velocity.z = msg.angular_velocity.z - self._gyro_z_bias
        corrected.angular_velocity_covariance = msg.angular_velocity_covariance

        # Apply accelerometer bias correction (if enabled)
        # ADDED 2026-01-08: Accelerometer bias support
        if self._apply_accel_bias:
            corrected.linear_acceleration.x = msg.linear_acceleration.x - self._accel_x_bias
            corrected.linear_acceleration.y = msg.linear_acceleration.y - self._accel_y_bias
            corrected.linear_acceleration.z = msg.linear_acceleration.z - self._accel_z_bias
        else:
            corrected.linear_acceleration = msg.linear_acceleration
        corrected.linear_acceleration_covariance = msg.linear_acceleration_covariance

        self._publisher.publish(corrected)


def main(args=None):
    rclpy.init(args=args)
    node = ImuBiasCorrector()
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
