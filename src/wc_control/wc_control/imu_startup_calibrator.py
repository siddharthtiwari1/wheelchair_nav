#!/usr/bin/env python3
"""
IMU Startup Calibrator - Calibrates gyro bias at system startup.

This node collects IMU data for a configurable duration when the system starts,
computes the mean gyro bias, and publishes it for the bias corrector to use.

Industry Standard Approach:
- Most consumer drones, smartphones, and industrial robots use startup calibration
- System must remain stationary for calibration period (typically 2-5 seconds)
- Wheelchair naturally starts stationary, making this ideal

Pipeline:
1. System boots with wheelchair stationary
2. This node collects raw IMU data for calibration_duration seconds
3. Computes mean gyro bias in camera frame
4. Publishes bias to /imu/calibrated_bias (latched)
5. imu_bias_corrector subscribes and updates its bias values

Usage:
- Launch with wheelchair_sensors.launch.py
- Wheelchair must be stationary at startup for accurate calibration
- Green LED or sound could indicate calibration complete (future)
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSPresetProfiles, DurabilityPolicy, ReliabilityPolicy
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Vector3Stamped
from std_msgs.msg import Bool
import numpy as np
from collections import deque


class ImuStartupCalibrator(Node):
    """Calibrates IMU gyro bias at startup by averaging static measurements."""

    def __init__(self):
        super().__init__('imu_startup_calibrator')

        # Parameters
        self.declare_parameter('input_topic', '/camera/imu')
        self.declare_parameter('bias_topic', '/imu/calibrated_bias')
        self.declare_parameter('status_topic', '/imu/calibration_status')
        self.declare_parameter('calibration_duration', 3.0)  # seconds
        self.declare_parameter('min_samples', 100)  # minimum samples for valid calibration
        self.declare_parameter('max_motion_threshold', 0.05)  # rad/s - reject if motion detected

        # Default bias values from static calibration (fallback if startup cal fails)
        self.declare_parameter('default_gyro_x_bias', -0.004302)
        self.declare_parameter('default_gyro_y_bias', 0.000787)
        self.declare_parameter('default_gyro_z_bias', 0.000948)

        self.input_topic = self.get_parameter('input_topic').value
        self.bias_topic = self.get_parameter('bias_topic').value
        self.status_topic = self.get_parameter('status_topic').value
        self.calibration_duration = self.get_parameter('calibration_duration').value
        self.min_samples = self.get_parameter('min_samples').value
        self.max_motion_threshold = self.get_parameter('max_motion_threshold').value

        self.default_bias = np.array([
            self.get_parameter('default_gyro_x_bias').value,
            self.get_parameter('default_gyro_y_bias').value,
            self.get_parameter('default_gyro_z_bias').value
        ])

        # State
        self.calibration_complete = False
        self.calibration_started = False
        self.start_time = None
        self.gyro_samples = []

        # QoS for latched topic (transient local)
        latched_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE
        )

        # Publishers
        self.bias_pub = self.create_publisher(
            Vector3Stamped, self.bias_topic, latched_qos)
        self.status_pub = self.create_publisher(
            Bool, self.status_topic, latched_qos)

        # Subscriber - use SENSOR_DATA QoS (BEST_EFFORT) to match RealSense
        self.imu_sub = self.create_subscription(
            Imu, self.input_topic, self.imu_callback,
            QoSPresetProfiles.SENSOR_DATA.value)

        self.get_logger().info(
            f'IMU Startup Calibrator initialized\n'
            f'  Input: {self.input_topic}\n'
            f'  Duration: {self.calibration_duration}s\n'
            f'  Min samples: {self.min_samples}\n'
            f'  Motion threshold: {self.max_motion_threshold} rad/s'
        )

        # Publish initial "not calibrated" status
        status_msg = Bool()
        status_msg.data = False
        self.status_pub.publish(status_msg)

    def imu_callback(self, msg: Imu):
        """Collect IMU samples during calibration period."""
        if self.calibration_complete:
            return

        current_time = self.get_clock().now()

        # Start calibration on first message
        if not self.calibration_started:
            self.calibration_started = True
            self.start_time = current_time
            self.get_logger().info('Starting IMU calibration - keep wheelchair stationary!')

        # Check if calibration duration has elapsed
        elapsed = (current_time - self.start_time).nanoseconds / 1e9

        if elapsed < self.calibration_duration:
            # Collect sample
            gyro = np.array([
                msg.angular_velocity.x,
                msg.angular_velocity.y,
                msg.angular_velocity.z
            ])
            self.gyro_samples.append(gyro)
        else:
            # Calibration period complete - compute bias
            self.compute_and_publish_bias()

    def compute_and_publish_bias(self):
        """Compute mean gyro bias from collected samples."""
        self.calibration_complete = True

        if len(self.gyro_samples) < self.min_samples:
            self.get_logger().warn(
                f'Calibration failed: only {len(self.gyro_samples)} samples '
                f'(need {self.min_samples}). Using default bias.'
            )
            calibrated_bias = self.default_bias
            calibration_valid = False
        else:
            samples = np.array(self.gyro_samples)
            mean_gyro = np.mean(samples, axis=0)
            std_gyro = np.std(samples, axis=0)

            # Check for motion during calibration
            max_std = np.max(std_gyro)
            if max_std > self.max_motion_threshold:
                self.get_logger().warn(
                    f'Motion detected during calibration (std={max_std:.4f} rad/s). '
                    f'Using default bias. Please keep wheelchair stationary at startup.'
                )
                calibrated_bias = self.default_bias
                calibration_valid = False
            else:
                calibrated_bias = mean_gyro
                calibration_valid = True

                self.get_logger().info(
                    f'Calibration complete!\n'
                    f'  Samples: {len(self.gyro_samples)}\n'
                    f'  Gyro bias (rad/s):\n'
                    f'    X: {calibrated_bias[0]:+.6f} (default: {self.default_bias[0]:+.6f})\n'
                    f'    Y: {calibrated_bias[1]:+.6f} (default: {self.default_bias[1]:+.6f})\n'
                    f'    Z: {calibrated_bias[2]:+.6f} (default: {self.default_bias[2]:+.6f})\n'
                    f'  Std dev: X={std_gyro[0]:.6f}, Y={std_gyro[1]:.6f}, Z={std_gyro[2]:.6f}'
                )

                # Compare to default
                diff = calibrated_bias - self.default_bias
                diff_deg = np.degrees(diff) * 60  # deg/min drift difference
                self.get_logger().info(
                    f'  Difference from default (deg/min drift):\n'
                    f'    X: {diff_deg[0]:+.2f}, Y: {diff_deg[1]:+.2f}, Z: {diff_deg[2]:+.2f}'
                )

        # Publish calibrated bias (latched)
        bias_msg = Vector3Stamped()
        bias_msg.header.stamp = self.get_clock().now().to_msg()
        bias_msg.header.frame_id = 'camera_imu_optical_frame'
        bias_msg.vector.x = float(calibrated_bias[0])
        bias_msg.vector.y = float(calibrated_bias[1])
        bias_msg.vector.z = float(calibrated_bias[2])
        self.bias_pub.publish(bias_msg)

        # Publish calibration status
        status_msg = Bool()
        status_msg.data = calibration_valid
        self.status_pub.publish(status_msg)

        # Unsubscribe from IMU - no longer needed
        self.destroy_subscription(self.imu_sub)
        self.get_logger().info('Calibration node going idle (bias published on latched topic)')


def main(args=None):
    rclpy.init(args=args)
    node = ImuStartupCalibrator()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
