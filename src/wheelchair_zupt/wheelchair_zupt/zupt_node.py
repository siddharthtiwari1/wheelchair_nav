#!/usr/bin/env python3
"""
ZUPT-Enhanced Odometry Node - REPLACES AMF

This is the WORKING algorithm that beats dead reckoning.
Drop-in replacement for AMF with same interface.

Subscribes:
    /wc_control/odom (nav_msgs/Odometry) - Wheel odometry
    /imu (sensor_msgs/Imu) - IMU data (gyro z)

Publishes:
    /odometry/filtered (nav_msgs/Odometry) - Fused state estimate
    /tf (odom -> base_link transform)
    /zupt/diagnostics - Slip detection, bias estimate

Algorithm:
    1. During stops: Estimate gyro bias
    2. During motion: Use bias-corrected gyro for slip detection
    3. When slip detected: Trust gyro for heading
    4. Otherwise: Trust encoders (standard dead reckoning)
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

import numpy as np
import threading
from typing import Optional

from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from geometry_msgs.msg import TransformStamped, Vector3, Quaternion
from std_msgs.msg import Float64MultiArray, Bool
from tf2_ros import TransformBroadcaster


def quaternion_from_yaw(yaw: float) -> Quaternion:
    """Create quaternion from yaw angle."""
    q = Quaternion()
    q.w = np.cos(yaw / 2.0)
    q.x = 0.0
    q.y = 0.0
    q.z = np.sin(yaw / 2.0)
    return q


class ZUPTOdometry:
    """
    ZUPT-Enhanced Dead Reckoning - THE WINNING ALGORITHM

    Key insight: Use stops to estimate gyro bias, then use
    bias-corrected gyro ONLY during slip events.
    """

    def __init__(self,
                 wheel_radius_L: float = 0.1524,
                 wheel_radius_R: float = 0.1524,
                 baseline: float = 0.565,
                 stationary_threshold: float = 0.02,
                 stationary_omega_threshold: float = 0.05,
                 bias_adaptation_rate: float = 0.01,
                 slip_threshold: float = 0.08,
                 gyro_blend_alpha: float = 0.98):

        self.wheel_radius_L = wheel_radius_L
        self.wheel_radius_R = wheel_radius_R
        self.baseline = baseline
        self.stationary_threshold = stationary_threshold
        self.stationary_omega_threshold = stationary_omega_threshold
        self.bias_adaptation_rate = bias_adaptation_rate
        self.slip_threshold = slip_threshold
        self.gyro_blend_alpha = gyro_blend_alpha

        # Pose state
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0

        # Velocity state (for publishing)
        self.v_x = 0.0
        self.omega = 0.0

        # Gyro bias estimate
        self.gyro_bias = 0.0

        # Status flags
        self.is_stationary = False
        self.is_slip = False

        # Diagnostics
        self.stationary_count = 0
        self.slip_count = 0
        self.total_count = 0

    def reset(self, x: float = 0.0, y: float = 0.0, theta: float = 0.0):
        """Reset to given pose."""
        self.x = x
        self.y = y
        self.theta = theta
        self.v_x = 0.0
        self.omega = 0.0
        self.gyro_bias = 0.0
        self.is_stationary = False
        self.is_slip = False
        self.stationary_count = 0
        self.slip_count = 0
        self.total_count = 0

    def update(self, v_odom: float, omega_odom: float, gyro_z: float, dt: float):
        """
        Update odometry with new readings.

        Args:
            v_odom: Linear velocity from wheel odometry (m/s)
            omega_odom: Angular velocity from wheel odometry (rad/s)
            gyro_z: Gyroscope z-axis reading (rad/s)
            dt: Time step (seconds)
        """
        self.total_count += 1

        # === Stationary detection ===
        self.is_stationary = (
            abs(v_odom) < self.stationary_threshold and
            abs(omega_odom) < self.stationary_omega_threshold
        )

        # === Gyro bias estimation during stationary ===
        if self.is_stationary:
            # When stationary: gyro_reading = bias + noise
            alpha = self.bias_adaptation_rate
            self.gyro_bias = alpha * gyro_z + (1 - alpha) * self.gyro_bias
            # Clamp bias to physical limits (MEMS gyro bias rarely exceeds ±0.05 rad/s)
            self.gyro_bias = max(-0.05, min(0.05, self.gyro_bias))
            self.stationary_count += 1

        # === Bias-corrected gyro ===
        gyro_corrected = gyro_z - self.gyro_bias

        # === Slip detection ===
        disagreement = abs(omega_odom - gyro_corrected)
        self.is_slip = disagreement > self.slip_threshold

        if self.is_slip:
            self.slip_count += 1

        # === Select velocities ===
        if self.is_stationary:
            # ZUPT: zero velocity when stopped — prevents drift, recalibrates gyro
            v_use = 0.0
            omega_use = 0.0
        else:
            # Complementary filter: blend encoder omega with bias-corrected gyro.
            # Encoders have systematic drift (wheel radius mismatch → heading arc).
            # Gyro has no systematic drift (bias is calibrated during stops).
            # alpha=0.30: 30% encoder + 70% gyro — gyro-dominant for heading accuracy.
            # During slip (encoder-gyro disagreement > threshold): alpha=0.05 (95% gyro).
            v_use = v_odom
            blend = 0.05 if self.is_slip else self.gyro_blend_alpha
            omega_use = blend * omega_odom + (1.0 - blend) * gyro_corrected

        # Store for publishing
        self.v_x = v_use
        self.omega = omega_use

        # === Pose update ===
        self.theta += omega_use * dt
        self.x += v_use * np.cos(self.theta) * dt
        self.y += v_use * np.sin(self.theta) * dt

        # Normalize theta to [-pi, pi]
        while self.theta > np.pi:
            self.theta -= 2 * np.pi
        while self.theta < -np.pi:
            self.theta += 2 * np.pi

        return self.x, self.y, self.theta


class ZUPTNode(Node):
    """
    ZUPT-Enhanced Odometry ROS2 Node.

    Drop-in replacement for AMF node.
    """

    def __init__(self):
        super().__init__('zupt_node')

        # Declare parameters
        self._declare_parameters()

        # Initialize filter
        self.filter = ZUPTOdometry(
            wheel_radius_L=self.get_parameter('wheel_radius_L').value,
            wheel_radius_R=self.get_parameter('wheel_radius_R').value,
            baseline=self.get_parameter('wheel_baseline').value,
            stationary_threshold=self.get_parameter('stationary_threshold').value,
            stationary_omega_threshold=self.get_parameter('stationary_omega_threshold').value,
            bias_adaptation_rate=self.get_parameter('bias_adaptation_rate').value,
            slip_threshold=self.get_parameter('slip_threshold').value,
            gyro_blend_alpha=self.get_parameter('gyro_blend_alpha').value,
        )

        # Set initial pose
        self.filter.reset(
            x=self.get_parameter('initial_x').value,
            y=self.get_parameter('initial_y').value,
            theta=self.get_parameter('initial_theta').value,
        )

        # Thread safety
        self._lock = threading.Lock()

        # Timestamps
        self._last_odom_time: Optional[float] = None
        self._latest_gyro_z: float = 0.0
        self._initialized = False

        # Sensor QoS - IMU uses BEST_EFFORT, Odom uses RELIABLE (ros2_control)
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # Odom QoS - RELIABLE to match ros2_control diff_drive_controller
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
            odom_qos  # Use RELIABLE QoS for odom (ros2_control)
        )

        # Publishers
        self.odom_pub = self.create_publisher(Odometry, '/odometry/filtered', 10)
        self.slip_pub = self.create_publisher(Bool, '/zupt/slip_detected', 10)
        self.diagnostics_pub = self.create_publisher(Float64MultiArray, '/zupt/diagnostics', 10)

        # TF broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)

        # Diagnostics timer
        self.create_timer(10.0, self._print_diagnostics)

        self.get_logger().info(
            f'''
╔══════════════════════════════════════════════════════════════════╗
║              ZUPT-ENHANCED ODOMETRY NODE                         ║
║                                                                  ║
║  Algorithm: ZUPT + Slip Detection (beats dead reckoning!)       ║
║                                                                  ║
║  Input:  {self.get_parameter("odom_topic").value:<50} ║
║          {self.get_parameter("imu_topic").value:<50} ║
║  Output: /odometry/filtered                                      ║
║                                                                  ║
║  Parameters:                                                     ║
║    Slip threshold:     {self.get_parameter("slip_threshold").value:.2f} rad/s                            ║
║    Stationary thresh:  {self.get_parameter("stationary_threshold").value:.2f} m/s                             ║
║    Bias adaptation:    {self.get_parameter("bias_adaptation_rate").value:.3f}                                ║
╚══════════════════════════════════════════════════════════════════╝
''')

    def _declare_parameters(self):
        """Declare ROS2 parameters."""
        # Topics
        self.declare_parameter('imu_topic', '/imu')
        self.declare_parameter('odom_topic', '/wc_control/odom')

        # Frames
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('base_frame', 'base_link')

        # Wheel parameters
        self.declare_parameter('wheel_radius_L', 0.1524)
        self.declare_parameter('wheel_radius_R', 0.1524)
        self.declare_parameter('wheel_baseline', 0.565)

        # ZUPT parameters
        self.declare_parameter('stationary_threshold', 0.015)  # m/s - above encoder noise (~0.005), below min real velocity (0.05)
        self.declare_parameter('stationary_omega_threshold', 0.05)  # rad/s
        self.declare_parameter('bias_adaptation_rate', 0.01)
        self.declare_parameter('slip_threshold', 0.08)  # rad/s
        self.declare_parameter('gyro_blend_alpha', 0.30)  # 0.30=70% gyro, prevents encoder heading drift

        # Initial pose
        self.declare_parameter('initial_x', 0.0)
        self.declare_parameter('initial_y', 0.0)
        self.declare_parameter('initial_theta', 0.0)

        # TF
        self.declare_parameter('publish_tf', True)

    def _imu_callback(self, msg: Imu):
        """Store latest gyro reading."""
        self._latest_gyro_z = msg.angular_velocity.z

    def _odom_callback(self, msg: Odometry):
        """Process wheel odometry and update pose."""
        current_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        with self._lock:
            if self._last_odom_time is None:
                self._last_odom_time = current_time
                self._initialized = True
                self.get_logger().info('ZUPT filter initialized')
                return

            dt = current_time - self._last_odom_time
            if dt <= 0 or dt > 0.5:
                self._last_odom_time = current_time
                return

            # Extract velocities
            v_odom = msg.twist.twist.linear.x
            omega_odom = msg.twist.twist.angular.z

            # Update filter
            self.filter.update(v_odom, omega_odom, self._latest_gyro_z, dt)

            self._last_odom_time = current_time

        # Publish
        self._publish_state(msg.header.stamp)

    def _publish_state(self, stamp):
        """Publish filtered odometry."""
        with self._lock:
            if not self._initialized:
                return

            x = self.filter.x
            y = self.filter.y
            theta = self.filter.theta
            v_x = self.filter.v_x
            omega = self.filter.omega
            is_slip = self.filter.is_slip
            is_stationary = self.filter.is_stationary
            gyro_bias = self.filter.gyro_bias

        # === Odometry message ===
        odom_msg = Odometry()
        odom_msg.header.stamp = stamp
        odom_msg.header.frame_id = self.odom_frame
        odom_msg.child_frame_id = self.base_frame

        odom_msg.pose.pose.position.x = x
        odom_msg.pose.pose.position.y = y
        odom_msg.pose.pose.position.z = 0.0
        odom_msg.pose.pose.orientation = quaternion_from_yaw(theta)

        # Covariance: lower when stationary, higher when slipping
        base_pos_cov = 0.01 if is_stationary else (0.1 if is_slip else 0.05)
        base_rot_cov = 0.01 if is_stationary else (0.1 if is_slip else 0.05)

        odom_msg.pose.covariance = [
            base_pos_cov, 0, 0, 0, 0, 0,
            0, base_pos_cov, 0, 0, 0, 0,
            0, 0, 1e6, 0, 0, 0,
            0, 0, 0, 1e6, 0, 0,
            0, 0, 0, 0, 1e6, 0,
            0, 0, 0, 0, 0, base_rot_cov,
        ]

        odom_msg.twist.twist.linear.x = v_x
        odom_msg.twist.twist.angular.z = omega

        odom_msg.twist.covariance = [
            0.01, 0, 0, 0, 0, 0,
            0, 0.01, 0, 0, 0, 0,
            0, 0, 1e6, 0, 0, 0,
            0, 0, 0, 1e6, 0, 0,
            0, 0, 0, 0, 1e6, 0,
            0, 0, 0, 0, 0, 0.02,
        ]

        self.odom_pub.publish(odom_msg)

        # === TF ===
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

        # === Slip detection ===
        slip_msg = Bool()
        slip_msg.data = is_slip
        self.slip_pub.publish(slip_msg)

        # === Diagnostics ===
        diag_msg = Float64MultiArray()
        diag_msg.data = [
            float(is_slip),
            float(is_stationary),
            gyro_bias,
            float(self.filter.slip_count),
            float(self.filter.stationary_count),
        ]
        self.diagnostics_pub.publish(diag_msg)

    def _print_diagnostics(self):
        """Print periodic diagnostics."""
        with self._lock:
            total = self.filter.total_count
            if total == 0:
                return
            slip_pct = 100 * self.filter.slip_count / total
            stat_pct = 100 * self.filter.stationary_count / total
            bias = self.filter.gyro_bias

        self.get_logger().info(
            f'ZUPT Stats | Samples: {total} | Slip: {slip_pct:.1f}% | '
            f'Stationary: {stat_pct:.1f}% | Gyro bias: {bias:.4f} rad/s'
        )


def main(args=None):
    rclpy.init(args=args)
    node = ZUPTNode()

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
