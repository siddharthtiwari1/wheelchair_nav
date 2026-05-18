#!/usr/bin/env python3

"""
Republish IMU data from camera_imu_optical_frame to base_link-aligned 'imu' frame.

IMU Filter (Madgwick) outputs:
- Orientation: quaternion relative to world frame (odom/enu)
- Angular velocity: vector in camera_imu_optical_frame
- Linear acceleration: vector in camera_imu_optical_frame

This node transforms all data to base_link frame for robot localization/EKF.
"""

from math import sqrt, atan2, asin
from typing import Tuple, Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSPresetProfiles
from sensor_msgs.msg import Imu

QuaternionTuple = Tuple[float, float, float, float]


def _normalize_quaternion(q: QuaternionTuple) -> QuaternionTuple:
    x, y, z, w = q
    norm = sqrt(x * x + y * y + z * z + w * w)
    if norm == 0.0:
        return 0.0, 0.0, 0.0, 1.0
    return x / norm, y / norm, z / norm, w / norm


def _quaternion_conjugate(q: QuaternionTuple) -> QuaternionTuple:
    x, y, z, w = q
    return -x, -y, -z, w


def _quaternion_multiply(a: QuaternionTuple, b: QuaternionTuple) -> QuaternionTuple:
    """Multiply two quaternions: result = a * b"""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def _rotate_vector(q: QuaternionTuple, vec: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """Rotate a vector by quaternion q using rotation matrix"""
    x, y, z, w = _normalize_quaternion(q)
    vx, vy, vz = vec

    # Convert quaternion to rotation matrix
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    # Apply rotation matrix to vector
    return (
        (1 - 2 * (yy + zz)) * vx + 2 * (xy - wz) * vy + 2 * (xz + wy) * vz,
        2 * (xy + wz) * vx + (1 - 2 * (xx + zz)) * vy + 2 * (yz - wx) * vz,
        2 * (xz - wy) * vx + 2 * (yz + wx) * vy + (1 - 2 * (xx + yy)) * vz,
    )


def _rotate_covariance(q: QuaternionTuple, cov: list) -> list:
    """
    Rotate a 3x3 covariance matrix using quaternion rotation.
    Formula: C' = R * C * R^T
    Input: 9-element list (row-major 3x3 matrix)
    Output: 9-element list (rotated covariance)
    """
    x, y, z, w = _normalize_quaternion(q)

    # Build rotation matrix from quaternion
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    R = [
        [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
        [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
        [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
    ]

    # Convert flat list to 3x3 matrix
    C = [[cov[i * 3 + j] for j in range(3)] for i in range(3)]

    # Compute R * C
    RC = [[sum(R[i][k] * C[k][j] for k in range(3)) for j in range(3)] for i in range(3)]

    # Compute (R * C) * R^T
    result = [[sum(RC[i][k] * R[j][k] for k in range(3)) for j in range(3)] for i in range(3)]

    # Flatten back to list
    return [result[i][j] for i in range(3) for j in range(3)]


def _quaternion_to_rpy(q: QuaternionTuple) -> Tuple[float, float, float]:
    """
    Convert quaternion to Roll-Pitch-Yaw (ZYX convention).
    Returns (roll, pitch, yaw) in radians.
    """
    x, y, z, w = _normalize_quaternion(q)

    # Roll (x-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = atan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = atan2(sinp, 0) if sinp >= 0 else -atan2(-sinp, 0)  # Use 90 degrees if out of range
    else:
        pitch = asin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = atan2(siny_cosp, cosy_cosp)

    return (roll, pitch, yaw)


class ImuWheelchairRepublisher(Node):
    """
    Transform IMU data from camera_imu_optical_frame to base_link frame.

    Handles the fact that:
    - Orientation quaternion is relative to world (odom) frame
    - Angular velocity & linear acceleration are in camera_imu_optical_frame

    Subscribes to /imu/data and publishes to /imu (base_link aligned).
    """

    def __init__(self) -> None:
        super().__init__('imu_wheelchair_republisher')

        self.declare_parameter('input_topic', '/imu/data')
        self.declare_parameter('output_topic', '/imu')
        self.declare_parameter('output_frame', 'imu')

        # Transform from camera_imu_optical_frame to base_link (REP-103)
        # camera_imu_optical_frame: X=right, Y=down, Z=forward (into scene)
        # base_link (REP-103): X=forward, Y=left, Z=up
        self.declare_parameter('orientation_quaternion', [-0.5, 0.5, -0.5, 0.5])
        self.declare_parameter('vector_quaternion', [-0.5, 0.5, -0.5, 0.5])
        self.declare_parameter('zero_on_start', True)

        self._input_topic = self.get_parameter('input_topic').get_parameter_value().string_value
        self._output_topic = self.get_parameter('output_topic').get_parameter_value().string_value
        self._output_frame = self.get_parameter('output_frame').get_parameter_value().string_value
        self._zero_on_start = self.get_parameter('zero_on_start').get_parameter_value().bool_value

        # Get quaternion transforms
        raw_orientation_quat = self.get_parameter('orientation_quaternion').get_parameter_value().double_array_value
        raw_vector_quat = self.get_parameter('vector_quaternion').get_parameter_value().double_array_value

        if len(raw_orientation_quat) != 4:
            raw_orientation_quat = [-0.5, 0.5, -0.5, 0.5]
        if len(raw_vector_quat) != 4:
            raw_vector_quat = [-0.5, 0.5, -0.5, 0.5]

        # For orientation transformation (quaternion multiplication with world->sensor)
        self._orientation_quaternion = _normalize_quaternion(tuple(raw_orientation_quat))
        # For vector rotation (angular velocity, linear acceleration)
        self._vector_quaternion = _normalize_quaternion(tuple(raw_vector_quat))

        self._initial_orientation: Optional[QuaternionTuple] = None
        self._initial_orientation_inv: Optional[QuaternionTuple] = None
        self._message_count = 0

        qos = QoSPresetProfiles.SENSOR_DATA.value
        self._publisher = self.create_publisher(Imu, self._output_topic, qos)
        self._subscription = self.create_subscription(Imu, self._input_topic, self._handle_imu, qos)

        self.get_logger().info(
            f'Republishing IMU {self._input_topic} -> {self._output_topic} '
            f'with base_link-aligned orientation'
        )

    def _handle_imu(self, msg: Imu) -> None:
        """Transform IMU data from sensor frame to base_link-aligned frame"""
        republished = Imu()
        republished.header = msg.header
        republished.header.frame_id = self._output_frame

        # Transform orientation
        # The orientation quaternion is: world -> camera_imu_optical_frame
        # We want: world -> base_link
        # Formula: q_world_to_base = q_world_to_sensor * q_sensor_to_base
        # But we have q_sensor_to_base, and q_world_to_sensor
        # So: q_world_to_base = q_world_to_sensor * q_sensor_to_base
        sensor_quat = (
            msg.orientation.x,
            msg.orientation.y,
            msg.orientation.z,
            msg.orientation.w,
        )

        # Apply frame rotation: q_result = q_sensor * q_sensor_to_base (for orientation)
        base_orientation = _quaternion_multiply(sensor_quat, self._orientation_quaternion)

        if self._zero_on_start and self._initial_orientation is None:
            self._initial_orientation = base_orientation
            self._initial_orientation_inv = _quaternion_conjugate(base_orientation)

        if self._zero_on_start and self._initial_orientation_inv is not None:
            aligned_orientation = _quaternion_multiply(self._initial_orientation_inv, base_orientation)
        else:
            aligned_orientation = base_orientation

        republished.orientation.x = aligned_orientation[0]
        republished.orientation.y = aligned_orientation[1]
        republished.orientation.z = aligned_orientation[2]
        republished.orientation.w = aligned_orientation[3]

        # Rotate angular velocity vector to base frame (use vector quaternion)
        angular = (msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z)
        rotated_angular = _rotate_vector(self._vector_quaternion, angular)
        republished.angular_velocity.x = rotated_angular[0]
        republished.angular_velocity.y = rotated_angular[1]
        # 2026-01-13: Removed negation - was causing opposite rotation
        # The quaternion rotation already handles the frame transform correctly
        republished.angular_velocity.z = rotated_angular[2]

        # Rotate linear acceleration vector to base frame (use vector quaternion)
        linear = (msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z)
        rotated_linear = _rotate_vector(self._vector_quaternion, linear)
        republished.linear_acceleration.x = rotated_linear[0]
        republished.linear_acceleration.y = rotated_linear[1]
        republished.linear_acceleration.z = rotated_linear[2]

        # Transform covariance matrices (rotate from sensor frame to base frame)
        # CRITICAL: robot_localization IGNORES data with zero covariance!
        # Madgwick filter publishes zeros, so we must provide reasonable defaults.

        # Orientation covariance
        ori_cov = list(msg.orientation_covariance)
        if all(c == 0.0 for c in ori_cov):
            # Default: ~3 degrees std dev for orientation
            ori_var = 0.003  # (0.05 rad)^2
            republished.orientation_covariance = [
                ori_var, 0.0, 0.0,
                0.0, ori_var, 0.0,
                0.0, 0.0, ori_var
            ]
        else:
            republished.orientation_covariance = _rotate_covariance(
                self._orientation_quaternion, ori_cov
            )

        # Angular velocity covariance
        ang_cov = list(msg.angular_velocity_covariance)
        if all(c == 0.0 for c in ang_cov):
            # Default: ~0.01 rad/s std dev for gyro
            ang_var = 0.0001  # (0.01 rad/s)^2
            republished.angular_velocity_covariance = [
                ang_var, 0.0, 0.0,
                0.0, ang_var, 0.0,
                0.0, 0.0, ang_var
            ]
        else:
            republished.angular_velocity_covariance = _rotate_covariance(
                self._vector_quaternion, ang_cov
            )

        # Linear acceleration covariance
        lin_cov = list(msg.linear_acceleration_covariance)
        if all(c == 0.0 for c in lin_cov):
            # Default: ~0.1 m/s^2 std dev for accelerometer
            lin_var = 0.01  # (0.1 m/s^2)^2
            republished.linear_acceleration_covariance = [
                lin_var, 0.0, 0.0,
                0.0, lin_var, 0.0,
                0.0, 0.0, lin_var
            ]
        else:
            republished.linear_acceleration_covariance = _rotate_covariance(
                self._vector_quaternion, lin_cov
            )

        # Log orientation (RPY in degrees) and acceleration every 30 messages
        self._message_count += 1
        if self._message_count % 30 == 0:
            rpy = _quaternion_to_rpy(aligned_orientation)
            roll_deg = rpy[0] * 180.0 / 3.14159265359
            pitch_deg = rpy[1] * 180.0 / 3.14159265359
            yaw_deg = rpy[2] * 180.0 / 3.14159265359

            self.get_logger().info(
                f"Orientation (RPY): Roll={roll_deg:.2f}°, Pitch={pitch_deg:.2f}°, Yaw={yaw_deg:.2f}° | "
                f"Accel: X={rotated_linear[0]:.3f}, Y={rotated_linear[1]:.3f}, Z={rotated_linear[2]:.3f} m/s²"
            )

        self._publisher.publish(republished)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ImuWheelchairRepublisher()
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
