#!/usr/bin/env python3
"""
Cable Tracing Deploy Node — Wheelchair adaptation.

Adapted from ugv_deploy.py (Diadem UGV) for differential-drive wheelchair.
Uses WireCNN (PyTorch Lightning) to predict velocity from camera frames.

Changes from original:
  - Velocity limits clamped to wheelchair safety limits (0.25 m/s max)
  - linear.y zeroed (diff-drive cannot move laterally)
  - Configurable camera index, checkpoint path via ROS params
  - Device auto-selects CUDA if available
  - Video recording to timestamped files

Usage:
  ros2 run cable_trace_deploy cable_trace_node
  # or standalone:
  python3 cable_trace_node.py
"""

import os
import sys
import csv
import time
import threading

import cv2
import numpy as np
import torch

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

# Import from same package
from cable_trace_deploy.cnn_train_vel import WireCNN


# Wheelchair safety limits
MAX_LINEAR_VEL = 0.25   # m/s (wheelchair max)
MAX_ANGULAR_VEL = 0.60  # rad/s (wheelchair max)


class CableTraceNode(Node):
    def __init__(self):
        super().__init__('cable_trace_node')

        # ROS parameters
        default_weights = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'weights.zip')
        self.declare_parameter('checkpoint_path', default_weights)
        self.declare_parameter('camera_index', -1)  # -1 = auto-detect
        self.declare_parameter('max_linear_vel', MAX_LINEAR_VEL)
        self.declare_parameter('max_angular_vel', MAX_ANGULAR_VEL)
        self.declare_parameter('publish_rate', 10.0)  # Hz
        self.declare_parameter('use_cuda', True)
        self.declare_parameter('record_video', True)
        self.declare_parameter('show_gui', True)

        # Read params (force int cast — LaunchConfiguration passes strings)
        ckpt_path = self.get_parameter('checkpoint_path').value
        self.camera_index = int(self.get_parameter('camera_index').value)
        self.max_lin = float(self.get_parameter('max_linear_vel').value)
        self.max_ang = float(self.get_parameter('max_angular_vel').value)
        publish_rate = self.get_parameter('publish_rate').value
        use_cuda = self.get_parameter('use_cuda').value
        self.record_video = self.get_parameter('record_video').value
        self.show_gui = self.get_parameter('show_gui').value

        # Publisher (uses 'cmd_vel' so launch file remappings work)
        self.pub = self.create_publisher(Twist, 'cmd_vel', 10)

        # Velocity state (thread-safe via GIL for simple float assignments)
        self.velx = 0.0
        self.velw = 0.0

        # Timer to publish at fixed rate
        self.timer = self.create_timer(1.0 / publish_rate, self.publish_velocity)

        # Camera settings
        self.WIDTH, self.HEIGHT, self.FPS = 640, 480, 15

        # Load model
        device = 'cuda' if (use_cuda and torch.cuda.is_available()) else 'cpu'
        self.get_logger().info(f'Loading model from {ckpt_path} on {device}')

        if not os.path.exists(ckpt_path):
            self.get_logger().fatal(f'Checkpoint not found: {ckpt_path}')
            raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')

        self.model = WireCNN.load_from_checkpoint(ckpt_path, map_location=device)
        self.model.eval()
        self.device = device
        self.get_logger().info(
            f'Model loaded. Max vel: linear={self.max_lin}, angular={self.max_ang}')

        # Start camera thread
        self.running = True
        self.camera_thread = threading.Thread(target=self.cv_loop, daemon=True)
        self.camera_thread.start()

    def publish_velocity(self):
        msg = Twist()
        msg.linear.x = float(self.velx)
        msg.linear.y = 0.0   # Diff-drive: no lateral movement
        msg.linear.z = 0.0
        msg.angular.z = float(self.velw)
        self.pub.publish(msg)

    def preprocess(self, frame):
        """Resize to 128x128 and normalize to [0, 1]."""
        img = cv2.resize(frame, (128, 128))
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        return img.unsqueeze(0).to(self.device)

    def find_camera(self):
        """Find camera by index or auto-detect Logitech."""
        if self.camera_index >= 0:
            self.get_logger().info(f'Using camera index {self.camera_index} (/dev/video{self.camera_index})')
            return self.camera_index
        # Auto-detect: scan all indices, skip RealSense (video4-9) and laptop webcam (video0-3)
        # Logitech C270 is typically at video10+
        for i in [10, 11, 12, 0, 1, 2]:
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    cap.release()
                    self.get_logger().info(f'Auto-detected camera at /dev/video{i}')
                    return i
            cap.release()
        return None

    def cv_loop(self):
        cam_index = self.find_camera()
        if cam_index is None:
            self.get_logger().error('No camera found! Node will publish zero velocity.')
            return

        cap = cv2.VideoCapture(cam_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.HEIGHT)
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # Disable autofocus
        cap.set(cv2.CAP_PROP_FOCUS, 0)      # Fixed focus at infinity

        # Video recorder
        out = None
        if self.record_video:
            os.makedirs('cable_trace_logs', exist_ok=True)
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            video_file = f'cable_trace_logs/trace_{timestamp}.avi'
            csv_file = f'cable_trace_logs/trace_{timestamp}.csv'
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(video_file, fourcc, self.FPS, (self.WIDTH, self.HEIGHT))
            csv_f = open(csv_file, 'w', newline='')
            csv_writer = csv.writer(csv_f)
            csv_writer.writerow(['time', 'velx', 'velw', 'raw_velx', 'raw_velw', 'raw_vely'])
            self.get_logger().info(f'Recording to {video_file}')

        start_time = time.time()
        frame_count = 0

        with torch.no_grad():
            while rclpy.ok() and self.running:
                ret, frame = cap.read()
                if not ret:
                    continue

                # Inference
                img = self.preprocess(frame)
                pred = self.model(img).squeeze().cpu().numpy()

                raw_velx = pred[0]
                raw_velw = pred[1]
                raw_vely = pred[2]

                # Clamp to wheelchair safety limits
                velx = float(np.clip(raw_velx, 0.0, self.max_lin))
                velw = float(np.clip(raw_velw, -self.max_ang, self.max_ang))

                self.velx = velx
                self.velw = velw

                # Log and record
                if self.record_video and out is not None:
                    out.write(frame)
                    csv_writer.writerow([
                        f'{time.time() - start_time:.3f}',
                        f'{velx:.4f}', f'{velw:.4f}',
                        f'{raw_velx:.4f}', f'{raw_velw:.4f}', f'{raw_vely:.4f}'
                    ])

                # OSD overlay
                if self.show_gui:
                    disp = frame.copy()
                    cv2.putText(disp, f'velx: {velx:.3f} m/s', (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(disp, f'velw: {velw:.3f} rad/s', (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(disp, f'FPS: {frame_count / max(time.time()-start_time, 1):.1f}',
                                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.imshow('Cable Trace', disp)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self.get_logger().info('User pressed Q — stopping')
                        self.running = False
                        break

                frame_count += 1

        # Cleanup
        cap.release()
        if out is not None:
            out.release()
            csv_f.close()
        if self.show_gui:
            cv2.destroyAllWindows()

        # Send zero velocity on exit
        self.velx = 0.0
        self.velw = 0.0
        self.get_logger().info(f'Camera loop ended. {frame_count} frames processed.')


def main():
    rclpy.init()
    node = CableTraceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.running = False
        # Publish zero velocity before shutdown
        msg = Twist()
        node.pub.publish(msg)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
