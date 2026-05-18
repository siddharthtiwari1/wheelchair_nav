#!/usr/bin/env python3
"""
Cable Tracer v3 Inference Node — temporal ResNet-SE with red mask + CBAM.

Maintains a 3-frame buffer for temporal context. Each frame is preprocessed
to 4 channels (RGB + red_mask). The 12-channel stack feeds the CNN.

Subscribes to: /logitech/image_raw (RGB)
Publishes to:  /cmd_vel (Twist)

Usage:
    python3 cable_tracer_node.py --ros-args \
        -p model_path:=models/cable_tracer/cable_tracer.pt \
        -p v_max:=0.15 \
        -p enabled:=false
"""

import collections
import json
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32

N_FRAMES = 3
CH_PER_FRAME = 4
IN_CHANNELS = N_FRAMES * CH_PER_FRAME

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ============================================================================
# MODEL — must match training architecture exactly
# ============================================================================

class SEBlock(nn.Module):
    def __init__(self, ch, reduction=4):
        super().__init__()
        mid = max(ch // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(ch, mid), nn.ReLU(),
            nn.Linear(mid, ch), nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.shape
        w = self.pool(x).view(b, c)
        return x * self.fc(w).view(b, c, 1, 1)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.bn = nn.BatchNorm2d(1)

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        mx = x.max(dim=1, keepdim=True)[0]
        return x * torch.sigmoid(self.bn(self.conv(torch.cat([avg, mx], dim=1))))


class ResBlockSE(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.se = SEBlock(out_ch)
        self.skip = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        ) if in_ch != out_ch else nn.Identity()
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        identity = self.skip(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        return self.pool(F.relu(out + identity))


class CableTracerCNN(nn.Module):
    def __init__(self, in_channels=IN_CHANNELS):
        super().__init__()
        self.block1 = ResBlockSE(in_channels, 24)
        self.block2 = ResBlockSE(24, 48)
        self.block3 = ResBlockSE(48, 64)
        self.spatial_attn = SpatialAttention()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.v_head = nn.Sequential(
            nn.Linear(64, 32), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(32, 1), nn.Tanh())
        self.omega_head = nn.Sequential(
            nn.Linear(64, 32), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(32, 1), nn.Tanh())
        self.confidence_head = nn.Sequential(
            nn.Linear(64, 16), nn.GELU(),
            nn.Linear(16, 1), nn.Sigmoid())

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.spatial_attn(x)
        x = self.pool(x).flatten(1)
        return torch.cat([self.v_head(x), self.omega_head(x),
                          self.confidence_head(x)], dim=1)


# ============================================================================
# PREPROCESSING
# ============================================================================

def extract_red_mask(img_rgb_01):
    """Extract red cable mask from RGB float [0,1]. Returns float [0,1] mask."""
    img_u8 = (img_rgb_01 * 255).astype(np.uint8)
    hsv = cv2.cvtColor(img_u8, cv2.COLOR_RGB2HSV)
    m1 = cv2.inRange(hsv, (0, 70, 50), (12, 255, 255))
    m2 = cv2.inRange(hsv, (168, 70, 50), (180, 255, 255))
    return ((m1 | m2) > 0).astype(np.float32)


def preprocess_frame(img_rgb, img_size=128):
    """Single frame → 4ch numpy array (HWC): ImageNet-normalized RGB + red_mask."""
    img = cv2.resize(img_rgb, (img_size, img_size)).astype(np.float32) / 255.0
    red_mask = extract_red_mask(img)
    img_norm = (img - IMAGENET_MEAN) / IMAGENET_STD
    return np.concatenate([img_norm, red_mask[:, :, np.newaxis]], axis=-1)


# ============================================================================
# ROS2 NODE
# ============================================================================

class CableTracerNode(Node):
    def __init__(self):
        super().__init__('cable_tracer_node')

        self.declare_parameter('model_path', 'models/cable_tracer/cable_tracer.pt')
        self.declare_parameter('v_max', 0.50)
        self.declare_parameter('omega_max', 0.15)
        self.declare_parameter('enabled', False)
        self.declare_parameter('image_topic', '/logitech/image_raw')
        self.declare_parameter('inference_hz', 10.0)
        self.declare_parameter('confidence_threshold', 0.3)

        model_path = self.get_parameter('model_path').value
        self.v_max = self.get_parameter('v_max').value
        self.omega_max = self.get_parameter('omega_max').value
        self.enabled = self.get_parameter('enabled').value
        image_topic = self.get_parameter('image_topic').value
        self.inference_interval = 1.0 / self.get_parameter('inference_hz').value
        self.confidence_threshold = self.get_parameter('confidence_threshold').value

        # Load model — auto-detect v3 (2-output) vs v4 (3-output with confidence)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = CableTracerCNN(in_channels=IN_CHANNELS).to(self.device)
        state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
        self.has_confidence = 'confidence_head.0.weight' in state_dict
        if not self.has_confidence:
            # v3 model — load without confidence head (strict=False)
            self.model.load_state_dict(state_dict, strict=False)
        else:
            self.model.load_state_dict(state_dict)
        self.model.eval()
        n_params = sum(p.numel() for p in self.model.parameters())
        version = 'v4 (confidence)' if self.has_confidence else 'v3'
        self.get_logger().info(
            f'Model loaded: {version}, {n_params:,} params on {self.device} '
            f'({N_FRAMES}-frame temporal, {IN_CHANNELS}ch)')

        # Temporal frame buffer
        self.frame_buffer = collections.deque(maxlen=N_FRAMES)

        self.bridge = CvBridge()
        self.last_inference_time = 0.0

        # Subscribers
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(Image, image_topic, self._image_cb, sensor_qos)
        self.create_subscription(Bool, '/cable_tracer/enable', self._enable_cb, 10)

        # Publishers
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.conf_pub = self.create_publisher(Float32, '/cable_tracer/confidence', 10)

        state = 'ENABLED' if self.enabled else 'DISABLED'
        self.get_logger().info(
            f'Cable tracer {state} | v_max={self.v_max} omega_max={self.omega_max} '
            f'conf_thresh={self.confidence_threshold}')
        self.get_logger().info(
            '  Publish /cable_tracer/enable (Bool) to start/stop')

    def _enable_cb(self, msg):
        self.enabled = msg.data
        state = 'ENABLED' if self.enabled else 'DISABLED'
        self.get_logger().info(f'Cable tracer {state}')
        if not self.enabled:
            self.cmd_pub.publish(Twist())
            self.frame_buffer.clear()

    @torch.no_grad()
    def _image_cb(self, msg):
        now = time.monotonic()
        if now - self.last_inference_time < self.inference_interval:
            return
        self.last_inference_time = now

        if not self.enabled:
            return

        # Convert ROS image → RGB numpy
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, 'rgb8')
        except Exception as e:
            self.get_logger().warn(f'Image conversion error: {e}')
            return

        # Preprocess and add to temporal buffer
        frame = preprocess_frame(cv_img)  # 128x128x4
        self.frame_buffer.append(frame)

        # Pad buffer at startup (duplicate first frame)
        while len(self.frame_buffer) < N_FRAMES:
            self.frame_buffer.appendleft(self.frame_buffer[0])

        # Stack temporal frames: N_FRAMES x 4ch = 12ch
        stacked = np.concatenate(list(self.frame_buffer), axis=-1)  # 128x128x12
        tensor = torch.from_numpy(
            np.transpose(stacked, (2, 0, 1))[np.newaxis].astype(np.float32)
        ).to(self.device)

        # Inference
        pred = self.model(tensor).cpu().numpy()[0]
        v_raw = float(pred[0]) * self.v_max
        omega_raw = float(pred[1]) * self.omega_max

        # Confidence gate (v4 only; v3 always publishes)
        if self.has_confidence:
            confidence = float(pred[2])
            conf_msg = Float32()
            conf_msg.data = confidence
            self.conf_pub.publish(conf_msg)
        else:
            confidence = 1.0

        cmd = Twist()
        if confidence >= self.confidence_threshold:
            cmd.linear.x = v_raw
            cmd.angular.z = omega_raw
        self.cmd_pub.publish(cmd)

        self.get_logger().debug(
            f'v={cmd.linear.x:.3f} w={cmd.angular.z:.3f} conf={confidence:.2f}')


def main(args=None):
    rclpy.init(args=args)
    node = CableTracerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cmd_pub.publish(Twist())
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
