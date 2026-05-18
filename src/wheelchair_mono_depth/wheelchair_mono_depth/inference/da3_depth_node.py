#!/usr/bin/env python3
"""ROS2 real-time monocular depth node — DA3 Metric Large, zero-lag.

Uses native depth-anything-3 with DIRECT forward() call (bypasses the slow
inference() API). Physics-grounded metric depth via focal scaling:
    depth_m = exp(logits) × (avg_focal / 300.0)

Performance on RTX 5050 at 504x280:
    Baseline:       ~60ms (16 FPS)
    + torch.compile + FP16 + lazy PC: ~50ms (20 FPS)
    At 364x210:     ~33ms (30 FPS)

Optimizations applied (see research audit 2026-03-04):
    1. Direct model.forward() instead of model.inference()  (~200ms saved)
    2. torch.compile(mode='reduce-overhead')                (~5ms saved)
    3. FP16 autocast (more accurate than BF16: 1mm vs 2.5mm mean error)
    4. cudnn.benchmark = True AFTER DA3 import (DA3 api.py overrides it)
    5. Lazy pointcloud (skip if no subscribers)             (~1.8ms saved)
    6. Direct rgb8 encoding (eliminates double color convert)
    7. EMA temporal filter for depth consistency
    8. QoS depth=1 (always process freshest frame)

Usage:
    ros2 run wheelchair_mono_depth da3_depth_node --ros-args \
        -p model_name:=da3-metric-large
"""

import time
import os

import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from cv_bridge import CvBridge


# DA3 focal scaling constant (from DA3 paper / alignment.py)
DA3_SCALE_FACTOR = 300.0

# ViT patch size — dimensions must be multiples of this
PATCH_SIZE = 14

# ImageNet normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class DA3DepthNode(Node):
    """Real-time DA3 monocular depth node with direct forward() inference."""

    def __init__(self):
        super().__init__('da3_depth_node')

        # --- Parameters ---
        self.declare_parameter('model_name', 'da3-metric-large')
        self.declare_parameter('max_depth', 6.0)
        self.declare_parameter('depth_correction', 1.0)  # multiply after focal scaling
        self.declare_parameter('inference_hz', 15.0)
        self.declare_parameter('process_width', 504)
        self.declare_parameter('rgb_topic', '/camera/color/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/color/camera_info')
        self.declare_parameter('output_depth_topic', '/camera/mono_depth/image_raw')
        self.declare_parameter('output_info_topic', '/camera/mono_depth/camera_info')
        self.declare_parameter('pointcloud_topic', '/camera/mono_depth/points')
        self.declare_parameter('compile_model', True)
        self.declare_parameter('temporal_alpha', 0.0)  # 0=off, 0.7=recommended

        self._model_name = self.get_parameter('model_name').value
        self._max_depth = self.get_parameter('max_depth').value
        self._depth_correction = self.get_parameter('depth_correction').value
        self._process_width = self.get_parameter('process_width').value
        self._compile_model = self.get_parameter('compile_model').value
        self._temporal_alpha = self.get_parameter('temporal_alpha').value
        inference_hz = self.get_parameter('inference_hz').value

        # Force process_width to multiple of 14
        self._process_width = (self._process_width // PATCH_SIZE) * PATCH_SIZE

        self._bridge = CvBridge()
        self._inference_interval = 1.0 / inference_hz
        self._last_inference_time = 0.0
        self._frame_count = 0
        self._total_latency = 0.0

        # Intrinsics from real camera
        self._fx = None
        self._fy = None
        self._cx = None
        self._cy = None
        self._cam_width = None
        self._cam_height = None

        # Pre-allocated buffers (set on first frame)
        self._uv_grid = None
        self._proc_w = None
        self._proc_h = None

        # EMA temporal filter state
        self._prev_depth = None

        # --- Load model ---
        self._load_model()

        # --- QoS ---
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,  # always process freshest frame
        )

        # --- Publishers ---
        depth_topic = self.get_parameter('output_depth_topic').value
        info_topic = self.get_parameter('output_info_topic').value
        pc_topic = self.get_parameter('pointcloud_topic').value

        self._depth_pub = self.create_publisher(Image, depth_topic, 10)
        self._info_pub = self.create_publisher(CameraInfo, info_topic, 10)
        self._pc_pub = self.create_publisher(PointCloud2, pc_topic, 10)

        # --- Subscribers ---
        rgb_topic = self.get_parameter('rgb_topic').value
        camera_info_topic = self.get_parameter('camera_info_topic').value

        self._rgb_sub = self.create_subscription(
            Image, rgb_topic, self._rgb_callback, sensor_qos)
        self._info_sub = self.create_subscription(
            CameraInfo, camera_info_topic, self._camera_info_cb, sensor_qos)

        self.get_logger().info(
            f'DA3DepthNode [{self._model_name}] DIRECT forward: '
            f'{rgb_topic} -> {depth_topic} '
            f'(process_width={self._process_width}, '
            f'focal_scaling x {self._depth_correction}, {inference_hz}Hz)')

    # ====================================================================
    # MODEL LOADING
    # ====================================================================
    def _load_model(self):
        """Load DA3 model with direct forward() path.

        Optimizations:
          - cudnn.benchmark set AFTER DA3 import (api.py sets it to False)
          - torch.compile(mode='reduce-overhead') for ~5ms savings
          - FP16 autocast (more accurate than DA3's default BF16)
        """
        import torch
        from torchvision import transforms as T

        os.environ['HF_HUB_OFFLINE'] = '1'

        self._device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.get_logger().info(f'Using device: {self._device}')

        from depth_anything_3.api import DepthAnything3

        # CRITICAL: set cudnn.benchmark AFTER import — DA3 api.py sets it False
        if self._device.type == 'cuda':
            torch.backends.cudnn.benchmark = True

        model_ids = {
            'da3-metric-large': 'depth-anything/DA3METRIC-LARGE',
            'da3-large': 'depth-anything/DA3-LARGE',
            'da3-small': 'depth-anything/DA3-SMALL',
        }
        model_id = model_ids.get(self._model_name)
        if not model_id:
            raise ValueError(
                f'Unknown model: {self._model_name}. '
                f'Available: {list(model_ids.keys())}')

        self.get_logger().info(f'Loading: {model_id}')
        self._model = DepthAnything3.from_pretrained(model_id)
        self._model = self._model.to(device=self._device)
        self._model.eval()

        # Extract inner model for direct calls (bypass API autocast)
        self._inner_model = self._model.model

        # torch.compile for kernel fusion (~5ms savings)
        if self._compile_model and self._device.type == 'cuda':
            self.get_logger().info(
                'Compiling model with torch.compile(reduce-overhead)...')
            self._inner_model = torch.compile(
                self._inner_model, mode='reduce-overhead', fullgraph=False)

        # Pre-build ImageNet normalization on GPU
        self._normalize = T.Normalize(
            mean=IMAGENET_MEAN, std=IMAGENET_STD)

        self._torch = torch  # cache import

        # Warmup — first forward is slow due to CUDA kernel compilation
        # With torch.compile, warmup takes 20-60s (one-time cost)
        proc_w = self._process_width
        proc_h = (int(480 * proc_w / 640) // PATCH_SIZE) * PATCH_SIZE
        dummy = torch.randn(1, 1, 3, proc_h, proc_w, device=self._device)
        n_warmup = 5 if self._compile_model else 1
        self.get_logger().info(
            f'Warming up ({n_warmup} passes, '
            f'compile={self._compile_model})...')
        for i in range(n_warmup):
            with torch.no_grad():
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    _ = self._inner_model(dummy, export_feat_layers=[])
            torch.cuda.synchronize()
        self.get_logger().info(f'Model ready: {model_id}')

    # ====================================================================
    # CAMERA INFO
    # ====================================================================
    def _camera_info_cb(self, msg: CameraInfo):
        """Capture real camera intrinsics once."""
        if self._fx is not None:
            return
        K = msg.k
        self._fx = K[0]
        self._fy = K[4]
        self._cx = K[2]
        self._cy = K[5]
        self._cam_width = msg.width
        self._cam_height = msg.height
        self.get_logger().info(
            f'Camera intrinsics: {msg.width}x{msg.height}, '
            f'fx={K[0]:.1f}, fy={K[4]:.1f}, '
            f'cx={K[2]:.1f}, cy={K[5]:.1f}')

    # ====================================================================
    # PRE-ALLOCATE BUFFERS
    # ====================================================================
    def _init_buffers(self, proc_w, proc_h):
        """Pre-allocate meshgrid for pointcloud generation."""
        self._proc_w = proc_w
        self._proc_h = proc_h
        u = np.arange(proc_w, dtype=np.float32)
        v = np.arange(proc_h, dtype=np.float32)
        self._u_grid, self._v_grid = np.meshgrid(u, v)
        self.get_logger().info(
            f'Buffers initialized: {proc_w}x{proc_h} '
            f'({proc_w * proc_h} pixels)')

    # ====================================================================
    # RGB CALLBACK
    # ====================================================================
    def _rgb_callback(self, msg: Image):
        """Process incoming RGB image — zero-lag direct forward path."""
        now = time.monotonic()
        if now - self._last_inference_time < self._inference_interval:
            return
        self._last_inference_time = now

        torch = self._torch
        t0 = time.monotonic()

        # Convert directly to RGB (avoids double color conversion)
        try:
            rgb = self._bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        except Exception as e:
            self.get_logger().warn(f'RGB conversion failed: {e}')
            return

        cam_h, cam_w = rgb.shape[:2]

        # Fallback intrinsics
        if self._fx is None:
            self._cam_width = cam_w
            self._cam_height = cam_h
            self._fx = cam_w * 0.9
            self._fy = cam_h * 0.9
            self._cx = cam_w / 2.0
            self._cy = cam_h / 2.0
            self.get_logger().warn(
                'Using approximate intrinsics',
                throttle_duration_sec=10.0)

        # Compute process dimensions (multiples of 14)
        scale = self._process_width / cam_w
        proc_w = self._process_width
        proc_h = (int(cam_h * scale) // PATCH_SIZE) * PATCH_SIZE

        # Scale intrinsics
        fx = self._fx * scale
        fy = self._fy * scale
        cx = self._cx * scale
        cy = self._cy * scale

        # Init buffers on first frame or resolution change
        if self._proc_w != proc_w or self._proc_h != proc_h:
            self._init_buffers(proc_w, proc_h)

        # --- RESIZE ---
        rgb_proc = cv2.resize(rgb, (proc_w, proc_h),
                              interpolation=cv2.INTER_AREA)

        # --- PREPROCESS (RGB → GPU → float/normalize) ---
        tensor = torch.from_numpy(rgb_proc).to(self._device, non_blocking=True)
        tensor = tensor.permute(2, 0, 1).float().div_(255.0)
        tensor = self._normalize(tensor)
        tensor = tensor.unsqueeze(0).unsqueeze(0)  # (1,1,3,H,W)

        # --- DIRECT FORWARD with FP16 autocast ---
        # FP16 is more accurate than DA3's default BF16 (1mm vs 2.5mm error)
        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                output = self._inner_model(tensor, export_feat_layers=[])

        # Extract canonical depth
        canonical = output["depth"].squeeze().cpu().numpy()  # (H', W')

        # Resize if model output differs from process resolution
        if canonical.shape != (proc_h, proc_w):
            canonical = cv2.resize(canonical, (proc_w, proc_h),
                                   interpolation=cv2.INTER_LINEAR)

        # --- FOCAL SCALING: physics-grounded metric conversion ---
        avg_focal = (fx + fy) / 2.0
        depth_m = canonical * (avg_focal / DA3_SCALE_FACTOR) * self._depth_correction

        # Clamp
        depth_m = np.clip(depth_m, 0.0, self._max_depth).astype(np.float32)

        # --- EMA TEMPORAL FILTER (reduces flickering) ---
        alpha = self._temporal_alpha
        if alpha > 0 and self._prev_depth is not None \
                and self._prev_depth.shape == depth_m.shape:
            depth_m = alpha * depth_m + (1.0 - alpha) * self._prev_depth
        if alpha > 0:
            self._prev_depth = depth_m.copy()

        # --- PUBLISH DEPTH IMAGE ---
        depth_mm = (depth_m * 1000.0).clip(0, 65535).astype(np.uint16)
        depth_msg = self._bridge.cv2_to_imgmsg(depth_mm, encoding='16UC1')
        depth_msg.header = msg.header
        self._depth_pub.publish(depth_msg)

        # --- PUBLISH CAMERA INFO ---
        info_msg = CameraInfo()
        info_msg.header = msg.header
        info_msg.width = proc_w
        info_msg.height = proc_h
        info_msg.distortion_model = 'plumb_bob'
        info_msg.d = [0.0, 0.0, 0.0, 0.0, 0.0]
        info_msg.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
        info_msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        info_msg.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
        self._info_pub.publish(info_msg)

        # --- PUBLISH POINTCLOUD (lazy — skip if no subscribers) ---
        if self._pc_pub.get_subscription_count() > 0:
            z = depth_m
            x = (self._u_grid - cx) * z / fx
            y = (self._v_grid - cy) * z / fy
            valid = z > 0.01
            points = np.stack(
                [x[valid], y[valid], z[valid]], axis=-1).astype(np.float32)

            pc_msg = PointCloud2()
            pc_msg.header = msg.header
            pc_msg.height = 1
            pc_msg.width = len(points)
            pc_msg.is_dense = True
            pc_msg.is_bigendian = False
            pc_msg.point_step = 12
            pc_msg.row_step = 12 * len(points)
            pc_msg.fields = [
                PointField(name='x', offset=0,
                           datatype=PointField.FLOAT32, count=1),
                PointField(name='y', offset=4,
                           datatype=PointField.FLOAT32, count=1),
                PointField(name='z', offset=8,
                           datatype=PointField.FLOAT32, count=1),
            ]
            pc_msg.data = points.tobytes()
            self._pc_pub.publish(pc_msg)

        # --- LATENCY TRACKING ---
        latency_ms = (time.monotonic() - t0) * 1000.0
        self._total_latency += latency_ms
        self._frame_count += 1
        if self._frame_count % 10 == 0:
            avg = self._total_latency / self._frame_count
            valid_depth = depth_m[depth_m > 0.01]
            if len(valid_depth) > 0:
                self.get_logger().info(
                    f'[{self._model_name}] Frame {self._frame_count}: '
                    f'{latency_ms:.0f}ms ({1000.0/avg:.1f} FPS) | '
                    f'{proc_w}x{proc_h}, '
                    f'depth=[{valid_depth.min():.2f}, {np.median(valid_depth):.2f}, '
                    f'{valid_depth.max():.2f}]m')


def main(args=None):
    rclpy.init(args=args)
    node = DA3DepthNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
