#!/usr/bin/env python3
"""Multi-camera DA3 Metric depth node — single model, round-robin inference.

One DA3 model instance shared across N cameras via round-robin scheduling.
VRAM: ~2.3GB (single model) vs ~6.9GB (3 separate models).

Each camera gets its own:
  - Process width (front=504px, sides=364px for speed)
  - UV grid buffers and intrinsics
  - Temporal filter state
  - Output publishers (depth, info, pointcloud)

Performance (RTX 5050):
  504px: ~60ms,  364px: ~33ms
  3-cam cycle: ~126ms → each camera ~8 FPS

No depth correction applied — raw DA3 focal scaling output.
Correction function to be derived from stereo-DA3 calibration data.

Usage:
    ros2 run wheelchair_mono_depth da3_multi_depth_node --ros-args \
        -p cameras.front.rgb_topic:=/camera/color/image_raw \
        -p cameras.front.process_width:=504 \
        -p cameras.left.rgb_topic:=/mapping_camera/color/image_raw \
        -p cameras.left.process_width:=364
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

# Default camera configurations
DEFAULT_CAMERAS = {
    'front': {
        'rgb_topic': '/camera/color/image_raw',
        'info_topic': '/camera/color/camera_info',
        'output_prefix': '/camera/mono_da3',
        'process_width': 504,
    },
    'left': {
        'rgb_topic': '/mapping_camera/color/image_raw',
        'info_topic': '/mapping_camera/color/camera_info',
        'output_prefix': '/mapping_camera/mono_da3',
        'process_width': 364,
    },
    'right': {
        'rgb_topic': '/right_camera/color/image_raw',
        'info_topic': '/right_camera/color/camera_info',
        'output_prefix': '/right_camera/mono_da3',
        'process_width': 364,
    },
}


class CameraState:
    """Per-camera state for round-robin inference."""

    def __init__(self, name):
        self.name = name

        # Latest RGB message (set by subscriber)
        self.latest_rgb = None
        self.has_new_frame = False

        # Intrinsics (from CameraInfo)
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None
        self.cam_width = None
        self.cam_height = None

        # Process dimensions (multiples of PATCH_SIZE)
        self.process_width = 504
        self.proc_w = None
        self.proc_h = None

        # Pre-allocated UV grid for pointcloud
        self.u_grid = None
        self.v_grid = None

        # EMA temporal filter
        self.prev_depth = None

        # Stats
        self.frame_count = 0
        self.total_latency = 0.0

        # Publishers (set during init)
        self.depth_pub = None
        self.info_pub = None
        self.pc_pub = None

    def init_buffers(self, proc_w, proc_h):
        """Pre-allocate meshgrid for pointcloud generation."""
        self.proc_w = proc_w
        self.proc_h = proc_h
        u = np.arange(proc_w, dtype=np.float32)
        v = np.arange(proc_h, dtype=np.float32)
        self.u_grid, self.v_grid = np.meshgrid(u, v)


class DA3MultiDepthNode(Node):
    """Multi-camera DA3 depth node with single model, round-robin inference."""

    def __init__(self):
        super().__init__('da3_multi_depth_node')

        # --- Global parameters ---
        self.declare_parameter('model_name', 'da3-metric-large')
        self.declare_parameter('max_depth', 8.0)
        self.declare_parameter('depth_correction', 1.0)
        self.declare_parameter('compile_model', False)
        self.declare_parameter('temporal_alpha', 0.0)
        self.declare_parameter('camera_names', ['front', 'left', 'right'])

        self._model_name = self.get_parameter('model_name').value
        self._max_depth = self.get_parameter('max_depth').value
        self._depth_correction = self.get_parameter('depth_correction').value
        self._compile_model = self.get_parameter('compile_model').value
        self._temporal_alpha = self.get_parameter('temporal_alpha').value
        camera_names = self.get_parameter('camera_names').value

        self._bridge = CvBridge()

        # --- Per-camera parameters ---
        self._cameras = []
        for name in camera_names:
            defaults = DEFAULT_CAMERAS.get(name, DEFAULT_CAMERAS['front'])

            self.declare_parameter(f'cameras.{name}.rgb_topic',
                                   defaults['rgb_topic'])
            self.declare_parameter(f'cameras.{name}.info_topic',
                                   defaults['info_topic'])
            self.declare_parameter(f'cameras.{name}.output_prefix',
                                   defaults['output_prefix'])
            self.declare_parameter(f'cameras.{name}.process_width',
                                   defaults['process_width'])

            cam = CameraState(name)
            cam.process_width = self.get_parameter(
                f'cameras.{name}.process_width').value
            # Force to multiple of PATCH_SIZE
            cam.process_width = (cam.process_width // PATCH_SIZE) * PATCH_SIZE
            self._cameras.append(cam)

        self._current_idx = 0

        # --- QoS ---
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # --- Create subscribers and publishers per camera ---
        for i, cam in enumerate(self._cameras):
            name = cam.name
            rgb_topic = self.get_parameter(f'cameras.{name}.rgb_topic').value
            info_topic = self.get_parameter(f'cameras.{name}.info_topic').value
            prefix = self.get_parameter(f'cameras.{name}.output_prefix').value

            # RGB subscriber
            self.create_subscription(
                Image, rgb_topic,
                lambda msg, c=cam: self._rgb_callback(c, msg),
                sensor_qos)

            # CameraInfo subscriber
            self.create_subscription(
                CameraInfo, info_topic,
                lambda msg, c=cam: self._info_callback(c, msg),
                sensor_qos)

            # Publishers
            cam.depth_pub = self.create_publisher(
                Image, f'{prefix}/image_raw', 10)
            cam.info_pub = self.create_publisher(
                CameraInfo, f'{prefix}/camera_info', 10)
            cam.pc_pub = self.create_publisher(
                PointCloud2, f'{prefix}/points', 10)

            self.get_logger().info(
                f'Camera [{name}]: {rgb_topic} -> {prefix}/points '
                f'(width={cam.process_width})')

        # --- Load model ---
        self._load_model()

        # --- Round-robin timer ---
        # Run at combined rate: cycle through all cameras
        cycle_ms = 10.0  # 100Hz polling, actual rate limited by inference time
        self._timer = self.create_timer(cycle_ms / 1000.0, self._cycle_callback)
        self._processing = False  # Guard against re-entrant calls

        self.get_logger().info(
            f'DA3MultiDepthNode [{self._model_name}]: '
            f'{len(self._cameras)} cameras, '
            f'correction={self._depth_correction}')

    # ====================================================================
    # MODEL LOADING (identical to da3_depth_node)
    # ====================================================================
    def _load_model(self):
        """Load DA3 model with direct forward() path."""
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

        # torch.compile for kernel fusion
        if self._compile_model and self._device.type == 'cuda':
            self.get_logger().info(
                'Compiling model with torch.compile(reduce-overhead)...')
            self._inner_model = torch.compile(
                self._inner_model, mode='reduce-overhead', fullgraph=False)

        # Pre-build ImageNet normalization on GPU
        self._normalize = T.Normalize(
            mean=IMAGENET_MEAN, std=IMAGENET_STD)

        self._torch = torch

        # Warmup with smallest process width for fast startup
        min_width = min(c.process_width for c in self._cameras)
        proc_h = (int(480 * min_width / 640) // PATCH_SIZE) * PATCH_SIZE
        dummy = torch.randn(1, 1, 3, proc_h, min_width, device=self._device)
        n_warmup = 5 if self._compile_model else 1
        self.get_logger().info(
            f'Warming up ({n_warmup} passes, '
            f'compile={self._compile_model})...')
        for _ in range(n_warmup):
            with torch.no_grad():
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    _ = self._inner_model(dummy, export_feat_layers=[])
            torch.cuda.synchronize()
        self.get_logger().info(f'Model ready: {model_id}')

    # ====================================================================
    # CALLBACKS
    # ====================================================================
    def _rgb_callback(self, cam, msg):
        """Store latest RGB frame."""
        cam.latest_rgb = msg
        cam.has_new_frame = True

    def _info_callback(self, cam, msg):
        """Capture real camera intrinsics once per camera."""
        if cam.fx is not None:
            return
        K = msg.k
        cam.fx = K[0]
        cam.fy = K[4]
        cam.cx = K[2]
        cam.cy = K[5]
        cam.cam_width = msg.width
        cam.cam_height = msg.height
        self.get_logger().info(
            f'[{cam.name}] Intrinsics: {msg.width}x{msg.height}, '
            f'fx={K[0]:.1f}, fy={K[4]:.1f}')

    # ====================================================================
    # ROUND-ROBIN CYCLE
    # ====================================================================
    def _cycle_callback(self):
        """Process one camera per cycle in round-robin order."""
        if self._processing:
            return
        self._processing = True

        try:
            # Find next camera with unprocessed data
            for _ in range(len(self._cameras)):
                cam = self._cameras[self._current_idx]
                self._current_idx = (
                    (self._current_idx + 1) % len(self._cameras))

                if cam.has_new_frame and cam.latest_rgb is not None:
                    self._process_camera(cam)
                    return
        finally:
            self._processing = False

    # ====================================================================
    # INFERENCE
    # ====================================================================
    def _process_camera(self, cam):
        """Run DA3 inference on one camera and publish results."""
        msg = cam.latest_rgb
        cam.has_new_frame = False

        torch = self._torch
        t0 = time.monotonic()

        # Convert to RGB
        try:
            rgb = self._bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        except Exception as e:
            self.get_logger().warn(
                f'[{cam.name}] RGB conversion failed: {e}')
            return

        cam_h, cam_w = rgb.shape[:2]

        # Fallback intrinsics
        if cam.fx is None:
            cam.cam_width = cam_w
            cam.cam_height = cam_h
            cam.fx = cam_w * 0.9
            cam.fy = cam_h * 0.9
            cam.cx = cam_w / 2.0
            cam.cy = cam_h / 2.0
            self.get_logger().warn(
                f'[{cam.name}] Using approximate intrinsics',
                throttle_duration_sec=10.0)

        # Compute process dimensions (multiples of PATCH_SIZE)
        scale = cam.process_width / cam_w
        proc_w = cam.process_width
        proc_h = (int(cam_h * scale) // PATCH_SIZE) * PATCH_SIZE

        # Scale intrinsics to process resolution
        fx = cam.fx * scale
        fy = cam.fy * scale
        cx = cam.cx * scale
        cy = cam.cy * scale

        # Init buffers on first frame or resolution change
        if cam.proc_w != proc_w or cam.proc_h != proc_h:
            cam.init_buffers(proc_w, proc_h)
            self.get_logger().info(
                f'[{cam.name}] Buffers: {proc_w}x{proc_h}')

        # --- RESIZE ---
        rgb_proc = cv2.resize(rgb, (proc_w, proc_h),
                              interpolation=cv2.INTER_AREA)

        # --- PREPROCESS ---
        tensor = torch.from_numpy(rgb_proc).to(
            self._device, non_blocking=True)
        tensor = tensor.permute(2, 0, 1).float().div_(255.0)
        tensor = self._normalize(tensor)
        tensor = tensor.unsqueeze(0).unsqueeze(0)  # (1,1,3,H,W)

        # --- DIRECT FORWARD with FP16 autocast ---
        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                output = self._inner_model(tensor, export_feat_layers=[])

        # Extract canonical depth
        canonical = output["depth"].squeeze().cpu().numpy()

        # Resize if model output differs
        if canonical.shape != (proc_h, proc_w):
            canonical = cv2.resize(canonical, (proc_w, proc_h),
                                   interpolation=cv2.INTER_LINEAR)

        # --- FOCAL SCALING: physics-grounded metric conversion ---
        avg_focal = (fx + fy) / 2.0
        depth_m = canonical * (avg_focal / DA3_SCALE_FACTOR) * \
            self._depth_correction

        # Clamp
        depth_m = np.clip(depth_m, 0.0, self._max_depth).astype(np.float32)

        # --- EMA TEMPORAL FILTER ---
        alpha = self._temporal_alpha
        if alpha > 0 and cam.prev_depth is not None \
                and cam.prev_depth.shape == depth_m.shape:
            depth_m = alpha * depth_m + (1.0 - alpha) * cam.prev_depth
        if alpha > 0:
            cam.prev_depth = depth_m.copy()

        # --- PUBLISH DEPTH IMAGE ---
        depth_mm = (depth_m * 1000.0).clip(0, 65535).astype(np.uint16)
        depth_msg = self._bridge.cv2_to_imgmsg(depth_mm, encoding='16UC1')
        depth_msg.header = msg.header
        cam.depth_pub.publish(depth_msg)

        # --- PUBLISH CAMERA INFO ---
        info_msg = CameraInfo()
        info_msg.header = msg.header
        info_msg.width = proc_w
        info_msg.height = proc_h
        info_msg.distortion_model = 'plumb_bob'
        info_msg.d = [0.0, 0.0, 0.0, 0.0, 0.0]
        info_msg.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
        info_msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        info_msg.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0,
                      0.0, 0.0, 1.0, 0.0]
        cam.info_pub.publish(info_msg)

        # --- PUBLISH POINTCLOUD (lazy) ---
        if cam.pc_pub.get_subscription_count() > 0:
            z = depth_m
            x = (cam.u_grid - cx) * z / fx
            y = (cam.v_grid - cy) * z / fy
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
            cam.pc_pub.publish(pc_msg)

        # --- LATENCY TRACKING ---
        latency_ms = (time.monotonic() - t0) * 1000.0
        cam.total_latency += latency_ms
        cam.frame_count += 1
        if cam.frame_count % 30 == 0:
            avg = cam.total_latency / cam.frame_count
            valid_depth = depth_m[depth_m > 0.01]
            if len(valid_depth) > 0:
                self.get_logger().info(
                    f'[{cam.name}] Frame {cam.frame_count}: '
                    f'{latency_ms:.0f}ms (avg {avg:.0f}ms) | '
                    f'{proc_w}x{proc_h}, '
                    f'depth=[{valid_depth.min():.2f}, '
                    f'{np.median(valid_depth):.2f}, '
                    f'{valid_depth.max():.2f}]m')


def main(args=None):
    rclpy.init(args=args)
    node = DA3MultiDepthNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        for cam in node._cameras:
            if cam.frame_count > 0:
                avg = cam.total_latency / cam.frame_count
                node.get_logger().info(
                    f'[{cam.name}] Total: {cam.frame_count} frames, '
                    f'avg={avg:.0f}ms ({1000.0/avg:.1f} FPS)')
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
