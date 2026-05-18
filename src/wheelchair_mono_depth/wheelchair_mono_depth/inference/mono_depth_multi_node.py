#!/usr/bin/env python3
"""Round-robin multi-camera monocular depth inference node.

Single node, single TensorRT/PyTorch engine, cycling through 3 cameras at 33ms
intervals. Achieves ~10Hz per camera with ~250MB GPU memory (vs 3 separate nodes
at ~750MB).

Output topics match RealSense exactly for zero-change integration with
scan_fusion_v7:
    /camera/depth/color/points        (front)
    /mapping_camera/depth/color/points (left)
    /right_camera/depth/color/points   (right)
"""

import os
import time
import json

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from cv_bridge import CvBridge

# Lazy imports
_torch = None


class CameraState:
    """Tracks state for one camera in the round-robin cycle."""

    def __init__(self, name, rgb_topic, depth_topic, info_topic, pc_topic,
                 intrinsics_file=''):
        self.name = name
        self.rgb_topic = rgb_topic
        self.depth_topic = depth_topic
        self.info_topic = info_topic
        self.pc_topic = pc_topic

        self.latest_rgb_msg = None
        self.processed = True  # Start as True so first frame triggers
        self.frame_count = 0
        self.total_latency = 0.0

        # Intrinsics
        self.fx = self.fy = self.cx = self.cy = None
        self.img_width = self.img_height = None
        if intrinsics_file and os.path.exists(intrinsics_file):
            with open(intrinsics_file) as f:
                intr = json.load(f)
            self.fx = intr['fx']
            self.fy = intr['fy']
            self.cx = intr['cx']
            self.cy = intr['cy']
            self.img_width = intr['width']
            self.img_height = intr['height']


class MonoDepthMultiNode(Node):
    """Single-engine round-robin depth inference across 3 cameras."""

    def __init__(self):
        super().__init__('mono_depth_multi_node')

        # Model parameters
        self.declare_parameter('model_path', '')
        self.declare_parameter('use_tensorrt', False)
        self.declare_parameter('encoder', 'vits')
        self.declare_parameter('max_depth', 6.0)
        self.declare_parameter('input_size', 518)
        self.declare_parameter('cycle_interval_ms', 33.0)
        self.declare_parameter('publish_pointcloud', True)

        # Per-camera topics (matching RealSense conventions)
        self.declare_parameter('front_rgb_topic', '/camera/color/image_raw')
        self.declare_parameter('front_depth_topic', '/camera/depth/image_rect_raw')
        self.declare_parameter('front_info_topic', '/camera/depth/camera_info')
        self.declare_parameter('front_pc_topic', '/camera/depth/color/points')
        self.declare_parameter('front_intrinsics', '')

        self.declare_parameter('left_rgb_topic', '/mapping_camera/color/image_raw')
        self.declare_parameter('left_depth_topic', '/mapping_camera/depth/image_rect_raw')
        self.declare_parameter('left_info_topic', '/mapping_camera/depth/camera_info')
        self.declare_parameter('left_pc_topic', '/mapping_camera/depth/color/points')
        self.declare_parameter('left_intrinsics', '')

        self.declare_parameter('right_rgb_topic', '/right_camera/color/image_raw')
        self.declare_parameter('right_depth_topic', '/right_camera/depth/image_rect_raw')
        self.declare_parameter('right_info_topic', '/right_camera/depth/camera_info')
        self.declare_parameter('right_pc_topic', '/right_camera/depth/color/points')
        self.declare_parameter('right_intrinsics', '')

        self.model_path = self.get_parameter('model_path').value
        self.use_tensorrt = self.get_parameter('use_tensorrt').value
        self.encoder = self.get_parameter('encoder').value
        self.max_depth = self.get_parameter('max_depth').value
        self.input_size = self.get_parameter('input_size').value
        cycle_ms = self.get_parameter('cycle_interval_ms').value
        self.publish_pc = self.get_parameter('publish_pointcloud').value

        self.bridge = CvBridge()

        # Initialize cameras
        self.cameras = []
        for prefix in ['front', 'left', 'right']:
            cam = CameraState(
                name=prefix,
                rgb_topic=self.get_parameter(f'{prefix}_rgb_topic').value,
                depth_topic=self.get_parameter(f'{prefix}_depth_topic').value,
                info_topic=self.get_parameter(f'{prefix}_info_topic').value,
                pc_topic=self.get_parameter(f'{prefix}_pc_topic').value,
                intrinsics_file=self.get_parameter(f'{prefix}_intrinsics').value,
            )
            self.cameras.append(cam)

        self.current_cam_idx = 0

        # QoS
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,  # Only keep latest — we process round-robin
        )

        # Create subscribers and publishers for each camera
        for cam in self.cameras:
            # RGB subscriber — just stores latest message
            self.create_subscription(
                Image, cam.rgb_topic,
                lambda msg, c=cam: self._rgb_callback(c, msg),
                sensor_qos,
            )

            # Depth image publisher
            cam.depth_pub = self.create_publisher(Image, cam.depth_topic, 10)

            # CameraInfo publisher
            cam.info_pub = self.create_publisher(CameraInfo, cam.info_topic, 10)

            # PointCloud2 publisher
            if self.publish_pc:
                cam.pc_pub = self.create_publisher(PointCloud2, cam.pc_topic, 10)

        # Load model (single engine for all cameras)
        self._load_model()

        # Round-robin timer
        self.timer = self.create_timer(cycle_ms / 1000.0, self._cycle_callback)

        cam_names = [c.name for c in self.cameras]
        self.get_logger().info(
            f'MonoDepthMultiNode: {cam_names}, cycle={cycle_ms}ms, '
            f'{"TensorRT" if self.use_tensorrt else "PyTorch"}, '
            f'input={self.input_size}x{self.input_size}')

    def _rgb_callback(self, cam, msg):
        """Store latest RGB message for round-robin processing."""
        cam.latest_rgb_msg = msg
        cam.processed = False

    def _cycle_callback(self):
        """Process one camera per cycle in round-robin order."""
        # Find next camera with unprocessed data
        for _ in range(len(self.cameras)):
            cam = self.cameras[self.current_cam_idx]
            self.current_cam_idx = (self.current_cam_idx + 1) % len(self.cameras)

            if cam.latest_rgb_msg is not None and not cam.processed:
                self._process_camera(cam)
                return

    def _process_camera(self, cam):
        """Run inference on one camera and publish results."""
        msg = cam.latest_rgb_msg
        cam.processed = True

        t0 = time.monotonic()

        try:
            rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().warn(f'[{cam.name}] RGB conversion failed: {e}')
            return

        orig_h, orig_w = rgb.shape[:2]

        # Auto-estimate intrinsics if not loaded
        if cam.fx is None:
            cam.img_width = orig_w
            cam.img_height = orig_h
            cam.fx = orig_w * 0.9
            cam.fy = orig_h * 0.9
            cam.cx = orig_w / 2.0
            cam.cy = orig_h / 2.0

        # Run inference
        depth_m = self._infer(rgb)
        if depth_m is None:
            return

        # Resize to original resolution
        if depth_m.shape != (orig_h, orig_w):
            import cv2
            depth_m = cv2.resize(depth_m, (orig_w, orig_h),
                                 interpolation=cv2.INTER_NEAREST)

        # Publish depth image (uint16 millimeters, RealSense format)
        depth_mm = (depth_m * 1000.0).clip(0, 65535).astype(np.uint16)
        depth_msg = self.bridge.cv2_to_imgmsg(depth_mm, encoding='16UC1')
        depth_msg.header = msg.header
        cam.depth_pub.publish(depth_msg)

        # Publish CameraInfo
        info_msg = CameraInfo()
        info_msg.header = msg.header
        info_msg.width = orig_w
        info_msg.height = orig_h
        info_msg.k = [
            cam.fx, 0.0, cam.cx,
            0.0, cam.fy, cam.cy,
            0.0, 0.0, 1.0,
        ]
        info_msg.p = [
            cam.fx, 0.0, cam.cx, 0.0,
            0.0, cam.fy, cam.cy, 0.0,
            0.0, 0.0, 1.0, 0.0,
        ]
        cam.info_pub.publish(info_msg)

        # Publish point cloud
        if self.publish_pc:
            pc_msg = self._depth_to_pointcloud(
                depth_m, msg.header, cam.fx, cam.fy, cam.cx, cam.cy
            )
            cam.pc_pub.publish(pc_msg)

        # Latency tracking
        latency = (time.monotonic() - t0) * 1000
        cam.total_latency += latency
        cam.frame_count += 1
        if cam.frame_count % 100 == 0:
            avg = cam.total_latency / cam.frame_count
            self.get_logger().info(
                f'[{cam.name}] Frame {cam.frame_count}: '
                f'avg latency={avg:.1f}ms ({1000/avg:.1f} FPS)')

    def _load_model(self):
        """Load the depth model (PyTorch or TensorRT)."""
        global _torch

        if not self.model_path:
            self.get_logger().error('No model_path specified')
            return

        import torch
        _torch = torch

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if self.use_tensorrt and self.model_path.endswith('.engine'):
            self._load_tensorrt()
        else:
            self._load_pytorch()

    def _load_pytorch(self):
        import torch
        from wheelchair_mono_depth.training.train import load_model

        ckpt = torch.load(self.model_path, map_location='cpu', weights_only=False)
        max_depth = ckpt.get('max_depth', self.max_depth)

        self.model = load_model(self.encoder, max_depth, self.model_path)
        self.model = self.model.to(self.device).eval()
        self.get_logger().info(f'PyTorch model loaded from {self.model_path}')

    def _load_tensorrt(self):
        try:
            import tensorrt as trt
            import pycuda.driver as cuda
            import pycuda.autoinit  # noqa: F401
        except ImportError:
            self.get_logger().error(
                'TensorRT/PyCUDA not available. Install on Jetson or use PyTorch.')
            return

        logger = trt.Logger(trt.Logger.WARNING)
        with open(self.model_path, 'rb') as f:
            engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())

        self.trt_context = engine.create_execution_context()
        self.trt_engine = engine

        self.trt_inputs = []
        self.trt_outputs = []
        self.trt_bindings = []

        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            shape = engine.get_tensor_shape(name)
            dtype = trt.nptype(engine.get_tensor_dtype(name))
            size = np.prod(shape)
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.trt_bindings.append(int(device_mem))

            if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.trt_inputs.append({
                    'host': host_mem, 'device': device_mem, 'shape': shape,
                })
            else:
                self.trt_outputs.append({
                    'host': host_mem, 'device': device_mem, 'shape': shape,
                })

        self.model = None
        self.get_logger().info(f'TensorRT engine loaded from {self.model_path}')

    def _infer(self, rgb):
        """Run depth inference on BGR image, returns float32 depth in meters."""
        import torch
        import cv2

        if self.use_tensorrt and hasattr(self, 'trt_context'):
            return self._infer_tensorrt(rgb)

        if not hasattr(self, 'model') or self.model is None:
            return None

        rgb_resized = cv2.resize(rgb, (self.input_size, self.input_size))
        rgb_rgb = rgb_resized[:, :, ::-1].copy()
        tensor = torch.from_numpy(rgb_rgb).permute(2, 0, 1).float() / 255.0

        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = (tensor - mean) / std
        tensor = tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            depth = self.model(tensor)

        if depth.dim() == 4:
            depth = depth.squeeze(0).squeeze(0)
        elif depth.dim() == 3:
            depth = depth.squeeze(0)

        return depth.cpu().numpy()

    def _infer_tensorrt(self, rgb):
        """Run inference using TensorRT engine."""
        import cv2
        try:
            import pycuda.driver as cuda
        except ImportError:
            return None

        rgb_resized = cv2.resize(rgb, (self.input_size, self.input_size))
        rgb_rgb = rgb_resized[:, :, ::-1].astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        rgb_rgb = (rgb_rgb - mean) / std
        input_data = np.ascontiguousarray(rgb_rgb.transpose(2, 0, 1)[np.newaxis])

        np.copyto(self.trt_inputs[0]['host'], input_data.ravel())
        cuda.memcpy_htod(self.trt_inputs[0]['device'], self.trt_inputs[0]['host'])

        self.trt_context.execute_v2(bindings=self.trt_bindings)

        cuda.memcpy_dtoh(self.trt_outputs[0]['host'], self.trt_outputs[0]['device'])
        output = self.trt_outputs[0]['host'].reshape(self.trt_outputs[0]['shape'])

        if output.ndim == 4:
            output = output[0, 0]
        elif output.ndim == 3:
            output = output[0]

        return output

    def _depth_to_pointcloud(self, depth_m, header, fx, fy, cx, cy):
        """Convert depth map to PointCloud2 (scan_fusion_v7 compatible)."""
        h, w = depth_m.shape

        u = np.arange(w, dtype=np.float32)
        v = np.arange(h, dtype=np.float32)
        u, v = np.meshgrid(u, v)

        z = depth_m
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy

        valid = z > 0.01
        x = x[valid]
        y = y[valid]
        z = z[valid]

        points = np.stack([x, y, z], axis=-1).astype(np.float32)

        msg = PointCloud2()
        msg.header = header
        msg.height = 1
        msg.width = len(points)
        msg.is_dense = True
        msg.is_bigendian = False
        msg.point_step = 12  # 3 * float32
        msg.row_step = msg.point_step * msg.width
        msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        msg.data = points.tobytes()

        return msg


def main(args=None):
    rclpy.init(args=args)
    node = MonoDepthMultiNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        for cam in node.cameras:
            if cam.frame_count > 0:
                avg = cam.total_latency / cam.frame_count
                node.get_logger().info(
                    f'[{cam.name}] Total: {cam.frame_count} frames, '
                    f'avg latency={avg:.1f}ms')
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
