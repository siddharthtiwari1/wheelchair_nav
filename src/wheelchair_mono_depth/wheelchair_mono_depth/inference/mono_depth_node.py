#!/usr/bin/env python3
"""ROS2 inference node for monocular depth estimation.

Subscribes to RGB images and produces depth images + point clouds that are
format-compatible with RealSense output, enabling transparent swap in the
existing scan_depth_fusion pipeline.

Supports both PyTorch (.pth) and TensorRT (.engine) backends.

Published topics match RealSense conventions:
    depth: sensor_msgs/Image, 16UC1, millimeters
    pointcloud: sensor_msgs/PointCloud2 (optional)
    camera_info: sensor_msgs/CameraInfo
"""

import os
import time
import json
import struct

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from std_msgs.msg import Header
from cv_bridge import CvBridge

# Lazy imports for torch/tensorrt (may not be available on all systems)
_torch = None
_model = None


class MonoDepthNode(Node):
    def __init__(self):
        super().__init__('mono_depth_node')

        # Parameters
        self.declare_parameter('model_path', '')
        self.declare_parameter('use_tensorrt', False)
        self.declare_parameter('encoder', 'vits')
        self.declare_parameter('max_depth', 10.0)
        self.declare_parameter('input_size', 518)
        self.declare_parameter('camera_name', 'camera')
        self.declare_parameter('rgb_topic', '/camera/color/image_raw')
        self.declare_parameter('output_depth_topic', '/camera/mono_depth/image_raw')
        self.declare_parameter('output_info_topic', '/camera/mono_depth/camera_info')
        self.declare_parameter('publish_pointcloud', False)
        self.declare_parameter('pointcloud_topic', '/camera/mono_depth/points')
        self.declare_parameter('inference_hz', 15.0)
        self.declare_parameter('intrinsics_file', '')

        self.model_path = self.get_parameter('model_path').value
        self.use_tensorrt = self.get_parameter('use_tensorrt').value
        self.encoder = self.get_parameter('encoder').value
        self.max_depth = self.get_parameter('max_depth').value
        self.input_size = self.get_parameter('input_size').value
        self.camera_name = self.get_parameter('camera_name').value
        self.publish_pc = self.get_parameter('publish_pointcloud').value
        inference_hz = self.get_parameter('inference_hz').value
        intrinsics_file = self.get_parameter('intrinsics_file').value

        self.bridge = CvBridge()
        self.inference_interval = 1.0 / inference_hz
        self.last_inference_time = 0.0
        self.frame_count = 0
        self.total_latency = 0.0

        # Load camera intrinsics
        self.fx = self.fy = self.cx = self.cy = None
        self.img_width = self.img_height = None
        if intrinsics_file and os.path.exists(intrinsics_file):
            self._load_intrinsics(intrinsics_file)

        # Load model
        self._load_model()

        # QoS
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )

        # Publishers
        depth_topic = self.get_parameter('output_depth_topic').value
        info_topic = self.get_parameter('output_info_topic').value
        pc_topic = self.get_parameter('pointcloud_topic').value

        self.depth_pub = self.create_publisher(Image, depth_topic, 10)
        self.info_pub = self.create_publisher(CameraInfo, info_topic, 10)
        if self.publish_pc:
            self.pc_pub = self.create_publisher(PointCloud2, pc_topic, 10)

        # Subscriber
        rgb_topic = self.get_parameter('rgb_topic').value
        self.rgb_sub = self.create_subscription(
            Image, rgb_topic, self._rgb_callback, sensor_qos)

        self.get_logger().info(
            f'MonoDepthNode [{self.camera_name}]: {rgb_topic} -> {depth_topic} '
            f'({"TensorRT" if self.use_tensorrt else "PyTorch"}, {inference_hz}Hz)')

    def _load_intrinsics(self, path):
        with open(path) as f:
            intr = json.load(f)
        self.fx = intr['fx']
        self.fy = intr['fy']
        self.cx = intr['cx']
        self.cy = intr['cy']
        self.img_width = intr['width']
        self.img_height = intr['height']
        self.get_logger().info(
            f'Loaded intrinsics: {self.img_width}x{self.img_height}, '
            f'fx={self.fx:.1f}, fy={self.fy:.1f}')

    def _load_model(self):
        global _torch, _model

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
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

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
            import pycuda.autoinit
        except ImportError:
            self.get_logger().error(
                'TensorRT/PyCUDA not available. Install on Jetson or use PyTorch backend.')
            return

        logger = trt.Logger(trt.Logger.WARNING)
        with open(self.model_path, 'rb') as f:
            engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())

        self.trt_context = engine.create_execution_context()
        self.trt_engine = engine

        # Allocate buffers
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
                self.trt_inputs.append({'host': host_mem, 'device': device_mem, 'shape': shape})
            else:
                self.trt_outputs.append({'host': host_mem, 'device': device_mem, 'shape': shape})

        self.model = None  # Signal TRT mode
        self.get_logger().info(f'TensorRT engine loaded from {self.model_path}')

    def _rgb_callback(self, msg: Image):
        now = time.monotonic()
        if now - self.last_inference_time < self.inference_interval:
            return
        self.last_inference_time = now

        t0 = time.monotonic()

        # Convert to numpy
        try:
            rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().warn(f'RGB conversion failed: {e}')
            return

        orig_h, orig_w = rgb.shape[:2]

        # Update intrinsics from first frame if not loaded from file
        if self.fx is None:
            # Approximate intrinsics (will be overwritten if CameraInfo arrives)
            self.img_width = orig_w
            self.img_height = orig_h
            self.fx = orig_w * 0.9
            self.fy = orig_h * 0.9
            self.cx = orig_w / 2.0
            self.cy = orig_h / 2.0

        # Run inference
        depth_m = self._infer(rgb)
        if depth_m is None:
            return

        # Resize to original resolution
        if depth_m.shape != (orig_h, orig_w):
            import cv2
            depth_m = cv2.resize(depth_m, (orig_w, orig_h),
                                 interpolation=cv2.INTER_NEAREST)

        # Convert to uint16 millimeters (RealSense format)
        depth_mm = (depth_m * 1000.0).clip(0, 65535).astype(np.uint16)

        # Publish depth image
        depth_msg = self.bridge.cv2_to_imgmsg(depth_mm, encoding='16UC1')
        depth_msg.header = msg.header
        self.depth_pub.publish(depth_msg)

        # Publish CameraInfo
        info_msg = CameraInfo()
        info_msg.header = msg.header
        info_msg.width = orig_w
        info_msg.height = orig_h
        info_msg.k = [self.fx, 0.0, self.cx, 0.0, self.fy, self.cy, 0.0, 0.0, 1.0]
        info_msg.p = [self.fx, 0.0, self.cx, 0.0, 0.0, self.fy, self.cy, 0.0, 0.0, 0.0, 1.0, 0.0]
        self.info_pub.publish(info_msg)

        # Publish point cloud (optional, for scan_depth_fusion compatibility)
        if self.publish_pc:
            pc_msg = self._depth_to_pointcloud(depth_m, msg.header)
            self.pc_pub.publish(pc_msg)

        # Latency tracking
        latency = (time.monotonic() - t0) * 1000
        self.total_latency += latency
        self.frame_count += 1
        if self.frame_count % 100 == 0:
            avg = self.total_latency / self.frame_count
            self.get_logger().info(
                f'[{self.camera_name}] Frame {self.frame_count}: '
                f'avg latency={avg:.1f}ms ({1000/avg:.1f} FPS)')

    def _infer(self, rgb):
        """Run depth inference on BGR image, returns float32 depth in meters."""
        import torch
        import cv2

        if self.use_tensorrt and hasattr(self, 'trt_context'):
            return self._infer_tensorrt(rgb)

        if self.model is None:
            return None

        # Preprocess: BGR -> RGB, resize, normalize
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

        # Preprocess
        rgb_resized = cv2.resize(rgb, (self.input_size, self.input_size))
        rgb_rgb = rgb_resized[:, :, ::-1].astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        rgb_rgb = (rgb_rgb - mean) / std
        input_data = np.ascontiguousarray(rgb_rgb.transpose(2, 0, 1)[np.newaxis])

        # Copy to device
        np.copyto(self.trt_inputs[0]['host'], input_data.ravel())
        cuda.memcpy_htod(self.trt_inputs[0]['device'], self.trt_inputs[0]['host'])

        # Execute
        self.trt_context.execute_v2(bindings=self.trt_bindings)

        # Copy output
        cuda.memcpy_dtoh(self.trt_outputs[0]['host'], self.trt_outputs[0]['device'])
        output = self.trt_outputs[0]['host'].reshape(self.trt_outputs[0]['shape'])

        if output.ndim == 4:
            output = output[0, 0]
        elif output.ndim == 3:
            output = output[0]

        return output

    def _depth_to_pointcloud(self, depth_m, header):
        """Convert depth map to PointCloud2 (compatible with scan_depth_fusion)."""
        h, w = depth_m.shape

        # Create pixel coordinate grids
        u = np.arange(w, dtype=np.float32)
        v = np.arange(h, dtype=np.float32)
        u, v = np.meshgrid(u, v)

        # Back-project to 3D
        z = depth_m
        x = (u - self.cx) * z / self.fx
        y = (v - self.cy) * z / self.fy

        # Filter invalid
        valid = z > 0.01
        x = x[valid]
        y = y[valid]
        z = z[valid]

        # Create PointCloud2
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
    node = MonoDepthNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
