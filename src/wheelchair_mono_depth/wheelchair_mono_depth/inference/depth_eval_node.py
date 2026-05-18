#!/usr/bin/env python3
"""Depth Anything V2 vs RealSense depth evaluation node.

Runs off-the-shelf DA V2 Metric Indoor (HuggingFace, no fine-tuning) on live
RGB from the front camera, compares predicted depth against RealSense aligned
depth ground truth, and optionally corrects predictions using LiDAR as an
affine anchor via RANSAC.

Three evaluation levels:
  1. Depth image metrics (AbsRel/RMSE/delta) per distance band
  2. LiDAR affine correction — RANSAC fit scale+bias, recompute metrics
  3. Virtual LaserScan comparison — back-project depth to laser frame

Outputs: CSV metrics, visualization PNGs, ROS LaserScan topics.
"""

import os
import csv
import time
import array as _array

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.time import Time

from sensor_msgs.msg import Image, CameraInfo, LaserScan
from cv_bridge import CvBridge
import message_filters
import tf2_ros


class DepthEvalNode(Node):
    """Evaluates DA V2 predicted depth against RealSense ground truth."""

    DISTANCE_BANDS = [
        ('0-1m', 0.0, 1.0),
        ('1-2m', 1.0, 2.0),
        ('2-3m', 2.0, 3.0),
        ('3-4m', 3.0, 4.0),
        ('4-6m', 4.0, 6.0),
        ('all', 0.0, 99.0),
    ]

    def __init__(self):
        super().__init__('depth_eval_node')

        # --- Parameters ---
        self.declare_parameter('hf_model_id',
                               'depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf')
        self.declare_parameter('hf_cache_dir', '')
        self.declare_parameter('rgb_topic', '/camera/color/image_raw')
        self.declare_parameter('aligned_depth_topic',
                               '/camera/aligned_depth_to_color/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/color/camera_info')
        self.declare_parameter('lidar_scan_topic', '/scan_filtered')
        self.declare_parameter('camera_scan_topic', '/scan_front_camera')
        self.declare_parameter('eval_hz', 2.0)
        self.declare_parameter('max_depth_m', 6.0)
        self.declare_parameter('virtual_scan_stride', 4)
        self.declare_parameter('virtual_scan_min_height', 0.10)
        self.declare_parameter('virtual_scan_max_height', 1.80)
        self.declare_parameter('ransac_residual_threshold_m', 0.10)
        self.declare_parameter('min_lidar_overlap_points', 20)
        self.declare_parameter('output_dir',
                               '/home/sidd/wheelchair_nav/eval_output/depth_eval')
        self.declare_parameter('viz_every_n_frames', 10)

        self._model_id = self.get_parameter('hf_model_id').value
        self._cache_dir = self.get_parameter('hf_cache_dir').value or None
        self._max_depth = self.get_parameter('max_depth_m').value
        self._eval_hz = self.get_parameter('eval_hz').value
        self._scan_stride = self.get_parameter('virtual_scan_stride').value
        self._scan_min_h = self.get_parameter('virtual_scan_min_height').value
        self._scan_max_h = self.get_parameter('virtual_scan_max_height').value
        self._ransac_thresh = self.get_parameter('ransac_residual_threshold_m').value
        self._min_lidar_pts = self.get_parameter('min_lidar_overlap_points').value
        self._output_dir = self.get_parameter('output_dir').value
        self._viz_every_n = self.get_parameter('viz_every_n_frames').value

        # --- Output directory ---
        os.makedirs(os.path.join(self._output_dir, 'viz'), exist_ok=True)

        # --- Load model ---
        self.get_logger().info(f'Loading DA V2 model: {self._model_id}')
        self._load_model()
        self.get_logger().info('Model loaded successfully')

        # --- State ---
        self._bridge = CvBridge()
        self._frame_id = 0
        self._last_eval_time = 0.0
        self._intrinsics = None   # (fx, fy, cx, cy, W, H)
        self._camera_frame = None
        self._latest_lidar = None
        self._latest_camera_scan = None
        self._tf_cache = {}

        # Running accumulators
        self._raw_tracker = _BandTracker(self.DISTANCE_BANDS)
        self._corr_tracker = _BandTracker(self.DISTANCE_BANDS)
        self._scan_errors = []
        self._scale_bias_history = []

        # --- TF ---
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        # --- QoS ---
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )

        # --- Subscribers ---
        rgb_sub = message_filters.Subscriber(
            self, Image,
            self.get_parameter('rgb_topic').value,
            qos_profile=sensor_qos)
        depth_sub = message_filters.Subscriber(
            self, Image,
            self.get_parameter('aligned_depth_topic').value,
            qos_profile=sensor_qos)
        self._sync = message_filters.ApproximateTimeSynchronizer(
            [rgb_sub, depth_sub], queue_size=5, slop=0.05)
        self._sync.registerCallback(self._sync_cb)

        self._info_sub = self.create_subscription(
            CameraInfo,
            self.get_parameter('camera_info_topic').value,
            self._camera_info_cb, sensor_qos)

        self.create_subscription(
            LaserScan,
            self.get_parameter('lidar_scan_topic').value,
            self._lidar_cb, sensor_qos)

        self.create_subscription(
            LaserScan,
            self.get_parameter('camera_scan_topic').value,
            self._camera_scan_cb, sensor_qos)

        # --- Publishers ---
        self._pub_vscan_raw = self.create_publisher(
            LaserScan, '/depth_eval/virtual_scan_raw', 10)
        self._pub_vscan_corr = self.create_publisher(
            LaserScan, '/depth_eval/virtual_scan_corrected', 10)
        self._pub_corr_depth = self.create_publisher(
            Image, '/camera/mono_depth_corrected/image_raw', 10)

        # --- CSV ---
        csv_path = os.path.join(self._output_dir, 'depth_eval_metrics.csv')
        self._csv_file = open(csv_path, 'w', newline='')
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow([
            'frame', 'timestamp',
            'abs_rel', 'rmse', 'delta_1', 'silog', 'n_valid',
            'corr_abs_rel', 'corr_rmse', 'corr_delta_1', 'corr_silog',
            'scale', 'bias', 'n_lidar_pts',
            'vscan_mean_err', 'vscan_rmse', 'vscan_n_bins',
            'inference_ms',
        ])

        self.get_logger().info(
            f'Depth eval node ready — output: {self._output_dir}, '
            f'eval_hz: {self._eval_hz}')

    # ------------------------------------------------------------------ model

    def _load_model(self):
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        self._device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.get_logger().info(f'Using device: {self._device}')

        kwargs = {}
        if self._cache_dir:
            kwargs['cache_dir'] = self._cache_dir

        self._processor = AutoImageProcessor.from_pretrained(
            self._model_id, **kwargs)
        self._model = AutoModelForDepthEstimation.from_pretrained(
            self._model_id, **kwargs)
        self._model.to(self._device)
        self._model.eval()

    # -------------------------------------------------------------- callbacks

    def _camera_info_cb(self, msg: CameraInfo):
        if self._intrinsics is not None:
            return
        K = msg.k
        self._intrinsics = (K[0], K[4], K[2], K[5], msg.width, msg.height)
        self._camera_frame = msg.header.frame_id
        self.get_logger().info(
            f'Intrinsics: fx={K[0]:.1f} fy={K[4]:.1f} '
            f'cx={K[2]:.1f} cy={K[5]:.1f} '
            f'{msg.width}x{msg.height} frame={self._camera_frame}')

    def _lidar_cb(self, msg: LaserScan):
        self._latest_lidar = msg

    def _camera_scan_cb(self, msg: LaserScan):
        self._latest_camera_scan = msg

    def _sync_cb(self, rgb_msg: Image, depth_msg: Image):
        import torch
        import cv2
        from PIL import Image as PILImage

        # Rate limit
        now = time.monotonic()
        if now - self._last_eval_time < 1.0 / self._eval_hz:
            return
        self._last_eval_time = now

        if self._intrinsics is None:
            self.get_logger().warn(
                'Waiting for camera_info...', throttle_duration_sec=5.0)
            return

        fx, fy, cx, cy, W, H = self._intrinsics

        # Convert images
        rgb_bgr = self._bridge.imgmsg_to_cv2(rgb_msg, 'bgr8')
        depth_raw = self._bridge.imgmsg_to_cv2(depth_msg, 'passthrough')
        gt_m = depth_raw.astype(np.float32) / 1000.0
        valid_gt = (gt_m > 0.01) & (gt_m < self._max_depth)

        # --- Inference ---
        t0 = time.monotonic()
        rgb_pil = PILImage.fromarray(cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB))
        inputs = self._processor(
            images=rgb_pil, return_tensors='pt').to(self._device)
        with torch.no_grad():
            outputs = self._model(**inputs)

        pred = outputs.predicted_depth.squeeze()  # (H', W')
        pred = torch.nn.functional.interpolate(
            pred.unsqueeze(0).unsqueeze(0),
            size=(H, W),
            mode='bicubic',
            align_corners=False,
        ).squeeze().cpu().numpy()  # (H, W) float32 meters
        inference_ms = (time.monotonic() - t0) * 1000.0

        # --- Level 1: raw metrics per distance band ---
        raw_metrics = self._compute_band_metrics(pred, gt_m, valid_gt)
        self._raw_tracker.update(raw_metrics)

        # --- Level 2: LiDAR affine correction ---
        scale, bias, n_lidar = np.nan, np.nan, 0
        corrected_m = pred.copy()
        corr_metrics = {}

        lidar = self._latest_lidar
        if lidar is not None:
            s, b, n = self._compute_lidar_correction(pred, lidar)
            if s is not None:
                scale, bias, n_lidar = s, b, n
                corrected_m = scale * pred + bias
                corrected_m = np.clip(corrected_m, 0.0, self._max_depth)
                corr_metrics = self._compute_band_metrics(
                    corrected_m, gt_m, valid_gt)
                self._corr_tracker.update(corr_metrics)
                self._scale_bias_history.append((scale, bias))

        # --- Level 3: virtual scan ---
        vscan_mean_err, vscan_rmse, vscan_n = np.nan, np.nan, 0

        if lidar is not None:
            pred_ranges = self._generate_virtual_scan(pred, lidar)
            gt_ranges = self._generate_virtual_scan(gt_m, lidar)

            if pred_ranges is not None and gt_ranges is not None:
                self._publish_scan(pred_ranges, lidar, self._pub_vscan_raw)

                if not np.isnan(scale):
                    corr_ranges = self._generate_virtual_scan(
                        corrected_m, lidar)
                    if corr_ranges is not None:
                        self._publish_scan(
                            corr_ranges, lidar, self._pub_vscan_corr)

                both_valid = (np.isfinite(pred_ranges)
                              & np.isfinite(gt_ranges)
                              & (pred_ranges > 0.01)
                              & (gt_ranges > 0.01))
                n_valid_bins = int(np.sum(both_valid))
                if n_valid_bins > 10:
                    errs = np.abs(
                        pred_ranges[both_valid] - gt_ranges[both_valid])
                    vscan_mean_err = float(np.mean(errs))
                    vscan_rmse = float(np.sqrt(np.mean(errs ** 2)))
                    vscan_n = n_valid_bins
                    self._scan_errors.append(
                        (vscan_mean_err, vscan_rmse, vscan_n))

            # Level 3b: compare against actual camera scan from scan_fusion
            cam_scan = self._latest_camera_scan
            if cam_scan is not None and pred_ranges is not None:
                actual = np.array(cam_scan.ranges, dtype=np.float32)
                if len(actual) == len(pred_ranges):
                    both = (np.isfinite(pred_ranges)
                            & np.isfinite(actual)
                            & (pred_ranges > 0.01)
                            & (actual > 0.01))
                    n_both = int(np.sum(both))
                    if n_both > 10:
                        diff = np.abs(
                            pred_ranges[both] - actual[both])
                        self.get_logger().info(
                            f'  vs camera_scan: meanErr='
                            f'{np.mean(diff):.4f}m, '
                            f'RMSE={np.sqrt(np.mean(diff**2)):.4f}m, '
                            f'n_bins={n_both}',
                            throttle_duration_sec=5.0)

        # --- Publish corrected depth image ---
        corr_uint16 = np.clip(
            corrected_m * 1000, 0, 65535).astype(np.uint16)
        corr_msg = self._bridge.cv2_to_imgmsg(corr_uint16, encoding='16UC1')
        corr_msg.header = depth_msg.header
        self._pub_corr_depth.publish(corr_msg)

        # --- CSV row ---
        raw_all = raw_metrics.get('all', {})
        corr_all = corr_metrics.get('all', {})
        self._csv_writer.writerow([
            self._frame_id,
            f'{rgb_msg.header.stamp.sec}.{rgb_msg.header.stamp.nanosec:09d}',
            f'{raw_all.get("abs_rel", np.nan):.5f}',
            f'{raw_all.get("rmse", np.nan):.4f}',
            f'{raw_all.get("delta_1", np.nan):.4f}',
            f'{raw_all.get("silog", np.nan):.5f}',
            raw_all.get('n_valid', 0),
            f'{corr_all.get("abs_rel", np.nan):.5f}',
            f'{corr_all.get("rmse", np.nan):.4f}',
            f'{corr_all.get("delta_1", np.nan):.4f}',
            f'{corr_all.get("silog", np.nan):.5f}',
            f'{scale:.4f}' if not np.isnan(scale) else '',
            f'{bias:.4f}' if not np.isnan(bias) else '',
            n_lidar,
            f'{vscan_mean_err:.4f}' if not np.isnan(vscan_mean_err) else '',
            f'{vscan_rmse:.4f}' if not np.isnan(vscan_rmse) else '',
            vscan_n,
            f'{inference_ms:.1f}',
        ])
        self._csv_file.flush()

        # --- Logging ---
        self._frame_id += 1
        if self._frame_id % 10 == 0:
            self._print_summary()

        # --- Visualization ---
        if self._frame_id % self._viz_every_n == 0:
            self._save_viz(rgb_bgr, gt_m, pred, corrected_m, valid_gt)

    # --------------------------------------------------------------- metrics

    def _compute_band_metrics(self, pred_m, gt_m, valid):
        """Compute depth metrics per distance band using torch."""
        import torch
        from wheelchair_mono_depth.training.metrics import compute_depth_metrics

        results = {}
        pred_t = torch.from_numpy(pred_m).unsqueeze(0).float()
        gt_t = torch.from_numpy(gt_m).unsqueeze(0).float()

        for name, lo, hi in self.DISTANCE_BANDS:
            if name == 'all':
                band_mask = valid
            else:
                band_mask = valid & (gt_m >= lo) & (gt_m < hi)
            mask_t = torch.from_numpy(band_mask).unsqueeze(0)
            m = compute_depth_metrics(
                pred_t, gt_t, mask_t, max_depth=self._max_depth)
            results[name] = m
        return results

    # -------------------------------------------------------- lidar correction

    def _get_tf(self, target_frame, source_frame):
        """Get cached static TF as (R, t) numpy arrays."""
        key = (target_frame, source_frame)
        if key in self._tf_cache:
            return self._tf_cache[key]
        try:
            tf_msg = self._tf_buffer.lookup_transform(
                target_frame, source_frame, Time())
            tr = tf_msg.transform.translation
            q = tf_msg.transform.rotation
            x, y, z, w = q.x, q.y, q.z, q.w
            R = np.array([
                [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
                [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
                [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)],
            ], dtype=np.float64)
            t_vec = np.array([tr.x, tr.y, tr.z], dtype=np.float64)
            self._tf_cache[key] = (R, t_vec)
            self.get_logger().info(
                f'TF {source_frame} -> {target_frame}: '
                f't=[{tr.x:.3f}, {tr.y:.3f}, {tr.z:.3f}]')
            return (R, t_vec)
        except Exception as e:
            self.get_logger().warn(
                f'TF {source_frame}->{target_frame} unavailable: {e}',
                throttle_duration_sec=5.0)
            return None

    def _compute_lidar_correction(self, pred_depth_m, lidar_msg):
        """Project LiDAR into camera image, RANSAC-fit scale+bias.

        Returns (scale, bias, n_inliers) or (None, None, 0).
        """
        if self._camera_frame is None:
            return None, None, 0

        tf_data = self._get_tf(self._camera_frame, lidar_msg.header.frame_id)
        if tf_data is None:
            return None, None, 0
        R, t = tf_data

        fx, fy, cx, cy, W, H = self._intrinsics

        # LiDAR scan -> 3D in laser frame (z=0 plane)
        n_rays = len(lidar_msg.ranges)
        angles = (lidar_msg.angle_min
                  + np.arange(n_rays) * lidar_msg.angle_increment)
        ranges = np.array(lidar_msg.ranges, dtype=np.float64)
        valid = (np.isfinite(ranges)
                 & (ranges > lidar_msg.range_min)
                 & (ranges < lidar_msg.range_max))
        angles = angles[valid]
        ranges = ranges[valid]

        pts_laser = np.stack([
            ranges * np.cos(angles),
            ranges * np.sin(angles),
            np.zeros_like(ranges),
        ], axis=-1)

        # Transform to camera optical frame
        pts_cam = pts_laser @ R.T + t

        # Keep only points in front of camera
        in_front = pts_cam[:, 2] > 0.1
        pts_cam = pts_cam[in_front]

        # Project to image plane
        u = (fx * pts_cam[:, 0] / pts_cam[:, 2] + cx).astype(int)
        v = (fy * pts_cam[:, 1] / pts_cam[:, 2] + cy).astype(int)
        Z_true = pts_cam[:, 2]

        in_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        u, v, Z_true = u[in_bounds], v[in_bounds], Z_true[in_bounds]

        if len(Z_true) < self._min_lidar_pts:
            return None, None, len(Z_true)

        # Predicted depth at LiDAR-projected pixels
        pred_at = pred_depth_m[v, u].astype(np.float64)

        # Filter invalid predictions
        pred_ok = pred_at > 0.01
        pred_at = pred_at[pred_ok]
        Z_true = Z_true[pred_ok]

        if len(Z_true) < self._min_lidar_pts:
            return None, None, len(Z_true)

        # RANSAC: Z_true = s * pred + b  =>  corrected = s * pred + b
        from sklearn.linear_model import RANSACRegressor, LinearRegression

        X = pred_at.reshape(-1, 1)
        y = Z_true
        ransac = RANSACRegressor(
            estimator=LinearRegression(),
            residual_threshold=self._ransac_thresh,
            min_samples=max(3, int(0.3 * len(y))),
            max_trials=200,
        )
        try:
            ransac.fit(X, y)
            s = float(ransac.estimator_.coef_[0])
            b = float(ransac.estimator_.intercept_)
            n_inliers = int(np.sum(ransac.inlier_mask_))
            return s, b, n_inliers
        except Exception as e:
            self.get_logger().warn(f'RANSAC failed: {e}')
            return None, None, len(y)

    # ---------------------------------------------------------- virtual scan

    def _generate_virtual_scan(self, depth_m, template_scan):
        """Back-project depth image to laser frame, return polar-binned ranges.

        Uses the same method as scan_fusion_v9: back-project, transform,
        height-filter, polar-bin with min-range.
        """
        if self._camera_frame is None:
            return None

        # Camera -> laser TF
        tf_data = self._get_tf(
            template_scan.header.frame_id, self._camera_frame)
        if tf_data is None:
            return None
        R, t = tf_data

        fx, fy, cx, cy, W, H = self._intrinsics
        stride = self._scan_stride

        # Back-project (downsampled for speed)
        cols = np.arange(0, W, stride)
        rows = np.arange(0, H, stride)
        uu, vv = np.meshgrid(
            cols.astype(np.float64), rows.astype(np.float64))
        Z = depth_m[::stride, ::stride].astype(np.float64)

        valid = (Z > 0.01) & (Z < self._max_depth)
        X = (uu - cx) * Z / fx
        Y = (vv - cy) * Z / fy

        pts_cam = np.stack(
            [X[valid], Y[valid], Z[valid]], axis=-1)  # (N, 3)

        # Transform to laser frame
        pts_laser = pts_cam @ R.T + t

        # Height filter
        h_mask = ((pts_laser[:, 2] >= self._scan_min_h)
                  & (pts_laser[:, 2] <= self._scan_max_h))
        pts_laser = pts_laser[h_mask]

        if len(pts_laser) < 10:
            return None

        # Project to 2D polar
        ranges = np.sqrt(pts_laser[:, 0]**2 + pts_laser[:, 1]**2)
        angles = np.arctan2(pts_laser[:, 1], pts_laser[:, 0])

        # Bin using lidar scan template geometry
        n_bins = int(round(
            (template_scan.angle_max - template_scan.angle_min)
            / template_scan.angle_increment)) + 1
        bin_idx = np.round(
            (angles - template_scan.angle_min)
            / template_scan.angle_increment).astype(int)
        valid_bins = (bin_idx >= 0) & (bin_idx < n_bins)
        bin_idx = bin_idx[valid_bins]
        ranges = ranges[valid_bins]

        out = np.full(n_bins, np.inf, dtype=np.float64)
        np.minimum.at(out, bin_idx, ranges)

        return out.astype(np.float32)

    def _publish_scan(self, ranges, template, publisher):
        """Publish a LaserScan using template metadata."""
        msg = LaserScan()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = template.header.frame_id
        msg.angle_min = template.angle_min
        msg.angle_max = template.angle_max
        msg.angle_increment = template.angle_increment
        msg.time_increment = template.time_increment
        msg.scan_time = template.scan_time
        msg.range_min = template.range_min
        msg.range_max = template.range_max
        msg.ranges = _array.array(
            'f', ranges.astype(np.float32).tobytes())
        msg.intensities = []
        publisher.publish(msg)

    # --------------------------------------------------------------- logging

    def _print_summary(self):
        """Print aggregate metrics every 10 frames."""
        self.get_logger().info(f'=== Frame {self._frame_id} summary ===')

        raw = self._raw_tracker.summary()
        if raw:
            self.get_logger().info(f'  RAW  (all): {raw.get("all", "n/a")}')
            for name, _, _ in self.DISTANCE_BANDS:
                if name != 'all' and name in raw:
                    self.get_logger().info(f'  RAW  ({name}): {raw[name]}')

        corr = self._corr_tracker.summary()
        if corr:
            self.get_logger().info(
                f'  CORR (all): {corr.get("all", "n/a")}')

        if self._scale_bias_history:
            scales = [sb[0] for sb in self._scale_bias_history]
            biases = [sb[1] for sb in self._scale_bias_history]
            self.get_logger().info(
                f'  Scale: {np.mean(scales):.3f} +/- {np.std(scales):.3f}, '
                f'Bias: {np.mean(biases):.3f} +/- {np.std(biases):.3f}')

        if self._scan_errors:
            mean_errs = [e[0] for e in self._scan_errors]
            rmses = [e[1] for e in self._scan_errors]
            self.get_logger().info(
                f'  VirtualScan: meanErr={np.mean(mean_errs):.4f}m, '
                f'RMSE={np.mean(rmses):.4f}m, '
                f'n_frames={len(self._scan_errors)}')

    # --------------------------------------------------------- visualization

    def _save_viz(self, rgb_bgr, gt_m, pred_m, corrected_m, valid):
        """Save a 2x3 visualization panel to disk."""
        import cv2

        def depth_to_color(d, vmax=None):
            vmax = vmax or self._max_depth
            d_norm = np.clip(d / vmax, 0, 1)
            d_u8 = (d_norm * 255).astype(np.uint8)
            return cv2.applyColorMap(d_u8, cv2.COLORMAP_TURBO)

        def error_map(p, g, mask, vmax=0.5):
            err = np.abs(p - g)
            err[~mask] = 0
            e_u8 = (np.clip(err / vmax, 0, 1) * 255).astype(np.uint8)
            return cv2.applyColorMap(e_u8, cv2.COLORMAP_HOT)

        panels = [
            rgb_bgr,
            depth_to_color(gt_m),
            depth_to_color(pred_m),
            depth_to_color(corrected_m),
            error_map(pred_m, gt_m, valid),
            error_map(corrected_m, gt_m, valid),
        ]
        labels = [
            'RGB', 'RealSense GT', 'DA V2 Raw',
            'DA V2 Corrected', 'Error Raw', 'Error Corrected',
        ]

        for panel, label in zip(panels, labels):
            cv2.putText(
                panel, label, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        row1 = np.hstack(panels[:3])
        row2 = np.hstack(panels[3:])
        grid = np.vstack([row1, row2])

        path = os.path.join(
            self._output_dir, 'viz', f'frame_{self._frame_id:06d}.png')
        cv2.imwrite(path, grid)

    # ------------------------------------------------------------ lifecycle

    def destroy_node(self):
        """Print final summary with pass/fail decision and close CSV."""
        self.get_logger().info('===== FINAL EVALUATION SUMMARY =====')
        self._print_summary()

        # Decision thresholds
        corr = self._corr_tracker.get_band('all')
        if corr and corr['count'] > 0:
            avg = corr['avg']
            self.get_logger().info('--- Decision Thresholds ---')

            absrel = avg.get('abs_rel', 1.0)
            rmse = avg.get('rmse', 1.0)
            delta1 = avg.get('delta_1', 0.0)

            absrel_ok = absrel < 0.10
            rmse_ok = rmse < 0.15
            delta1_ok = delta1 > 0.80

            scale_ok = True
            if self._scale_bias_history:
                s_mean = np.mean([sb[0] for sb in self._scale_bias_history])
                scale_ok = 0.8 <= s_mean <= 1.2

            scan_ok = True
            if self._scan_errors:
                scan_mean = np.mean([e[0] for e in self._scan_errors])
                scan_ok = scan_mean < 0.08

            all_pass = (absrel_ok and rmse_ok and delta1_ok
                        and scale_ok and scan_ok)

            self.get_logger().info(
                f'  AbsRel < 0.10: '
                f'{"PASS" if absrel_ok else "FAIL"} ({absrel:.4f})')
            self.get_logger().info(
                f'  RMSE < 0.15m:  '
                f'{"PASS" if rmse_ok else "FAIL"} ({rmse:.4f})')
            self.get_logger().info(
                f'  delta1 > 0.80: '
                f'{"PASS" if delta1_ok else "FAIL"} ({delta1:.4f})')
            self.get_logger().info(
                f'  Scale 0.8-1.2: '
                f'{"PASS" if scale_ok else "FAIL"}')
            self.get_logger().info(
                f'  ScanErr<0.08m: '
                f'{"PASS" if scan_ok else "FAIL"}')
            self.get_logger().info(
                f'  VERDICT: '
                f'{"SKIP fine-tuning" if all_pass else "NEED fine-tuning"}')

        self._csv_file.close()
        super().destroy_node()


class _BandTracker:
    """Tracks running average of metrics per distance band."""

    def __init__(self, bands):
        self._bands = bands
        self._data = {name: {'sum': {}, 'count': 0} for name, _, _ in bands}

    def update(self, band_metrics):
        for name, _, _ in self._bands:
            if name not in band_metrics:
                continue
            m = band_metrics[name]
            if m.get('n_valid', 0) < 10:
                continue
            d = self._data[name]
            for k, v in m.items():
                if k == 'n_valid':
                    continue
                d['sum'][k] = d['sum'].get(k, 0.0) + float(v)
            d['count'] += 1

    def summary(self):
        result = {}
        for name, _, _ in self._bands:
            d = self._data[name]
            if d['count'] == 0:
                continue
            avg = {k: v / d['count'] for k, v in d['sum'].items()}
            result[name] = (
                f"AbsRel={avg.get('abs_rel', 0):.4f} "
                f"RMSE={avg.get('rmse', 0):.4f} "
                f"d<1.25={avg.get('delta_1', 0):.3f} "
                f"n={d['count']}")
        return result

    def get_band(self, name):
        d = self._data.get(name)
        if d is None or d['count'] == 0:
            return None
        avg = {k: v / d['count'] for k, v in d['sum'].items()}
        return {'avg': avg, 'count': d['count']}


def main(args=None):
    rclpy.init(args=args)
    node = DepthEvalNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
