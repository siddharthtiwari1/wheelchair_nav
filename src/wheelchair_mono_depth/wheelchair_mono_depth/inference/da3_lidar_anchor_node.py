#!/usr/bin/env python3
"""LiDAR-anchored DA3 depth correction node.

Uses the 2D lidar scan as ground truth on the horizontal scan plane to compute
a dynamic depth correction for DA3 monocular depth.

The camera COMPLEMENTS the lidar — it sees above/below the scan plane
(overhangs, tables, low obstacles) while the lidar anchors its scale.

Two correction modes (controlled by `use_band_correction` parameter):

**Per-distance-band affine (default, use_band_correction=True)**:
  Fits separate affine models corrected = a_i * depth + b_i for each distance
  band (near/mid/far/very-far). Uses boundary blending within band_blend_m of
  each edge to avoid discontinuities. Falls back to global scale for bands
  with insufficient pairs.

**Global scale (use_band_correction=False)**:
  Single scale factor: corrected = scale * depth, where
  scale = median(lidar_range / da3_range).

Algorithm per frame:
  1. Find closest lidar scan (within sync_slop_sec)
  2. Project DA3 depth pixels to laser frame (stride=4 subsample)
  3. Keep pixels near scan plane (|z_laser| < plane_tolerance_m)
  4. Match angular bins -> (da3_range, lidar_range, da3_depth) triples
  5. Fit global scale + per-band affine coefficients
  6. EMA smooth all coefficients independently
  7. Apply correction with boundary blending
  8. Fallback: scale=1.0 (pass-through) if stale >stale_frames

Note: DA3 depth already has depth_correction=1.142 applied. This node
provides an ADDITIONAL dynamic correction on top of that. If 1.142 is
perfect, the global scale will converge to ~1.0.

Subscribes:
  /scan_filtered (LaserScan, 10Hz)
  /camera/mono_da3/image_raw (Image, 16UC1 depth in mm)
  /camera/mono_da3/camera_info (CameraInfo)

Publishes:
  /camera/mono_da3_corrected/image_raw (Image, 16UC1)
  /camera/mono_da3_corrected/camera_info (CameraInfo)
  /camera/mono_da3_corrected/points (PointCloud2)

Usage:
  ros2 run wheelchair_mono_depth da3_lidar_anchor_node
"""

import time
from collections import deque

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField, LaserScan
from cv_bridge import CvBridge

try:
    import tf2_ros
    HAS_TF2 = True
except ImportError:
    HAS_TF2 = False


class DA3LidarAnchorNode(Node):
    """Corrects DA3 monocular depth using lidar ground truth."""

    def __init__(self):
        super().__init__('da3_lidar_anchor_node')

        # --- Parameters ---
        self.declare_parameter('lidar_topic', '/scan_filtered')
        self.declare_parameter('depth_topic', '/camera/mono_da3/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/mono_da3/camera_info')
        self.declare_parameter('output_depth_topic', '/camera/mono_da3_corrected/image_raw')
        self.declare_parameter('output_info_topic', '/camera/mono_da3_corrected/camera_info')
        self.declare_parameter('output_pc_topic', '/camera/mono_da3_corrected/points')

        self.declare_parameter('sync_slop_sec', 0.10)
        self.declare_parameter('plane_tolerance_m', 0.15)
        self.declare_parameter('stride', 4)
        self.declare_parameter('min_pairs', 15)
        self.declare_parameter('ema_alpha', 0.10)
        self.declare_parameter('stale_frames', 30)
        self.declare_parameter('max_depth', 6.0)
        self.declare_parameter('angular_bin_deg', 0.5)
        self.declare_parameter('min_range', 0.3)
        self.declare_parameter('camera_frame', 'camera_depth_optical_frame')
        self.declare_parameter('laser_frame', 'laser')

        # Per-distance-band affine correction
        self.declare_parameter('use_band_correction', True)
        self.declare_parameter('band_boundaries',
                               [0.3, 1.0, 2.0, 3.5, 6.0])
        self.declare_parameter('band_blend_m', 0.2)

        self._sync_slop = self.get_parameter('sync_slop_sec').value
        self._plane_tol = self.get_parameter('plane_tolerance_m').value
        self._stride = self.get_parameter('stride').value
        self._min_pairs = self.get_parameter('min_pairs').value
        self._ema_alpha = self.get_parameter('ema_alpha').value
        self._stale_frames = self.get_parameter('stale_frames').value
        self._max_depth = self.get_parameter('max_depth').value
        self._angular_bin_deg = self.get_parameter('angular_bin_deg').value
        self._min_range = self.get_parameter('min_range').value
        self._camera_frame = self.get_parameter('camera_frame').value
        self._laser_frame = self.get_parameter('laser_frame').value

        self._use_band_correction = self.get_parameter('use_band_correction').value
        self._band_boundaries = list(
            self.get_parameter('band_boundaries').value)
        self._band_blend_m = self.get_parameter('band_blend_m').value

        self._bridge = CvBridge()

        # --- State ---
        self._lidar_buf = deque(maxlen=30)
        self._cam_info = None
        self._fx = self._fy = self._cx = self._cy = None

        # Dynamic scale factor: corrected = scale * depth (global fallback)
        self._scale = None  # None until first fit
        self._frames_since_fit = 0
        self._fit_count = 0
        self._frame_count = 0

        # Per-band affine coefficients: corrected = a * depth + b
        # _band_boundaries = [b0, b1, b2, ..., bN] defines N-1 bands
        n_bands = len(self._band_boundaries) - 1
        self._band_coeffs = [None] * n_bands  # Each: (a, b) or None
        self._band_fit_counts = [0] * n_bands

        # Pre-allocated pixel grids
        self._u_grid = None
        self._v_grid = None

        # TF
        self._tf_buffer = None
        self._tf_listener = None
        self._cam_to_laser = None
        if HAS_TF2:
            self._tf_buffer = tf2_ros.Buffer()
            self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        # --- QoS ---
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # --- Subscribers ---
        lidar_topic = self.get_parameter('lidar_topic').value
        depth_topic = self.get_parameter('depth_topic').value
        info_topic = self.get_parameter('camera_info_topic').value

        self.create_subscription(
            LaserScan, lidar_topic, self._lidar_cb, sensor_qos)
        self.create_subscription(
            Image, depth_topic, self._depth_cb, sensor_qos)
        self.create_subscription(
            CameraInfo, info_topic, self._info_cb, sensor_qos)

        # --- Publishers ---
        out_depth = self.get_parameter('output_depth_topic').value
        out_info = self.get_parameter('output_info_topic').value
        out_pc = self.get_parameter('output_pc_topic').value

        self._depth_pub = self.create_publisher(Image, out_depth, 10)
        self._info_pub = self.create_publisher(CameraInfo, out_info, 10)
        self._pc_pub = self.create_publisher(PointCloud2, out_pc, 10)

        self.get_logger().info(
            f'DA3LidarAnchor: {depth_topic} + {lidar_topic} -> {out_depth}')
        self.get_logger().info(
            f'  sync_slop={self._sync_slop}s, min_pairs={self._min_pairs}, '
            f'ema_alpha={self._ema_alpha}')
        if self._use_band_correction:
            self.get_logger().info(
                f'  Band correction ON: boundaries={self._band_boundaries}, '
                f'blend={self._band_blend_m}m')
        else:
            self.get_logger().info('  Band correction OFF: global scale only')

    # ===================================================================
    # CALLBACKS
    # ===================================================================
    def _lidar_cb(self, msg: LaserScan):
        self._lidar_buf.append(msg)

    def _info_cb(self, msg: CameraInfo):
        if self._cam_info is not None:
            return
        self._cam_info = msg
        K = msg.k
        self._fx = K[0]
        self._fy = K[4]
        self._cx = K[2]
        self._cy = K[5]
        self.get_logger().info(
            f'CameraInfo: {msg.width}x{msg.height}, '
            f'fx={K[0]:.1f}, fy={K[4]:.1f}')

    def _depth_cb(self, msg: Image):
        """Main processing: correct depth using lidar anchor."""
        if self._cam_info is None:
            return

        self._frame_count += 1
        t0 = time.monotonic()

        # Decode depth (16UC1, mm)
        try:
            depth_mm = self._bridge.imgmsg_to_cv2(msg, desired_encoding='16UC1')
        except Exception as e:
            self.get_logger().warn(f'Depth decode failed: {e}',
                                  throttle_duration_sec=5.0)
            return

        depth_m = depth_mm.astype(np.float32) / 1000.0
        h, w = depth_m.shape

        # Init pixel grids
        if self._u_grid is None or self._u_grid.shape != (h, w):
            u = np.arange(w, dtype=np.float32)
            v = np.arange(h, dtype=np.float32)
            self._u_grid, self._v_grid = np.meshgrid(u, v)

        # Try to get TF camera->laser
        if self._cam_to_laser is None:
            self._cam_to_laser = self._lookup_transform()

        # Find closest lidar scan
        depth_sec = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        lidar_msg = self._find_closest_lidar(depth_sec)

        # Try scale fit
        if lidar_msg is not None and self._cam_to_laser is not None:
            pairs = self._extract_pairs(depth_m, lidar_msg)
            if pairs is not None and len(pairs) >= self._min_pairs:
                self._update_scale(pairs)
                if self._use_band_correction:
                    self._update_band_coeffs(pairs)
                self._frames_since_fit = 0

        self._frames_since_fit += 1

        # Apply correction
        if self._use_band_correction:
            corrected = self._apply_band_correction(depth_m)
        else:
            scale = self._get_active_scale()
            corrected = (depth_m * scale).astype(np.float32)
        corrected = np.clip(corrected, 0.0, self._max_depth)

        # Publish depth image
        corrected_mm = (corrected * 1000.0).clip(0, 65535).astype(np.uint16)
        depth_out = self._bridge.cv2_to_imgmsg(corrected_mm, encoding='16UC1')
        depth_out.header = msg.header
        self._depth_pub.publish(depth_out)

        # Publish camera info
        info_out = CameraInfo()
        info_out.header = msg.header
        info_out.width = self._cam_info.width
        info_out.height = self._cam_info.height
        info_out.distortion_model = self._cam_info.distortion_model
        info_out.d = list(self._cam_info.d)
        info_out.k = list(self._cam_info.k)
        info_out.r = list(self._cam_info.r)
        info_out.p = list(self._cam_info.p)
        self._info_pub.publish(info_out)

        # Publish point cloud
        self._publish_pointcloud(corrected, msg.header)

        # Logging
        latency_ms = (time.monotonic() - t0) * 1000.0
        if self._frame_count % 10 == 0:
            active = (self._scale is not None and
                      self._frames_since_fit < self._stale_frames)
            if self._use_band_correction:
                mode = 'band' if active else 'passthru'
            else:
                mode = 'lidar' if active else 'passthru'
            global_s = self._get_active_scale()
            self.get_logger().info(
                f'Frame {self._frame_count}: {latency_ms:.1f}ms, '
                f'mode={mode}, global_scale={global_s:.4f}, '
                f'fits={self._fit_count}, stale={self._frames_since_fit}')
            # Per-band logging
            if self._use_band_correction:
                bb = self._band_boundaries
                for i in range(len(bb) - 1):
                    c = self._band_coeffs[i]
                    if c is not None:
                        a_i, b_i = c
                        mid = (bb[i] + bb[i + 1]) / 2.0
                        eff_scale = a_i + b_i / mid if mid > 0 else a_i
                        self.get_logger().info(
                            f'  Band {i} [{bb[i]:.1f}-{bb[i+1]:.1f}m]: '
                            f'n_fits={self._band_fit_counts[i]}, '
                            f'a={a_i:.4f}, b={b_i:.4f}, '
                            f'eff_scale@mid={eff_scale:.4f}')
                    else:
                        self.get_logger().info(
                            f'  Band {i} [{bb[i]:.1f}-{bb[i+1]:.1f}m]: '
                            f'no fit (using global)')

    # ===================================================================
    # TRANSFORM LOOKUP
    # ===================================================================
    def _lookup_transform(self):
        """Look up camera_depth_optical_frame -> laser transform."""
        if self._tf_buffer is None:
            return None
        try:
            tf_msg = self._tf_buffer.lookup_transform(
                self._laser_frame, self._camera_frame,
                rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=1.0))
            t = tf_msg.transform.translation
            q = tf_msg.transform.rotation
            mat = self._quat_to_matrix(q.x, q.y, q.z, q.w)
            mat[0, 3] = t.x
            mat[1, 3] = t.y
            mat[2, 3] = t.z
            self.get_logger().info(
                f'TF {self._camera_frame} -> {self._laser_frame}: '
                f't=[{t.x:.3f}, {t.y:.3f}, {t.z:.3f}]')
            return mat
        except Exception as e:
            self.get_logger().warn(
                f'TF lookup failed: {e}', throttle_duration_sec=5.0)
            return None

    @staticmethod
    def _quat_to_matrix(x, y, z, w):
        """Quaternion to 4x4 homogeneous rotation matrix."""
        mat = np.eye(4, dtype=np.float64)
        mat[0, 0] = 1 - 2*(y*y + z*z)
        mat[0, 1] = 2*(x*y - z*w)
        mat[0, 2] = 2*(x*z + y*w)
        mat[1, 0] = 2*(x*y + z*w)
        mat[1, 1] = 1 - 2*(x*x + z*z)
        mat[1, 2] = 2*(y*z - x*w)
        mat[2, 0] = 2*(x*z - y*w)
        mat[2, 1] = 2*(y*z + x*w)
        mat[2, 2] = 1 - 2*(x*x + y*y)
        return mat

    # ===================================================================
    # LIDAR MATCHING
    # ===================================================================
    def _find_closest_lidar(self, target_sec):
        """Find closest lidar scan within sync slop."""
        best = None
        best_dt = self._sync_slop + 1.0
        for msg in self._lidar_buf:
            t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            dt = abs(t - target_sec)
            if dt < best_dt:
                best_dt = dt
                best = msg
        return best if best_dt <= self._sync_slop else None

    def _extract_pairs(self, depth_m, lidar_msg):
        """Extract (da3_range, lidar_range, da3_depth) triples at matching bins.

        Returns:
            np.ndarray of shape (N, 3): [da3_range, lidar_range, da3_depth]
            where da3_depth is the raw camera-frame depth (z_cam) for band
            assignment, and da3_range is the projected laser-plane range.
            Returns None if insufficient valid pairs.
        """
        h, w = depth_m.shape
        stride = self._stride
        fx, fy = self._fx, self._fy
        cx, cy = self._cx, self._cy

        # Subsample pixel coordinates
        vs = np.arange(0, h, stride)
        us = np.arange(0, w, stride)
        uu, vv = np.meshgrid(us, vs)
        uu = uu.ravel()
        vv = vv.ravel()

        # Get depths at subsampled pixels
        d = depth_m[vv, uu]
        valid = (d > self._min_range) & (d < self._max_depth)
        if np.sum(valid) < self._min_pairs:
            return None

        uu = uu[valid].astype(np.float32)
        vv = vv[valid].astype(np.float32)
        d = d[valid]

        # Back-project to camera 3D (optical frame: x-right, y-down, z-forward)
        x_cam = (uu - cx) * d / fx
        y_cam = (vv - cy) * d / fy
        z_cam = d

        # Transform to laser frame
        pts_cam = np.stack([x_cam, y_cam, z_cam, np.ones_like(d)], axis=0)
        pts_laser = self._cam_to_laser @ pts_cam

        x_l = pts_laser[0]
        y_l = pts_laser[1]
        z_l = pts_laser[2]

        # Filter: keep points near scan plane
        on_plane = np.abs(z_l) < self._plane_tol
        if np.sum(on_plane) < self._min_pairs:
            return None

        x_l = x_l[on_plane]
        y_l = y_l[on_plane]
        da3_depth_on_plane = d[on_plane]  # raw DA3 depth for band assignment
        da3_range = np.sqrt(x_l**2 + y_l**2)
        da3_angle = np.arctan2(y_l, x_l)

        # Build lidar lookup
        lidar_ranges = np.array(lidar_msg.ranges, dtype=np.float32)
        angle_min = lidar_msg.angle_min
        angle_inc = lidar_msg.angle_increment
        n_bins = len(lidar_ranges)

        # Map DA3 angles to lidar bin indices
        bin_idx = ((da3_angle - angle_min) / angle_inc).astype(np.int32)
        in_range = (bin_idx >= 0) & (bin_idx < n_bins)
        bin_idx = bin_idx[in_range]
        da3_range = da3_range[in_range]
        da3_depth_matched = da3_depth_on_plane[in_range]

        # Get lidar ranges at matched bins
        lidar_r = lidar_ranges[bin_idx]
        lidar_valid = np.isfinite(lidar_r) & \
                      (lidar_r >= max(lidar_msg.range_min, self._min_range)) & \
                      (lidar_r <= lidar_msg.range_max)

        if np.sum(lidar_valid) < self._min_pairs:
            return None

        da3_r = da3_range[lidar_valid]
        lid_r = lidar_r[lidar_valid]
        da3_d = da3_depth_matched[lidar_valid]

        # Bin-average to reduce noise: one pair per angular bin
        bin_deg = self._angular_bin_deg
        x_l_final = x_l[in_range][lidar_valid]
        y_l_final = y_l[in_range][lidar_valid]
        da3_angles_valid = np.arctan2(y_l_final, x_l_final)
        angle_bins = (da3_angles_valid / np.deg2rad(bin_deg)).astype(np.int32)
        unique_bins = np.unique(angle_bins)

        triples = []
        for b in unique_bins:
            mask = angle_bins == b
            if np.sum(mask) >= 2:  # need >=2 points per bin for reliable median
                triples.append((
                    np.median(da3_r[mask]),
                    np.median(lid_r[mask]),
                    np.median(da3_d[mask]),
                ))

        if len(triples) < self._min_pairs:
            return None

        return np.array(triples, dtype=np.float64)  # (N, 3): [da3, lidar, depth]

    # ===================================================================
    # SCALE FIT
    # ===================================================================
    def _update_scale(self, pairs):
        """Compute median scale factor and EMA smooth it.

        scale = median(lidar_range / da3_range) for all matched pairs.
        pairs: (N, 3) array with columns [da3_range, lidar_range, da3_depth].
        """
        da3_r = pairs[:, 0]
        lid_r = pairs[:, 1]

        # Compute per-pair ratios
        ratios = lid_r / np.maximum(da3_r, 0.01)

        # Reject outliers: > 2sigma from median
        med = np.median(ratios)
        dev = np.abs(ratios - med)
        sigma = np.median(dev) * 1.4826  # MAD-based robust sigma
        inlier = dev < 2.0 * max(sigma, 0.01)

        if np.sum(inlier) < self._min_pairs:
            return

        new_scale = float(np.median(ratios[inlier]))

        # Sanity: scale should be in [0.7, 1.4]
        # (DA3 already has 1.142x correction, so scale should be near 1.0)
        if new_scale < 0.7 or new_scale > 1.4:
            return

        # EMA smooth
        alpha = self._ema_alpha
        if self._scale is None:
            self._scale = new_scale
        else:
            self._scale = alpha * new_scale + (1.0 - alpha) * self._scale

        self._fit_count += 1

    def _get_active_scale(self):
        """Return the active scale factor."""
        if self._scale is not None and \
           self._frames_since_fit < self._stale_frames:
            return self._scale
        # Fallback: pass-through (DA3 already has depth_correction applied)
        return 1.0

    # ===================================================================
    # PER-BAND AFFINE CORRECTION
    # ===================================================================
    def _update_band_coeffs(self, pairs):
        """Fit per-band affine coefficients: corrected = a * da3 + b.

        pairs: (N, 3) array with columns [da3_range, lidar_range, da3_depth].
        Uses da3_depth (col 2) for band assignment, da3_range (col 0) and
        lidar_range (col 1) for the affine fit.
        """
        da3_r = pairs[:, 0]
        lid_r = pairs[:, 1]
        da3_d = pairs[:, 2]  # raw DA3 depth for band assignment

        bb = self._band_boundaries
        alpha = self._ema_alpha
        n_bands = len(bb) - 1

        for i in range(n_bands):
            lo, hi = bb[i], bb[i + 1]
            band_mask = (da3_d >= lo) & (da3_d < hi)
            n_in_band = int(np.sum(band_mask))

            if n_in_band < self._min_pairs:
                # Insufficient pairs: keep existing coefficients (or None)
                continue

            da3_band = da3_r[band_mask]
            lid_band = lid_r[band_mask]

            # MAD-based outlier rejection within band
            ratios = lid_band / np.maximum(da3_band, 0.01)
            med = np.median(ratios)
            dev = np.abs(ratios - med)
            sigma = np.median(dev) * 1.4826
            inlier = dev < 2.0 * max(sigma, 0.01)

            if np.sum(inlier) < self._min_pairs:
                continue

            da3_in = da3_band[inlier]
            lid_in = lid_band[inlier]

            # Fit affine: lidar = a * da3 + b via least squares
            A = np.column_stack([da3_in, np.ones(len(da3_in))])
            result, _, _, _ = np.linalg.lstsq(A, lid_in, rcond=None)
            new_a = float(result[0])
            new_b = float(result[1])

            # Sanity: a should be in [0.5, 1.8], |b| < 1.0
            if new_a < 0.5 or new_a > 1.8 or abs(new_b) > 1.0:
                continue

            # EMA smooth
            if self._band_coeffs[i] is None:
                self._band_coeffs[i] = (new_a, new_b)
            else:
                old_a, old_b = self._band_coeffs[i]
                self._band_coeffs[i] = (
                    alpha * new_a + (1.0 - alpha) * old_a,
                    alpha * new_b + (1.0 - alpha) * old_b,
                )

            self._band_fit_counts[i] += 1

    def _apply_band_correction(self, depth_m):
        """Apply per-band affine correction with boundary blending.

        For each pixel with depth d, find its band and apply corrected = a*d + b.
        Within band_blend_m of a boundary, linearly interpolate between
        adjacent bands to avoid discontinuities.

        Falls back to global scale for bands without fitted coefficients.
        """
        corrected = np.empty_like(depth_m, dtype=np.float32)
        bb = self._band_boundaries
        n_bands = len(bb) - 1
        blend = self._band_blend_m
        global_scale = self._get_active_scale()

        # Get effective (a, b) for each band, falling back to global
        eff_coeffs = []
        for i in range(n_bands):
            if self._band_coeffs[i] is not None:
                eff_coeffs.append(self._band_coeffs[i])
            else:
                eff_coeffs.append((global_scale, 0.0))

        # Process each pixel
        d = depth_m
        # Start with global fallback for out-of-range pixels
        corrected[:] = d * global_scale

        for i in range(n_bands):
            lo, hi = bb[i], bb[i + 1]
            a_i, b_i = eff_coeffs[i]

            # Core region: [lo + blend, hi - blend) -- pure band correction
            core_lo = lo + blend if i > 0 else lo
            core_hi = hi - blend if i < n_bands - 1 else hi
            core_mask = (d >= core_lo) & (d < core_hi) & (d > 0)
            corrected[core_mask] = a_i * d[core_mask] + b_i

            # Lower blend region: [lo, lo + blend) -- blend with band i-1
            if i > 0 and blend > 0:
                blend_mask = (d >= lo) & (d < lo + blend) & (d > 0)
                if np.any(blend_mask):
                    d_blend = d[blend_mask]
                    # t=0 at lo (fully band i-1), t=1 at lo+blend (fully band i)
                    t = (d_blend - lo) / blend
                    a_prev, b_prev = eff_coeffs[i - 1]
                    val_prev = a_prev * d_blend + b_prev
                    val_curr = a_i * d_blend + b_i
                    corrected[blend_mask] = (1.0 - t) * val_prev + t * val_curr

        return corrected

    # ===================================================================
    # POINTCLOUD
    # ===================================================================
    def _publish_pointcloud(self, depth_m, header):
        """Publish corrected depth as PointCloud2."""
        fx, fy = self._fx, self._fy
        cx, cy = self._cx, self._cy

        z = depth_m
        x = (self._u_grid - cx) * z / fx
        y = (self._v_grid - cy) * z / fy
        valid = z > 0.01
        points = np.stack([x[valid], y[valid], z[valid]], axis=-1).astype(np.float32)

        pc_msg = PointCloud2()
        pc_msg.header = header
        pc_msg.height = 1
        pc_msg.width = len(points)
        pc_msg.is_dense = True
        pc_msg.is_bigendian = False
        pc_msg.point_step = 12
        pc_msg.row_step = 12 * len(points)
        pc_msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        pc_msg.data = points.tobytes()
        self._pc_pub.publish(pc_msg)


def main(args=None):
    rclpy.init(args=args)
    node = DA3LidarAnchorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
