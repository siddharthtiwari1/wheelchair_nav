#!/usr/bin/env python3
"""Benchmark node comparing stereo vs monocular depth-derived LaserScans.

Subscribes to /scan_stereo, /scan_mono, and /scan_mono_d2l with BEST_EFFORT QoS
(matching pointcloud_to_laserscan / depthimage_to_laserscan publishers), computes
per-frame quality metrics, writes CSV logs, publishes a diff scan, and REPUBLISHES
all scans as RELIABLE for RViz (which defaults to RELIABLE subscriptions).

Metrics computed:
  - MAE, RMSE, max error across valid overlapping bins
  - Pearson correlation coefficient
  - Valid bin count (stereo vs mono)
  - False positive/negative rates (threshold configurable)
  - Distance-banded metrics: [0-1m], [1-2m], [2-3m], [3-5m]
  - CPU% (via psutil)
  - GPU memory (via torch.cuda if available)
"""

import os
import csv
import time
import array as _array
from collections import deque

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import LaserScan


class DepthScanBenchmarkNode(Node):
    """Compare stereo and mono LaserScans with QoS bridging."""

    DISTANCE_BANDS = [
        ('0-1m', 0.0, 1.0),
        ('1-2m', 1.0, 2.0),
        ('2-3m', 2.0, 3.0),
        ('3-5m', 3.0, 5.0),
    ]

    def __init__(self):
        super().__init__('depth_scan_benchmark_node')

        # --- Parameters ---
        self.declare_parameter('stereo_scan_topic', '/scan_stereo')
        self.declare_parameter('mono_scan_topic', '/scan_mono')
        self.declare_parameter('mono_d2l_scan_topic', '/scan_mono_d2l')
        self.declare_parameter('output_dir',
                               '/home/sidd/wheelchair_nav/eval_output/depth_comparison')
        self.declare_parameter('sync_slop', 0.15)
        self.declare_parameter('fp_threshold_m', 0.5)
        self.declare_parameter('summary_interval', 50)

        stereo_topic = self.get_parameter('stereo_scan_topic').value
        mono_topic = self.get_parameter('mono_scan_topic').value
        d2l_topic = self.get_parameter('mono_d2l_scan_topic').value
        self._output_dir = self.get_parameter('output_dir').value
        self._sync_slop = self.get_parameter('sync_slop').value
        self._fp_thresh = self.get_parameter('fp_threshold_m').value
        self._summary_interval = self.get_parameter('summary_interval').value

        # --- State ---
        self._frame_count = 0
        self._accum = {
            'mae': [], 'rmse': [], 'max_err': [], 'corr': [],
            'n_valid_stereo': [], 'n_valid_mono': [], 'n_overlap': [],
            'fp_rate': [], 'fn_rate': [],
            'cpu_pct': [], 'gpu_mb': [],
            # D2L comparison
            'mae_d2l': [], 'rmse_d2l': [], 'corr_d2l': [],
        }
        for name, _, _ in self.DISTANCE_BANDS:
            self._accum[f'mae_{name}'] = []
            self._accum[f'rmse_{name}'] = []
            # D2L (corrected) per-band metrics
            self._accum[f'mae_d2l_{name}'] = []
            self._accum[f'rmse_d2l_{name}'] = []

        # --- Message buffers (manual sync instead of message_filters) ---
        self._stereo_buf = deque(maxlen=20)
        self._mono_buf = deque(maxlen=20)
        self._d2l_buf = deque(maxlen=20)
        self._last_stereo_stamp = 0.0
        self._last_mono_stamp = 0.0

        # --- Output directory ---
        os.makedirs(self._output_dir, exist_ok=True)

        # --- CSV ---
        timestamp_str = time.strftime('%Y%m%d_%H%M%S')
        csv_path = os.path.join(
            self._output_dir, f'{timestamp_str}_comparison.csv')
        self._csv_file = open(csv_path, 'w', newline='')
        self._csv_writer = csv.writer(self._csv_file)
        header = [
            'frame', 'timestamp',
            'mae', 'rmse', 'max_error', 'pearson_corr',
            'n_valid_stereo', 'n_valid_mono', 'n_overlap',
            'fp_rate', 'fn_rate',
        ]
        for name, _, _ in self.DISTANCE_BANDS:
            header.extend([f'mae_{name}', f'rmse_{name}'])
        header.extend(['mae_d2l', 'rmse_d2l', 'corr_d2l'])
        for name, _, _ in self.DISTANCE_BANDS:
            header.extend([f'mae_d2l_{name}', f'rmse_d2l_{name}'])
        header.extend(['cpu_pct', 'gpu_mem_mb'])
        self._csv_writer.writerow(header)

        # --- QoS: BEST_EFFORT to match pc2ls/d2l publishers ---
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # --- Subscribers (plain rclpy, NOT message_filters) ---
        self._stereo_sub = self.create_subscription(
            LaserScan, stereo_topic, self._stereo_cb, sensor_qos)
        self._mono_sub = self.create_subscription(
            LaserScan, mono_topic, self._mono_cb, sensor_qos)
        self._d2l_sub = self.create_subscription(
            LaserScan, d2l_topic, self._d2l_cb, sensor_qos)

        # --- RELIABLE publishers for RViz (bridge BEST_EFFORT → RELIABLE) ---
        reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        self._stereo_relay_pub = self.create_publisher(
            LaserScan, '/benchmark/scan_stereo_reliable', reliable_qos)
        self._mono_relay_pub = self.create_publisher(
            LaserScan, '/benchmark/scan_mono_reliable', reliable_qos)
        self._d2l_relay_pub = self.create_publisher(
            LaserScan, '/benchmark/scan_d2l_reliable', reliable_qos)
        self._diff_pub = self.create_publisher(
            LaserScan, '/benchmark/diff_scan', reliable_qos)

        # --- Timer for manual sync matching at 10Hz ---
        self._match_timer = self.create_timer(0.1, self._match_cb)

        # --- Resource monitoring ---
        try:
            import psutil
            self._process = psutil.Process()
            self._process.cpu_percent()
        except ImportError:
            self._process = None
            self.get_logger().warn('psutil not available, CPU monitoring disabled')

        self._gpu_available = False
        try:
            import torch
            if torch.cuda.is_available():
                self._gpu_available = True
        except ImportError:
            pass

        self.get_logger().info(
            f'DepthScanBenchmark: {stereo_topic} vs {mono_topic} vs {d2l_topic} '
            f'(sync_slop={self._sync_slop}s, fp_thresh={self._fp_thresh}m)')
        self.get_logger().info(
            f'Relaying scans as RELIABLE: /benchmark/scan_*_reliable')
        self.get_logger().info(f'CSV output: {csv_path}')

    @staticmethod
    def _stamp_to_sec(stamp):
        return stamp.sec + stamp.nanosec * 1e-9

    def _stereo_cb(self, msg: LaserScan):
        self._stereo_buf.append(msg)
        self._stereo_relay_pub.publish(msg)

    def _mono_cb(self, msg: LaserScan):
        self._mono_buf.append(msg)
        self._mono_relay_pub.publish(msg)

    def _d2l_cb(self, msg: LaserScan):
        self._d2l_buf.append(msg)
        self._d2l_relay_pub.publish(msg)

    def _find_closest(self, buf, target_sec, slop):
        """Find message in buffer closest to target_sec within slop."""
        best = None
        best_dt = slop + 1.0
        for msg in buf:
            dt = abs(self._stamp_to_sec(msg.header.stamp) - target_sec)
            if dt < best_dt:
                best_dt = dt
                best = msg
        return best if best_dt <= slop else None

    def _match_cb(self):
        """Timer callback: try to match stereo + mono scans by timestamp."""
        if not self._stereo_buf or not self._mono_buf:
            return

        # Use latest stereo as reference
        stereo_msg = self._stereo_buf[-1]
        stereo_sec = self._stamp_to_sec(stereo_msg.header.stamp)

        # Skip if we already processed this timestamp
        if abs(stereo_sec - self._last_stereo_stamp) < 0.01:
            return

        # Find closest mono scan
        mono_msg = self._find_closest(self._mono_buf, stereo_sec, self._sync_slop)
        if mono_msg is None:
            return

        mono_sec = self._stamp_to_sec(mono_msg.header.stamp)
        if abs(mono_sec - self._last_mono_stamp) < 0.01:
            return

        self._last_stereo_stamp = stereo_sec
        self._last_mono_stamp = mono_sec

        # Find closest d2l scan (optional)
        d2l_msg = self._find_closest(self._d2l_buf, stereo_sec, self._sync_slop)

        # Run comparison
        self._compare(stereo_msg, mono_msg, d2l_msg)

    def _compare(self, stereo_msg: LaserScan, mono_msg: LaserScan,
                 d2l_msg=None):
        """Compare synchronized stereo and mono scans."""
        stereo = np.array(stereo_msg.ranges, dtype=np.float32)
        mono = np.array(mono_msg.ranges, dtype=np.float32)

        min_len = min(len(stereo), len(mono))
        stereo = stereo[:min_len]
        mono = mono[:min_len]

        r_min = stereo_msg.range_min
        r_max = stereo_msg.range_max

        valid_s = np.isfinite(stereo) & (stereo >= r_min) & (stereo <= r_max)
        valid_m = np.isfinite(mono) & (mono >= r_min) & (mono <= r_max)
        overlap = valid_s & valid_m

        n_valid_s = int(np.sum(valid_s))
        n_valid_m = int(np.sum(valid_m))
        n_overlap = int(np.sum(overlap))

        # --- Overlap metrics ---
        mae = rmse_val = max_err = corr = np.nan
        fp_rate = fn_rate = 0.0

        if n_overlap > 10:
            diff = np.abs(stereo[overlap] - mono[overlap])
            mae = float(np.mean(diff))
            rmse_val = float(np.sqrt(np.mean(diff ** 2)))
            max_err = float(np.max(diff))
            if np.std(stereo[overlap]) > 0.001 and np.std(mono[overlap]) > 0.001:
                corr = float(np.corrcoef(stereo[overlap], mono[overlap])[0, 1])

        if n_valid_s > 0:
            fn_mask = valid_s & ~valid_m
            fn_rate = float(np.sum(fn_mask)) / n_valid_s

        if n_valid_m > 0:
            fp_mask = valid_m & ~valid_s
            fp_rate = float(np.sum(fp_mask)) / n_valid_m

        # --- Distance-banded metrics ---
        band_metrics = {}
        for name, lo, hi in self.DISTANCE_BANDS:
            band_mask = overlap & (stereo >= lo) & (stereo < hi)
            n_band = int(np.sum(band_mask))
            if n_band > 5:
                band_diff = np.abs(stereo[band_mask] - mono[band_mask])
                band_metrics[f'mae_{name}'] = float(np.mean(band_diff))
                band_metrics[f'rmse_{name}'] = float(
                    np.sqrt(np.mean(band_diff ** 2)))
            else:
                band_metrics[f'mae_{name}'] = np.nan
                band_metrics[f'rmse_{name}'] = np.nan

        # --- D2L comparison ---
        mae_d2l = rmse_d2l = corr_d2l = np.nan
        d2l_band_metrics = {}
        for name, _, _ in self.DISTANCE_BANDS:
            d2l_band_metrics[f'mae_d2l_{name}'] = np.nan
            d2l_band_metrics[f'rmse_d2l_{name}'] = np.nan
        if d2l_msg is not None:
            d2l = np.array(d2l_msg.ranges, dtype=np.float32)
            d2l_len = min(len(stereo), len(d2l))
            s_d2l = stereo[:d2l_len]
            d_d2l = d2l[:d2l_len]
            valid_d = np.isfinite(d_d2l) & (d_d2l >= r_min) & (d_d2l <= r_max)
            valid_sd = valid_s[:d2l_len] & valid_d
            n_d2l_overlap = int(np.sum(valid_sd))
            if n_d2l_overlap > 10:
                dd = np.abs(s_d2l[valid_sd] - d_d2l[valid_sd])
                mae_d2l = float(np.mean(dd))
                rmse_d2l = float(np.sqrt(np.mean(dd ** 2)))
                if np.std(s_d2l[valid_sd]) > 0.001 and np.std(d_d2l[valid_sd]) > 0.001:
                    corr_d2l = float(np.corrcoef(
                        s_d2l[valid_sd], d_d2l[valid_sd])[0, 1])

            # Per-band D2L metrics (use stereo as ground truth for banding)
            for name, lo, hi in self.DISTANCE_BANDS:
                band_mask = valid_sd & (s_d2l >= lo) & (s_d2l < hi)
                n_band = int(np.sum(band_mask))
                if n_band > 5:
                    bd = np.abs(s_d2l[band_mask] - d_d2l[band_mask])
                    d2l_band_metrics[f'mae_d2l_{name}'] = float(np.mean(bd))
                    d2l_band_metrics[f'rmse_d2l_{name}'] = float(
                        np.sqrt(np.mean(bd ** 2)))

        # --- Resource monitoring ---
        cpu_pct = 0.0
        gpu_mb = 0.0
        if self._process is not None:
            try:
                cpu_pct = self._process.cpu_percent()
            except Exception:
                pass

        if self._gpu_available:
            try:
                import torch
                gpu_mb = torch.cuda.memory_allocated() / (1024 * 1024)
            except Exception:
                pass

        # --- Publish diff scan (RELIABLE for RViz) ---
        self._publish_diff_scan(stereo, mono, valid_s, valid_m, stereo_msg)

        # --- CSV row ---
        row = [
            self._frame_count,
            f'{stereo_msg.header.stamp.sec}.{stereo_msg.header.stamp.nanosec:09d}',
            f'{mae:.4f}' if not np.isnan(mae) else '',
            f'{rmse_val:.4f}' if not np.isnan(rmse_val) else '',
            f'{max_err:.4f}' if not np.isnan(max_err) else '',
            f'{corr:.4f}' if not np.isnan(corr) else '',
            n_valid_s, n_valid_m, n_overlap,
            f'{fp_rate:.4f}', f'{fn_rate:.4f}',
        ]
        for name, _, _ in self.DISTANCE_BANDS:
            row.append(f'{band_metrics.get(f"mae_{name}", np.nan):.4f}'
                        if not np.isnan(band_metrics.get(f'mae_{name}', np.nan))
                        else '')
            row.append(f'{band_metrics.get(f"rmse_{name}", np.nan):.4f}'
                        if not np.isnan(band_metrics.get(f'rmse_{name}', np.nan))
                        else '')
        row.extend([
            f'{mae_d2l:.4f}' if not np.isnan(mae_d2l) else '',
            f'{rmse_d2l:.4f}' if not np.isnan(rmse_d2l) else '',
            f'{corr_d2l:.4f}' if not np.isnan(corr_d2l) else '',
        ])
        for name, _, _ in self.DISTANCE_BANDS:
            v = d2l_band_metrics.get(f'mae_d2l_{name}', np.nan)
            row.append(f'{v:.4f}' if not np.isnan(v) else '')
            v = d2l_band_metrics.get(f'rmse_d2l_{name}', np.nan)
            row.append(f'{v:.4f}' if not np.isnan(v) else '')
        row.extend([f'{cpu_pct:.1f}', f'{gpu_mb:.1f}'])
        self._csv_writer.writerow(row)
        self._csv_file.flush()

        # --- Accumulate ---
        if not np.isnan(mae):
            self._accum['mae'].append(mae)
            self._accum['rmse'].append(rmse_val)
            self._accum['max_err'].append(max_err)
        if not np.isnan(corr):
            self._accum['corr'].append(corr)
        if not np.isnan(mae_d2l):
            self._accum['mae_d2l'].append(mae_d2l)
            self._accum['rmse_d2l'].append(rmse_d2l)
        if not np.isnan(corr_d2l):
            self._accum['corr_d2l'].append(corr_d2l)
        self._accum['n_valid_stereo'].append(n_valid_s)
        self._accum['n_valid_mono'].append(n_valid_m)
        self._accum['n_overlap'].append(n_overlap)
        self._accum['fp_rate'].append(fp_rate)
        self._accum['fn_rate'].append(fn_rate)
        self._accum['cpu_pct'].append(cpu_pct)
        self._accum['gpu_mb'].append(gpu_mb)

        for name, _, _ in self.DISTANCE_BANDS:
            v = band_metrics.get(f'mae_{name}', np.nan)
            if not np.isnan(v):
                self._accum[f'mae_{name}'].append(v)
            v = band_metrics.get(f'rmse_{name}', np.nan)
            if not np.isnan(v):
                self._accum[f'rmse_{name}'].append(v)
            # D2L per-band
            v = d2l_band_metrics.get(f'mae_d2l_{name}', np.nan)
            if not np.isnan(v):
                self._accum[f'mae_d2l_{name}'].append(v)
            v = d2l_band_metrics.get(f'rmse_d2l_{name}', np.nan)
            if not np.isnan(v):
                self._accum[f'rmse_d2l_{name}'].append(v)

        # --- Log every frame ---
        self._frame_count += 1
        if n_overlap > 10:
            self.get_logger().info(
                f'Frame {self._frame_count}: MAE={mae:.3f}m, RMSE={rmse_val:.3f}m, '
                f'corr={corr:.3f}, overlap={n_overlap}/{n_valid_s} bins'
                + (f', corrected MAE={mae_d2l:.3f}m' if not np.isnan(mae_d2l) else ''))
        else:
            self.get_logger().info(
                f'Frame {self._frame_count}: overlap={n_overlap} bins (too few)')

        if self._frame_count % self._summary_interval == 0:
            self._print_summary()

    def _publish_diff_scan(self, stereo, mono, valid_s, valid_m,
                           template: LaserScan):
        """Publish |stereo - mono| as a LaserScan for RViz visualization."""
        overlap = valid_s & valid_m
        diff_ranges = np.full(len(stereo), float('inf'), dtype=np.float32)

        if np.any(overlap):
            diff_ranges[overlap] = np.abs(stereo[overlap] - mono[overlap])

        msg = LaserScan()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = template.header.frame_id
        msg.angle_min = template.angle_min
        msg.angle_max = template.angle_max
        msg.angle_increment = template.angle_increment
        msg.time_increment = template.time_increment
        msg.scan_time = template.scan_time
        msg.range_min = 0.0
        msg.range_max = 5.0
        msg.ranges = _array.array('f', diff_ranges.tobytes())
        msg.intensities = []
        self._diff_pub.publish(msg)

    def _print_summary(self):
        """Print aggregate metrics."""
        self.get_logger().info(
            f'\n{"=" * 60}\n'
            f'  BENCHMARK SUMMARY — {self._frame_count} frames\n'
            f'{"=" * 60}')

        def _fmt(key):
            vals = self._accum.get(key, [])
            if not vals:
                return 'n/a'
            return f'{np.mean(vals):.4f} +/- {np.std(vals):.4f}'

        self.get_logger().info(
            f'  === DA3 RAW vs STEREO ===')
        self.get_logger().info(
            f'  MAE:         {_fmt("mae")}')
        self.get_logger().info(
            f'  RMSE:        {_fmt("rmse")}')
        self.get_logger().info(
            f'  Max Error:   {_fmt("max_err")}')
        self.get_logger().info(
            f'  Correlation: {_fmt("corr")}')
        self.get_logger().info(
            f'  FP Rate:     {_fmt("fp_rate")}')
        self.get_logger().info(
            f'  FN Rate:     {_fmt("fn_rate")}')

        self.get_logger().info(
            f'  === DA3 CORRECTED vs STEREO ===')
        self.get_logger().info(
            f'  MAE:         {_fmt("mae_d2l")}')
        self.get_logger().info(
            f'  RMSE:        {_fmt("rmse_d2l")}')
        self.get_logger().info(
            f'  Correlation: {_fmt("corr_d2l")}')

        self.get_logger().info(
            f'  === RAW per-band (stereo vs mono) ===')
        for name, _, _ in self.DISTANCE_BANDS:
            mae_vals = self._accum.get(f'mae_{name}', [])
            rmse_vals = self._accum.get(f'rmse_{name}', [])
            if mae_vals:
                self.get_logger().info(
                    f'  [{name}] MAE={np.mean(mae_vals):.4f}, '
                    f'RMSE={np.mean(rmse_vals):.4f}, '
                    f'n={len(mae_vals)}')

        self.get_logger().info(
            f'  === CORRECTED per-band (stereo vs d2l) ===')
        for name, _, _ in self.DISTANCE_BANDS:
            mae_vals = self._accum.get(f'mae_d2l_{name}', [])
            rmse_vals = self._accum.get(f'rmse_d2l_{name}', [])
            if mae_vals:
                self.get_logger().info(
                    f'  [{name}] MAE={np.mean(mae_vals):.4f}, '
                    f'RMSE={np.mean(rmse_vals):.4f}, '
                    f'n={len(mae_vals)}')

        overlap_vals = self._accum['n_overlap']
        stereo_vals = self._accum['n_valid_stereo']
        mono_vals = self._accum['n_valid_mono']
        self.get_logger().info(
            f'  Avg bins: stereo={np.mean(stereo_vals):.0f}, '
            f'mono={np.mean(mono_vals):.0f}, '
            f'overlap={np.mean(overlap_vals):.0f}')

        cpu_vals = self._accum['cpu_pct']
        gpu_vals = self._accum['gpu_mb']
        if cpu_vals:
            self.get_logger().info(
                f'  CPU: {np.mean(cpu_vals):.1f}%, '
                f'GPU mem: {np.mean(gpu_vals):.1f} MB')

        self.get_logger().info(f'{"=" * 60}\n')

    def destroy_node(self):
        """Print final summary and close CSV."""
        if self._frame_count > 0:
            self.get_logger().info('===== FINAL BENCHMARK RESULTS =====')
            self._print_summary()
        self._csv_file.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = DepthScanBenchmarkNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
