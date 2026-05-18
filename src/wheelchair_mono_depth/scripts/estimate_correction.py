#!/usr/bin/env python3
"""Live DA3 depth correction function estimator.

Subscribes to LiDAR (/scan_filtered) and DA3 fused (/scan_mono_fused),
collects paired (lidar_range, da3_range) per angular bin, then fits
candidate correction functions and reports the best one.

Usage (while 3cam launch is running):
    python3 src/wheelchair_mono_depth/scripts/estimate_correction.py

Collects data for --seconds (default 30), then fits and plots.
"""

import argparse
import time
import sys

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import LaserScan


class CorrectionEstimator(Node):
    def __init__(self, collect_seconds=30):
        super().__init__('correction_estimator')

        self._collect_seconds = collect_seconds
        self._start_time = None

        # Storage for paired measurements
        self._lidar_ranges = []   # ground truth
        self._da3_ranges = []     # DA3 fused

        # Latest scans
        self._latest_lidar = None
        self._latest_da3 = None

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.declare_parameter('truth_topic', '/scan_stereo_fused')
        self.declare_parameter('da3_topic', '/scan_mono_fused')
        truth_topic = self.get_parameter('truth_topic').value
        da3_topic = self.get_parameter('da3_topic').value
        self.get_logger().info(
            f'Comparing: {truth_topic} (truth) vs {da3_topic} (DA3)')

        self.create_subscription(
            LaserScan, truth_topic, self._lidar_cb, sensor_qos)
        self.create_subscription(
            LaserScan, da3_topic, self._da3_cb, sensor_qos)

        # Pair them at 10Hz
        self.create_timer(0.1, self._pair_callback)

        self.get_logger().info(
            f'Collecting paired data for {collect_seconds}s...')

    def _lidar_cb(self, msg):
        self._latest_lidar = msg

    def _da3_cb(self, msg):
        self._latest_da3 = msg

    def _pair_callback(self):
        if self._latest_lidar is None or self._latest_da3 is None:
            return

        if self._start_time is None:
            self._start_time = time.monotonic()
            self.get_logger().info('First pair received, collecting...')

        elapsed = time.monotonic() - self._start_time
        if elapsed > self._collect_seconds:
            self.get_logger().info(
                f'Collection done: {len(self._lidar_ranges)} pairs')
            self._fit_and_report()
            raise SystemExit(0)

        lidar = np.array(self._latest_lidar.ranges, dtype=np.float32)
        da3 = np.array(self._latest_da3.ranges, dtype=np.float32)

        if len(lidar) != len(da3):
            return

        # Find bins where both have valid (finite, > 0.3m) data
        valid = (
            np.isfinite(lidar) & np.isfinite(da3) &
            (lidar > 0.3) & (lidar < 8.0) &
            (da3 > 0.3) & (da3 < 10.0)
        )

        if valid.sum() > 0:
            self._lidar_ranges.extend(lidar[valid].tolist())
            self._da3_ranges.extend(da3[valid].tolist())

        if int(elapsed) % 5 == 0 and int(elapsed * 10) % 50 == 0:
            self.get_logger().info(
                f'  {elapsed:.0f}s: {len(self._lidar_ranges)} pairs')

    def _fit_and_report(self):
        lidar = np.array(self._lidar_ranges)
        da3 = np.array(self._da3_ranges)
        n = len(lidar)

        print(f'\n{"="*60}')
        print(f'  CORRECTION FUNCTION ESTIMATION — {n:,} range pairs')
        print(f'{"="*60}\n')

        if n < 100:
            print('ERROR: Not enough data pairs. Run longer or check topics.')
            return

        # ============================================================
        # Per-band analysis (signed bias)
        # ============================================================
        bands = [(0.3, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 5.0), (5.0, 8.0)]
        print('Per-band analysis (positive = DA3 undershoots):')
        print(f'  {"Band":>10s}  {"N":>7s}  {"Bias":>8s}  {"MAE":>8s}  '
              f'{"Scale":>8s}  {"StdDev":>8s}')

        band_centers = []
        band_scales = []
        band_ns = []

        for lo, hi in bands:
            mask = (lidar >= lo) & (lidar < hi)
            if mask.sum() < 10:
                print(f'  [{lo:.1f}-{hi:.1f}m]  {"<10":>7s}  —')
                continue

            l_band = lidar[mask]
            d_band = da3[mask]
            bias = np.mean(l_band - d_band)
            mae = np.mean(np.abs(l_band - d_band))
            scale = np.mean(l_band / d_band)
            std = np.std(l_band - d_band)

            print(f'  [{lo:.1f}-{hi:.1f}m]  {mask.sum():>7d}  '
                  f'{bias:>+8.3f}  {mae:>8.3f}  {scale:>8.4f}  {std:>8.3f}')

            band_centers.append((lo + hi) / 2)
            band_scales.append(scale)
            band_ns.append(mask.sum())

        # ============================================================
        # Fit candidate correction functions: lidar = f(da3)
        # ============================================================
        print(f'\n{"="*60}')
        print('  CANDIDATE CORRECTION FUNCTIONS')
        print(f'{"="*60}\n')

        results = []

        # 1. Global scale: lidar = a * da3
        a_global = np.sum(lidar * da3) / np.sum(da3 * da3)
        pred = a_global * da3
        mae_global = np.mean(np.abs(lidar - pred))
        rmse_global = np.sqrt(np.mean((lidar - pred) ** 2))
        results.append(('Global scale', mae_global, rmse_global,
                         f'd_corr = {a_global:.4f} * d_da3'))
        print(f'  1. Global scale: d_corr = {a_global:.4f} * d_da3')
        print(f'     MAE={mae_global:.4f}m  RMSE={rmse_global:.4f}m\n')

        # 2. Affine: lidar = a * da3 + b
        A = np.vstack([da3, np.ones(n)]).T
        coef_affine, _, _, _ = np.linalg.lstsq(A, lidar, rcond=None)
        a_aff, b_aff = coef_affine
        pred = a_aff * da3 + b_aff
        mae_aff = np.mean(np.abs(lidar - pred))
        rmse_aff = np.sqrt(np.mean((lidar - pred) ** 2))
        results.append(('Affine', mae_aff, rmse_aff,
                         f'd_corr = {a_aff:.4f} * d_da3 + {b_aff:.4f}'))
        print(f'  2. Affine: d_corr = {a_aff:.4f} * d_da3 + {b_aff:.4f}')
        print(f'     MAE={mae_aff:.4f}m  RMSE={rmse_aff:.4f}m\n')

        # 3. Quadratic: lidar = a*da3^2 + b*da3 + c
        A = np.vstack([da3**2, da3, np.ones(n)]).T
        coef_quad, _, _, _ = np.linalg.lstsq(A, lidar, rcond=None)
        a_q, b_q, c_q = coef_quad
        pred = a_q * da3**2 + b_q * da3 + c_q
        mae_quad = np.mean(np.abs(lidar - pred))
        rmse_quad = np.sqrt(np.mean((lidar - pred) ** 2))
        results.append(('Quadratic', mae_quad, rmse_quad,
                         f'd_corr = {a_q:.6f}*d^2 + {b_q:.4f}*d + {c_q:.4f}'))
        print(f'  3. Quadratic: d_corr = {a_q:.6f}*d^2 + {b_q:.4f}*d + '
              f'{c_q:.4f}')
        print(f'     MAE={mae_quad:.4f}m  RMSE={rmse_quad:.4f}m\n')

        # 4. Cubic: lidar = a*da3^3 + b*da3^2 + c*da3 + d
        A = np.vstack([da3**3, da3**2, da3, np.ones(n)]).T
        coef_cub, _, _, _ = np.linalg.lstsq(A, lidar, rcond=None)
        a_c, b_c, c_c, d_c = coef_cub
        pred = a_c * da3**3 + b_c * da3**2 + c_c * da3 + d_c
        mae_cub = np.mean(np.abs(lidar - pred))
        rmse_cub = np.sqrt(np.mean((lidar - pred) ** 2))
        results.append(('Cubic', mae_cub, rmse_cub,
                         f'd_corr = {a_c:.6f}*d^3 + {b_c:.6f}*d^2 + '
                         f'{c_c:.4f}*d + {d_c:.4f}'))
        print(f'  4. Cubic: d_corr = {a_c:.6f}*d^3 + {b_c:.6f}*d^2 + '
              f'{c_c:.4f}*d + {d_c:.4f}')
        print(f'     MAE={mae_cub:.4f}m  RMSE={rmse_cub:.4f}m\n')

        # 5. Power law: lidar = a * da3^b  (fit in log space)
        pos = (da3 > 0.1) & (lidar > 0.1)
        log_da3 = np.log(da3[pos])
        log_lidar = np.log(lidar[pos])
        A = np.vstack([log_da3, np.ones(pos.sum())]).T
        coef_pow, _, _, _ = np.linalg.lstsq(A, log_lidar, rcond=None)
        b_pow = coef_pow[0]
        a_pow = np.exp(coef_pow[1])
        pred_all = a_pow * da3 ** b_pow
        mae_pow = np.mean(np.abs(lidar - pred_all))
        rmse_pow = np.sqrt(np.mean((lidar - pred_all) ** 2))
        results.append(('Power law', mae_pow, rmse_pow,
                         f'd_corr = {a_pow:.4f} * d_da3^{b_pow:.4f}'))
        print(f'  5. Power law: d_corr = {a_pow:.4f} * d_da3^{b_pow:.4f}')
        print(f'     MAE={mae_pow:.4f}m  RMSE={rmse_pow:.4f}m\n')

        # ============================================================
        # WINNER
        # ============================================================
        results.sort(key=lambda x: x[1])  # sort by MAE
        print(f'{"="*60}')
        print(f'  RANKING (by MAE)')
        print(f'{"="*60}')
        for i, (name, mae, rmse, formula) in enumerate(results):
            marker = ' <<<< BEST' if i == 0 else ''
            print(f'  {i+1}. {name:12s}  MAE={mae:.4f}  RMSE={rmse:.4f}'
                  f'{marker}')
            print(f'     {formula}')

        best_name, best_mae, best_rmse, best_formula = results[0]
        print(f'\n{"="*60}')
        print(f'  BEST: {best_name}')
        print(f'  {best_formula}')
        print(f'  MAE={best_mae:.4f}m  RMSE={best_rmse:.4f}m')
        print(f'{"="*60}')

        # ============================================================
        # Per-band residuals for best model
        # ============================================================
        print(f'\nPer-band residuals after {best_name} correction:')
        # Re-compute best prediction
        if best_name == 'Global scale':
            pred_best = a_global * da3
        elif best_name == 'Affine':
            pred_best = a_aff * da3 + b_aff
        elif best_name == 'Quadratic':
            pred_best = a_q * da3**2 + b_q * da3 + c_q
        elif best_name == 'Cubic':
            pred_best = a_c * da3**3 + b_c * da3**2 + c_c * da3 + d_c
        elif best_name == 'Power law':
            pred_best = a_pow * da3 ** b_pow

        print(f'  {"Band":>10s}  {"N":>7s}  {"MAE_raw":>8s}  '
              f'{"MAE_corr":>8s}  {"Improve":>8s}')
        for lo, hi in bands:
            mask = (lidar >= lo) & (lidar < hi)
            if mask.sum() < 10:
                continue
            mae_raw = np.mean(np.abs(lidar[mask] - da3[mask]))
            mae_corr = np.mean(np.abs(lidar[mask] - pred_best[mask]))
            improve = (1 - mae_corr / mae_raw) * 100 if mae_raw > 0 else 0
            print(f'  [{lo:.1f}-{hi:.1f}m]  {mask.sum():>7d}  '
                  f'{mae_raw:>8.3f}  {mae_corr:>8.3f}  '
                  f'{improve:>+7.1f}%')

        # Save raw data for further analysis
        outfile = '/home/sidd/wheelchair_nav/eval_output/da3_correction_data.npz'
        np.savez(outfile, lidar=lidar, da3=da3)
        print(f'\nRaw data saved to: {outfile}')
        print(f'Total pairs: {n:,}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seconds', type=int, default=30,
                        help='Collection time in seconds')
    args = parser.parse_args()

    rclpy.init()
    node = CorrectionEstimator(collect_seconds=args.seconds)
    try:
        rclpy.spin(node)
    except SystemExit:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
