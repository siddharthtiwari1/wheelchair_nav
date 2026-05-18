#!/usr/bin/env python3
"""
Comprehensive Evaluation Script for KinoFlow.

Evaluates a trained KinoFlow model on held-out data with 10 metrics:
    1. Trajectory MSE       - Overall prediction accuracy
    2. First-step accuracy  - What actually gets executed
    3. ADE (Avg Disp Error) - Average position error along trajectory
    4. FDE (Final Disp Error) - Endpoint position error
    5. Collision rate        - % of trajectories with occupied cells
    6. Jerk (smoothness)     - RMS jerk over trajectory
    7. Comfort (ISO 2631)    - RMS acceleration vs comfort thresholds
    8. Goal progress         - How much closer to goal after trajectory
    9. Diversity (K-sample)  - Spread of multi-sample trajectories
   10. Inference latency     - Wall-clock time per inference

Usage:
    python evaluate_kinoflow.py \
        --model_path checkpoints_kinoflow/best_kinoflow.pth \
        --data_dir /path/to/test_data \
        --output_dir eval_results
"""

import os
import sys
import argparse
import json
import time
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from wheelchair_e2e.models.kinoflow_net import KinoFlowNet
from wheelchair_e2e.training.dataset import CFMTrajectoryDataset


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate KinoFlow model')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='eval_results')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--n_samples', type=int, default=8,
                        help='Number of trajectory samples for diversity')
    parser.add_argument('--n_latency_trials', type=int, default=100,
                        help='Number of forward passes for latency benchmark')
    parser.add_argument('--device', type=str, default='cuda')
    return parser.parse_args()


def evaluate(model, dataloader, device, n_samples=8):
    """
    Run full evaluation on test set.

    Returns:
        metrics: dict of aggregated metrics
        per_sample: dict of per-sample metrics (for analysis)
    """
    model.eval()

    # Accumulators
    all_traj_mse = []
    all_first_v_mse = []
    all_first_w_mse = []
    all_ade = []
    all_fde = []
    all_collision_rate = []
    all_jerk = []
    all_rms_accel = []
    all_rms_centripetal = []
    all_rms_angular_accel = []
    all_goal_progress = []
    all_diversity = []
    all_nh_violation = []

    with torch.no_grad():
        for batch in dataloader:
            bev = batch['bev'].to(device)
            odom = batch['odom'].to(device)
            vel_traj_gt = batch['velocity_traj'].to(device)

            B, H, _ = vel_traj_gt.shape

            # Generate trajectory (single sample for accuracy metrics)
            gen_vel, gen_poses, _ = model.generate(bev, odom)

            # Ground truth poses
            gt_poses = model.integrate_trajectory(vel_traj_gt)

            # --- 1. Trajectory MSE ---
            traj_mse = ((gen_vel - vel_traj_gt) ** 2).mean(dim=(1, 2))
            all_traj_mse.extend(traj_mse.cpu().numpy().tolist())

            # --- 2. First-step accuracy ---
            fv_mse = ((gen_vel[:, 0, 0] - vel_traj_gt[:, 0, 0]) ** 2)
            fw_mse = ((gen_vel[:, 0, 1] - vel_traj_gt[:, 0, 1]) ** 2)
            all_first_v_mse.extend(fv_mse.cpu().numpy().tolist())
            all_first_w_mse.extend(fw_mse.cpu().numpy().tolist())

            # --- 3. ADE ---
            disp = torch.sqrt(
                (gen_poses[:, :, 0] - gt_poses[:, :, 0]) ** 2
                + (gen_poses[:, :, 1] - gt_poses[:, :, 1]) ** 2)
            ade = disp.mean(dim=1)
            all_ade.extend(ade.cpu().numpy().tolist())

            # --- 4. FDE ---
            fde = disp[:, -1]
            all_fde.extend(fde.cpu().numpy().tolist())

            # --- 5. Collision rate ---
            occupancy = torch.max(bev[:, 0], bev[:, 1])
            grid_size = bev.shape[-1]
            center = grid_size // 2
            resolution = 0.05
            safety_px = 6  # 0.3m / 0.05

            for b in range(B):
                collision = False
                for t in range(H):
                    px = int(gen_poses[b, t, 0].item() / resolution + center)
                    py = int(gen_poses[b, t, 1].item() / resolution + center)
                    px = max(0, min(grid_size - 1, px))
                    py = max(0, min(grid_size - 1, py))
                    x_lo = max(0, px - safety_px)
                    x_hi = min(grid_size, px + safety_px + 1)
                    y_lo = max(0, py - safety_px)
                    y_hi = min(grid_size, py + safety_px + 1)
                    patch = occupancy[b, y_lo:y_hi, x_lo:x_hi]
                    if patch.numel() > 0 and patch.max().item() > 0.5:
                        collision = True
                        break
                all_collision_rate.append(1.0 if collision else 0.0)

            # --- 6. Jerk ---
            dt = model.dt
            if H >= 3:
                accel = (gen_vel[:, 1:, :] - gen_vel[:, :-1, :]) / dt
                jerk = (accel[:, 1:, :] - accel[:, :-1, :]) / dt
                rms_jerk = torch.sqrt((jerk ** 2).mean(dim=(1, 2)))
                all_jerk.extend(rms_jerk.cpu().numpy().tolist())

            # --- 7. Comfort (ISO 2631-1) ---
            if H >= 2:
                v = gen_vel[:, :, 0]
                omega = gen_vel[:, :, 1]

                # Fore-aft acceleration
                a_x = (v[:, 1:] - v[:, :-1]) / dt
                rms_ax = torch.sqrt((a_x ** 2).mean(dim=1))
                all_rms_accel.extend(rms_ax.cpu().numpy().tolist())

                # Centripetal: v * omega
                a_c = v[:, :-1] * omega[:, :-1]
                rms_ac = torch.sqrt((a_c ** 2).mean(dim=1))
                all_rms_centripetal.extend(rms_ac.cpu().numpy().tolist())

                # Angular acceleration
                alpha = (omega[:, 1:] - omega[:, :-1]) / dt
                rms_alpha = torch.sqrt((alpha ** 2).mean(dim=1))
                all_rms_angular_accel.extend(rms_alpha.cpu().numpy().tolist())

            # --- 8. Goal progress ---
            # Approximate: endpoint should be closer to where GT goes
            gt_endpoint = gt_poses[:, -1, :2]  # (B, 2)
            gen_endpoint = gen_poses[:, -1, :2]
            # Progress = distance moved in direction of GT endpoint
            gt_dist = torch.norm(gt_endpoint, dim=1)
            gen_towards = (gen_endpoint * gt_endpoint).sum(dim=1) / (
                gt_dist + 1e-6)
            progress = gen_towards / (gt_dist + 1e-6)
            all_goal_progress.extend(progress.cpu().numpy().tolist())

            # --- 9. Diversity (multi-sample) ---
            if n_samples > 1:
                # Generate K samples for first item in batch
                single_bev = bev[:1]
                single_odom = odom[:1]

                K = n_samples
                cond, _ = model.encode(single_bev, single_odom)
                cond_k = cond.expand(K, -1)
                z0 = torch.randn(K, model.traj_dim, device=device)

                from wheelchair_e2e.models.kinoflow_net import _euler_sample_from
                raw = _euler_sample_from(
                    model.vector_field, cond_k, z0,
                    n_steps=model.n_euler_steps, device=device)
                raw = raw.view(K, model.horizon, 2)
                vel_k = model.scale_trajectory(raw)
                poses_k = model.integrate_trajectory(vel_k)

                # Diversity = mean pairwise distance of endpoints
                endpoints = poses_k[:, -1, :2]  # (K, 2)
                dists = torch.cdist(endpoints.unsqueeze(0),
                                    endpoints.unsqueeze(0)).squeeze(0)
                # Mean of upper triangle
                mask = torch.triu(torch.ones(K, K, device=device), diagonal=1)
                mean_diversity = (dists * mask).sum() / mask.sum()
                all_diversity.append(mean_diversity.item())

            # --- 10. Non-holonomic violation ---
            if H >= 2:
                x_dot = (gen_poses[:, 1:, 0] - gen_poses[:, :-1, 0]) / dt
                y_dot = (gen_poses[:, 1:, 1] - gen_poses[:, :-1, 1]) / dt
                theta = gen_poses[:, :-1, 2]
                v_lat = -x_dot * torch.sin(theta) + y_dot * torch.cos(theta)
                nh_viol = torch.sqrt((v_lat ** 2).mean(dim=1))
                all_nh_violation.extend(nh_viol.cpu().numpy().tolist())

    # Aggregate
    def safe_mean(lst):
        return float(np.mean(lst)) if lst else 0.0

    def safe_std(lst):
        return float(np.std(lst)) if lst else 0.0

    # ISO 2631-1 comfort classification
    rms_a_mean = safe_mean(all_rms_accel)
    if rms_a_mean < 0.315:
        comfort_class = "Not uncomfortable"
    elif rms_a_mean < 0.63:
        comfort_class = "A little uncomfortable"
    elif rms_a_mean < 1.0:
        comfort_class = "Fairly uncomfortable"
    elif rms_a_mean < 1.6:
        comfort_class = "Uncomfortable"
    else:
        comfort_class = "Very uncomfortable (HAZARDOUS)"

    metrics = {
        'trajectory_mse': {
            'mean': safe_mean(all_traj_mse),
            'std': safe_std(all_traj_mse),
        },
        'first_step_accuracy': {
            'v_mse': safe_mean(all_first_v_mse),
            'w_mse': safe_mean(all_first_w_mse),
            'v_rmse': float(np.sqrt(safe_mean(all_first_v_mse))),
            'w_rmse': float(np.sqrt(safe_mean(all_first_w_mse))),
        },
        'ade': {
            'mean': safe_mean(all_ade),
            'std': safe_std(all_ade),
        },
        'fde': {
            'mean': safe_mean(all_fde),
            'std': safe_std(all_fde),
        },
        'collision_rate': safe_mean(all_collision_rate),
        'jerk': {
            'rms_mean': safe_mean(all_jerk),
            'rms_std': safe_std(all_jerk),
        },
        'comfort_iso2631': {
            'rms_fore_aft_accel': rms_a_mean,
            'rms_centripetal_accel': safe_mean(all_rms_centripetal),
            'rms_angular_accel': safe_mean(all_rms_angular_accel),
            'comfort_class': comfort_class,
        },
        'goal_progress': {
            'mean': safe_mean(all_goal_progress),
            'std': safe_std(all_goal_progress),
        },
        'diversity': {
            'mean_pairwise_endpoint_dist': safe_mean(all_diversity),
        },
        'nonholonomic_violation': {
            'rms_lateral_vel': safe_mean(all_nh_violation),
        },
        'n_samples_evaluated': len(all_traj_mse),
    }

    return metrics


def benchmark_latency(model, device, horizon=10, n_trials=100):
    """Benchmark inference latency."""
    bev = torch.randn(1, 5, 200, 200, device=device)
    odom = torch.randn(1, 30, device=device)

    # Warmup
    for _ in range(10):
        with torch.no_grad():
            model.generate(bev, odom)

    # Single sample latency
    if device.type == 'cuda':
        torch.cuda.synchronize()

    times_single = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        with torch.no_grad():
            model.generate(bev, odom)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        times_single.append((time.perf_counter() - t0) * 1000)

    # Multi-sample latency (K=8)
    times_multi = []
    bev_occ = torch.randn(1, 200, 200, device=device)
    for _ in range(n_trials):
        t0 = time.perf_counter()
        with torch.no_grad():
            model.generate_multi_sample(
                bev, odom, bev_occupancy=bev_occ,
                goal_dx=2.0, goal_dy=0.0)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        times_multi.append((time.perf_counter() - t0) * 1000)

    return {
        'single_sample': {
            'mean_ms': float(np.mean(times_single)),
            'std_ms': float(np.std(times_single)),
            'p50_ms': float(np.percentile(times_single, 50)),
            'p95_ms': float(np.percentile(times_single, 95)),
            'p99_ms': float(np.percentile(times_single, 99)),
            'max_hz': 1000.0 / float(np.mean(times_single)),
        },
        'multi_sample_K8': {
            'mean_ms': float(np.mean(times_multi)),
            'std_ms': float(np.std(times_multi)),
            'p50_ms': float(np.percentile(times_multi, 50)),
            'p95_ms': float(np.percentile(times_multi, 95)),
            'p99_ms': float(np.percentile(times_multi, 99)),
            'max_hz': 1000.0 / float(np.mean(times_multi)),
        },
    }


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Evaluating on: {device}")

    # Load model
    ckpt = torch.load(args.model_path, map_location=device, weights_only=False)
    model_args = ckpt.get('args', {})

    model = KinoFlowNet(
        bev_channels=5,
        v_max=model_args.get('v_max', 0.25),
        w_max=model_args.get('w_max', 1.0),
        horizon=model_args.get('horizon', 10),
        cfm_hidden=model_args.get('cfm_hidden', 512),
        cfm_layers=model_args.get('cfm_layers', 4),
        n_euler_steps=model_args.get('n_euler_steps', 3),
        dt=model_args.get('dt', 0.1),
        n_samples=args.n_samples,
    ).to(device)

    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    epoch = ckpt.get('epoch', '?')
    phase = ckpt.get('phase', '?')
    print(f"Loaded model: epoch={epoch}, phase={phase}")
    print(f"Parameters: {model.get_param_count()}")

    # Load test dataset
    test_dataset = CFMTrajectoryDataset(
        args.data_dir, horizon=model_args.get('horizon', 10))
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=2, pin_memory=True)
    print(f"Test set: {len(test_dataset)} samples")

    # Evaluate
    print("\nRunning evaluation...")
    metrics = evaluate(model, test_loader, device, n_samples=args.n_samples)

    # Latency benchmark
    print("Running latency benchmark...")
    latency = benchmark_latency(
        model, device,
        horizon=model_args.get('horizon', 10),
        n_trials=args.n_latency_trials)
    metrics['latency'] = latency

    # Print results
    print("\n" + "=" * 60)
    print("KINOFLOW EVALUATION RESULTS")
    print("=" * 60)
    print(f"Samples evaluated: {metrics['n_samples_evaluated']}")
    print(f"\n1. Trajectory MSE:     {metrics['trajectory_mse']['mean']:.6f} "
          f"(+/- {metrics['trajectory_mse']['std']:.6f})")
    print(f"2. First-step RMSE:    "
          f"v={metrics['first_step_accuracy']['v_rmse']:.4f} m/s, "
          f"w={metrics['first_step_accuracy']['w_rmse']:.4f} rad/s")
    print(f"3. ADE:                {metrics['ade']['mean']:.4f} m "
          f"(+/- {metrics['ade']['std']:.4f})")
    print(f"4. FDE:                {metrics['fde']['mean']:.4f} m "
          f"(+/- {metrics['fde']['std']:.4f})")
    print(f"5. Collision rate:     {metrics['collision_rate']*100:.1f}%")
    print(f"6. RMS Jerk:           {metrics['jerk']['rms_mean']:.4f} m/s^3")
    print(f"7. Comfort (ISO 2631): {metrics['comfort_iso2631']['comfort_class']}")
    print(f"   - Fore-aft accel:   "
          f"{metrics['comfort_iso2631']['rms_fore_aft_accel']:.3f} m/s^2")
    print(f"   - Centripetal:      "
          f"{metrics['comfort_iso2631']['rms_centripetal_accel']:.3f} m/s^2")
    print(f"   - Angular accel:    "
          f"{metrics['comfort_iso2631']['rms_angular_accel']:.3f} rad/s^2")
    print(f"8. Goal progress:      {metrics['goal_progress']['mean']:.3f}")
    print(f"9. Diversity (K={args.n_samples}):   "
          f"{metrics['diversity']['mean_pairwise_endpoint_dist']:.4f} m")
    print(f"10. NH violation:      "
          f"{metrics['nonholonomic_violation']['rms_lateral_vel']:.6f} m/s")
    print(f"\nLatency (single):      "
          f"{latency['single_sample']['mean_ms']:.1f} ms "
          f"({latency['single_sample']['max_hz']:.0f} Hz)")
    print(f"Latency (K={args.n_samples}):         "
          f"{latency['multi_sample_K8']['mean_ms']:.1f} ms "
          f"({latency['multi_sample_K8']['max_hz']:.0f} Hz)")
    print("=" * 60)

    # Save results
    results = {
        'model_path': args.model_path,
        'epoch': epoch,
        'phase': phase,
        'model_args': model_args,
        'metrics': metrics,
    }

    output_path = os.path.join(args.output_dir, 'eval_results.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()
