#!/usr/bin/env python3
"""
End-to-end smoke test for the KinoFlow training pipeline.

Verifies the full pipeline works without errors:
    1. Load synthetic data via CFMTrajectoryDataset
    2. Train KinoFlowNet for 2 epochs (Phase 1 only)
    3. Generate trajectories from trained model
    4. Train DualSpaceScoringTransformer for 2 epochs
    5. Run combined inference (generate_multi_sample)

NO ROS dependency. Runs on CPU or GPU.

Usage:
    # First generate synthetic data:
    python -m wheelchair_e2e.scripts.generate_synthetic_data \
        --output_dir /tmp/kinoflow_test_data --n_episodes 50

    # Then run smoke test:
    python -m wheelchair_e2e.scripts.smoke_test_pipeline \
        --data_dir /tmp/kinoflow_test_data

    # Quick test (fewer samples, CPU only):
    python -m wheelchair_e2e.scripts.smoke_test_pipeline \
        --data_dir /tmp/kinoflow_test_data --device cpu --batch_size 8
"""

import os
import sys
import argparse
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from wheelchair_e2e.models.kinoflow_net import KinoFlowNet, _euler_sample_from
from wheelchair_e2e.models.flow_matching import euler_sample
from wheelchair_e2e.models.scoring_network import DualSpaceScoringTransformer
from wheelchair_e2e.training.kinoflow_losses import KinoFlowLoss
from wheelchair_e2e.training.dataset import CFMTrajectoryDataset


def parse_args():
    parser = argparse.ArgumentParser(
        description='Smoke test KinoFlow pipeline end-to-end')
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--device', type=str, default='auto',
                        help='cpu, cuda, or auto')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--horizon', type=int, default=10)
    parser.add_argument('--n_samples', type=int, default=8,
                        help='K candidates for scorer')
    parser.add_argument('--max_train_samples', type=int, default=500,
                        help='Limit training samples for speed')
    return parser.parse_args()


def step_header(step_num, title):
    print(f"\n{'='*60}")
    print(f"  Step {step_num}: {title}")
    print(f"{'='*60}")


def main():
    args = parse_args()

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    passed = 0
    failed = 0

    # ================================================================
    # Step 1: Load Dataset
    # ================================================================
    step_header(1, "Load CFMTrajectoryDataset")
    t0 = time.time()

    try:
        dataset = CFMTrajectoryDataset(
            data_dir=args.data_dir, horizon=args.horizon)
        print(f"  Dataset size: {len(dataset)} valid trajectory samples")

        # Limit for speed
        n_use = min(len(dataset), args.max_train_samples)
        subset = Subset(dataset, list(range(n_use)))
        loader = DataLoader(subset, batch_size=args.batch_size,
                            shuffle=True, num_workers=0)

        # Verify a batch
        batch = next(iter(loader))
        assert batch['bev'].shape[1:] == (5, 200, 200), \
            f"BEV shape wrong: {batch['bev'].shape}"
        assert batch['odom'].shape[1:] == (30,), \
            f"Odom shape wrong: {batch['odom'].shape}"
        assert batch['velocity_traj'].shape[1:] == (args.horizon, 2), \
            f"Vel traj shape wrong: {batch['velocity_traj'].shape}"

        print(f"  Batch shapes: bev={batch['bev'].shape}, "
              f"odom={batch['odom'].shape}, "
              f"vel_traj={batch['velocity_traj'].shape}")
        print(f"  PASSED ({time.time()-t0:.1f}s)")
        passed += 1
    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback; traceback.print_exc()
        failed += 1
        return failed == 0

    # ================================================================
    # Step 2: Train KinoFlowNet (2 epochs, Phase 1)
    # ================================================================
    step_header(2, "Train KinoFlowNet (2 epochs, flow loss only)")
    t0 = time.time()

    try:
        model = KinoFlowNet(
            v_max=0.25, w_max=1.0, horizon=args.horizon,
            n_euler_steps=3, cfm_hidden=256, cfm_layers=2,
            n_samples=args.n_samples
        ).to(device)

        param_count = sum(p.numel() for p in model.parameters())
        print(f"  Model parameters: {param_count:,}")

        loss_fn = KinoFlowLoss(
            dt=0.1, v_max=0.25, w_max=1.0,
            lambda_flow=1.0, lambda_kinematic=0.0,
            lambda_nonholonomic=0.0, lambda_collision=0.0,
            lambda_jerk=0.0, lambda_comfort=0.0
        ).to(device)
        loss_fn.set_phase(1)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        for epoch in range(2):
            model.train()
            epoch_loss = 0.0
            n_batches = 0

            for batch in loader:
                bev = batch['bev'].to(device)
                odom = batch['odom'].to(device)
                vel_traj = batch['velocity_traj'].to(device)

                B, H, _ = vel_traj.shape
                cond, _ = model(bev, odom, hidden=None)

                x_1 = vel_traj.reshape(B, -1)
                x_0 = torch.randn_like(x_1)
                t = torch.rand(B, device=device)

                loss, loss_dict = loss_fn(
                    model.vector_field, x_0, x_1, t, cond)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / n_batches
            print(f"  Epoch {epoch+1}/2: loss={avg_loss:.4f}")

            assert not np.isnan(avg_loss), "NaN loss detected!"
            assert avg_loss < 100, f"Loss too high: {avg_loss}"

        # Save checkpoint
        ckpt_path = '/tmp/kinoflow_smoke_test.pt'
        torch.save(model.state_dict(), ckpt_path)
        print(f"  Saved checkpoint: {ckpt_path}")
        print(f"  PASSED ({time.time()-t0:.1f}s)")
        passed += 1
    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback; traceback.print_exc()
        failed += 1
        return failed == 0

    # ================================================================
    # Step 3: Generate trajectories from trained model
    # ================================================================
    step_header(3, "Generate trajectories (single + multi-sample)")
    t0 = time.time()

    try:
        model.eval()
        batch = next(iter(loader))
        bev = batch['bev'].to(device)
        odom = batch['odom'].to(device)

        with torch.no_grad():
            # Generate single trajectory (batched)
            gen_vel, gen_poses, _ = model.generate(bev, odom)
            print(f"  Single trajectory: vel={gen_vel.shape}, "
                  f"poses={gen_poses.shape}")

            # Generate K candidates for first sample (inference API)
            best_vel, best_poses, all_vel, all_poses, best_idx, scores, _ = \
                model.generate_multi_sample(bev[:1], odom[:1])
            K = all_vel.shape[0]
            print(f"  Multi-sample: K={K}, all_vel={all_vel.shape}, "
                  f"best_idx={best_idx}, scores range="
                  f"[{scores.min():.3f}, {scores.max():.3f}]")

            # Verify ranges
            v_gen = gen_vel[:, :, 0]
            w_gen = gen_vel[:, :, 1]
            print(f"  Generated v range: [{v_gen.min():.3f}, {v_gen.max():.3f}]")
            print(f"  Generated w range: [{w_gen.min():.3f}, {w_gen.max():.3f}]")

            assert not torch.isnan(gen_vel).any(), "NaN in generated velocities!"
            assert not torch.isnan(gen_poses).any(), "NaN in generated poses!"
            assert all_vel.shape == (K, args.horizon, 2)

        print(f"  PASSED ({time.time()-t0:.1f}s)")
        passed += 1
    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback; traceback.print_exc()
        failed += 1
        return failed == 0

    # ================================================================
    # Step 4: Train Scorer (2 epochs)
    # ================================================================
    step_header(4, "Train DualSpaceScoringTransformer (2 epochs)")
    t0 = time.time()

    try:
        scorer = DualSpaceScoringTransformer(
            embed_dim=64, n_heads=2, n_layers=2,
            backbone_dim=512, horizon=args.horizon,
        ).to(device)

        scorer_params = sum(p.numel() for p in scorer.parameters())
        print(f"  Scorer parameters: {scorer_params:,}")

        scorer_optimizer = torch.optim.AdamW(scorer.parameters(), lr=3e-4)
        model.eval()

        for epoch in range(2):
            scorer.train()
            epoch_loss = 0.0
            epoch_correct = 0
            n_samples_total = 0
            n_batches = 0

            for batch in loader:
                bev = batch['bev'].to(device)
                odom = batch['odom'].to(device)
                vel_traj_gt = batch['velocity_traj'].to(device)
                B = bev.shape[0]

                # Process each sample individually (scorer API is per-sample)
                batch_loss = torch.tensor(0.0, device=device)

                for i in range(B):
                    bev_i = bev[i:i+1]
                    odom_i = odom[i:i+1]
                    expert_i = vel_traj_gt[i]  # (H, 2)

                    # Generate K candidates
                    with torch.no_grad():
                        cond, _ = model.encode(bev_i, odom_i)
                        cond_k = cond.expand(args.n_samples, -1)
                        z0 = torch.randn(args.n_samples, model.traj_dim,
                                         device=device)
                        raw = _euler_sample_from(
                            model.vector_field, cond_k, z0,
                            n_steps=model.n_euler_steps, device=device)
                        raw = raw.view(args.n_samples, model.horizon, 2)
                        vel_trajs = model.scale_trajectory(raw)
                        poses_trajs = model.integrate_trajectory(vel_trajs)
                        scene_feat = model.encoder(bev_i).squeeze(0)

                    # Find expert-closest
                    with torch.no_grad():
                        expert_poses = model.integrate_trajectory(
                            expert_i.unsqueeze(0))  # (1, H, 3)
                        dists = []
                        for k in range(args.n_samples):
                            d = (poses_trajs[k, :, :2] -
                                 expert_poses[0, :, :2]).pow(2).sum(-1).sqrt().mean()
                            dists.append(d)
                        target_idx = torch.argmin(torch.stack(dists))

                    # BEV occupancy and goal
                    bev_occ = torch.max(bev_i[0, 0], bev_i[0, 1])
                    goal_ch = bev_i[0, 2]
                    if goal_ch.max() > 0.01:
                        gy, gx = torch.where(goal_ch == goal_ch.max())
                        goal_dx = (gx[0].item() - 100) * 0.05
                        goal_dy = (gy[0].item() - 100) * 0.05
                    else:
                        goal_dx, goal_dy = 1.0, 0.0

                    # Score
                    scores = scorer.score_with_context(
                        vel_trajs, poses_trajs, bev_occ,
                        scene_feat, goal_dx, goal_dy)

                    # CE loss
                    ce_loss = F.cross_entropy(
                        scores.unsqueeze(0), target_idx.unsqueeze(0))
                    batch_loss = batch_loss + ce_loss

                    if scores.argmax() == target_idx:
                        epoch_correct += 1
                    n_samples_total += 1

                batch_loss = batch_loss / B
                scorer_optimizer.zero_grad()
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(scorer.parameters(), 1.0)
                scorer_optimizer.step()

                epoch_loss += batch_loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            acc = epoch_correct / max(n_samples_total, 1)
            print(f"  Epoch {epoch+1}/2: CE loss={avg_loss:.4f}, "
                  f"acc={acc:.1%}")

            assert not np.isnan(avg_loss), "NaN loss in scorer!"

        scorer_ckpt = '/tmp/scorer_smoke_test.pt'
        torch.save(scorer.state_dict(), scorer_ckpt)
        print(f"  Saved scorer: {scorer_ckpt}")
        print(f"  PASSED ({time.time()-t0:.1f}s)")
        passed += 1
    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback; traceback.print_exc()
        failed += 1
        return failed == 0

    # ================================================================
    # Step 5: Combined inference (generate_multi_sample with scorer)
    # ================================================================
    step_header(5, "Combined inference: generate_multi_sample + learned scorer")
    t0 = time.time()

    try:
        model.eval()
        scorer.eval()

        # Attach learned scorer to model
        model.set_learned_scorer(scorer)

        batch = next(iter(loader))
        bev = batch['bev'].to(device)
        odom = batch['odom'].to(device)

        with torch.no_grad():
            # Full inference pipeline (single sample, like real-time)
            bev_occ = torch.max(bev[0, 0], bev[0, 1])  # (200, 200)
            best_vel, best_poses, all_vel, all_poses, best_idx, scores, _ = \
                model.generate_multi_sample(
                    bev[:1], odom[:1],
                    bev_occupancy=bev_occ.unsqueeze(0))

            cmd_v = best_vel[0, 0, 0].item()
            cmd_w = best_vel[0, 0, 1].item()

            print(f"  Selected trajectory: idx={best_idx}")
            print(f"  First command: v={cmd_v:.4f} m/s, w={cmd_w:.4f} rad/s")
            print(f"  All scores: {scores.cpu().numpy().round(3)}")
            ep_dist = torch.sqrt(
                best_poses[0, -1, 0]**2 + best_poses[0, -1, 1]**2)
            print(f"  Endpoint distance: {ep_dist:.4f} m")

            assert not torch.isnan(best_vel).any(), "NaN in output!"
            assert best_vel.shape == (1, args.horizon, 2)

        # Timing benchmark
        n_warmup = 3
        n_trials = 10
        for _ in range(n_warmup):
            with torch.no_grad():
                model.generate_multi_sample(bev[:1], odom[:1])

        times = []
        for _ in range(n_trials):
            t_start = time.time()
            with torch.no_grad():
                model.generate_multi_sample(bev[:1], odom[:1])
            times.append(time.time() - t_start)

        avg_ms = np.mean(times) * 1000
        print(f"  Inference latency: {avg_ms:.1f}ms ({1000/avg_ms:.0f} Hz)")

        print(f"  PASSED ({time.time()-t0:.1f}s)")
        passed += 1
    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback; traceback.print_exc()
        failed += 1

    # ================================================================
    # Summary
    # ================================================================
    print(f"\n{'='*60}")
    print(f"  SMOKE TEST RESULTS: {passed}/5 passed, {failed}/5 failed")
    print(f"{'='*60}")

    if failed == 0:
        print("\n  ALL CLEAR — KinoFlow pipeline is end-to-end functional!")
        print("  Ready for real data collection and training.")
    else:
        print(f"\n  {failed} step(s) failed. Check errors above.")

    return failed == 0


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
