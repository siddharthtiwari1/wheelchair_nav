"""
Training Script for DualSpaceScoringTransformer.

Trains AFTER the KinoFlow CFM model is trained (Phase 4).
Uses the frozen CFM to generate K candidates per training sample,
then trains the scorer to pick the one closest to the expert trajectory.

Training pipeline:
    1. Load trained KinoFlow model (frozen)
    2. For each (BEV, odom, expert_trajectory) sample:
       a. Generate K=8 candidate trajectories via frozen CFM
       b. Find which candidate is closest to expert (L2 distance)
       c. Train scorer: CrossEntropy(scores, best_idx)
    3. + auxiliary comfort loss: penalize selecting high-jerk trajectories

Usage:
    python -m wheelchair_e2e.training.train_scorer \
        --data_dir /path/to/training_data \
        --kinoflow_ckpt /path/to/kinoflow_phase3.pt \
        --output_dir /path/to/scorer_checkpoints \
        --epochs 30 --batch_size 64
"""

import os
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from wheelchair_e2e.models.kinoflow_net import KinoFlowNet
from wheelchair_e2e.models.scoring_network import DualSpaceScoringTransformer
from wheelchair_e2e.training.dataset import CFMTrajectoryDataset


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train DualSpaceScoringTransformer for KinoFlow')

    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--kinoflow_ckpt', type=str, required=True,
                        help='Path to trained KinoFlow checkpoint (phase 2 or 3)')
    parser.add_argument('--output_dir', type=str, default='checkpoints_scorer')
    parser.add_argument('--val_split', type=float, default=0.1)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--n_samples', type=int, default=8,
                        help='K candidates per sample')
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--n_layers', type=int, default=3)
    parser.add_argument('--comfort_weight', type=float, default=0.1,
                        help='Weight for auxiliary comfort loss')
    parser.add_argument('--wandb', action='store_true')

    return parser.parse_args()


def find_closest_to_expert(candidate_poses, expert_traj):
    """Find which candidate trajectory is closest to the expert.

    Args:
        candidate_poses: (K, H, 3) pose trajectories [x, y, θ]
        expert_traj: (H, 2) expert velocity trajectory [v, ω]
                     (we compare in position space after integration)

    Returns:
        best_idx: index of closest candidate
    """
    K = candidate_poses.shape[0]
    expert_xy = expert_traj[:, :2] if expert_traj.shape[1] >= 3 else expert_traj

    # Compare endpoint distance (FDE) + average distance (ADE)
    dists = []
    for k in range(K):
        cand_xy = candidate_poses[k, :, :2]
        H_min = min(cand_xy.shape[0], expert_xy.shape[0])
        ade = (cand_xy[:H_min] - expert_xy[:H_min]).pow(2).sum(dim=-1).sqrt().mean()
        fde = (cand_xy[-1] - expert_xy[-1]).pow(2).sum().sqrt()
        dists.append(0.5 * ade + 0.5 * fde)

    return torch.argmin(torch.stack(dists))


def train_epoch(kinoflow, scorer, dataloader, optimizer, device, args):
    """Train scorer for one epoch."""
    scorer.train()
    kinoflow.eval()

    total_loss = 0.0
    total_ce_loss = 0.0
    total_comfort_loss = 0.0
    total_correct = 0
    total_samples = 0
    n_batches = 0

    for batch in dataloader:
        bev = batch['bev'].to(device)
        odom = batch['odom'].to(device)
        expert_traj = batch['velocity_traj'].to(device)  # (B, H, 2)
        B = bev.shape[0]

        # Process each sample in the batch individually
        # (scorer handles K candidates at once, not batched across samples)
        batch_loss = torch.tensor(0.0, device=device)
        batch_correct = 0

        for i in range(B):
            bev_i = bev[i:i+1]         # (1, 5, 200, 200)
            odom_i = odom[i:i+1]       # (1, 30)
            expert_i = expert_traj[i]   # (H, 2)

            # Generate K candidates from frozen CFM
            with torch.no_grad():
                cond, _ = kinoflow.encode(bev_i, odom_i)
                cond_k = cond.expand(args.n_samples, -1)
                z0 = torch.randn(args.n_samples, kinoflow.traj_dim, device=device)

                from wheelchair_e2e.models.kinoflow_net import _euler_sample_from
                raw = _euler_sample_from(
                    kinoflow.vector_field, cond_k, z0,
                    n_steps=kinoflow.n_euler_steps, device=device
                )
                raw = raw.view(args.n_samples, kinoflow.horizon, 2)
                vel_trajs = kinoflow.scale_trajectory(raw)       # (K, H, 2)
                poses_trajs = kinoflow.integrate_trajectory(vel_trajs)  # (K, H, 3)

                # Scene features from backbone (reuse!)
                scene_features = kinoflow.encoder(bev_i).squeeze(0)  # (512,)

            # Find expert-closest candidate (label)
            with torch.no_grad():
                # Integrate expert velocity to get expert poses for comparison
                expert_vel = expert_i.unsqueeze(0)  # (1, H, 2)
                expert_poses = kinoflow.integrate_trajectory(
                    kinoflow.scale_trajectory(expert_vel)
                )  # (1, H, 3)
                target_idx = find_closest_to_expert(poses_trajs, expert_poses[0])

            # BEV occupancy for comfort features
            bev_occ = torch.max(bev_i[0, 0], bev_i[0, 1])  # (200, 200)

            # Get relative goal from BEV (approximate from goal channel)
            goal_ch = bev_i[0, 2]  # (200, 200) goal Gaussian
            if goal_ch.max() > 0.01:
                gy, gx = torch.where(goal_ch == goal_ch.max())
                goal_dx = (gx[0].item() - 100) * 0.05
                goal_dy = (gy[0].item() - 100) * 0.05
            else:
                goal_dx, goal_dy = 1.0, 0.0  # default forward

            # Score candidates
            scores = scorer.score_with_context(
                vel_trajs, poses_trajs, bev_occ,
                scene_features, goal_dx, goal_dy
            )  # (K,)

            # Cross-entropy loss: predict expert-closest
            ce_loss = F.cross_entropy(scores.unsqueeze(0), target_idx.unsqueeze(0))

            # Auxiliary comfort loss: penalize selecting high-jerk trajectory
            selected_idx = scores.argmax()
            selected_vel = vel_trajs[selected_idx]
            if selected_vel.shape[0] >= 3:
                accel = selected_vel[1:] - selected_vel[:-1]
                jerk = accel[1:] - accel[:-1]
                comfort_loss = jerk.pow(2).mean()
            else:
                comfort_loss = torch.tensor(0.0, device=device)

            loss = ce_loss + args.comfort_weight * comfort_loss
            batch_loss = batch_loss + loss

            # Track accuracy
            if selected_idx == target_idx:
                batch_correct += 1

        # Average loss over batch
        batch_loss = batch_loss / B
        optimizer.zero_grad()
        batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(scorer.parameters(), 1.0)
        optimizer.step()

        total_loss += batch_loss.item()
        total_correct += batch_correct
        total_samples += B
        n_batches += 1

    return {
        'loss': total_loss / max(n_batches, 1),
        'accuracy': total_correct / max(total_samples, 1),
    }


@torch.no_grad()
def validate(kinoflow, scorer, dataloader, device, args):
    """Validate scorer."""
    scorer.eval()
    kinoflow.eval()

    total_correct = 0
    total_samples = 0
    total_loss = 0.0
    n_batches = 0

    for batch in dataloader:
        bev = batch['bev'].to(device)
        odom = batch['odom'].to(device)
        expert_traj = batch['velocity_traj'].to(device)
        B = bev.shape[0]

        for i in range(B):
            bev_i = bev[i:i+1]
            odom_i = odom[i:i+1]
            expert_i = expert_traj[i]

            cond, _ = kinoflow.encode(bev_i, odom_i)
            cond_k = cond.expand(args.n_samples, -1)
            z0 = torch.randn(args.n_samples, kinoflow.traj_dim, device=device)

            from wheelchair_e2e.models.kinoflow_net import _euler_sample_from
            raw = _euler_sample_from(
                kinoflow.vector_field, cond_k, z0,
                n_steps=kinoflow.n_euler_steps, device=device
            )
            raw = raw.view(args.n_samples, kinoflow.horizon, 2)
            vel_trajs = kinoflow.scale_trajectory(raw)
            poses_trajs = kinoflow.integrate_trajectory(vel_trajs)
            scene_features = kinoflow.encoder(bev_i).squeeze(0)

            expert_vel = expert_i.unsqueeze(0)
            expert_poses = kinoflow.integrate_trajectory(
                kinoflow.scale_trajectory(expert_vel)
            )
            target_idx = find_closest_to_expert(poses_trajs, expert_poses[0])

            bev_occ = torch.max(bev_i[0, 0], bev_i[0, 1])

            goal_ch = bev_i[0, 2]
            if goal_ch.max() > 0.01:
                gy, gx = torch.where(goal_ch == goal_ch.max())
                goal_dx = (gx[0].item() - 100) * 0.05
                goal_dy = (gy[0].item() - 100) * 0.05
            else:
                goal_dx, goal_dy = 1.0, 0.0

            scores = scorer.score_with_context(
                vel_trajs, poses_trajs, bev_occ,
                scene_features, goal_dx, goal_dy
            )

            loss = F.cross_entropy(scores.unsqueeze(0), target_idx.unsqueeze(0))
            total_loss += loss.item()

            if scores.argmax() == target_idx:
                total_correct += 1
            total_samples += 1

        n_batches += 1

    return {
        'loss': total_loss / max(total_samples, 1),
        'accuracy': total_correct / max(total_samples, 1),
    }


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # --- Load frozen KinoFlow model ---
    ckpt = torch.load(args.kinoflow_ckpt, map_location=device, weights_only=False)

    # Infer model config from checkpoint
    horizon = ckpt.get('horizon', 10)
    v_max = ckpt.get('v_max', 0.25)
    w_max = ckpt.get('w_max', 1.0)

    kinoflow = KinoFlowNet(
        bev_channels=5, v_max=v_max, w_max=w_max,
        horizon=horizon, n_samples=args.n_samples,
    ).to(device)
    kinoflow.load_state_dict(ckpt['model_state_dict'])
    kinoflow.eval()
    for p in kinoflow.parameters():
        p.requires_grad = False
    print(f"Loaded frozen KinoFlow from {args.kinoflow_ckpt}")

    # --- Create scorer ---
    scorer = DualSpaceScoringTransformer(
        embed_dim=args.embed_dim,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        backbone_dim=512,
        horizon=horizon,
        dropout=0.1,
    ).to(device)

    pc = scorer.get_param_count()
    print(f"Scorer params: {pc['total']:,} "
          f"(traj_enc={pc['traj_encoder']:,}, "
          f"transformer={pc['transformer']:,}, "
          f"head={pc['score_head']:,})")

    # --- Data ---
    dataset = CFMTrajectoryDataset(args.data_dir, horizon=horizon)
    val_size = max(1, int(len(dataset) * args.val_split))
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=2, pin_memory=True)

    print(f"Data: {train_size} train, {val_size} val")

    # --- Optimizer ---
    optimizer = AdamW(scorer.parameters(), lr=args.lr,
                      weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # --- WandB ---
    if args.wandb:
        import wandb
        wandb.init(project='kinoflow-scorer', config=vars(args))

    # --- Training loop ---
    best_val_acc = 0.0

    for epoch in range(args.epochs):
        t0 = time.time()
        train_metrics = train_epoch(
            kinoflow, scorer, train_loader, optimizer, device, args)

        val_metrics = validate(kinoflow, scorer, val_loader, device, args)
        scheduler.step()

        elapsed = time.time() - t0
        print(f"Epoch {epoch+1}/{args.epochs}: "
              f"train_loss={train_metrics['loss']:.4f} "
              f"train_acc={train_metrics['accuracy']:.3f} "
              f"val_loss={val_metrics['loss']:.4f} "
              f"val_acc={val_metrics['accuracy']:.3f} "
              f"({elapsed:.1f}s)")

        if args.wandb:
            wandb.log({
                'epoch': epoch + 1,
                'train/loss': train_metrics['loss'],
                'train/accuracy': train_metrics['accuracy'],
                'val/loss': val_metrics['loss'],
                'val/accuracy': val_metrics['accuracy'],
                'lr': scheduler.get_last_lr()[0],
            })

        # Save best
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': scorer.state_dict(),
                'val_accuracy': best_val_acc,
                'embed_dim': args.embed_dim,
                'n_heads': args.n_heads,
                'n_layers': args.n_layers,
                'horizon': horizon,
            }, os.path.join(args.output_dir, 'scorer_best.pt'))
            print(f"  -> Saved best (val_acc={best_val_acc:.3f})")

        # Periodic save
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': scorer.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': val_metrics['accuracy'],
                'embed_dim': args.embed_dim,
                'n_heads': args.n_heads,
                'n_layers': args.n_layers,
                'horizon': horizon,
            }, os.path.join(args.output_dir, f'scorer_epoch{epoch+1}.pt'))

    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.3f}")


if __name__ == '__main__':
    main()
