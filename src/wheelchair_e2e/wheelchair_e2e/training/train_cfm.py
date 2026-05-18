"""
Training Script for CFM-BEV: Conditional Flow Matching Velocity Trajectories.

Usage:
    python -m wheelchair_e2e.training.train_cfm \
        --data_dir /path/to/training_data \
        --output_dir /path/to/checkpoints \
        --epochs 50 --batch_size 128 --horizon 10

Training pipeline:
    1. Load BEV grids (5ch) + odom + velocity trajectory labels (H steps)
    2. Encode BEV+odom via ResNet-18 + GRU → conditioning features
    3. Sample noise x_0, interpolate x_t = (1-t)*x_0 + t*x_1
    4. Train vector field to predict x_1 - x_0 (flow matching loss)
    5. Combined loss: flow_matching + λ_c*collision + λ_j*jerk

Key difference from Nav2 DWB:
    Nav2: sample 1000 (v,ω) → simulate trajectories → score vs costmap → pick best
    Ours: BEV → single forward pass → generate velocity trajectory directly
"""

import os
import argparse
import json
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from wheelchair_e2e.models.cfm_velocity_net import CFMVelocityNet
from wheelchair_e2e.models.flow_matching import flow_matching_loss, euler_sample
from wheelchair_e2e.training.dataset import CFMTrajectoryDataset
from wheelchair_e2e.training.augmentation import ChauffeurNetAugmentor


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train CFM-BEV Velocity Trajectory Model')
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='checkpoints_cfm')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=128,
                        help='~128 on RTX 5050 8GB, 32 on Jetson')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--freeze_epochs', type=int, default=10)
    parser.add_argument('--v_max', type=float, default=0.25)
    parser.add_argument('--w_max', type=float, default=1.0)
    parser.add_argument('--horizon', type=int, default=10,
                        help='Velocity trajectory horizon (steps at 10Hz)')
    parser.add_argument('--n_euler_steps', type=int, default=3,
                        help='Euler ODE steps at inference')
    parser.add_argument('--cfm_hidden', type=int, default=512)
    parser.add_argument('--cfm_layers', type=int, default=4)
    parser.add_argument('--lambda_collision', type=float, default=5.0)
    parser.add_argument('--lambda_jerk', type=float, default=0.1)
    parser.add_argument('--augment', action='store_true', default=True)
    parser.add_argument('--no_augment', action='store_false', dest='augment')
    parser.add_argument('--val_split', type=float, default=0.1)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--resume', type=str, default=None)
    return parser.parse_args()


def trajectory_jerk_loss(vel_traj):
    """
    Jerk loss over generated trajectory.
    Penalizes velocity changes between consecutive steps.

    Args:
        vel_traj: (B, H, 2) velocity trajectory
    Returns:
        scalar jerk loss
    """
    if vel_traj.shape[1] < 2:
        return torch.tensor(0.0, device=vel_traj.device)
    diffs = vel_traj[:, 1:, :] - vel_traj[:, :-1, :]  # (B, H-1, 2)
    return (diffs ** 2).mean()


def trajectory_collision_loss(vel_traj, bev, dt=0.1, safety_radius=0.3,
                              resolution=0.05):
    """
    Collision loss over the full generated trajectory.
    Forward-projects each step and checks BEV occupancy.

    Args:
        vel_traj: (B, H, 2) velocity trajectory
        bev: (B, 5, 200, 200) BEV grid
        dt: time step between trajectory steps
    Returns:
        scalar collision loss
    """
    B, H, _ = vel_traj.shape
    grid_size = bev.shape[-1]
    center = grid_size // 2
    safety_px = int(safety_radius / resolution)

    # Max of obstacle channels (0=lidar, 1=depth)
    occupancy = torch.max(bev[:, 0], bev[:, 1])  # (B, 200, 200)

    total_loss = torch.tensor(0.0, device=vel_traj.device)
    cum_x = torch.zeros(B, device=vel_traj.device)
    cum_y = torch.zeros(B, device=vel_traj.device)
    cum_theta = torch.zeros(B, device=vel_traj.device)

    for t in range(H):
        v = vel_traj[:, t, 0]
        omega = vel_traj[:, t, 1]

        # Differential drive forward projection from current accumulated pose
        cum_theta = cum_theta + omega * dt
        cum_x = cum_x + v * dt * torch.cos(cum_theta)
        cum_y = cum_y + v * dt * torch.sin(cum_theta)

        # Convert to pixel coords
        px = (cum_x / resolution + center).long().clamp(0, grid_size - 1)
        py = (cum_y / resolution + center).long().clamp(0, grid_size - 1)

        # Check occupancy at projected positions
        for b in range(B):
            x_lo = max(0, px[b].item() - safety_px)
            x_hi = min(grid_size, px[b].item() + safety_px + 1)
            y_lo = max(0, py[b].item() - safety_px)
            y_hi = min(grid_size, py[b].item() + safety_px + 1)
            patch = occupancy[b, y_lo:y_hi, x_lo:x_hi]
            if patch.numel() > 0:
                total_loss = total_loss + patch.max()

    return total_loss / (B * H)


def train_one_epoch(model, dataloader, optimizer, device, args):
    """Train one epoch with flow matching + environment losses."""
    model.train()
    total_losses = {}
    n_batches = 0

    for batch in dataloader:
        bev = batch['bev'].to(device)           # (B, 5, 200, 200)
        odom = batch['odom'].to(device)          # (B, 30)
        vel_traj = batch['velocity_traj'].to(device)  # (B, H, 2)

        # 1. Encode BEV + odom → conditioning
        cond, _ = model(bev, odom, hidden=None)  # (B, 256)

        # 2. Flow matching loss
        B = vel_traj.shape[0]
        x_1 = vel_traj.reshape(B, -1)           # (B, H*2) target
        x_0 = torch.randn_like(x_1)             # noise
        t = torch.rand(B, device=device)         # flow time

        fm_loss = flow_matching_loss(
            model.vector_field, x_0, x_1, t, cond)

        # 3. Generate trajectory for environment losses
        with torch.no_grad():
            gen_traj, _ = model.generate(bev, odom)  # (B, H, 2)

        # Make gen_traj require grad for collision/jerk backprop through VF
        # (We use the FM loss as primary, env losses as auxiliary)
        jerk_loss = trajectory_jerk_loss(gen_traj)
        coll_loss = trajectory_collision_loss(gen_traj, bev)

        loss = fm_loss + args.lambda_collision * coll_loss \
            + args.lambda_jerk * jerk_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        losses = {
            'total': loss.item(),
            'fm': fm_loss.item(),
            'collision': coll_loss.item(),
            'jerk': jerk_loss.item(),
        }
        for k, v in losses.items():
            total_losses[k] = total_losses.get(k, 0) + v
        n_batches += 1

    return {k: v / n_batches for k, v in total_losses.items()}


@torch.no_grad()
def validate(model, dataloader, device, args):
    """Validate: generate trajectories, measure quality."""
    model.eval()
    total_losses = {}
    all_v_first = []
    all_w_first = []
    n_batches = 0

    for batch in dataloader:
        bev = batch['bev'].to(device)
        odom = batch['odom'].to(device)
        vel_traj_gt = batch['velocity_traj'].to(device)

        # Generate trajectory
        gen_traj, _ = model.generate(bev, odom)  # (B, H, 2)

        # MSE between generated and ground truth trajectories
        traj_mse = ((gen_traj - vel_traj_gt) ** 2).mean()
        v_mse = ((gen_traj[:, :, 0] - vel_traj_gt[:, :, 0]) ** 2).mean()
        w_mse = ((gen_traj[:, :, 1] - vel_traj_gt[:, :, 1]) ** 2).mean()

        # First-step accuracy (what actually gets executed)
        first_v_mse = ((gen_traj[:, 0, 0] - vel_traj_gt[:, 0, 0]) ** 2).mean()
        first_w_mse = ((gen_traj[:, 0, 1] - vel_traj_gt[:, 0, 1]) ** 2).mean()

        jerk = trajectory_jerk_loss(gen_traj)

        losses = {
            'total': traj_mse.item(),
            'v_mse': v_mse.item(),
            'w_mse': w_mse.item(),
            'first_v_mse': first_v_mse.item(),
            'first_w_mse': first_w_mse.item(),
            'jerk': jerk.item(),
        }
        for k, v in losses.items():
            total_losses[k] = total_losses.get(k, 0) + v
        n_batches += 1

        all_v_first.extend(gen_traj[:, 0, 0].cpu().numpy().tolist())
        all_w_first.extend(gen_traj[:, 0, 1].cpu().numpy().tolist())

    avg = {k: v / n_batches for k, v in total_losses.items()}
    avg['pred_v_mean'] = np.mean(all_v_first)
    avg['pred_v_std'] = np.std(all_v_first)
    avg['pred_w_mean'] = np.mean(all_w_first)
    avg['pred_w_std'] = np.std(all_w_first)
    return avg


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available()
                          else 'cpu')
    print(f"Training CFM-BEV on: {device}")

    # --- Dataset ---
    augmentor = None
    if args.augment:
        augmentor = ChauffeurNetAugmentor(
            max_pos=0.3, max_heading=30, prob=0.5,
            v_max=args.v_max, w_max=args.w_max)

    full_dataset = CFMTrajectoryDataset(
        args.data_dir, horizon=args.horizon, augment=augmentor)
    print(f"CFM dataset: {len(full_dataset)} valid samples "
          f"(horizon={args.horizon})")

    n_val = int(len(full_dataset) * args.val_split)
    n_train = len(full_dataset) - n_val
    train_set, val_set = random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size,
        shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size,
        shuffle=False, num_workers=2, pin_memory=True)
    print(f"Train: {n_train}, Val: {n_val}")

    # --- Model ---
    model = CFMVelocityNet(
        bev_channels=5, v_max=args.v_max, w_max=args.w_max,
        horizon=args.horizon, cfm_hidden=args.cfm_hidden,
        cfm_layers=args.cfm_layers,
        n_euler_steps=args.n_euler_steps,
    ).to(device)
    print(f"Model parameters: {model.get_param_count()}")

    # --- Optimizer: freeze backbone first ---
    for param in model.encoder.parameters():
        param.requires_grad = False

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Resume
    start_epoch = 0
    best_val_loss = float('inf')
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_val_loss = ckpt.get('best_val_loss', float('inf'))
        print(f"Resumed from epoch {start_epoch}")

    # --- Training loop ---
    history = []

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        # Unfreeze backbone after freeze_epochs
        if epoch == args.freeze_epochs:
            print("Unfreezing backbone with 10x lower LR")
            for param in model.encoder.parameters():
                param.requires_grad = True
            optimizer = AdamW([
                {'params': model.encoder.parameters(),
                 'lr': args.lr * 0.1},
                {'params': model.odom_encoder.parameters()},
                {'params': model.gru.parameters()},
                {'params': model.vector_field.parameters()},
            ], lr=args.lr, weight_decay=args.weight_decay)
            scheduler = CosineAnnealingLR(
                optimizer, T_max=args.epochs - epoch)

        train_losses = train_one_epoch(
            model, train_loader, optimizer, device, args)
        val_losses = validate(model, val_loader, device, args)

        scheduler.step()
        elapsed = time.time() - t0

        print(f"Epoch {epoch:3d}/{args.epochs} "
              f"[{elapsed:.1f}s] "
              f"fm={train_losses['fm']:.4f} "
              f"val={val_losses['total']:.4f} "
              f"first_v={val_losses['first_v_mse']:.5f} "
              f"first_w={val_losses['first_w_mse']:.5f} "
              f"jerk={val_losses['jerk']:.4f}")

        history.append({
            'epoch': epoch,
            'train': train_losses,
            'val': val_losses,
            'lr': optimizer.param_groups[0]['lr'],
            'time': elapsed,
        })

        if val_losses['total'] < best_val_loss:
            best_val_loss = val_losses['total']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'args': vars(args),
                'param_counts': model.get_param_count(),
            }, os.path.join(args.output_dir, 'best_cfm_model.pth'))
            print(f"  -> Saved best (val={best_val_loss:.4f})")

        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
            }, os.path.join(args.output_dir, f'cfm_ckpt_{epoch}.pth'))

    with open(os.path.join(args.output_dir, 'cfm_history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\nCFM training complete. Best val loss: {best_val_loss:.4f}")


if __name__ == '__main__':
    main()
