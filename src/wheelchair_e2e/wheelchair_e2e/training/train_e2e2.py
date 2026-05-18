"""
Training Script for E2E-2: BEV-Velocity CNN

Usage:
    python -m wheelchair_e2e.training.train_e2e2 \
        --data_dir /path/to/training_data \
        --output_dir /path/to/checkpoints \
        --epochs 50 \
        --batch_size 64 \
        --lr 1e-4

Training pipeline (ChauffeurNet-inspired):
    1. Load BEV grids + velocity labels
    2. Apply ChauffeurNet perturbation augmentation (5-10x data)
    3. Train with combined loss (imitation + collision + jerk + dropout)
    4. Freeze backbone for first 10 epochs, then unfreeze with 10x lower LR
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

from wheelchair_e2e.models.bev_velocity_net import BEVVelocityNet
from wheelchair_e2e.training.dataset import BEVVelocityDataset, SequentialBEVDataset
from wheelchair_e2e.training.augmentation import ChauffeurNetAugmentor
from wheelchair_e2e.training.losses import E2EVelocityLoss


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train E2E-2 BEV-Velocity CNN')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to preprocessed training data')
    parser.add_argument('--output_dir', type=str, default='checkpoints',
                        help='Path to save checkpoints')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Max ~384 on RTX 5050 8GB, 64 on Jetson')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--freeze_epochs', type=int, default=10,
                        help='Epochs to freeze backbone')
    parser.add_argument('--v_max', type=float, default=0.25)
    parser.add_argument('--w_max', type=float, default=1.0)
    parser.add_argument('--augment', action='store_true', default=True,
                        help='Enable ChauffeurNet augmentation')
    parser.add_argument('--no_augment', action='store_false',
                        dest='augment')
    parser.add_argument('--imitation_dropout', type=float, default=0.5)
    parser.add_argument('--lambda_collision', type=float, default=10.0)
    parser.add_argument('--lambda_jerk', type=float, default=0.1)
    parser.add_argument('--val_split', type=float, default=0.1)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')
    parser.add_argument('--seq_len', type=int, default=5,
                        help='Sequence length for GRU training (1=no sequence)')
    return parser.parse_args()


def train_one_epoch(model, dataloader, optimizer, criterion, device,
                    sequential=False):
    """Train for one epoch. If sequential=True, unrolls GRU over time steps."""
    model.train()
    total_losses = {}
    n_batches = 0

    for batch in dataloader:
        bev = batch['bev'].to(device)
        odom = batch['odom'].to(device)
        target = batch['velocity'].to(device)

        if sequential and bev.dim() == 5:
            # Sequential mode: bev is (B, T, 4, 200, 200)
            B, T = bev.shape[0], bev.shape[1]
            hidden = None
            seq_loss = torch.tensor(0.0, device=device)
            seq_losses = {}
            prev_vel = None

            for t in range(T):
                pred_vel, hidden = model(
                    bev[:, t], odom[:, t], hidden=hidden)
                hidden = hidden.detach()  # truncated BPTT

                step_loss, step_dict = criterion(
                    pred_vel, target[:, t], bev=bev[:, t],
                    prev_vel=prev_vel)
                seq_loss = seq_loss + step_loss
                prev_vel = pred_vel.detach()

                for k, v in step_dict.items():
                    seq_losses[k] = seq_losses.get(k, 0) + v

            loss = seq_loss / T
            loss_dict = {k: v / T for k, v in seq_losses.items()}
        else:
            # Single-step mode (fallback)
            pred_vel, _ = model(bev, odom, hidden=None)
            loss, loss_dict = criterion(
                pred_vel, target, bev=bev, prev_vel=None)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        for k, v in loss_dict.items():
            total_losses[k] = total_losses.get(k, 0) + v
        n_batches += 1

    return {k: v / n_batches for k, v in total_losses.items()}


@torch.no_grad()
def validate(model, dataloader, criterion, device):
    """Validate, return average losses and velocity stats."""
    model.eval()
    total_losses = {}
    all_pred_v = []
    all_pred_w = []
    n_batches = 0

    for batch in dataloader:
        bev = batch['bev'].to(device)
        odom = batch['odom'].to(device)
        target = batch['velocity'].to(device)

        pred_vel, _ = model(bev, odom, hidden=None)
        loss, loss_dict = criterion(pred_vel, target, bev=bev)

        for k, v in loss_dict.items():
            total_losses[k] = total_losses.get(k, 0) + v
        n_batches += 1

        all_pred_v.extend(pred_vel[:, 0].cpu().numpy().tolist())
        all_pred_w.extend(pred_vel[:, 1].cpu().numpy().tolist())

    avg_losses = {k: v / n_batches for k, v in total_losses.items()}

    # Velocity distribution stats
    avg_losses['pred_v_mean'] = np.mean(all_pred_v)
    avg_losses['pred_v_std'] = np.std(all_pred_v)
    avg_losses['pred_w_mean'] = np.mean(all_pred_w)
    avg_losses['pred_w_std'] = np.std(all_pred_w)

    return avg_losses


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available()
                          else 'cpu')
    print(f"Training on: {device}")

    # --- Dataset ---
    augmentor = None
    if args.augment:
        augmentor = ChauffeurNetAugmentor(
            max_pos=0.3, max_heading=30, prob=0.5,
            v_max=args.v_max, w_max=args.w_max)

    sequential = args.seq_len > 1
    if sequential:
        full_dataset = SequentialBEVDataset(
            args.data_dir, seq_len=args.seq_len, augment=augmentor)
        print(f"Sequential mode: seq_len={args.seq_len}, "
              f"{len(full_dataset)} valid sequences")
    else:
        full_dataset = BEVVelocityDataset(
            args.data_dir, augment=augmentor)

    # Train/val split
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
    model = BEVVelocityNet(
        v_max=args.v_max, w_max=args.w_max).to(device)
    param_counts = model.get_param_count()
    print(f"Model parameters: {param_counts}")

    # --- Loss ---
    criterion = E2EVelocityLoss(
        lambda_collision=args.lambda_collision,
        lambda_jerk=args.lambda_jerk,
        imitation_dropout=args.imitation_dropout)

    # --- Optimizer ---
    # Phase 1: freeze backbone
    for param in model.encoder.parameters():
        param.requires_grad = False

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Resume from checkpoint
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
                {'params': model.velocity_head.parameters()},
            ], lr=args.lr, weight_decay=args.weight_decay)
            scheduler = CosineAnnealingLR(
                optimizer, T_max=args.epochs - epoch)

        # Train
        train_losses = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            sequential=sequential)

        # Validate
        val_losses = validate(model, val_loader, criterion, device)

        scheduler.step()
        elapsed = time.time() - t0

        # Log
        print(f"Epoch {epoch:3d}/{args.epochs} "
              f"[{elapsed:.1f}s] "
              f"train_loss={train_losses['total']:.4f} "
              f"val_loss={val_losses['total']:.4f} "
              f"v_mse={val_losses['v_mse']:.4f} "
              f"w_mse={val_losses['w_mse']:.4f} "
              f"pred_v={val_losses['pred_v_mean']:.3f}±"
              f"{val_losses['pred_v_std']:.3f}")

        history.append({
            'epoch': epoch,
            'train': train_losses,
            'val': val_losses,
            'lr': optimizer.param_groups[0]['lr'],
            'time': elapsed
        })

        # Save best model
        if val_losses['total'] < best_val_loss:
            best_val_loss = val_losses['total']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'args': vars(args),
                'param_counts': param_counts,
            }, os.path.join(args.output_dir, 'best_model.pth'))
            print(f"  -> Saved best model (val_loss={best_val_loss:.4f})")

        # Save periodic checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
            }, os.path.join(args.output_dir, f'ckpt_epoch_{epoch}.pth'))

    # Save training history
    with open(os.path.join(args.output_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
