"""
3-Phase Training Script for KinoFlow v2: Modular Kinodynamic Flow Matching.

Phase 1 (epochs 0-9):     All trainable, flow matching loss only
Phase 2 (epochs 10-24):   Add kinematic + non-holonomic + collision losses
Phase 3 (epochs 25-49):   Add jerk + comfort losses

No frozen/unfrozen backbone phases needed — no pretrained ResNet-18.
All ~1.3M params train from scratch. 50% faster than v1 due to 10x fewer params.

Usage:
    python -m wheelchair_e2e.training.train_kinoflow \
        --data_dir /path/to/training_data \
        --output_dir /path/to/checkpoints \
        --epochs 50 --batch_size 256 --horizon 10

    # Legacy v1 mode (ResNet-18 BEV encoder):
    python -m wheelchair_e2e.training.train_kinoflow \
        --data_dir /path/to/training_data \
        --output_dir /path/to/checkpoints \
        --legacy
"""

import os
import argparse
import json
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from wheelchair_e2e.models.kinoflow_net import ModularKinoFlowNet, KinoFlowNet
from wheelchair_e2e.models.flow_matching import euler_sample
from wheelchair_e2e.training.kinoflow_losses import KinoFlowLoss
from wheelchair_e2e.training.dataset import (
    CFMTrajectoryDataset, ModularKinoFlowDataset
)
from wheelchair_e2e.training.augmentation import (
    ChauffeurNetAugmentor, ModularTrajectoryAugmentor
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train KinoFlow v2: Modular Kinodynamic Flow Matching')

    # Data
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='checkpoints_kinoflow')
    parser.add_argument('--val_split', type=float, default=0.1)

    # Training
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=256,
                        help='256 on RTX 3090, 128 on RTX 5050, 32 on Jetson')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--grad_clip', type=float, default=1.0)

    # Phase boundaries
    parser.add_argument('--phase1_epochs', type=int, default=10)
    parser.add_argument('--phase2_epochs', type=int, default=15)

    # Model architecture
    parser.add_argument('--v_max', type=float, default=0.25)
    parser.add_argument('--w_max', type=float, default=1.0)
    parser.add_argument('--horizon', type=int, default=10)
    parser.add_argument('--n_euler_steps', type=int, default=3)
    parser.add_argument('--n_samples', type=int, default=8)
    parser.add_argument('--dt', type=float, default=0.1)
    parser.add_argument('--scan_points', type=int, default=720)
    parser.add_argument('--temporal_frames', type=int, default=5)
    parser.add_argument('--d_model', type=int, default=128)

    # Ablation flags
    parser.add_argument('--no_dynamic', action='store_true')
    parser.add_argument('--no_temporal', action='store_true')
    parser.add_argument('--no_goal', action='store_true')
    parser.add_argument('--no_velocity', action='store_true')
    parser.add_argument('--mlp_vectorfield', action='store_true')
    parser.add_argument('--concat_fusion', action='store_true')
    parser.add_argument('--monolithic', action='store_true',
                        help='Use legacy v1 architecture (ResNet-18)')
    parser.add_argument('--legacy', action='store_true',
                        help='Use legacy v1 KinoFlowNet + CFMTrajectoryDataset')

    # Loss weights
    parser.add_argument('--lambda_flow', type=float, default=1.0)
    parser.add_argument('--lambda_kinematic', type=float, default=0.5)
    parser.add_argument('--lambda_nonholonomic', type=float, default=1.0)
    parser.add_argument('--lambda_collision', type=float, default=5.0)
    parser.add_argument('--lambda_jerk', type=float, default=0.1)
    parser.add_argument('--lambda_comfort', type=float, default=0.5)
    parser.add_argument('--imitation_dropout', type=float, default=0.5)

    # Augmentation
    parser.add_argument('--augment', action='store_true', default=True)
    parser.add_argument('--no_augment', action='store_false', dest='augment')

    # System
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--wandb', action='store_true', default=False)
    parser.add_argument('--wandb_project', type=str, default='kinoflow-wheelchair')

    return parser.parse_args()


def get_phase(epoch, phase1_epochs, phase2_epochs):
    if epoch < phase1_epochs:
        return 1
    elif epoch < phase1_epochs + phase2_epochs:
        return 2
    return 3


def train_one_epoch_modular(model, loss_fn, dataloader, optimizer, device,
                            phase, n_euler_steps, grad_clip):
    """Train one epoch with modular v2 inputs."""
    model.train()
    loss_fn.train()
    loss_fn.set_phase(phase)

    total_losses = {}
    n_batches = 0

    for batch in dataloader:
        scan = batch['scan_current'].to(device)
        residuals = batch['scan_residuals'].to(device)
        goal = batch['goal_features'].to(device)
        odom = batch['odom'].to(device)
        vel_traj = batch['velocity_traj'].to(device)
        bev = batch['bev'].to(device)

        B, H, _ = vel_traj.shape

        # 1. Encode
        cond, _ = model(scan, residuals, goal, odom, hidden=None)

        # 2. Flow matching inputs
        x_1 = vel_traj.reshape(B, -1)
        x_0 = torch.randn_like(x_1)
        t = torch.rand(B, device=device)

        # 3. Generate trajectory for env losses (phases 2-3)
        gen_vel_traj = None
        gen_poses = None
        if phase >= 2:
            with torch.no_grad():
                gen_vel_traj, gen_poses, _ = model.generate(
                    scan, residuals, goal, odom)

        # 4. Combined loss
        loss, loss_dict = loss_fn(
            model.vector_field, x_0, x_1, t, cond,
            gen_vel_traj=gen_vel_traj,
            gen_poses=gen_poses,
            bev=bev,
        )

        # 5. Backward + optimize
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()

        for k, v in loss_dict.items():
            total_losses[k] = total_losses.get(k, 0) + v
        n_batches += 1

    return {k: v / n_batches for k, v in total_losses.items()}


def train_one_epoch_legacy(model, loss_fn, dataloader, optimizer, device,
                           phase, n_euler_steps, grad_clip):
    """Train one epoch with legacy v1 BEV inputs."""
    model.train()
    loss_fn.train()
    loss_fn.set_phase(phase)

    total_losses = {}
    n_batches = 0

    for batch in dataloader:
        bev = batch['bev'].to(device)
        odom = batch['odom'].to(device)
        vel_traj = batch['velocity_traj'].to(device)

        B, H, _ = vel_traj.shape

        cond, _ = model(bev, odom, hidden=None)

        x_1 = vel_traj.reshape(B, -1)
        x_0 = torch.randn_like(x_1)
        t = torch.rand(B, device=device)

        gen_vel_traj = None
        gen_poses = None
        if phase >= 2:
            with torch.no_grad():
                gen_vel_traj, gen_poses, _ = model.generate(bev, odom)

        loss, loss_dict = loss_fn(
            model.vector_field, x_0, x_1, t, cond,
            gen_vel_traj=gen_vel_traj,
            gen_poses=gen_poses,
            bev=bev,
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()

        for k, v in loss_dict.items():
            total_losses[k] = total_losses.get(k, 0) + v
        n_batches += 1

    return {k: v / n_batches for k, v in total_losses.items()}


@torch.no_grad()
def validate_modular(model, loss_fn, dataloader, device, n_euler_steps):
    """Validate modular v2 model."""
    model.eval()
    loss_fn.eval()

    total_metrics = {}
    all_v_first = []
    all_w_first = []
    n_batches = 0

    for batch in dataloader:
        scan = batch['scan_current'].to(device)
        residuals = batch['scan_residuals'].to(device)
        goal = batch['goal_features'].to(device)
        odom = batch['odom'].to(device)
        vel_traj_gt = batch['velocity_traj'].to(device)

        B, H, _ = vel_traj_gt.shape

        gen_vel_traj, gen_poses, _ = model.generate(
            scan, residuals, goal, odom)
        gt_poses = model.integrate_trajectory(vel_traj_gt)

        metrics = _compute_metrics(gen_vel_traj, vel_traj_gt, gen_poses,
                                   gt_poses, H)
        for k, v in metrics.items():
            total_metrics[k] = total_metrics.get(k, 0) + v
        n_batches += 1

        all_v_first.extend(gen_vel_traj[:, 0, 0].cpu().numpy().tolist())
        all_w_first.extend(gen_vel_traj[:, 0, 1].cpu().numpy().tolist())

    avg = {k: v / n_batches for k, v in total_metrics.items()}
    avg['pred_v_mean'] = np.mean(all_v_first)
    avg['pred_v_std'] = np.std(all_v_first)
    avg['pred_w_mean'] = np.mean(all_w_first)
    avg['pred_w_std'] = np.std(all_w_first)
    return avg


@torch.no_grad()
def validate_legacy(model, loss_fn, dataloader, device, n_euler_steps):
    """Validate legacy v1 model."""
    model.eval()
    loss_fn.eval()

    total_metrics = {}
    all_v_first = []
    all_w_first = []
    n_batches = 0

    for batch in dataloader:
        bev = batch['bev'].to(device)
        odom = batch['odom'].to(device)
        vel_traj_gt = batch['velocity_traj'].to(device)

        B, H, _ = vel_traj_gt.shape

        gen_vel_traj, gen_poses, _ = model.generate(bev, odom)
        gt_poses = model.integrate_trajectory(vel_traj_gt)

        metrics = _compute_metrics(gen_vel_traj, vel_traj_gt, gen_poses,
                                   gt_poses, H)
        for k, v in metrics.items():
            total_metrics[k] = total_metrics.get(k, 0) + v
        n_batches += 1

        all_v_first.extend(gen_vel_traj[:, 0, 0].cpu().numpy().tolist())
        all_w_first.extend(gen_vel_traj[:, 0, 1].cpu().numpy().tolist())

    avg = {k: v / n_batches for k, v in total_metrics.items()}
    avg['pred_v_mean'] = np.mean(all_v_first)
    avg['pred_v_std'] = np.std(all_v_first)
    avg['pred_w_mean'] = np.mean(all_w_first)
    avg['pred_w_std'] = np.std(all_w_first)
    return avg


def _compute_metrics(gen_vel_traj, vel_traj_gt, gen_poses, gt_poses, H):
    """Compute validation metrics (shared between modular and legacy)."""
    traj_mse = ((gen_vel_traj - vel_traj_gt) ** 2).mean()
    v_mse = ((gen_vel_traj[:, :, 0] - vel_traj_gt[:, :, 0]) ** 2).mean()
    w_mse = ((gen_vel_traj[:, :, 1] - vel_traj_gt[:, :, 1]) ** 2).mean()
    first_v_mse = ((gen_vel_traj[:, 0, 0] - vel_traj_gt[:, 0, 0]) ** 2).mean()
    first_w_mse = ((gen_vel_traj[:, 0, 1] - vel_traj_gt[:, 0, 1]) ** 2).mean()
    ade = torch.sqrt(
        (gen_poses[:, :, 0] - gt_poses[:, :, 0]) ** 2
        + (gen_poses[:, :, 1] - gt_poses[:, :, 1]) ** 2
    ).mean()
    fde = torch.sqrt(
        (gen_poses[:, -1, 0] - gt_poses[:, -1, 0]) ** 2
        + (gen_poses[:, -1, 1] - gt_poses[:, -1, 1]) ** 2
    ).mean()

    jerk_metric = torch.tensor(0.0)
    if H >= 3:
        accel = (gen_vel_traj[:, 1:, :] - gen_vel_traj[:, :-1, :]) / 0.1
        jerk = (accel[:, 1:, :] - accel[:, :-1, :]) / 0.1
        jerk_metric = (jerk ** 2).mean()

    nh_violation = torch.tensor(0.0)
    if H >= 2:
        x_dot = (gen_poses[:, 1:, 0] - gen_poses[:, :-1, 0]) / 0.1
        y_dot = (gen_poses[:, 1:, 1] - gen_poses[:, :-1, 1]) / 0.1
        theta = gen_poses[:, :-1, 2]
        v_lateral = -x_dot * torch.sin(theta) + y_dot * torch.cos(theta)
        nh_violation = (v_lateral ** 2).mean()

    rms_accel = torch.tensor(0.0)
    if H >= 2:
        a = (gen_vel_traj[:, 1:, 0] - gen_vel_traj[:, :-1, 0]) / 0.1
        rms_accel = torch.sqrt((a ** 2).mean())

    return {
        'traj_mse': traj_mse.item(),
        'v_mse': v_mse.item(),
        'w_mse': w_mse.item(),
        'first_v_mse': first_v_mse.item(),
        'first_w_mse': first_w_mse.item(),
        'ade': ade.item(),
        'fde': fde.item(),
        'jerk': jerk_metric.item(),
        'nh_violation': nh_violation.item(),
        'rms_accel': rms_accel.item(),
    }


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    use_legacy = args.legacy or args.monolithic
    arch_name = "KinoFlow v1 (legacy)" if use_legacy else "KinoFlow v2 (modular)"

    print(f"{arch_name} training on: {device}")
    print(f"3-phase schedule: P1={args.phase1_epochs}, "
          f"P2={args.phase2_epochs}, "
          f"P3={args.epochs - args.phase1_epochs - args.phase2_epochs}")

    # Ablation flags
    if not use_legacy:
        ablation_flags = []
        for flag in ['no_dynamic', 'no_temporal', 'no_goal', 'no_velocity',
                      'mlp_vectorfield', 'concat_fusion']:
            if getattr(args, flag, False):
                ablation_flags.append(flag)
        if ablation_flags:
            print(f"Ablation: {', '.join(ablation_flags)}")

    # --- Optional wandb ---
    wandb_run = None
    if args.wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project=args.wandb_project, config=vars(args),
                name=f'kinoflow_{"v1" if use_legacy else "v2"}_'
                     f'H{args.horizon}_K{args.n_samples}')
        except ImportError:
            print("wandb not installed, continuing without logging")

    # --- Dataset ---
    if use_legacy:
        augmentor = None
        if args.augment:
            augmentor = ChauffeurNetAugmentor(
                max_pos=0.3, max_heading=30, prob=0.5,
                v_max=args.v_max, w_max=args.w_max)
        full_dataset = CFMTrajectoryDataset(
            args.data_dir, horizon=args.horizon, augment=augmentor)
    else:
        augmentor = None
        if args.augment:
            augmentor = ModularTrajectoryAugmentor(
                mirror_prob=0.5, noise_prob=0.3)
        full_dataset = ModularKinoFlowDataset(
            args.data_dir, horizon=args.horizon,
            temporal_frames=args.temporal_frames,
            scan_points=args.scan_points, augment=augmentor)

    print(f"Dataset: {len(full_dataset)} samples (horizon={args.horizon})")

    n_val = int(len(full_dataset) * args.val_split)
    n_train = len(full_dataset) - n_val
    train_set, val_set = random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size,
        shuffle=False, num_workers=max(1, args.num_workers // 2),
        pin_memory=True)
    print(f"Train: {n_train}, Val: {n_val}")

    # --- Model ---
    if use_legacy:
        if args.monolithic:
            model = ModularKinoFlowNet(
                v_max=args.v_max, w_max=args.w_max,
                horizon=args.horizon, n_euler_steps=args.n_euler_steps,
                dt=args.dt, n_samples=args.n_samples,
                monolithic=True,
            ).to(device)
        else:
            model = KinoFlowNet(
                bev_channels=5, v_max=args.v_max, w_max=args.w_max,
                horizon=args.horizon, n_euler_steps=args.n_euler_steps,
                dt=args.dt, n_samples=args.n_samples,
            ).to(device)
    else:
        model = ModularKinoFlowNet(
            v_max=args.v_max, w_max=args.w_max,
            horizon=args.horizon, n_euler_steps=args.n_euler_steps,
            dt=args.dt, n_samples=args.n_samples,
            scan_points=args.scan_points,
            temporal_frames=args.temporal_frames,
            d_model=args.d_model,
            no_dynamic=args.no_dynamic,
            no_temporal=args.no_temporal,
            no_goal=args.no_goal,
            no_velocity=args.no_velocity,
            mlp_vectorfield=args.mlp_vectorfield,
            concat_fusion=args.concat_fusion,
        ).to(device)

    param_counts = model.get_param_count()
    print(f"Model parameters: {param_counts}")
    total_params = param_counts['total']
    print(f"Total: {total_params:,} ({total_params / 1e6:.2f}M)")

    # --- Loss function ---
    loss_fn = KinoFlowLoss(
        lambda_flow=args.lambda_flow,
        lambda_kinematic=args.lambda_kinematic,
        lambda_nonholonomic=args.lambda_nonholonomic,
        lambda_collision=args.lambda_collision,
        lambda_jerk=args.lambda_jerk,
        lambda_comfort=args.lambda_comfort,
        imitation_dropout=args.imitation_dropout,
        dt=args.dt, v_max=args.v_max, w_max=args.w_max,
    )

    # --- Optimizer (no frozen backbone phases for v2!) ---
    if use_legacy and not args.monolithic:
        # Legacy: freeze backbone in P1
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
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_val_loss = ckpt.get('best_val_loss', float('inf'))
        print(f"Resumed from epoch {start_epoch}")

    # Save config
    config = vars(args)
    config['param_counts'] = param_counts
    config['architecture'] = arch_name
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    # Select train/validate functions
    train_fn = train_one_epoch_legacy if use_legacy else train_one_epoch_modular
    val_fn = validate_legacy if use_legacy else validate_modular

    # --- Training loop ---
    history = []
    prev_phase = 0

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        phase = get_phase(epoch, args.phase1_epochs, args.phase2_epochs)

        if phase != prev_phase:
            print(f"\n{'='*60}")
            print(f"PHASE {phase} starting at epoch {epoch}")
            print(f"{'='*60}")

            if phase == 2:
                print("  Activating: kinematic + NH + collision losses")
            elif phase == 3:
                print("  Activating: jerk + comfort losses")
                if use_legacy and not args.monolithic:
                    print("  Unfreezing backbone (10x lower LR)")
                    for param in model.encoder.parameters():
                        param.requires_grad = True
                    optimizer = AdamW([
                        {'params': model.encoder.parameters(),
                         'lr': args.lr * 0.1},
                        {'params': [p for n, p in model.named_parameters()
                                    if not n.startswith('encoder')]},
                    ], lr=args.lr, weight_decay=args.weight_decay)
                    scheduler = CosineAnnealingLR(
                        optimizer, T_max=args.epochs - epoch)

            prev_phase = phase

        train_losses = train_fn(
            model, loss_fn, train_loader, optimizer, device,
            phase=phase, n_euler_steps=args.n_euler_steps,
            grad_clip=args.grad_clip)

        val_metrics = val_fn(
            model, loss_fn, val_loader, device,
            n_euler_steps=args.n_euler_steps)

        scheduler.step()
        elapsed = time.time() - t0

        print(f"Epoch {epoch:3d}/{args.epochs} P{phase} "
              f"[{elapsed:.1f}s] "
              f"flow={train_losses.get('flow', 0):.4f} "
              f"traj_mse={val_metrics['traj_mse']:.4f} "
              f"ade={val_metrics['ade']:.4f} "
              f"fde={val_metrics['fde']:.4f} "
              f"jerk={val_metrics['jerk']:.4f} "
              f"rms_a={val_metrics['rms_accel']:.3f}")

        record = {
            'epoch': epoch, 'phase': phase,
            'train': train_losses, 'val': val_metrics,
            'lr': optimizer.param_groups[0]['lr'], 'time': elapsed,
        }
        history.append(record)

        if wandb_run is not None:
            log_dict = {'epoch': epoch, 'phase': phase, 'lr': record['lr']}
            for k, v in train_losses.items():
                log_dict[f'train/{k}'] = v
            for k, v in val_metrics.items():
                log_dict[f'val/{k}'] = v
            wandb_run.log(log_dict)

        val_score = val_metrics['traj_mse']
        if val_score < best_val_loss:
            best_val_loss = val_score
            torch.save({
                'epoch': epoch, 'phase': phase,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'args': vars(args),
                'param_counts': param_counts,
                'val_metrics': val_metrics,
                'architecture': arch_name,
            }, os.path.join(args.output_dir, 'best_kinoflow.pth'))
            print(f"  -> Saved best (traj_mse={best_val_loss:.4f})")

        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch, 'phase': phase,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'val_metrics': val_metrics,
            }, os.path.join(args.output_dir,
                           f'kinoflow_P{phase}_ep{epoch}.pth'))

    # Save final
    torch.save({
        'epoch': args.epochs - 1,
        'model_state_dict': model.state_dict(),
        'args': vars(args),
        'param_counts': param_counts,
        'architecture': arch_name,
    }, os.path.join(args.output_dir, 'kinoflow_final.pth'))

    with open(os.path.join(args.output_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    if wandb_run is not None:
        wandb_run.finish()

    print(f"\n{arch_name} training complete.")
    print(f"  Best traj MSE: {best_val_loss:.4f}")
    print(f"  Checkpoints: {args.output_dir}")


if __name__ == '__main__':
    main()
