#!/usr/bin/env python3
"""
Train Cable Tracer v4 — Temporal ResNet-SE with Red Mask + CBAM + Confidence Head.

Architecture upgrades over v3:
  1. Temporal stacking: 3 consecutive frames (model sees cable motion)
  2. Red mask channel: HSV-extracted cable prior (4th channel per frame)
  3. SE attention: per-channel feature weighting in each ResBlock
  4. CBAM spatial attention: focus on cable region after backbone
  5. Dual GELU heads: separate v and omega prediction paths
  6. EMA weights: exponential moving average for smoother generalization
  7. Confidence head: outputs cable visibility score [0, 1] — gates velocity at inference

Input:  12 channels (3 frames x [3 RGB + 1 red_mask])
Output: [v, omega, confidence] — v/omega in [-1, 1], confidence in [0, 1]
Params: ~117K

Usage:
    python3 train_cable_tracer.py --data data/rgb_vel_20260319_162633 data/rgb_vel_20260319_162910
    python3 train_cable_tracer.py --data run1/ run2/ --epochs 150 --lr 1e-3 --batch 32

Output:
    models/cable_tracer/cable_tracer.pt           — EMA-smoothed weights
    models/cable_tracer/cable_tracer_config.json   — normalization + architecture info
    models/cable_tracer/training_log.csv           — per-epoch metrics
"""

import argparse
import csv
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

N_FRAMES = 3       # temporal stack depth
CH_PER_FRAME = 4   # 3 RGB + 1 red_mask
IN_CHANNELS = N_FRAMES * CH_PER_FRAME  # 12

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ============================================================================
# MODEL — Temporal ResNet-SE with CBAM + Dual Heads
# ============================================================================

class SEBlock(nn.Module):
    """Squeeze-and-Excitation: learns per-channel feature importance."""
    def __init__(self, ch, reduction=4):
        super().__init__()
        mid = max(ch // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(ch, mid), nn.ReLU(),
            nn.Linear(mid, ch), nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w


class SpatialAttention(nn.Module):
    """CBAM spatial attention: learns WHERE in the image to focus."""
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.bn = nn.BatchNorm2d(1)

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        mx = x.max(dim=1, keepdim=True)[0]
        attn = torch.sigmoid(self.bn(self.conv(torch.cat([avg, mx], dim=1))))
        return x * attn


class ResBlockSE(nn.Module):
    """Residual block with SE channel attention."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.se = SEBlock(out_ch)
        self.skip = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        ) if in_ch != out_ch else nn.Identity()
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        identity = self.skip(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out = F.relu(out + identity)
        return self.pool(out)


class CableTracerCNN(nn.Module):
    """
    Input:  12ch tensor (3 temporal frames x [3 RGB + 1 red_mask])
    Output: [v, omega, confidence] — v/omega in [-1, 1], confidence in [0, 1]

    Pipeline:
        ResBlockSE(12→24)  128→64     (temporal fusion + color detection)
        ResBlockSE(24→48)  64→32      (cable shape extraction)
        ResBlockSE(48→64)  32→16      (spatial layout encoding)
        CBAM spatial attention         (focus on cable region)
        AdaptiveAvgPool → 64
        Dual heads: v_head, omega_head (separate GELU + dropout paths)
        Confidence head: cable visibility score [0, 1]
    """
    def __init__(self, in_channels=IN_CHANNELS):
        super().__init__()
        self.block1 = ResBlockSE(in_channels, 24)  # 128→64
        self.block2 = ResBlockSE(24, 48)            # 64→32
        self.block3 = ResBlockSE(48, 64)            # 32→16
        self.spatial_attn = SpatialAttention()
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.v_head = nn.Sequential(
            nn.Linear(64, 32), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(32, 1), nn.Tanh(),
        )
        self.omega_head = nn.Sequential(
            nn.Linear(64, 32), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(32, 1), nn.Tanh(),
        )
        self.confidence_head = nn.Sequential(
            nn.Linear(64, 16), nn.GELU(),
            nn.Linear(16, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.spatial_attn(x)
        x = self.pool(x).flatten(1)
        v = self.v_head(x)
        omega = self.omega_head(x)
        confidence = self.confidence_head(x)
        return torch.cat([v, omega, confidence], dim=1)


# ============================================================================
# LOSS — weighted MSE + direction consistency penalty
# ============================================================================

def cable_tracer_loss(preds, targets, cable_visible=None,
                      v_weight=1.0, omega_weight=3.0, dir_weight=2.0,
                      conf_weight=1.0):
    """
    Loss with confidence head. When cable_visible labels are provided:
      - Velocity loss computed only on cable-visible samples
      - BCE loss on confidence head for all samples
    """
    pred_conf = preds[:, 2]  # confidence output

    if cable_visible is not None:
        # BCE for confidence prediction
        conf_loss = F.binary_cross_entropy(pred_conf, cable_visible)

        # Velocity loss only on cable-visible samples
        vis_mask = cable_visible > 0.5
        if vis_mask.sum() > 0:
            v_loss = F.mse_loss(preds[vis_mask, 0], targets[vis_mask, 0])
            omega_loss = F.mse_loss(preds[vis_mask, 1], targets[vis_mask, 1])
            sign_disagree = (preds[vis_mask, 1] * targets[vis_mask, 1] < 0).float()
            dir_penalty = (sign_disagree * (preds[vis_mask, 1] - targets[vis_mask, 1]).pow(2)).mean()
        else:
            v_loss = torch.tensor(0.0, device=preds.device)
            omega_loss = torch.tensor(0.0, device=preds.device)
            dir_penalty = torch.tensor(0.0, device=preds.device)

        return (v_weight * v_loss + omega_weight * omega_loss +
                dir_weight * dir_penalty + conf_weight * conf_loss)
    else:
        # Backward compat: no cable_visible labels, all samples are positive
        v_loss = F.mse_loss(preds[:, 0], targets[:, 0])
        omega_loss = F.mse_loss(preds[:, 1], targets[:, 1])
        sign_disagree = (preds[:, 1] * targets[:, 1] < 0).float()
        dir_penalty = (sign_disagree * (preds[:, 1] - targets[:, 1]).pow(2)).mean()
        return v_weight * v_loss + omega_weight * omega_loss + dir_weight * dir_penalty


# ============================================================================
# EMA — exponential moving average of model weights
# ============================================================================

class ModelEMA:
    """Maintains an exponential moving average of model weights for smoother predictions."""
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        for k, v in model.state_dict().items():
            self.shadow[k] = v.clone().detach()

    def update(self, model):
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if v.is_floating_point():
                    self.shadow[k].mul_(self.decay).add_(v, alpha=1 - self.decay)
                else:
                    self.shadow[k].copy_(v)

    def state_dict(self):
        return {k: v.clone() for k, v in self.shadow.items()}


# ============================================================================
# DATASET — temporal stacking + red mask + augmentation
# ============================================================================

def extract_red_mask(img_rgb_01):
    """Extract red cable mask from RGB float [0,1] image. Returns float [0,1] mask."""
    img_u8 = (img_rgb_01 * 255).astype(np.uint8)
    hsv = cv2.cvtColor(img_u8, cv2.COLOR_RGB2HSV)
    # Red wraps around in HSV hue space
    m1 = cv2.inRange(hsv, (0, 70, 50), (12, 255, 255))
    m2 = cv2.inRange(hsv, (168, 70, 50), (180, 255, 255))
    return ((m1 | m2) > 0).astype(np.float32)


class CableTracerDataset(Dataset):
    """
    Temporal dataset: returns N consecutive frames with red mask channels.

    Each sample is a tuple of:
        - stacked tensor: (N_FRAMES * 4) x H x W  (3 frames x [RGB + red_mask])
        - velocity: [v_norm, omega_norm] in [-1, 1]

    Temporal frames are clamped to run boundaries (never crosses between runs).
    Augmentation (flip, jitter, noise) is applied consistently across all frames.
    """
    def __init__(self, data_dirs, img_size=128, v_max=0.50, omega_max=0.15,
                 augment=False, n_frames=N_FRAMES):
        self.img_size = img_size
        self.v_max = v_max
        self.omega_max = omega_max
        self.augment = augment
        self.n_frames = n_frames

        self.all_image_paths = []   # flat list of ALL image paths in temporal order
        self.samples = []           # (global_frame_idx, v_norm, omega_norm, cable_visible)
        self.run_boundaries = []    # [(start, end), ...] for clamping temporal lookback

        global_offset = 0

        for data_dir in data_dirs:
            data_dir = Path(data_dir)
            csv_path = data_dir / 'velocities.csv'
            img_dir = data_dir / 'images'

            if not csv_path.exists():
                print(f'WARNING: No velocities.csv in {data_dir}, skipping')
                continue

            # ALL images sorted by frame_id (needed for temporal context)
            all_imgs = sorted(img_dir.glob('*.jpg'),
                              key=lambda p: p.stem.split('_')[0])
            if not all_imgs:
                continue

            # frame_id → local index within this run
            fid_to_local = {}
            for local_idx, p in enumerate(all_imgs):
                fid = p.stem.split('_')[0]
                fid_to_local[fid] = local_idx
                self.all_image_paths.append(str(p))

            run_start = global_offset
            run_end = global_offset + len(all_imgs)
            self.run_boundaries.append((run_start, run_end))

            # Load velocity labels
            with open(csv_path) as f:
                for row in csv.DictReader(f):
                    fid = row['frame_id']
                    v = float(row['v_actual'])
                    omega = float(row['omega_actual'])
                    # cable_visible: default 1 for backward compat
                    cable_vis = int(row.get('cable_visible', 1))

                    if fid not in fid_to_local:
                        continue

                    # For negative samples (cable_visible=0), keep them
                    # (they ARE the training signal for confidence head).
                    # For positive samples, skip stationary frames.
                    if cable_vis == 1 and abs(v) < 0.01 and abs(omega) < 0.01:
                        continue

                    gidx = global_offset + fid_to_local[fid]
                    v_norm = np.clip(v / self.v_max, -1.0, 1.0)
                    omega_norm = np.clip(omega / self.omega_max, -1.0, 1.0)
                    self.samples.append((gidx, v_norm, omega_norm, cable_vis))

            global_offset += len(all_imgs)

        n_frames_total = len(self.all_image_paths)
        n_runs = len(self.run_boundaries)
        n_neg = sum(1 for s in self.samples if s[3] == 0)
        print(f'Loaded {len(self.samples)} samples '
              f'({n_frames_total} total frames, {n_neg} negatives) from {n_runs} runs')
        if not self.samples:
            raise ValueError('No samples found! Check data directories.')

    def _get_run_start(self, global_idx):
        for start, end in self.run_boundaries:
            if start <= global_idx < end:
                return start
        return global_idx

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        global_idx, v_norm, omega_norm, cable_vis = self.samples[idx]
        run_start = self._get_run_start(global_idx)

        # Augmentation params (consistent across all frames in this sample)
        do_flip = self.augment and random.random() < 0.5
        brightness = 1.0 + random.uniform(-0.20, 0.20) if self.augment else 1.0
        contrast = 1.0 + random.uniform(-0.15, 0.15) if self.augment else 1.0

        # Load N temporal frames: [t-(N-1), ..., t-1, t]
        channels = []
        for offset in range(-(self.n_frames - 1), 1):
            src_idx = max(run_start, global_idx + offset)

            img = cv2.imread(self.all_image_paths[src_idx])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.img_size, self.img_size))
            img = img.astype(np.float32) / 255.0

            # Augmentation (consistent flip + jitter across temporal frames)
            if do_flip:
                img = np.ascontiguousarray(img[:, ::-1, :])
            if self.augment:
                img = np.clip(
                    contrast * (img - 0.5) + 0.5 + (brightness - 1.0),
                    0.0, 1.0)

            # Red mask BEFORE noise (clean color signal)
            red_mask = extract_red_mask(img)

            # Gaussian noise (independent per frame — real camera behavior)
            if self.augment:
                img = np.clip(
                    img + np.random.randn(*img.shape).astype(np.float32) * 0.02,
                    0.0, 1.0)

            # ImageNet normalize RGB channels
            img_norm = (img - IMAGENET_MEAN) / IMAGENET_STD

            # Append 4 channels: [R, G, B, red_mask]
            channels.append(img_norm)
            channels.append(red_mask[:, :, np.newaxis])

        # Stack: N_FRAMES * 4ch = 12ch, HWC → CHW
        stacked = np.concatenate(channels, axis=-1).astype(np.float32)
        stacked = np.transpose(stacked, (2, 0, 1))

        if do_flip:
            omega_norm = -omega_norm

        velocity = np.array([v_norm, omega_norm], dtype=np.float32)
        cable_visible = np.float32(cable_vis)
        return torch.from_numpy(stacked), torch.from_numpy(velocity), cable_visible


# ============================================================================
# TRAINING
# ============================================================================

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # Load dataset
    dataset = CableTracerDataset(
        data_dirs=args.data, img_size=128,
        v_max=args.v_max, omega_max=args.omega_max, n_frames=N_FRAMES,
    )

    # Data statistics
    v_vals = [s[1] for s in dataset.samples if s[3] == 1]
    w_vals = [s[2] for s in dataset.samples if s[3] == 1]
    n_neg = sum(1 for s in dataset.samples if s[3] == 0)
    print(f'Cable-positive: {len(v_vals)}, Negatives: {n_neg}')
    if v_vals:
        print(f'v_norm range: [{min(v_vals):.2f}, {max(v_vals):.2f}], '
              f'mean={np.mean(v_vals):.2f}')
        print(f'w_norm range: [{min(w_vals):.2f}, {max(w_vals):.2f}], '
              f'mean={np.mean(w_vals):.2f}')

    # Split 85/15
    n_val = max(1, int(len(dataset) * 0.15))
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42))
    print(f'Train: {n_train} (augmented), Val: {n_val}')

    train_loader = DataLoader(
        train_set, batch_size=args.batch, shuffle=True,
        num_workers=4, pin_memory=True)
    val_loader = DataLoader(
        val_set, batch_size=args.batch, shuffle=False,
        num_workers=4, pin_memory=True)

    # Model
    model = CableTracerCNN(in_channels=IN_CHANNELS).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'\nArchitecture: Temporal ResNet-SE + CBAM + DualHead')
    print(f'Input: {N_FRAMES} frames x {CH_PER_FRAME}ch '
          f'(RGB+RedMask) = {IN_CHANNELS}ch')
    print(f'Params: {n_params:,}')
    print(f'Loss: MSE(v)*{args.v_weight:.1f} + MSE(w)*{args.omega_weight:.1f} '
          f'+ dir_penalty*{args.dir_weight:.1f} + BCE(conf)*{args.conf_weight:.1f}')
    print(f'EMA decay: {args.ema_decay}\n')

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5)
    ema = ModelEMA(model, decay=args.ema_decay)

    # Output
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    log_path = out_dir / 'training_log.csv'
    log_file = open(log_path, 'w', newline='')
    log_writer = csv.writer(log_file)
    log_writer.writerow(['epoch', 'train_loss', 'val_loss', 'lr',
                         'val_mse_v', 'val_mse_w', 'val_dir_match',
                         'val_conf_acc'])

    best_val_loss = float('inf')
    best_dir_match = 0.0

    for epoch in range(args.epochs):
        # --- Train ---
        model.train()
        dataset.augment = True
        train_losses = []
        for images, velocities, cable_vis in train_loader:
            images = images.to(device)
            velocities = velocities.to(device)
            cable_vis = cable_vis.to(device)

            preds = model(images)
            loss = cable_tracer_loss(
                preds, velocities, cable_visible=cable_vis,
                v_weight=args.v_weight,
                omega_weight=args.omega_weight,
                dir_weight=args.dir_weight,
                conf_weight=args.conf_weight)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            ema.update(model)
            train_losses.append(loss.item())

        avg_train = np.mean(train_losses)

        # --- Validate (with EMA weights) ---
        dataset.augment = False
        orig_state = {k: v.clone() for k, v in model.state_dict().items()}
        model.load_state_dict(ema.state_dict())
        model.eval()

        val_losses = []
        all_preds, all_targets, all_cable_vis = [], [], []
        with torch.no_grad():
            for images, velocities, cable_vis in val_loader:
                images = images.to(device)
                velocities = velocities.to(device)
                cable_vis = cable_vis.to(device)
                preds = model(images)
                loss = cable_tracer_loss(
                    preds, velocities, cable_visible=cable_vis,
                    v_weight=args.v_weight,
                    omega_weight=args.omega_weight,
                    dir_weight=args.dir_weight,
                    conf_weight=args.conf_weight)
                val_losses.append(loss.item())
                all_preds.append(preds.cpu())
                all_targets.append(velocities.cpu())
                all_cable_vis.append(cable_vis.cpu())

        # Restore training weights
        model.load_state_dict(orig_state)

        avg_val = np.mean(val_losses)
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # Per-component metrics
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        all_cable_vis = torch.cat(all_cable_vis)

        # Velocity metrics on cable-visible samples only
        vis_mask = all_cable_vis > 0.5
        if vis_mask.sum() > 0:
            mse_v = F.mse_loss(all_preds[vis_mask, 0], all_targets[vis_mask, 0]).item()
            mse_w = F.mse_loss(all_preds[vis_mask, 1], all_targets[vis_mask, 1]).item()
            dir_match = (torch.sign(all_preds[vis_mask, 1]) == torch.sign(
                all_targets[vis_mask, 1])).float().mean().item() * 100
        else:
            mse_v = mse_w = 0.0
            dir_match = 0.0

        # Confidence accuracy
        pred_cable = (all_preds[:, 2] > 0.5).float()
        conf_acc = (pred_cable == all_cable_vis).float().mean().item() * 100

        log_writer.writerow([
            epoch, f'{avg_train:.6f}', f'{avg_val:.6f}', f'{current_lr:.6f}',
            f'{mse_v:.6f}', f'{mse_w:.6f}', f'{dir_match:.1f}',
            f'{conf_acc:.1f}'])

        if epoch % 5 == 0 or avg_val < best_val_loss or dir_match > best_dir_match:
            print(f'Epoch {epoch:3d}/{args.epochs} | '
                  f'train={avg_train:.5f} val={avg_val:.5f} '
                  f'mse_v={mse_v:.5f} mse_w={mse_w:.5f} '
                  f'dir={dir_match:.0f}% conf={conf_acc:.0f}% lr={current_lr:.1e}')

        # Save best by val loss (EMA weights)
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(ema.state_dict(), out_dir / 'cable_tracer.pt')
            print(f'  -> Saved best model (val_loss={best_val_loss:.5f})')

        # Also track best direction match
        if dir_match > best_dir_match:
            best_dir_match = dir_match
            torch.save(ema.state_dict(), out_dir / 'cable_tracer_best_dir.pt')
            print(f'  -> Saved best direction model (dir={best_dir_match:.0f}%)')

    log_file.close()

    # Save final epoch model
    final_name = f'cable_tracer_epoch{args.epochs}_final.pt'
    torch.save(ema.state_dict(), out_dir / final_name)
    print(f'Final epoch saved: {final_name}')

    # Save config
    config = {
        'model': 'CableTracerCNN_v4_temporal_SE_CBAM_confidence',
        'params': n_params,
        'img_size': 128,
        'n_frames': N_FRAMES,
        'in_channels': IN_CHANNELS,
        'v_max': args.v_max,
        'omega_max': args.omega_max,
        'v_weight': args.v_weight,
        'omega_weight': args.omega_weight,
        'dir_weight': args.dir_weight,
        'conf_weight': args.conf_weight,
        'ema_decay': args.ema_decay,
        'best_val_loss': best_val_loss,
        'best_dir_match': best_dir_match,
        'epochs_trained': args.epochs,
        'train_samples': n_train,
        'val_samples': n_val,
    }
    with open(out_dir / 'cable_tracer_config.json', 'w') as f:
        json.dump(config, f, indent=2)

    print(f'\nDone! Model saved to {out_dir}/cable_tracer.pt')
    print(f'Best val loss: {best_val_loss:.5f}')
    print(f'Best direction match: {best_dir_match:.0f}%')


def main():
    parser = argparse.ArgumentParser(
        description='Train Cable Tracer v3 — Temporal ResNet-SE + CBAM')
    parser.add_argument('--data', nargs='+', required=True,
                        help='Data directories from rgb_velocity_recorder')
    parser.add_argument('--output', default='models/cable_tracer',
                        help='Output directory for model + logs')
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--v_max', type=float, default=0.50)
    parser.add_argument('--omega_max', type=float, default=0.15)
    parser.add_argument('--v_weight', type=float, default=1.0,
                        help='Loss weight for linear velocity')
    parser.add_argument('--omega_weight', type=float, default=3.0,
                        help='Loss weight for angular velocity')
    parser.add_argument('--dir_weight', type=float, default=2.0,
                        help='Loss weight for direction consistency penalty')
    parser.add_argument('--conf_weight', type=float, default=1.0,
                        help='Loss weight for confidence/cable_visible BCE')
    parser.add_argument('--ema_decay', type=float, default=0.999,
                        help='EMA decay for weight averaging')
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
