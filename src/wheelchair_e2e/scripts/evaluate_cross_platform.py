#!/usr/bin/env python3
"""
Cross-Platform Trajectory Evaluation — Cable Tracer v3.

Loads the wheelchair-trained v3 model and evaluates it on wheelchair, UGV, and
quadruped datasets by feeding images one-by-one through the model, integrating
predicted velocities into XY trajectories, and comparing against ground truth.

Ground truth:
  - Wheelchair: direct x,y,theta from odometry CSV (also shows integrated GT)
  - UGV/Quadruped: integrate recorded velx,velw to obtain GT trajectory

Output per dataset:
  - GT trajectory (green solid) vs Predicted trajectory (red dashed)
  - Velocity time series (v and omega: GT vs predicted)
  - Metrics: MSE, direction match, ATE, endpoint error

Usage:
    python3 evaluate_cross_platform.py
    python3 evaluate_cross_platform.py --model models/cable_tracer/cable_tracer.pt
    python3 evaluate_cross_platform.py --datasets wheelchair ugv
"""

import argparse
import collections
import csv
import json
import os
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# v3 MODEL — 2 outputs [v, omega], no confidence head
# ============================================================================

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

N_FRAMES = 3
CH_PER_FRAME = 4
IN_CHANNELS = N_FRAMES * CH_PER_FRAME


class SEBlock(nn.Module):
    def __init__(self, ch, reduction=4):
        super().__init__()
        mid = max(ch // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(ch, mid), nn.ReLU(),
            nn.Linear(mid, ch), nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.shape
        w = self.pool(x).view(b, c)
        return x * self.fc(w).view(b, c, 1, 1)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.bn = nn.BatchNorm2d(1)

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        mx = x.max(dim=1, keepdim=True)[0]
        return x * torch.sigmoid(self.bn(self.conv(torch.cat([avg, mx], dim=1))))


class ResBlockSE(nn.Module):
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
        return self.pool(F.relu(out + identity))


class CableTracerV3(nn.Module):
    """v3: outputs [v, omega] only (no confidence head)."""
    def __init__(self, in_channels=IN_CHANNELS):
        super().__init__()
        self.block1 = ResBlockSE(in_channels, 24)
        self.block2 = ResBlockSE(24, 48)
        self.block3 = ResBlockSE(48, 64)
        self.spatial_attn = SpatialAttention()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.v_head = nn.Sequential(
            nn.Linear(64, 32), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(32, 1), nn.Tanh())
        self.omega_head = nn.Sequential(
            nn.Linear(64, 32), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(32, 1), nn.Tanh())

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.spatial_attn(x)
        x = self.pool(x).flatten(1)
        return torch.cat([self.v_head(x), self.omega_head(x)], dim=1)


# ============================================================================
# PREPROCESSING
# ============================================================================

def extract_red_mask(img_rgb_01):
    img_u8 = (img_rgb_01 * 255).astype(np.uint8)
    hsv = cv2.cvtColor(img_u8, cv2.COLOR_RGB2HSV)
    m1 = cv2.inRange(hsv, (0, 70, 50), (12, 255, 255))
    m2 = cv2.inRange(hsv, (168, 70, 50), (180, 255, 255))
    return ((m1 | m2) > 0).astype(np.float32)


def preprocess_frame(img_rgb, img_size=128):
    """Single frame → 4ch: ImageNet-normalized RGB + red_mask."""
    img = cv2.resize(img_rgb, (img_size, img_size)).astype(np.float32) / 255.0
    red_mask = extract_red_mask(img)
    img_norm = (img - IMAGENET_MEAN) / IMAGENET_STD
    return np.concatenate([img_norm, red_mask[:, :, np.newaxis]], axis=-1)


def integrate_trajectory(v_arr, omega_arr, dt):
    """Integrate velocity → XY trajectory. Returns (xs, ys, thetas) arrays."""
    x, y, theta = 0.0, 0.0, 0.0
    xs, ys, thetas = [0.0], [0.0], [0.0]
    dts = dt if isinstance(dt, (list, np.ndarray)) else [dt] * len(v_arr)
    for v, omega, t in zip(v_arr, omega_arr, dts):
        theta += omega * t
        x += v * np.cos(theta) * t
        y += v * np.sin(theta) * t
        xs.append(x)
        ys.append(y)
        thetas.append(theta)
    return np.array(xs), np.array(ys), np.array(thetas)


# ============================================================================
# FRAME ITERATORS — memory-efficient, one frame at a time
# ============================================================================

def wheelchair_frames(data_dir):
    """Yield BGR frames from wheelchair images/ directory in order."""
    data_dir = Path(data_dir)
    img_dir = data_dir / 'images'
    for img_path in sorted(img_dir.glob('*.jpg'), key=lambda p: p.stem.split('_')[0]):
        img = cv2.imread(str(img_path))
        if img is not None:
            yield img


def video_frames(video_path):
    """Yield BGR frames from an MP4 video file."""
    cap = cv2.VideoCapture(str(video_path))
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yield frame
    cap.release()


# ============================================================================
# DATA LOADERS — return (frame_iterator, gt_v, gt_omega, gt_xy_or_None, dt, name, n_frames)
# ============================================================================

def load_wheelchair(data_dir):
    """Load wheelchair dataset. GT from direct x,y,theta in CSV."""
    data_dir = Path(data_dir)
    csv_path = data_dir / 'velocities.csv'

    gt_v, gt_omega = [], []
    gt_x, gt_y, gt_theta = [], [], []
    frame_ids = []

    with open(csv_path) as f:
        for row in csv.DictReader(f):
            frame_ids.append(row['frame_id'])
            gt_v.append(float(row['v_actual']))
            gt_omega.append(float(row['omega_actual']))
            if 'x' in row and 'y' in row:
                gt_x.append(float(row['x']))
                gt_y.append(float(row['y']))
                gt_theta.append(float(row['theta']))

    has_pose = len(gt_x) == len(gt_v) and len(gt_x) > 0
    gt_xy = (np.array(gt_x), np.array(gt_y), np.array(gt_theta)) if has_pose else None

    return (lambda: wheelchair_frames(data_dir),
            np.array(gt_v), np.array(gt_omega), gt_xy, 0.1, data_dir.name,
            len(frame_ids))


UGV_DEG_RECS = {22, 23}

def load_ugv(video_path, csv_path, rec_num):
    """Load UGV dataset. GT from integrated velx/velw."""
    csv_rows = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            t = float(row['time'])
            vx = float(row['velx'])
            vw = float(row['velw'])
            if rec_num in UGV_DEG_RECS:
                vw /= 57.2957795
            vx = np.clip(vx, -0.5, 0.5)
            vw = np.clip(vw, -1.0, 1.0)
            csv_rows.append((t, vx, vw))

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if n_frames <= 0:
        return None

    # Sync each video frame to closest CSV row
    gt_v, gt_omega = [], []
    for fi in range(n_frames):
        frame_time = fi / fps
        best_idx = min(range(len(csv_rows)),
                       key=lambda ci: abs(csv_rows[ci][0] - frame_time))
        _, vx, vw = csv_rows[best_idx]
        gt_v.append(vx)
        gt_omega.append(vw)

    dt = 1.0 / fps
    name = f'ugv_rec{rec_num}'
    return (lambda vp=video_path: video_frames(vp),
            np.array(gt_v), np.array(gt_omega), None, dt, name, n_frames)


def load_quadruped(video_path, csv_path, name):
    """Load quadruped FV dataset. GT from integrated velocities."""
    csv_rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        # Detect columns
        time_col = v_col = w_col = None
        for h in headers:
            hl = h.lower().strip()
            if hl in ('time', 'timestamp', 't'):
                time_col = h
            elif hl in ('velx', 'vx', 'v', 'v_actual'):
                v_col = h
            elif hl in ('velw', 'vw', 'omega', 'omega_actual', 'w'):
                w_col = h
        if not all([time_col, v_col, w_col]) and len(headers) >= 5:
            time_col, v_col, w_col = headers[0], headers[3], headers[4]
        if not all([time_col, v_col, w_col]):
            return None
        for row in reader:
            try:
                t = float(row[time_col])
                vx = np.clip(float(row[v_col]), -0.5, 0.5)
                vw = np.clip(float(row[w_col]), -1.0, 1.0)
                csv_rows.append((t, vx, vw))
            except (ValueError, KeyError):
                continue

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if n_frames <= 0 or not csv_rows:
        return None

    gt_v, gt_omega = [], []
    for fi in range(n_frames):
        frame_time = fi / fps
        best_idx = min(range(len(csv_rows)),
                       key=lambda ci: abs(csv_rows[ci][0] - frame_time))
        _, vx, vw = csv_rows[best_idx]
        gt_v.append(vx)
        gt_omega.append(vw)

    dt = 1.0 / fps
    return (lambda vp=video_path: video_frames(vp),
            np.array(gt_v), np.array(gt_omega), None, dt, name, n_frames)


# ============================================================================
# INFERENCE — sequential frame-by-frame with temporal buffer
# ============================================================================

@torch.no_grad()
def run_sequential_inference(model, frame_iter, v_max, omega_max, device,
                             n_total=None):
    """Feed frames one-by-one through model. Returns (pred_v, pred_omega) arrays."""
    frame_buffer = collections.deque(maxlen=N_FRAMES)
    pred_v, pred_omega = [], []

    for i, frame_bgr in enumerate(frame_iter):
        img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        processed = preprocess_frame(img_rgb)  # 128x128x4
        frame_buffer.append(processed)

        while len(frame_buffer) < N_FRAMES:
            frame_buffer.appendleft(frame_buffer[0])

        stacked = np.concatenate(list(frame_buffer), axis=-1)  # 128x128x12
        tensor = torch.from_numpy(
            np.transpose(stacked, (2, 0, 1))[np.newaxis].astype(np.float32)
        ).to(device)

        pred = model(tensor).cpu().numpy()[0]
        pred_v.append(float(pred[0]) * v_max)
        pred_omega.append(float(pred[1]) * omega_max)

        if n_total and (i + 1) % 200 == 0:
            print(f'      frame {i+1}/{n_total}')

    return np.array(pred_v), np.array(pred_omega)


# ============================================================================
# PLOTTING
# ============================================================================

def plot_dataset(name, gt_v, gt_omega, pred_v, pred_omega, gt_xy, dt, out_path):
    """Per-dataset evaluation: trajectory + velocity time series + metrics."""
    n = min(len(pred_v), len(gt_v))
    pred_v = pred_v[:n]
    pred_omega = pred_omega[:n]
    gt_v = gt_v[:n]
    gt_omega = gt_omega[:n]

    # Integrate predicted trajectory
    pred_xs, pred_ys, _ = integrate_trajectory(pred_v, pred_omega, dt)

    # Ground truth trajectory
    if gt_xy is not None:
        gt_xs, gt_ys, gt_thetas = gt_xy[0][:n], gt_xy[1][:n], gt_xy[2][:n]
        # Shift so start is at origin (match integrated pred which starts at 0,0)
        gt_xs = gt_xs - gt_xs[0]
        gt_ys = gt_ys - gt_ys[0]
        gt_label = 'GT (odometry)'
        # Also show integrated GT for comparison
        igt_xs, igt_ys, _ = integrate_trajectory(gt_v, gt_omega, dt)
    else:
        gt_xs, gt_ys, _ = integrate_trajectory(gt_v, gt_omega, dt)
        gt_label = 'GT (integrated)'
        igt_xs = igt_ys = None

    # Ensure same length for comparison
    n_traj = min(len(pred_xs), len(gt_xs))
    pred_xs_c = pred_xs[:n_traj]
    pred_ys_c = pred_ys[:n_traj]
    gt_xs_c = gt_xs[:n_traj]
    gt_ys_c = gt_ys[:n_traj]

    # Metrics
    mse_v = np.mean((pred_v - gt_v) ** 2)
    mse_omega = np.mean((pred_omega - gt_omega) ** 2)

    sig = np.abs(gt_omega) > 0.005
    dir_match = (np.mean(np.sign(pred_omega[sig]) == np.sign(gt_omega[sig])) * 100
                 if sig.sum() > 0 else 0.0)

    ate = np.mean(np.sqrt((pred_xs_c - gt_xs_c)**2 + (pred_ys_c - gt_ys_c)**2))
    endpoint_err = np.sqrt((pred_xs_c[-1] - gt_xs_c[-1])**2 +
                           (pred_ys_c[-1] - gt_ys_c[-1])**2)
    gt_len = np.sum(np.sqrt(np.diff(gt_xs_c)**2 + np.diff(gt_ys_c)**2))

    # --- Figure ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1. Trajectory
    ax = axes[0, 0]
    ax.plot(gt_xs_c, gt_ys_c, 'g-', lw=2.5, alpha=0.8, label=gt_label)
    if igt_xs is not None:
        n_i = min(len(igt_xs), n_traj)
        ax.plot(igt_xs[:n_i], igt_ys[:n_i], 'b--', lw=1.5, alpha=0.5,
                label='GT (integrated vel)')
    ax.plot(pred_xs_c, pred_ys_c, 'r--', lw=2, alpha=0.85, label='Predicted')
    ax.plot(0, 0, 'ko', ms=10, zorder=5, label='Start')
    ax.plot(gt_xs_c[-1], gt_ys_c[-1], 'gs', ms=10, zorder=5, label='GT End')
    ax.plot(pred_xs_c[-1], pred_ys_c[-1], 'r^', ms=10, zorder=5, label='Pred End')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(f'Trajectory — {name}\n'
                 f'ATE={ate:.4f}m  EndErr={endpoint_err:.4f}m  Length={gt_len:.3f}m')
    ax.legend(fontsize=8, loc='best')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # 2. Linear velocity time series
    ax = axes[0, 1]
    t = np.arange(n) * dt
    ax.plot(t, gt_v, 'g-', alpha=0.6, lw=1, label='GT v')
    ax.plot(t, pred_v, 'r-', alpha=0.7, lw=1, label='Pred v')
    ax.fill_between(t, gt_v, pred_v, alpha=0.15, color='orange')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('v (m/s)')
    ax.set_title(f'Linear Velocity — MSE={mse_v:.5f}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Angular velocity time series
    ax = axes[1, 0]
    ax.plot(t, gt_omega, 'g-', alpha=0.6, lw=1, label='GT omega')
    ax.plot(t, pred_omega, 'r-', alpha=0.7, lw=1, label='Pred omega')
    ax.fill_between(t, gt_omega, pred_omega, alpha=0.15, color='orange')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('omega (rad/s)')
    ax.set_title(f'Angular Velocity — MSE={mse_omega:.5f}  Dir={dir_match:.0f}%')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Metrics table
    ax = axes[1, 1]
    ax.axis('off')
    metrics = (
        f"Dataset:  {name}\n"
        f"Frames:   {n}\n"
        f"Duration: {n*dt:.1f}s  (dt={dt:.4f}s)\n"
        f"GT Traj:  {gt_len:.3f}m\n"
        f"\n"
        f"--- Velocity ---\n"
        f"MSE v:      {mse_v:.6f}\n"
        f"MSE omega:  {mse_omega:.6f}\n"
        f"MAE v:      {np.mean(np.abs(pred_v - gt_v)):.5f}\n"
        f"MAE omega:  {np.mean(np.abs(pred_omega - gt_omega)):.5f}\n"
        f"Dir match:  {dir_match:.1f}%\n"
        f"\n"
        f"--- Trajectory ---\n"
        f"ATE:        {ate:.4f}m\n"
        f"Endpoint:   {endpoint_err:.4f}m\n"
        f"ATE/Len:    {ate/max(gt_len, 1e-6)*100:.1f}%\n"
    )
    ax.text(0.05, 0.95, metrics, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    fig.suptitle(f'Cable Tracer v3 — {name}', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches='tight')
    plt.close(fig)

    return {
        'name': name, 'n': n, 'dt': dt,
        'mse_v': mse_v, 'mse_omega': mse_omega, 'dir_match': dir_match,
        'ate': ate, 'endpoint_err': endpoint_err, 'gt_len': gt_len,
        'pred_xs': pred_xs_c, 'pred_ys': pred_ys_c,
        'gt_xs': gt_xs_c, 'gt_ys': gt_ys_c,
    }


def plot_summary(results, out_path):
    """All datasets: overlaid trajectories + comparison table."""
    if not results:
        return

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    colors = plt.cm.Set1(np.linspace(0, 0.8, len(results)))

    # Left: all trajectories
    ax = axes[0]
    for i, r in enumerate(results):
        ax.plot(r['gt_xs'], r['gt_ys'], '-', color=colors[i], lw=2, alpha=0.5)
        ax.plot(r['pred_xs'], r['pred_ys'], '--', color=colors[i], lw=2, alpha=0.85,
                label=r['name'])
    ax.plot(0, 0, 'ko', ms=10, zorder=5)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('All Trajectories — solid=GT, dashed=Predicted')
    ax.legend(fontsize=9)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # Right: metrics table
    ax = axes[1]
    ax.axis('off')
    cols = ['Dataset', 'Frames', 'MSE_v', 'MSE_w', 'Dir%', 'ATE(m)', 'EndErr(m)', 'Len(m)']
    rows = []
    for r in results:
        rows.append([
            r['name'][:18], str(r['n']),
            f'{r["mse_v"]:.5f}', f'{r["mse_omega"]:.5f}',
            f'{r["dir_match"]:.0f}', f'{r["ate"]:.4f}',
            f'{r["endpoint_err"]:.4f}', f'{r["gt_len"]:.3f}',
        ])
    tbl = ax.table(cellText=rows, colLabels=cols, loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.3, 1.8)
    for j in range(len(cols)):
        tbl[0, j].set_facecolor('#2E75B6')
        tbl[0, j].set_text_props(color='white', fontweight='bold')
    # Alternate row colors
    for i in range(1, len(rows) + 1):
        color = '#F2F2F2' if i % 2 == 0 else 'white'
        for j in range(len(cols)):
            tbl[i, j].set_facecolor(color)
    ax.set_title('Cross-Platform Metrics', fontsize=12, pad=20)

    fig.suptitle('Cable Tracer v3 — Cross-Platform Trajectory Evaluation',
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches='tight')
    plt.close(fig)


# ============================================================================
# DATA DISCOVERY
# ============================================================================

WHEELCHAIR_DATA = Path('/home/sidd/wheelchair_nav/data')
UGV_DATA = Path('/opt/nvidia/home_data/Downloads/UGV_data/UGV_Data')
QUAD_ZIP = Path('/opt/nvidia/home_data/Downloads/Quadruped_data-20260318T100924Z-1-001.zip')
QUAD_RAR_DIR = Path('/opt/nvidia/home_data/Downloads/Quadruped_data')

UGV_RECS = [2, 22, 23]


def find_ugv_files(rec_num):
    """Find trial video + CSV for a UGV recording."""
    if not UGV_DATA.exists():
        return None, None

    # Exact match patterns: recording_N_ or recording_N. or recording_N/
    import re
    pattern = re.compile(rf'recording_{rec_num}[_./\s]', re.IGNORECASE)

    vid = None
    csv_f = None
    for f in UGV_DATA.rglob('*'):
        if not pattern.search(f.name) and not pattern.search(f.parent.name):
            continue
        fl = f.name.lower()
        if f.suffix.lower() == '.mp4' and 'av' not in fl:
            if 'trial' in fl or vid is None:
                vid = f
        elif f.suffix.lower() == '.csv':
            csv_f = f

    return vid, csv_f


def find_quadruped_data(output_base):
    """Extract quadruped data if needed, return list of (video, csv, name) tuples."""
    results = []

    # Check if RAR exists
    rar_path = QUAD_RAR_DIR / 'csv_av_fv_data_quadruped.rar'
    if not rar_path.exists() and QUAD_ZIP.exists():
        import zipfile
        print('  Extracting ZIP...')
        with zipfile.ZipFile(QUAD_ZIP) as zf:
            zf.extractall(QUAD_ZIP.parent)

    if not rar_path.exists():
        return results

    # Extract RAR
    raw_dir = output_base / '_quad_raw'
    if not raw_dir.exists():
        print('  Extracting RAR...')
        try:
            raw_dir.mkdir(parents=True, exist_ok=True)
            from unrar.cffi import rarfile as cffi_rarfile
            rf = cffi_rarfile.RarFile(str(rar_path))
            for info in rf.infolist():
                out_file = raw_dir / info.filename
                if info.is_dir():
                    out_file.mkdir(parents=True, exist_ok=True)
                    continue
                out_file.parent.mkdir(parents=True, exist_ok=True)
                out_file.write_bytes(rf.read(info.filename))
            print(f'  Extracted {len(rf.namelist())} files to {raw_dir}')
        except Exception as e:
            print(f'  RAR extraction failed: {e}')
            import traceback; traceback.print_exc()
            return results

    # Find FV (front view) videos with paired CSVs
    for f in sorted(raw_dir.rglob('*.mp4')):
        fl = f.name.lower()
        if 'fv' in fl and 'av' not in fl:
            csv_candidates = list(f.parent.glob('*.csv'))
            if csv_candidates:
                idx = len(results) + 1
                results.append((f, csv_candidates[0], f'quad_{idx}'))

    return results


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Cross-Platform Trajectory Evaluation — Cable Tracer v3')
    parser.add_argument('--model', default='models/cable_tracer/cable_tracer.pt')
    parser.add_argument('--output', default='models/cable_tracer/cross_platform_eval')
    parser.add_argument('--v_max', type=float, default=None)
    parser.add_argument('--omega_max', type=float, default=None)
    parser.add_argument('--datasets', nargs='+',
                        default=['wheelchair', 'ugv', 'quadruped'],
                        choices=['wheelchair', 'ugv', 'quadruped'])
    args = parser.parse_args()

    # Load config
    config_path = Path(args.model).parent / 'cable_tracer_config.json'
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        v_max = args.v_max or config.get('v_max', 0.50)
        omega_max = args.omega_max or config.get('omega_max', 0.15)
    else:
        v_max = args.v_max or 0.50
        omega_max = args.omega_max or 0.15

    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CableTracerV3(in_channels=IN_CHANNELS).to(device)
    model.load_state_dict(
        torch.load(args.model, map_location=device, weights_only=True))
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f'Model: {n_params:,} params on {device}')
    print(f'v_max={v_max}, omega_max={omega_max}\n')

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    # === WHEELCHAIR ===
    if 'wheelchair' in args.datasets:
        print('=== Wheelchair ===')
        for wc_dir in sorted(WHEELCHAIR_DATA.glob('rgb_vel_*')):
            csv_path = wc_dir / 'velocities.csv'
            if not csv_path.exists():
                continue
            print(f'  {wc_dir.name}')
            try:
                frame_fn, gt_v, gt_omega, gt_xy, dt, name, n_frames = \
                    load_wheelchair(wc_dir)
                print(f'    {n_frames} frames, dt={dt}s')
                print(f'    Running inference...')
                pred_v, pred_omega = run_sequential_inference(
                    model, frame_fn(), v_max, omega_max, device, n_frames)
                out_path = out_dir / f'traj_{name}.png'
                result = plot_dataset(
                    name, gt_v, gt_omega, pred_v, pred_omega, gt_xy, dt, out_path)
                all_results.append(result)
                print(f'    ATE={result["ate"]:.4f}m  Dir={result["dir_match"]:.0f}%  '
                      f'-> {out_path.name}')
            except Exception as e:
                print(f'    ERROR: {e}')
                import traceback; traceback.print_exc()

    # === UGV ===
    if 'ugv' in args.datasets:
        print('\n=== UGV ===')
        for rec in UGV_RECS:
            vid, csv_f = find_ugv_files(rec)
            if vid is None or csv_f is None:
                print(f'  rec{rec}: not found, skipping')
                continue
            print(f'  rec{rec}: {vid.name}')
            try:
                data = load_ugv(vid, csv_f, rec)
                if data is None:
                    print(f'    ERROR: Could not load')
                    continue
                frame_fn, gt_v, gt_omega, gt_xy, dt, name, n_frames = data
                print(f'    {n_frames} frames, dt={dt:.4f}s ({1/dt:.1f} fps)')
                print(f'    Running inference...')
                pred_v, pred_omega = run_sequential_inference(
                    model, frame_fn(), v_max, omega_max, device, n_frames)
                out_path = out_dir / f'traj_{name}.png'
                result = plot_dataset(
                    name, gt_v, gt_omega, pred_v, pred_omega, gt_xy, dt, out_path)
                all_results.append(result)
                print(f'    ATE={result["ate"]:.4f}m  Dir={result["dir_match"]:.0f}%  '
                      f'-> {out_path.name}')
            except Exception as e:
                print(f'    ERROR: {e}')
                import traceback; traceback.print_exc()

    # === QUADRUPED ===
    if 'quadruped' in args.datasets:
        print('\n=== Quadruped ===')
        quad_data = find_quadruped_data(out_dir)
        if not quad_data:
            print('  No quadruped data available')
        for vid, csv_f, name in quad_data:
            print(f'  {name}: {vid.name}')
            try:
                data = load_quadruped(vid, csv_f, name)
                if data is None:
                    print(f'    ERROR: Could not load')
                    continue
                frame_fn, gt_v, gt_omega, gt_xy, dt, name, n_frames = data
                print(f'    {n_frames} frames, dt={dt:.4f}s ({1/dt:.1f} fps)')
                print(f'    Running inference...')
                pred_v, pred_omega = run_sequential_inference(
                    model, frame_fn(), v_max, omega_max, device, n_frames)
                out_path = out_dir / f'traj_{name}.png'
                result = plot_dataset(
                    name, gt_v, gt_omega, pred_v, pred_omega, gt_xy, dt, out_path)
                all_results.append(result)
                print(f'    ATE={result["ate"]:.4f}m  Dir={result["dir_match"]:.0f}%  '
                      f'-> {out_path.name}')
            except Exception as e:
                print(f'    ERROR: {e}')
                import traceback; traceback.print_exc()

    # === SUMMARY ===
    if all_results:
        print(f'\n{"="*75}')
        print(f'{"Dataset":<20} {"N":>6} {"MSE_v":>9} {"MSE_w":>9} '
              f'{"Dir%":>5} {"ATE(m)":>8} {"EndErr":>8} {"Len(m)":>7}')
        print('-' * 75)
        for r in all_results:
            print(f'{r["name"]:<20} {r["n"]:>6} {r["mse_v"]:>9.5f} '
                  f'{r["mse_omega"]:>9.5f} {r["dir_match"]:>5.0f} '
                  f'{r["ate"]:>8.4f} {r["endpoint_err"]:>8.4f} '
                  f'{r["gt_len"]:>7.3f}')

        summary_path = out_dir / 'cross_platform_summary.png'
        plot_summary(all_results, summary_path)
        print(f'\nSummary: {summary_path}')
        print(f'All plots: {out_dir}/')
    else:
        print('\nNo datasets evaluated!')


if __name__ == '__main__':
    main()
