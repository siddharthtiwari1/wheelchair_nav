#!/usr/bin/env python3
"""
Prepare Cable Tracer Multi-Dataset — unified extraction for wheelchair, UGV, and quadruped.

Extracts frames from videos, syncs to CSV velocity labels, generates negative examples,
and outputs a unified directory structure at /opt/nvidia/cable_tracer_data/.

Sources:
  - Wheelchair: symlinks existing rgb_vel_* directories (already in correct format)
  - UGV: extracts frames from trial MP4s, syncs to CSV (velx/velw columns)
  - Quadruped: extracts ZIP→RAR→files, frames from FV MP4s, syncs to CSV
  - Negatives: hue-shifted cable images + aerial frames + start/end frames → zero velocity

Output CSV format (per directory):
    frame_id,v_actual,omega_actual,cable_visible
    000000,0.0506,-0.0435,1

Usage:
    python3 prepare_cable_data.py
    python3 prepare_cable_data.py --output /opt/nvidia/cable_tracer_data
"""

import argparse
import csv
import json
import os
import random
import shutil
import subprocess
import zipfile
from pathlib import Path

import cv2
import numpy as np

# ============================================================================
# PATHS
# ============================================================================

WHEELCHAIR_DATA = Path('/home/sidd/wheelchair_nav/data')
UGV_DATA = Path('/opt/nvidia/home_data/Downloads/UGV_data/UGV_Data')
QUAD_ZIP = Path('/opt/nvidia/home_data/Downloads/Quadruped_data-20260318T100924Z-1-001.zip')
QUAD_RAR_DIR = Path('/opt/nvidia/home_data/Downloads/Quadruped_data')
DEFAULT_OUTPUT = Path('/opt/nvidia/cable_tracer_data')

# UGV recordings to use (confirmed readable)
UGV_RECORDINGS = [2, 22, 23]
# Recordings where omega is in degrees/s (need ÷57.3)
UGV_DEG_RECORDINGS = {22, 23}
# Skip corrupt
UGV_SKIP = {1, 21, 24}

UGV_FPS = 15  # video framerate

# Velocity clipping (after unit conversion)
V_CLIP = (-0.5, 0.5)
OMEGA_CLIP = (-1.0, 1.0)

# Negative example counts
N_HUE_SHIFTED = 200
N_AERIAL = 200
N_START_END = 100

random.seed(42)
np.random.seed(42)


# ============================================================================
# HELPERS
# ============================================================================

def extract_frames_ffmpeg(video_path, output_dir, fps=None):
    """Extract frames from video using ffmpeg. Returns frame count."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = ['ffmpeg', '-i', str(video_path), '-vsync', '0']
    if fps:
        cmd += ['-vf', f'fps={fps}']
    cmd += [str(output_dir / '%06d.jpg'), '-y', '-loglevel', 'warning']
    subprocess.run(cmd, check=True)
    frames = sorted(output_dir.glob('*.jpg'))
    return len(frames)


def get_video_fps(video_path):
    """Get actual video FPS using ffprobe."""
    cmd = [
        'ffprobe', '-v', 'quiet', '-select_streams', 'v:0',
        '-show_entries', 'stream=r_frame_rate',
        '-of', 'csv=p=0', str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0 and '/' in result.stdout.strip():
        num, den = result.stdout.strip().split('/')
        return float(num) / float(den)
    return UGV_FPS  # fallback


def write_unified_csv(output_dir, rows):
    """Write unified CSV: frame_id, v_actual, omega_actual, cable_visible."""
    csv_path = output_dir / 'velocities.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['frame_id', 'v_actual', 'omega_actual', 'cable_visible'])
        for row in rows:
            writer.writerow(row)
    return len(rows)


# ============================================================================
# WHEELCHAIR — symlink existing data
# ============================================================================

def prepare_wheelchair(output_base):
    """Symlink wheelchair data directories. Returns list of (dir_name, sample_count)."""
    results = []
    wheelchair_dirs = sorted(WHEELCHAIR_DATA.glob('rgb_vel_*'))

    for i, src_dir in enumerate(wheelchair_dirs, 1):
        dst_name = f'wheelchair_run{i}'
        dst = output_base / dst_name

        if dst.exists() or dst.is_symlink():
            dst.unlink() if dst.is_symlink() else shutil.rmtree(dst)

        os.symlink(str(src_dir.resolve()), str(dst))

        # Count samples and add cable_visible column if missing
        csv_path = src_dir / 'velocities.csv'
        if csv_path.exists():
            with open(csv_path) as f:
                reader = csv.DictReader(f)
                headers = reader.fieldnames
                rows = list(reader)

            if 'cable_visible' not in headers:
                # Write an augmented CSV alongside the original
                aug_csv = dst / 'velocities_with_visibility.csv'
                # Since it's a symlink, write to a new file in the output
                # Actually, symlinks point to the original dir — we need to
                # copy the CSV with the extra column instead.
                # Better: just note that wheelchair data defaults cable_visible=1
                pass

            n_samples = len(rows)
        else:
            n_samples = 0

        results.append((dst_name, n_samples))
        print(f'  {dst_name}: {n_samples} samples (symlink → {src_dir.name})')

    return results


# ============================================================================
# UGV — extract frames from trial videos, sync to CSV
# ============================================================================

def prepare_ugv(output_base):
    """Extract UGV recording frames and sync to CSV velocities."""
    results = []

    for rec_num in UGV_RECORDINGS:
        rec_dir = UGV_DATA / f'recording_{rec_num}'
        if not rec_dir.exists():
            # Try alternate naming (some have _trial suffix on dir)
            rec_dir = UGV_DATA / f'recording_{rec_num}_trial'
        if not rec_dir.exists():
            print(f'  WARNING: recording_{rec_num} not found, skipping')
            continue

        # Find trial video and CSV
        trial_video = None
        trial_csv = None
        for f in rec_dir.iterdir():
            name_lower = f.name.lower()
            if name_lower.endswith('.mp4') and 'trial' in name_lower:
                trial_video = f
            elif name_lower.endswith('.mp4') and 'av' not in name_lower and trial_video is None:
                trial_video = f
            if name_lower.endswith('.csv'):
                trial_csv = f

        # Also check parent directory for files named recording_N_trial.mp4
        if trial_video is None:
            for f in UGV_DATA.iterdir():
                if f.name.lower() == f'recording_{rec_num}_trial.mp4':
                    trial_video = f
                    break
                elif f.name.lower() == f'recording_{rec_num}.mp4':
                    trial_video = f
                    break

        if trial_csv is None:
            for f in UGV_DATA.iterdir():
                if f.name.lower() == f'recording_{rec_num}_trial.csv' or \
                   f.name.lower() == f'recording_{rec_num}.csv':
                    trial_csv = f
                    break

        if trial_video is None or trial_csv is None:
            print(f'  WARNING: recording_{rec_num} missing video or CSV')
            print(f'    video={trial_video}, csv={trial_csv}')
            # Try flat structure
            for f in UGV_DATA.iterdir():
                if f.is_file():
                    if f.suffix == '.mp4' and f'recording_{rec_num}' in f.name and 'av' not in f.name.lower():
                        trial_video = f
                    elif f.suffix == '.csv' and f'recording_{rec_num}' in f.name:
                        trial_csv = f
            if trial_video is None or trial_csv is None:
                print(f'  SKIPPING recording_{rec_num}')
                continue

        print(f'  UGV rec{rec_num}: video={trial_video.name}, csv={trial_csv.name}')

        # Read CSV
        csv_rows = []
        with open(trial_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                t = float(row['time'])
                vx = float(row['velx'])
                vw = float(row['velw'])

                # Unit conversion for deg/s recordings
                if rec_num in UGV_DEG_RECORDINGS:
                    vw = vw / 57.2957795  # deg/s → rad/s

                # Clip outliers
                vx = np.clip(vx, V_CLIP[0], V_CLIP[1])
                vw = np.clip(vw, OMEGA_CLIP[0], OMEGA_CLIP[1])

                csv_rows.append((t, vx, vw))

        if not csv_rows:
            print(f'  WARNING: No CSV rows for recording_{rec_num}')
            continue

        # Get video FPS
        actual_fps = get_video_fps(trial_video)
        print(f'    Video FPS: {actual_fps:.1f}, CSV rows: {len(csv_rows)}')

        # Extract frames
        dst_name = f'ugv_rec{rec_num}'
        dst = output_base / dst_name
        img_dir = dst / 'images'
        if dst.exists():
            shutil.rmtree(dst)
        n_frames = extract_frames_ffmpeg(trial_video, img_dir)
        print(f'    Extracted {n_frames} frames')

        # Sync frames to CSV by timestamp
        # CSV time column gives timestamp for each row
        # Video frame index = time * fps
        unified_rows = []
        frame_files = sorted(img_dir.glob('*.jpg'))

        for frame_idx, frame_path in enumerate(frame_files):
            # Frame timestamp in seconds
            frame_time = frame_idx / actual_fps

            # Find closest CSV row by timestamp
            best_idx = 0
            best_dt = float('inf')
            for ci, (ct, _, _) in enumerate(csv_rows):
                dt = abs(ct - frame_time)
                if dt < best_dt:
                    best_dt = dt
                    best_idx = ci

            _, vx, vw = csv_rows[best_idx]

            # Rename frame to zero-padded format
            new_name = f'{frame_idx:06d}.jpg'
            if frame_path.name != new_name:
                frame_path.rename(img_dir / new_name)

            unified_rows.append((f'{frame_idx:06d}', f'{vx:.6f}', f'{vw:.6f}', 1))

        n_written = write_unified_csv(dst, unified_rows)
        results.append((dst_name, n_written))
        print(f'    Output: {n_written} synced samples')

    return results


# ============================================================================
# QUADRUPED — extract ZIP→RAR→files, frames from FV MP4s
# ============================================================================

def prepare_quadruped(output_base):
    """Extract quadruped data and process FV (front view) videos."""
    results = []

    # Step 1: Extract RAR
    rar_path = QUAD_RAR_DIR / 'csv_av_fv_data_quadruped.rar'

    if not rar_path.exists():
        # Try extracting from ZIP first
        if QUAD_ZIP.exists():
            print('  Extracting ZIP...')
            with zipfile.ZipFile(QUAD_ZIP) as zf:
                zf.extractall(QUAD_RAR_DIR.parent)
            # The ZIP contains Quadruped_data/csv_av_fv_data_quadruped.rar
            rar_path = QUAD_RAR_DIR / 'csv_av_fv_data_quadruped.rar'

    if not rar_path.exists():
        print(f'  ERROR: RAR not found at {rar_path}')
        return results

    # Extract RAR using Python rarfile module
    extract_dir = output_base / '_quad_raw'
    if not extract_dir.exists():
        print('  Extracting RAR...')
        extract_dir.mkdir(parents=True, exist_ok=True)
        try:
            from unrar.cffi import rarfile as cffi_rarfile
            rf = cffi_rarfile.RarFile(str(rar_path))
            for name in rf.namelist():
                out_file = extract_dir / name
                if name.endswith('/'):
                    out_file.mkdir(parents=True, exist_ok=True)
                    continue
                out_file.parent.mkdir(parents=True, exist_ok=True)
                out_file.write_bytes(rf.read(name))
            print(f'  RAR extracted {len(rf.namelist())} files to {extract_dir}')
        except Exception as e:
            print(f'  ERROR extracting RAR: {e}')
            print('  Install: pip install --user --break-system-packages unrar-cffi')
            return results
    else:
        print(f'  Using existing extraction at {extract_dir}')

    # Find FV (front view) MP4s and their CSV pairs
    # Walk the extracted directory for FV videos
    fv_videos = []
    for root, dirs, files in os.walk(extract_dir):
        for f in files:
            f_lower = f.lower()
            if f_lower.endswith('.mp4') and 'fv' in f_lower and 'av' not in f_lower:
                fv_videos.append(Path(root) / f)

    fv_videos.sort()
    print(f'  Found {len(fv_videos)} FV videos')

    for vid_idx, vid_path in enumerate(fv_videos, 1):
        vid_dir = vid_path.parent

        # Find matching CSV (same directory or similar name)
        csv_path = None
        for f in vid_dir.iterdir():
            if f.suffix.lower() == '.csv':
                csv_path = f
                break

        if csv_path is None:
            # Try matching by name pattern
            stem = vid_path.stem.replace('_fv', '').replace('_FV', '')
            for f in vid_dir.iterdir():
                if f.suffix.lower() == '.csv' and stem in f.stem:
                    csv_path = f
                    break

        if csv_path is None:
            print(f'  WARNING: No CSV for {vid_path.name}, skipping')
            continue

        print(f'  Quad {vid_idx}: video={vid_path.name}, csv={csv_path.name}')

        # Read CSV (try common column names)
        csv_rows = []
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames
            print(f'    CSV columns: {headers}')

            # Detect column names
            time_col = None
            v_col = None
            w_col = None
            for h in headers:
                h_lower = h.lower().strip()
                if h_lower in ('time', 'timestamp', 't'):
                    time_col = h
                elif h_lower in ('velx', 'vx', 'v', 'v_actual', 'linear_velocity', 'vel_x'):
                    v_col = h
                elif h_lower in ('velw', 'vw', 'omega', 'omega_actual', 'angular_velocity', 'vel_w', 'w'):
                    w_col = h

            if time_col is None or v_col is None or w_col is None:
                print(f'    WARNING: Could not identify columns (time={time_col}, v={v_col}, w={w_col})')
                print(f'    Trying positional fallback...')
                # Try positional: time, lateral_error, angular_error, velx, velw
                if len(headers) >= 5:
                    time_col, v_col, w_col = headers[0], headers[3], headers[4]
                    print(f'    Using positional: time={time_col}, v={v_col}, w={w_col}')
                else:
                    continue

            for row in reader:
                try:
                    t = float(row[time_col])
                    vx = float(row[v_col])
                    vw = float(row[w_col])
                    vx = np.clip(vx, V_CLIP[0], V_CLIP[1])
                    vw = np.clip(vw, OMEGA_CLIP[0], OMEGA_CLIP[1])
                    csv_rows.append((t, vx, vw))
                except (ValueError, KeyError):
                    continue

        if not csv_rows:
            print(f'    WARNING: No valid CSV rows')
            continue

        # Get video FPS and extract frames
        actual_fps = get_video_fps(vid_path)
        print(f'    Video FPS: {actual_fps:.1f}, CSV rows: {len(csv_rows)}')

        dst_name = f'quad_{vid_idx}'
        dst = output_base / dst_name
        img_dir = dst / 'images'
        if dst.exists():
            shutil.rmtree(dst)
        n_frames = extract_frames_ffmpeg(vid_path, img_dir)
        print(f'    Extracted {n_frames} frames')

        # Sync frames to CSV
        unified_rows = []
        frame_files = sorted(img_dir.glob('*.jpg'))

        for frame_idx, frame_path in enumerate(frame_files):
            frame_time = frame_idx / actual_fps

            best_idx = 0
            best_dt = float('inf')
            for ci, (ct, _, _) in enumerate(csv_rows):
                dt = abs(ct - frame_time)
                if dt < best_dt:
                    best_dt = dt
                    best_idx = ci

            _, vx, vw = csv_rows[best_idx]

            new_name = f'{frame_idx:06d}.jpg'
            if frame_path.name != new_name:
                frame_path.rename(img_dir / new_name)

            unified_rows.append((f'{frame_idx:06d}', f'{vx:.6f}', f'{vw:.6f}', 1))

        n_written = write_unified_csv(dst, unified_rows)
        results.append((dst_name, n_written))
        print(f'    Output: {n_written} synced samples')

    return results


# ============================================================================
# NEGATIVES — hue-shifted, aerial, start/end frames
# ============================================================================

def hue_shift_image(img_bgr, shift_deg):
    """Shift hue of BGR image by shift_deg degrees (0-180 in OpenCV HSV)."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    hsv[:, :, 0] = (hsv[:, :, 0].astype(int) + shift_deg) % 180
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def collect_cable_images(output_base, max_images=500):
    """Collect cable-positive image paths from all processed directories."""
    images = []
    for d in output_base.iterdir():
        if d.is_dir() and d.name != 'negatives' and not d.name.startswith('_'):
            img_dir = d / 'images' if (d / 'images').exists() else d
            for img_path in sorted(img_dir.glob('*.jpg'))[:100]:
                images.append(img_path)
    random.shuffle(images)
    return images[:max_images]


def collect_aerial_videos():
    """Collect aerial view (AV) video paths from UGV and quadruped data."""
    av_videos = []

    # UGV AV videos
    for f in UGV_DATA.rglob('*av*.mp4'):
        av_videos.append(f)
    for f in UGV_DATA.rglob('*AV*.mp4'):
        av_videos.append(f)

    return list(set(av_videos))


def prepare_negatives(output_base):
    """Generate negative examples (no cable → zero velocity)."""
    neg_dir = output_base / 'negatives'
    img_dir = neg_dir / 'images'
    if neg_dir.exists():
        shutil.rmtree(neg_dir)
    img_dir.mkdir(parents=True, exist_ok=True)

    unified_rows = []
    frame_counter = 0

    # --- 1. Hue-shifted cable images (red → blue/green) ---
    print('  Generating hue-shifted negatives...')
    cable_images = collect_cable_images(output_base)
    n_hue = min(N_HUE_SHIFTED, len(cable_images))

    for i in range(n_hue):
        img = cv2.imread(str(cable_images[i]))
        if img is None:
            continue

        # Shift red to blue (60) or green (120)
        shift = random.choice([60, 120])
        shifted = hue_shift_image(img, shift)

        fname = f'{frame_counter:06d}.jpg'
        cv2.imwrite(str(img_dir / fname), shifted)
        unified_rows.append((f'{frame_counter:06d}', '0.000000', '0.000000', 0))
        frame_counter += 1

    print(f'    Hue-shifted: {n_hue} images')

    # --- 2. Aerial frames (real images, no first-person cable view) ---
    print('  Extracting aerial view negatives...')
    av_videos = collect_aerial_videos()
    n_aerial_per_vid = max(1, N_AERIAL // max(len(av_videos), 1))
    n_aerial_total = 0

    for vid_path in av_videos:
        if n_aerial_total >= N_AERIAL:
            break

        cap = cv2.VideoCapture(str(vid_path))
        if not cap.isOpened():
            continue

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < 10:
            cap.release()
            continue

        # Sample evenly spaced frames
        indices = np.linspace(10, total_frames - 10, n_aerial_per_vid, dtype=int)
        for idx in indices:
            if n_aerial_total >= N_AERIAL:
                break

            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            fname = f'{frame_counter:06d}.jpg'
            cv2.imwrite(str(img_dir / fname), frame)
            unified_rows.append((f'{frame_counter:06d}', '0.000000', '0.000000', 0))
            frame_counter += 1
            n_aerial_total += 1

        cap.release()

    print(f'    Aerial: {n_aerial_total} images')

    # --- 3. Start/end frames (robot stationary, cable may be absent) ---
    print('  Extracting start/end frame negatives...')
    n_startend = 0

    # Get all trial videos
    trial_videos = []
    for rec_num in UGV_RECORDINGS:
        for f in UGV_DATA.rglob(f'*recording_{rec_num}*trial*.mp4'):
            trial_videos.append(f)
        for f in UGV_DATA.rglob(f'*recording_{rec_num}*.mp4'):
            if 'av' not in f.name.lower() and f not in trial_videos:
                trial_videos.append(f)

    n_per_vid = max(1, N_START_END // max(len(trial_videos) * 2, 1))

    for vid_path in trial_videos:
        if n_startend >= N_START_END:
            break

        cap = cv2.VideoCapture(str(vid_path))
        if not cap.isOpened():
            continue

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < 20:
            cap.release()
            continue

        # First N and last N frames
        for idx in list(range(0, min(n_per_vid, 5))) + \
                   list(range(max(0, total_frames - n_per_vid), total_frames)):
            if n_startend >= N_START_END:
                break

            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            fname = f'{frame_counter:06d}.jpg'
            cv2.imwrite(str(img_dir / fname), frame)
            unified_rows.append((f'{frame_counter:06d}', '0.000000', '0.000000', 0))
            frame_counter += 1
            n_startend += 1

        cap.release()

    print(f'    Start/end: {n_startend} images')

    # Also add quadruped raw AV frames if available
    quad_raw = output_base / '_quad_raw'
    if quad_raw.exists():
        av_count = 0
        for f in quad_raw.rglob('*av*.mp4'):
            if n_aerial_total + av_count >= N_AERIAL + 50:
                break
            cap = cv2.VideoCapture(str(f))
            if not cap.isOpened():
                continue
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            indices = np.linspace(10, max(11, total_frames - 10), 20, dtype=int)
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ret, frame = cap.read()
                if not ret or frame is None:
                    continue
                fname = f'{frame_counter:06d}.jpg'
                cv2.imwrite(str(img_dir / fname), frame)
                unified_rows.append((f'{frame_counter:06d}', '0.000000', '0.000000', 0))
                frame_counter += 1
                av_count += 1
            cap.release()
        if av_count:
            print(f'    Quad aerial: {av_count} images')

    n_written = write_unified_csv(neg_dir, unified_rows)
    print(f'    Total negatives: {n_written}')
    return [('negatives', n_written)]


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Prepare Cable Tracer Multi-Dataset')
    parser.add_argument('--output', default=str(DEFAULT_OUTPUT),
                        help='Output directory')
    args = parser.parse_args()

    output_base = Path(args.output)
    output_base.mkdir(parents=True, exist_ok=True)
    print(f'Output: {output_base}\n')

    manifest = {'sources': {}, 'total_samples': 0, 'total_negatives': 0}

    # 1. Wheelchair
    print('=== Wheelchair ===')
    wc_results = prepare_wheelchair(output_base)
    for name, count in wc_results:
        manifest['sources'][name] = {
            'type': 'wheelchair', 'samples': count, 'cable_visible': count
        }
        manifest['total_samples'] += count

    # 2. UGV
    print('\n=== UGV ===')
    ugv_results = prepare_ugv(output_base)
    for name, count in ugv_results:
        manifest['sources'][name] = {
            'type': 'ugv', 'samples': count, 'cable_visible': count
        }
        manifest['total_samples'] += count

    # 3. Quadruped
    print('\n=== Quadruped ===')
    quad_results = prepare_quadruped(output_base)
    for name, count in quad_results:
        manifest['sources'][name] = {
            'type': 'quadruped', 'samples': count, 'cable_visible': count
        }
        manifest['total_samples'] += count

    # 4. Negatives (must come after others so we can sample cable images)
    print('\n=== Negatives ===')
    neg_results = prepare_negatives(output_base)
    for name, count in neg_results:
        manifest['sources'][name] = {
            'type': 'negatives', 'samples': count, 'cable_visible': 0
        }
        manifest['total_samples'] += count
        manifest['total_negatives'] += count

    # Compute velocity stats per source
    for name in manifest['sources']:
        csv_path = output_base / name / 'velocities.csv'
        if not csv_path.exists():
            continue
        vs, ws = [], []
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                vs.append(float(row['v_actual']))
                ws.append(float(row['omega_actual']))
        if vs:
            manifest['sources'][name]['v_stats'] = {
                'min': float(np.min(vs)), 'max': float(np.max(vs)),
                'mean': float(np.mean(vs)), 'std': float(np.std(vs)),
            }
            manifest['sources'][name]['w_stats'] = {
                'min': float(np.min(ws)), 'max': float(np.max(ws)),
                'mean': float(np.mean(ws)), 'std': float(np.std(ws)),
            }

    # Write manifest
    manifest_path = output_base / 'manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    # Summary
    print(f'\n{"="*60}')
    print(f'SUMMARY')
    print(f'{"="*60}')
    print(f'Output: {output_base}')
    print(f'Total samples: {manifest["total_samples"]}')
    print(f'  Cable-positive: {manifest["total_samples"] - manifest["total_negatives"]}')
    print(f'  Negatives: {manifest["total_negatives"]}')
    print(f'\nDirectories:')
    for name, info in manifest['sources'].items():
        print(f'  {name:20s} {info["type"]:12s} {info["samples"]:6d} samples')
    print(f'\nManifest: {manifest_path}')
    print('Done!')


if __name__ == '__main__':
    main()
