#!/usr/bin/env python3
"""Offline batch inference: DA3Metric-Large on collected RGB images.

Runs DA3Metric-Large on all RGB frames in collected sessions, saves predicted
depth maps alongside stereo GT. Creates teacher pseudo-labels AND baseline
comparison for fine-tuning evaluation.

Output: XXXXXX_da3_depth.png (uint16 mm, same format as stereo depth)

Usage:
    python3 -m wheelchair_mono_depth.scripts.batch_da3_inference \
        --data_dir /home/sidd/wheelchair_nav/mono_depth_data \
        --session 20260304_154230 \
        --cameras front left right

    # Process all sessions:
    python3 -m wheelchair_mono_depth.scripts.batch_da3_inference \
        --data_dir /home/sidd/wheelchair_nav/mono_depth_data
"""

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

# DA3 constants (from da3_depth_node.py)
DA3_SCALE_FACTOR = 300.0
PATCH_SIZE = 14
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_da3_model(device, compile_model=True):
    """Load DA3Metric-Large with optimizations from da3_depth_node.py."""
    import torch
    from torchvision import transforms as T

    os.environ['HF_HUB_OFFLINE'] = '1'

    from depth_anything_3.api import DepthAnything3

    # CRITICAL: set cudnn.benchmark AFTER import — DA3 api.py sets it False
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    model = DepthAnything3.from_pretrained('depth-anything/DA3METRIC-LARGE')
    model = model.to(device=device).eval()

    # Extract inner model for direct forward() calls
    inner_model = model.model

    if compile_model and device.type == 'cuda':
        print('Compiling model with torch.compile(reduce-overhead)...')
        inner_model = torch.compile(
            inner_model, mode='reduce-overhead', fullgraph=False)

    normalize = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    # Warmup
    proc_w = 504
    proc_h = (int(480 * proc_w / 640) // PATCH_SIZE) * PATCH_SIZE
    dummy = torch.randn(1, 1, 3, proc_h, proc_w, device=device)
    n_warmup = 5 if compile_model else 1
    print(f'Warming up ({n_warmup} passes)...')
    for _ in range(n_warmup):
        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                _ = inner_model(dummy, export_feat_layers=[])
        torch.cuda.synchronize()
    print('Model ready.')

    return inner_model, normalize


def load_intrinsics(session_dir, camera):
    """Load camera intrinsics from session data."""
    path = os.path.join(session_dir, camera, 'intrinsics.json')
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    return data


def infer_depth(inner_model, normalize, rgb, fx, fy, device,
                process_width=504, max_depth=6.0):
    """Run DA3Metric-Large on a single RGB image. Returns depth in meters."""
    import torch

    cam_h, cam_w = rgb.shape[:2]
    scale = process_width / cam_w
    proc_w = process_width
    proc_h = (int(cam_h * scale) // PATCH_SIZE) * PATCH_SIZE

    # Scale intrinsics
    fx_scaled = fx * scale
    fy_scaled = fy * scale

    # Resize
    rgb_proc = cv2.resize(rgb, (proc_w, proc_h), interpolation=cv2.INTER_AREA)

    # Preprocess: RGB -> tensor -> normalize
    tensor = torch.from_numpy(rgb_proc).to(device, non_blocking=True)
    tensor = tensor.permute(2, 0, 1).float().div_(255.0)
    tensor = normalize(tensor)
    tensor = tensor.unsqueeze(0).unsqueeze(0)  # (1,1,3,H,W)

    # Direct forward with FP16
    with torch.no_grad():
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            output = inner_model(tensor, export_feat_layers=[])

    canonical = output["depth"].squeeze().cpu().numpy()

    # Resize if model output differs
    if canonical.shape != (proc_h, proc_w):
        canonical = cv2.resize(canonical, (proc_w, proc_h),
                               interpolation=cv2.INTER_LINEAR)

    # Focal scaling: physics-grounded metric conversion
    avg_focal = (fx_scaled + fy_scaled) / 2.0
    depth_m = canonical * (avg_focal / DA3_SCALE_FACTOR)

    # Resize back to original resolution
    depth_m = cv2.resize(depth_m, (cam_w, cam_h),
                         interpolation=cv2.INTER_LINEAR)

    return np.clip(depth_m, 0.0, max_depth).astype(np.float32)


def process_session(session_dir, inner_model, normalize, device, cameras,
                    process_width, max_depth, skip_existing):
    """Process all frames in a session."""
    import torch

    session_name = os.path.basename(session_dir)
    total_frames = 0
    total_time = 0.0

    for camera in cameras:
        cam_dir = os.path.join(session_dir, camera)
        if not os.path.isdir(cam_dir):
            print(f'  [{camera}] directory not found, skipping')
            continue

        # Load intrinsics
        intrinsics = load_intrinsics(session_dir, camera)
        if intrinsics is None:
            print(f'  [{camera}] no intrinsics.json, skipping')
            continue

        fx = intrinsics['fx']
        fy = intrinsics['fy']

        # Find all RGB frames
        files = os.listdir(cam_dir)
        rgb_files = sorted([f for f in files if f.endswith('_rgb.png')])

        if not rgb_files:
            print(f'  [{camera}] no RGB frames found')
            continue

        skipped = 0
        processed = 0

        for rgb_file in rgb_files:
            frame_id = rgb_file[:6]
            da3_path = os.path.join(cam_dir, f'{frame_id}_da3_depth.png')

            if skip_existing and os.path.exists(da3_path):
                skipped += 1
                continue

            rgb_path = os.path.join(cam_dir, rgb_file)
            rgb = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
            if rgb is None:
                print(f'    WARNING: failed to load {rgb_path}')
                continue

            # Convert BGR -> RGB
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

            t0 = time.monotonic()
            depth_m = infer_depth(inner_model, normalize, rgb, fx, fy,
                                  device, process_width, max_depth)
            elapsed = time.monotonic() - t0
            total_time += elapsed

            # Save as uint16 mm (same format as stereo depth)
            depth_mm = (depth_m * 1000.0).clip(0, 65535).astype(np.uint16)
            cv2.imwrite(da3_path, depth_mm)

            processed += 1
            total_frames += 1

        print(f'  [{camera}] {processed} processed, {skipped} skipped '
              f'(fx={fx:.1f}, fy={fy:.1f})')

    return total_frames, total_time


def find_sessions(data_dir, session_filter=None):
    """Find valid session directories."""
    sessions = []
    for d in sorted(os.listdir(data_dir)):
        session_path = os.path.join(data_dir, d)
        if not os.path.isdir(session_path):
            continue
        if not os.path.exists(os.path.join(session_path, 'metadata.json')):
            continue
        if session_filter and d not in session_filter:
            continue
        sessions.append(session_path)
    return sessions


def main():
    parser = argparse.ArgumentParser(
        description='Batch DA3Metric-Large inference on collected data')
    parser.add_argument('--data_dir', required=True,
                        help='Root directory with session folders')
    parser.add_argument('--session', nargs='*', default=None,
                        help='Specific session IDs to process (default: all)')
    parser.add_argument('--cameras', nargs='*',
                        default=['front', 'left', 'right'],
                        help='Cameras to process')
    parser.add_argument('--process_width', type=int, default=504,
                        help='Processing width (multiple of 14)')
    parser.add_argument('--max_depth', type=float, default=6.0,
                        help='Maximum depth in meters')
    parser.add_argument('--no_compile', action='store_true',
                        help='Disable torch.compile (faster startup)')
    parser.add_argument('--skip_existing', action='store_true', default=True,
                        help='Skip frames that already have DA3 depth')
    parser.add_argument('--no_skip', action='store_true',
                        help='Reprocess all frames even if DA3 depth exists')
    args = parser.parse_args()

    if args.no_skip:
        args.skip_existing = False

    # Force process_width to multiple of 14
    args.process_width = (args.process_width // PATCH_SIZE) * PATCH_SIZE

    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # Find sessions
    sessions = find_sessions(args.data_dir, args.session)
    if not sessions:
        print(f'No valid sessions found in {args.data_dir}')
        sys.exit(1)

    print(f'Found {len(sessions)} session(s)')
    for s in sessions:
        print(f'  {os.path.basename(s)}')

    # Load model
    inner_model, normalize = load_da3_model(
        device, compile_model=not args.no_compile)

    # Process sessions
    grand_total_frames = 0
    grand_total_time = 0.0

    for session_dir in sessions:
        session_name = os.path.basename(session_dir)
        print(f'\nProcessing: {session_name}')

        n_frames, elapsed = process_session(
            session_dir, inner_model, normalize, device,
            args.cameras, args.process_width, args.max_depth,
            args.skip_existing)

        grand_total_frames += n_frames
        grand_total_time += elapsed

        if n_frames > 0:
            fps = n_frames / elapsed
            print(f'  Total: {n_frames} frames in {elapsed:.1f}s '
                  f'({fps:.1f} FPS)')

    print(f'\n{"="*60}')
    print(f'Done. {grand_total_frames} frames processed in '
          f'{grand_total_time:.1f}s')
    if grand_total_frames > 0:
        print(f'Average: {grand_total_frames / grand_total_time:.1f} FPS')


if __name__ == '__main__':
    main()
