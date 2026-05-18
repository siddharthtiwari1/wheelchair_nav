"""PyTorch Dataset for wheelchair monocular depth fine-tuning.

Loads paired RGB-depth data collected by data_collection_node.py.
Supports multiple sessions and cameras, with DPT-compatible transforms.

Camera-aware confidence weighting:
  - D455 (front, left): weight = 1.0 at all depths
  - D435i (right): weight = min(1.0, 2.0 / depth_m) beyond 2m
  This accounts for the D435i's worse depth noise (92mm@3.5m vs 49mm for D455).

Expected directory structure:
    session_dir/camera_name/XXXXXX_rgb.png    (8-bit BGR)
    session_dir/camera_name/XXXXXX_depth.png  (16-bit uint16 millimeters)
"""

import os
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


CAMERAS = ['front', 'left', 'right']

# Camera type mapping: which cameras use D435i (worse depth at range)
D435I_CAMERAS = {'right'}


class WheelchairDepthDataset(Dataset):
    """Dataset for wheelchair monocular depth fine-tuning.

    Args:
        session_dirs: list of session directory paths
        transform: callable(rgb, depth) -> (rgb_tensor, depth_tensor, mask)
        cameras: list of camera names to include (default: all 3)
        min_valid_ratio: minimum fraction of valid depth pixels (skip if below)
        max_depth: maximum depth in meters (for filtering)
    """

    def __init__(self, session_dirs, transform=None, cameras=None,
                 min_valid_ratio=0.3, max_depth=10.0):
        self.transform = transform
        self.max_depth = max_depth
        self.min_valid_ratio = min_valid_ratio
        cameras = cameras or CAMERAS

        self.samples = []  # list of (rgb_path, depth_path, camera_id)

        for session_dir in session_dirs:
            session_dir = str(session_dir)
            for cam in cameras:
                cam_dir = os.path.join(session_dir, cam)
                if not os.path.isdir(cam_dir):
                    continue

                # Find all frame IDs that have both RGB and depth
                files = os.listdir(cam_dir)
                rgb_ids = {f[:6] for f in files if f.endswith('_rgb.png')}
                depth_ids = {f[:6] for f in files if f.endswith('_depth.png')}
                common_ids = sorted(rgb_ids & depth_ids)

                for fid in common_ids:
                    self.samples.append((
                        os.path.join(cam_dir, f'{fid}_rgb.png'),
                        os.path.join(cam_dir, f'{fid}_depth.png'),
                        cam,
                    ))

        print(f'WheelchairDepthDataset: {len(self.samples)} samples '
              f'from {len(session_dirs)} sessions')

    def __len__(self):
        return len(self.samples)

    def _compute_camera_weights(self, depth_m, camera_id):
        """Generate per-pixel confidence weights based on camera type and depth.

        D435i (right camera) has ~1.9x worse depth noise than D455, especially
        beyond 2m where error scales as z^2/(f*b). We down-weight those pixels.

        Args:
            depth_m: numpy HxW float32 depth in meters
            camera_id: 'front', 'left', or 'right'

        Returns:
            weights: numpy HxW float32, values in (0, 1]
        """
        weights = np.ones_like(depth_m, dtype=np.float32)

        if camera_id in D435I_CAMERAS:
            # Beyond 2m, weight decreases linearly: min(1.0, 2.0/z)
            # At 2m: weight=1.0, at 4m: weight=0.5, at 6m: weight=0.33
            far_mask = depth_m > 2.0
            if far_mask.any():
                weights[far_mask] = np.minimum(
                    1.0, 2.0 / depth_m[far_mask].clip(min=1e-3)
                )

        return weights

    def __getitem__(self, idx):
        rgb_path, depth_path, camera_id = self.samples[idx]

        # Load RGB (BGR uint8)
        rgb = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        if rgb is None:
            raise RuntimeError(f'Failed to load RGB: {rgb_path}')

        # Load depth (uint16 millimeters)
        depth_mm = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth_mm is None:
            raise RuntimeError(f'Failed to load depth: {depth_path}')

        # Convert to meters
        depth = depth_mm.astype(np.float32) / 1000.0

        # Compute camera confidence weights
        camera_weights = self._compute_camera_weights(depth, camera_id)

        if self.transform:
            rgb_tensor, depth_tensor, valid_mask = self.transform(rgb, depth)
            # Resize camera_weights to match transform output size
            h, w = depth_tensor.shape[-2:]
            weights_tensor = torch.from_numpy(
                cv2.resize(camera_weights, (w, h), interpolation=cv2.INTER_NEAREST)
            ).unsqueeze(0).float()
            return rgb_tensor, depth_tensor, valid_mask, weights_tensor

        # Default: minimal conversion
        rgb_tensor = torch.from_numpy(rgb[:, :, ::-1].copy()).permute(2, 0, 1).float() / 255.0
        depth_tensor = torch.from_numpy(depth).unsqueeze(0).float()
        valid_mask = (depth_tensor > 0) & (depth_tensor <= self.max_depth)
        weights_tensor = torch.from_numpy(camera_weights).unsqueeze(0).float()

        return rgb_tensor, depth_tensor, valid_mask, weights_tensor


def get_session_split(data_root, val_ratio=0.1):
    """Split sessions into train/val (session-stratified, no temporal leakage).

    Args:
        data_root: root directory containing session subdirectories
        val_ratio: fraction of sessions for validation

    Returns:
        train_sessions, val_sessions: lists of absolute session directory paths
    """
    sessions = sorted([
        os.path.join(data_root, d)
        for d in os.listdir(data_root)
        if os.path.isdir(os.path.join(data_root, d))
        and os.path.exists(os.path.join(data_root, d, 'metadata.json'))
    ])

    if len(sessions) < 2:
        print(f'Warning: only {len(sessions)} sessions, using all for training')
        return sessions, sessions

    n_val = max(1, int(len(sessions) * val_ratio))
    val_sessions = sessions[-n_val:]
    train_sessions = sessions[:-n_val]

    print(f'Split: {len(train_sessions)} train sessions, {len(val_sessions)} val sessions')
    return train_sessions, val_sessions
