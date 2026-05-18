"""
PyTorch Datasets for BEV-Velocity training.

Supports three modes:
    1. Single-step: BEV (5, 200, 200) + odom (30,) → velocity (2,)
    2. CFM trajectory: BEV (5, 200, 200) + odom (30,) → velocity_traj (H, 2)
    3. Modular KinoFlow v2: scan (720,) + residuals (4, 720) + goal (4,) + odom (30,)
       → velocity_traj (H, 2)

Data is created by scripts/bag_to_bev_velocity.py from recorded rosbags.
Mode 3 uses additional files: scan_ranges.npy, scan_odom.npy, goal_relative.npy.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset


class BEVVelocityDataset(Dataset):
    """
    Dataset for single-step BEV-Velocity training (baseline model).

    Expected directory structure:
        data_dir/
            bev_000000.npy      # (5, 200, 200) float32
            odom_000000.npy     # (30,) float32 [v, w, theta] x 10 steps
            labels.npy          # (N, 2) float32 [v, omega] per timestep
            metadata.npz        # timestamps, segment_ids, etc.
    """

    def __init__(self, data_dir, transform=None, augment=None):
        self.data_dir = data_dir
        self.transform = transform
        self.augment = augment

        labels_path = os.path.join(data_dir, 'labels.npy')
        self.labels = np.load(labels_path)  # (N, 2)
        self.n_samples = len(self.labels)

        meta_path = os.path.join(data_dir, 'metadata.npz')
        if os.path.exists(meta_path):
            meta = np.load(meta_path)
            self.timestamps = meta.get('timestamps', None)
            self.goal_positions = meta.get('goal_positions', None)
        else:
            self.timestamps = None
            self.goal_positions = None

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        bev = np.load(os.path.join(self.data_dir, f'bev_{idx:06d}.npy'))

        odom_path = os.path.join(self.data_dir, f'odom_{idx:06d}.npy')
        if os.path.exists(odom_path):
            odom = np.load(odom_path)
        else:
            odom = np.zeros(30, dtype=np.float32)

        label = self.labels[idx]

        if self.augment is not None:
            bev, label = self.augment(bev, label)
        if self.transform is not None:
            bev = self.transform(bev)

        return {
            'bev': torch.from_numpy(bev).float(),
            'odom': torch.from_numpy(odom).float(),
            'velocity': torch.from_numpy(label).float(),
        }


class SequentialBEVDataset(Dataset):
    """
    Sequential dataset for GRU training — returns sequences of
    consecutive BEV frames that respect segment boundaries.

    Segments are driving episodes separated by 5-second stops.
    Sequences never cross segment boundaries so the GRU doesn't
    learn spurious transitions between unrelated episodes.
    """

    def __init__(self, data_dir, seq_len=5, transform=None,
                 augment=None):
        self.data_dir = data_dir
        self.seq_len = seq_len
        self.transform = transform
        self.augment = augment

        self.labels = np.load(os.path.join(data_dir, 'labels.npy'))
        self.n_total = len(self.labels)

        # Load segment boundaries to avoid cross-segment sequences
        meta_path = os.path.join(data_dir, 'metadata.npz')
        if os.path.exists(meta_path):
            meta = np.load(meta_path, allow_pickle=True)
            seg_ids = meta.get('segment_ids', None)
            if seg_ids is not None:
                self.segment_ids = seg_ids
            else:
                self.segment_ids = np.zeros(self.n_total, dtype=np.int32)
        else:
            self.segment_ids = np.zeros(self.n_total, dtype=np.int32)

        # Build valid start indices (sequences that stay within one segment)
        self.valid_starts = []
        for i in range(self.n_total - seq_len + 1):
            seg_slice = self.segment_ids[i:i + seq_len]
            if np.all(seg_slice == seg_slice[0]):
                self.valid_starts.append(i)

        self.valid_starts = np.array(self.valid_starts)

    def __len__(self):
        return len(self.valid_starts)

    def __getitem__(self, idx):
        start = self.valid_starts[idx]
        bevs = []
        odoms = []
        velocities = []

        for t in range(self.seq_len):
            i = start + t
            bev = np.load(
                os.path.join(self.data_dir, f'bev_{i:06d}.npy'))
            odom_path = os.path.join(self.data_dir, f'odom_{i:06d}.npy')
            if os.path.exists(odom_path):
                odom = np.load(odom_path)
            else:
                odom = np.zeros(30, dtype=np.float32)

            bevs.append(bev)
            odoms.append(odom)
            velocities.append(self.labels[i])

        return {
            'bev': torch.from_numpy(np.stack(bevs)).float(),
            'odom': torch.from_numpy(np.stack(odoms)).float(),
            'velocity': torch.from_numpy(np.stack(velocities)).float(),
        }


class CFMTrajectoryDataset(Dataset):
    """
    Dataset for CFM velocity trajectory training.

    Returns current BEV + odom as input, and a future velocity trajectory
    of H steps as the target. The CFM learns to generate these trajectories.

    Each sample:
        Input:  BEV at time t (5, 200, 200), odom at time t (30,)
        Target: velocity trajectory [(v,ω)]_{t:t+H} shape (H, 2)

    Respects segment boundaries — trajectories never cross 5s stops.
    """

    def __init__(self, data_dir, horizon=10, transform=None, augment=None):
        """
        Args:
            data_dir: Path to preprocessed training data
            horizon: Number of future velocity steps (H)
            transform: Optional transform for BEV grids
            augment: Optional ChauffeurNet augmentation
        """
        self.data_dir = data_dir
        self.horizon = horizon
        self.transform = transform
        self.augment = augment

        self.labels = np.load(os.path.join(data_dir, 'labels.npy'))
        self.n_total = len(self.labels)

        # Load segment boundaries
        meta_path = os.path.join(data_dir, 'metadata.npz')
        if os.path.exists(meta_path):
            meta = np.load(meta_path, allow_pickle=True)
            seg_ids = meta.get('segment_ids', None)
            self.segment_ids = seg_ids if seg_ids is not None else \
                np.zeros(self.n_total, dtype=np.int32)
        else:
            self.segment_ids = np.zeros(self.n_total, dtype=np.int32)

        # Valid indices: need H future steps within same segment
        self.valid_indices = []
        for i in range(self.n_total - horizon):
            seg_slice = self.segment_ids[i:i + horizon + 1]
            if np.all(seg_slice == seg_slice[0]):
                self.valid_indices.append(i)

        self.valid_indices = np.array(self.valid_indices)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        i = self.valid_indices[idx]

        # Current BEV and odom
        bev = np.load(os.path.join(self.data_dir, f'bev_{i:06d}.npy'))

        odom_path = os.path.join(self.data_dir, f'odom_{i:06d}.npy')
        if os.path.exists(odom_path):
            odom = np.load(odom_path)
        else:
            odom = np.zeros(30, dtype=np.float32)

        # Future velocity trajectory: H steps starting from current frame
        vel_traj = self.labels[i:i + self.horizon]  # (H, 2)

        if self.augment is not None:
            bev, vel_traj[0] = self.augment(bev, vel_traj[0])
        if self.transform is not None:
            bev = self.transform(bev)

        return {
            'bev': torch.from_numpy(bev).float(),
            'odom': torch.from_numpy(odom).float(),
            'velocity': torch.from_numpy(vel_traj[0].copy()).float(),
            'velocity_traj': torch.from_numpy(vel_traj.copy()).float(),
        }


class ModularKinoFlowDataset(Dataset):
    """Dataset for KinoFlow v2 modular architecture training.

    Loads scan temporal stacks + goal features + odom for the modular
    encoder pipeline (E1-E4). Also loads BEV for collision scoring.

    Expected directory structure (in addition to standard files):
        data_dir/
            scan_ranges.npy     # (N, 720) float32 — polar scan ranges per frame
            scan_odom.npy       # (N, 3) float32 — (x, y, theta) per frame
            goal_relative.npy   # (N, 4) float32 — [norm_dist, norm_bearing, cos_b, sin_b]
            bev_000000.npy      # (5, 200, 200) — still needed for collision scoring
            odom_000000.npy     # (30,) — odom history
            labels.npy          # (N, 2) — [v, omega]
            metadata.npz        # timestamps, segment_ids, etc.

    Each sample returns:
        scan_current:   (720,)   current polar scan
        scan_residuals: (4, 720) ego-compensated temporal residuals
        goal_features:  (4,)     [norm_dist, norm_bearing, cos_b, sin_b]
        odom:           (30,)    odom history
        bev:            (5, 200, 200) BEV (for collision loss)
        velocity_traj:  (H, 2)   target trajectory
    """

    def __init__(self, data_dir, horizon=10, temporal_frames=5,
                 scan_points=720, transform=None, augment=None):
        self.data_dir = data_dir
        self.horizon = horizon
        self.temporal_frames = temporal_frames
        self.n_residuals = temporal_frames - 1
        self.scan_points = scan_points
        self.transform = transform
        self.augment = augment

        self.labels = np.load(os.path.join(data_dir, 'labels.npy'))
        self.n_total = len(self.labels)

        # Load scan ranges and odom for temporal residuals
        scan_path = os.path.join(data_dir, 'scan_ranges.npy')
        self.has_scans = os.path.exists(scan_path)
        if self.has_scans:
            self.scan_ranges = np.load(scan_path)  # (N, 720)
        else:
            self.scan_ranges = None

        odom_poses_path = os.path.join(data_dir, 'scan_odom.npy')
        if os.path.exists(odom_poses_path):
            self.scan_odom = np.load(odom_poses_path)  # (N, 3)
        else:
            self.scan_odom = None

        # Load goal features
        goal_path = os.path.join(data_dir, 'goal_relative.npy')
        if os.path.exists(goal_path):
            self.goal_features = np.load(goal_path)  # (N, 4)
        else:
            self.goal_features = None

        # Segment boundaries
        meta_path = os.path.join(data_dir, 'metadata.npz')
        if os.path.exists(meta_path):
            meta = np.load(meta_path, allow_pickle=True)
            seg_ids = meta.get('segment_ids', None)
            self.segment_ids = seg_ids if seg_ids is not None else \
                np.zeros(self.n_total, dtype=np.int32)
        else:
            self.segment_ids = np.zeros(self.n_total, dtype=np.int32)

        # Valid indices: need H future steps + temporal_frames past steps
        # within same segment
        self.valid_indices = []
        lookback = temporal_frames - 1
        for i in range(lookback, self.n_total - horizon):
            # Check both past (for temporal stack) and future (for trajectory)
            seg_slice = self.segment_ids[i - lookback:i + horizon + 1]
            if np.all(seg_slice == seg_slice[0]):
                self.valid_indices.append(i)

        self.valid_indices = np.array(self.valid_indices)

    def __len__(self):
        return len(self.valid_indices)

    def _compute_residuals(self, idx):
        """Compute temporal scan residuals for a given index."""
        N = self.scan_points
        T = self.temporal_frames

        if self.scan_ranges is None or self.scan_odom is None:
            return np.zeros((self.n_residuals, N), dtype=np.float32)

        # Gather T frames ending at idx
        frame_indices = list(range(idx - T + 1, idx + 1))
        scans = [self.scan_ranges[fi] for fi in frame_indices]

        # Compute odom deltas between consecutive frames
        odom_deltas = []
        for j in range(T - 1):
            fi_prev = frame_indices[j]
            fi_curr = frame_indices[j + 1]
            prev_pose = self.scan_odom[fi_prev]
            curr_pose = self.scan_odom[fi_curr]
            delta = curr_pose - prev_pose
            odom_deltas.append(delta.astype(np.float32))

        # Build angles (assume uniform 0 to 2*pi for 720 points)
        angles = np.linspace(-np.pi, np.pi, N, dtype=np.float32)

        from wheelchair_e2e.models.dynamic_encoder import build_temporal_residuals
        residuals = build_temporal_residuals(scans, odom_deltas, angles)
        return residuals

    def __getitem__(self, idx):
        i = self.valid_indices[idx]

        # Current scan
        if self.scan_ranges is not None:
            scan_current = self.scan_ranges[i].copy()
        else:
            scan_current = np.zeros(self.scan_points, dtype=np.float32)

        # Temporal residuals
        scan_residuals = self._compute_residuals(i)

        # Goal features
        if self.goal_features is not None:
            goal_feat = self.goal_features[i].copy()
        else:
            goal_feat = np.zeros(4, dtype=np.float32)

        # Odom history
        odom_path = os.path.join(self.data_dir, f'odom_{i:06d}.npy')
        if os.path.exists(odom_path):
            odom = np.load(odom_path)
        else:
            odom = np.zeros(30, dtype=np.float32)

        # BEV (still needed for collision scoring)
        bev_path = os.path.join(self.data_dir, f'bev_{i:06d}.npy')
        if os.path.exists(bev_path):
            bev = np.load(bev_path)
        else:
            bev = np.zeros((5, 200, 200), dtype=np.float32)

        # Future velocity trajectory
        vel_traj = self.labels[i:i + self.horizon].copy()

        # Augmentation (mirror operates on scan + trajectory)
        if self.augment and callable(self.augment):
            scan_current, scan_residuals, vel_traj = self.augment(
                scan_current, scan_residuals, vel_traj)

        # Replace inf/nan in scan with 0
        scan_current = np.where(
            np.isfinite(scan_current), scan_current, 0.0
        ).astype(np.float32)

        return {
            'scan_current': torch.from_numpy(scan_current).float(),
            'scan_residuals': torch.from_numpy(scan_residuals).float(),
            'goal_features': torch.from_numpy(goal_feat).float(),
            'odom': torch.from_numpy(odom).float(),
            'bev': torch.from_numpy(bev).float(),
            'velocity': torch.from_numpy(vel_traj[0].copy()).float(),
            'velocity_traj': torch.from_numpy(vel_traj).float(),
        }
