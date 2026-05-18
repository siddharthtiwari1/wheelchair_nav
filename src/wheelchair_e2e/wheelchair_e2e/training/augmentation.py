"""
Data Augmentation for KinoFlow Training.

Contains:
1. ChauffeurNet-style perturbation (recovery behavior, single-step + trajectory)
2. Mirror augmentation (horizontal flip, 2x free data, ~3x success rate improvement)

ChauffeurNet (Waymo RSS'19):
    Perturb wheelchair pose in BEV, compute corrective velocity labels.
    Teaches recovery behavior. Multiplies data 5-10x.

Mirror augmentation (Ratatouille, Sep'25):
    Flip BEV left-right, negate omega. Indoor environments are roughly
    symmetric. 2x free data, shown to give 3x success rate improvement
    and 6x fewer collisions in Ratatouille paper.
"""

import numpy as np
from scipy.ndimage import affine_transform


def perturb_and_relabel(bev, v_label, omega_label,
                        max_pos_perturb=0.3,
                        max_heading_perturb=30,
                        resolution=0.05,
                        v_max=0.25, w_max=1.0):
    """
    ChauffeurNet-style data augmentation.

    Perturb wheelchair pose, re-render BEV, compute corrective velocity.

    Args:
        bev: (C, 200, 200) BEV grid (C=5 channels)
        v_label: original linear velocity label (m/s)
        omega_label: original angular velocity label (rad/s)
        max_pos_perturb: max position perturbation (meters)
        max_heading_perturb: max heading perturbation (degrees)
        resolution: BEV grid resolution (m/pixel)
        v_max: maximum linear velocity
        w_max: maximum angular velocity

    Returns:
        bev_perturbed: (4, 200, 200) transformed BEV
        v_corrective: corrective linear velocity
        omega_corrective: corrective angular velocity
    """
    # Random perturbation
    dx = np.random.uniform(-max_pos_perturb, max_pos_perturb)
    dy = np.random.uniform(-max_pos_perturb, max_pos_perturb)
    dtheta = np.radians(np.random.uniform(
        -max_heading_perturb, max_heading_perturb))

    # Shift BEV grid (translate + rotate)
    bev_perturbed = affine_transform_bev(bev, dx, dy, dtheta, resolution)

    # Corrective velocity: drive back to original trajectory
    # Simple P-controller target (becomes the new label)
    kp_v = 0.5     # proportional gain for position
    kp_omega = 2.0  # proportional gain for heading

    dist = np.sqrt(dx**2 + dy**2)
    angle_to_original = np.arctan2(-dy, -dx) - dtheta

    v_corrective = np.clip(kp_v * dist, 0, v_max)
    omega_corrective = np.clip(
        kp_omega * angle_to_original, -w_max, w_max)

    return bev_perturbed, v_corrective, omega_corrective


def affine_transform_bev(bev, dx, dy, dtheta, resolution=0.05):
    """
    Apply affine transformation to BEV grid.

    Simulates viewing the scene from a perturbed pose.

    Args:
        bev: (C, H, W) BEV grid
        dx, dy: position offset in meters
        dtheta: heading offset in radians
        resolution: m/pixel

    Returns:
        bev_transformed: (C, H, W) transformed grid
    """
    C, H, W = bev.shape
    center = H // 2

    # Convert meters to pixels
    dx_px = dx / resolution
    dy_px = dy / resolution

    cos_t = np.cos(dtheta)
    sin_t = np.sin(dtheta)

    # Affine matrix for rotation + translation
    # We transform each channel independently
    bev_out = np.zeros_like(bev)

    for c in range(C):
        # Build transformation matrix for scipy.ndimage.affine_transform
        # This applies: output[o] = input[A @ o + b]
        # We want to rotate around center then translate
        A = np.array([[cos_t, -sin_t],
                      [sin_t, cos_t]])

        # Offset: rotate around center, then translate
        offset = np.array([center, center]) - A @ np.array(
            [center + dy_px, center + dx_px])

        bev_out[c] = affine_transform(
            bev[c], A, offset=offset,
            order=1, mode='constant', cval=0.0
        )

    return bev_out


class ChauffeurNetAugmentor:
    """
    Augmentation pipeline with configurable perturbation parameters.
    """

    def __init__(self, max_pos=0.3, max_heading=30,
                 prob=0.5, resolution=0.05,
                 v_max=0.25, w_max=1.0):
        """
        Args:
            max_pos: max position perturbation (meters)
            max_heading: max heading perturbation (degrees)
            prob: probability of applying perturbation
            resolution: BEV grid resolution
            v_max: max linear velocity
            w_max: max angular velocity
        """
        self.max_pos = max_pos
        self.max_heading = max_heading
        self.prob = prob
        self.resolution = resolution
        self.v_max = v_max
        self.w_max = w_max

    def __call__(self, bev, label):
        """
        Apply augmentation with probability self.prob.

        Args:
            bev: (C, 200, 200) numpy array (C=5 channels)
            label: (2,) numpy array [v, omega]

        Returns:
            bev: possibly perturbed
            label: possibly corrective velocity
        """
        if np.random.random() < self.prob:
            bev, v_new, w_new = perturb_and_relabel(
                bev, label[0], label[1],
                max_pos_perturb=self.max_pos,
                max_heading_perturb=self.max_heading,
                resolution=self.resolution,
                v_max=self.v_max,
                w_max=self.w_max
            )
            label = np.array([v_new, w_new], dtype=np.float32)

        return bev, label


# ========================================================================
# MIRROR AUGMENTATION (Flaw 5 fix)
# ========================================================================

def mirror_augment(bev, label):
    """
    Mirror augmentation: flip BEV horizontally, negate angular velocity.

    Indoor environments are approximately left-right symmetric.
    This gives 2x free training data. Ratatouille (Sep'25) showed
    3x success rate improvement and 6x fewer collisions.

    Args:
        bev: (C, H, W) BEV grid
        label: (2,) [v, omega] single-step label
               OR (T, 2) [v, omega] trajectory label

    Returns:
        bev_mirrored: (C, H, W) horizontally flipped
        label_mirrored: label with negated omega
    """
    # Flip BEV left-right (axis=-1 is the W dimension)
    bev_mirrored = np.flip(bev, axis=-1).copy()

    # Negate angular velocity (omega → -omega)
    label_mirrored = label.copy()
    if label.ndim == 1:
        # Single step: (2,) = [v, omega]
        label_mirrored[1] = -label_mirrored[1]
    elif label.ndim == 2:
        # Trajectory: (T, 2) = [[v, omega], ...]
        label_mirrored[:, 1] = -label_mirrored[:, 1]

    return bev_mirrored, label_mirrored


class MirrorAugmentor:
    """Apply mirror augmentation with given probability."""

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, bev, label):
        if np.random.random() < self.prob:
            return mirror_augment(bev, label)
        return bev, label


# ========================================================================
# TRAJECTORY-LEVEL CHAUFFEURNET PERTURBATION (Flaw 8 fix)
# ========================================================================

def perturb_trajectory(bev, traj_label, max_pos=0.3, max_heading=30,
                       resolution=0.05, v_max=0.25, w_max=1.0, dt=0.1):
    """
    ChauffeurNet-style perturbation for trajectory-level labels.

    Unlike single-step perturb_and_relabel, this:
    1. Perturbs the initial pose
    2. Computes a full corrective trajectory (not just one velocity)
    3. Blends corrective trajectory with original expert trajectory

    This teaches the model to recover from deviations over multiple steps,
    not just at a single timestep.

    Args:
        bev: (C, H, W) BEV grid
        traj_label: (T, 2) trajectory [v, omega] per timestep
        max_pos: max position perturbation (meters)
        max_heading: max heading perturbation (degrees)
        resolution: BEV grid resolution
        v_max: max linear velocity
        w_max: max angular velocity
        dt: time between trajectory steps

    Returns:
        bev_perturbed: (C, H, W) transformed BEV
        traj_corrective: (T, 2) corrective trajectory
    """
    T = traj_label.shape[0]

    # Random initial perturbation
    dx = np.random.uniform(-max_pos, max_pos)
    dy = np.random.uniform(-max_pos, max_pos)
    dtheta = np.radians(np.random.uniform(-max_heading, max_heading))

    # Perturb BEV
    bev_perturbed = affine_transform_bev(bev, dx, dy, dtheta, resolution)

    # Compute corrective trajectory
    # The correction decays over time (exponential blend back to expert)
    traj_corrective = traj_label.copy()

    # Current perturbation state
    cur_dx, cur_dy, cur_dtheta = dx, dy, dtheta
    kp_v = 0.5
    kp_omega = 2.0
    decay = 0.7  # Each step reduces perturbation by 30%

    for t in range(T):
        dist = np.sqrt(cur_dx**2 + cur_dy**2)
        angle_to_origin = np.arctan2(-cur_dy, -cur_dx) - cur_dtheta

        # Corrective component
        v_correct = np.clip(kp_v * dist, 0, v_max)
        w_correct = np.clip(kp_omega * angle_to_origin, -w_max, w_max)

        # Blend: corrective at start, expert trajectory at end
        blend = decay ** t  # Goes from 1.0 → 0 exponentially
        traj_corrective[t, 0] = blend * v_correct + (1 - blend) * traj_label[t, 0]
        traj_corrective[t, 1] = blend * w_correct + (1 - blend) * traj_label[t, 1]

        # Simulate: reduce perturbation by the corrective velocity
        cur_dx -= traj_corrective[t, 0] * np.cos(cur_dtheta) * dt
        cur_dy -= traj_corrective[t, 0] * np.sin(cur_dtheta) * dt
        cur_dtheta -= traj_corrective[t, 1] * dt
        cur_dx *= decay
        cur_dy *= decay
        cur_dtheta *= decay

    return bev_perturbed, traj_corrective


class TrajectoryAugmentor:
    """Combined augmentation pipeline for trajectory-level training.

    Applies both ChauffeurNet perturbation AND mirror augmentation.
    """

    def __init__(self, perturb_prob=0.5, mirror_prob=0.5,
                 max_pos=0.3, max_heading=30,
                 resolution=0.05, v_max=0.25, w_max=1.0, dt=0.1):
        self.perturb_prob = perturb_prob
        self.mirror_prob = mirror_prob
        self.max_pos = max_pos
        self.max_heading = max_heading
        self.resolution = resolution
        self.v_max = v_max
        self.w_max = w_max
        self.dt = dt

    def __call__(self, bev, traj_label):
        """
        Args:
            bev: (C, H, W) BEV grid
            traj_label: (T, 2) trajectory [v, omega]

        Returns:
            bev: augmented BEV
            traj_label: augmented trajectory
        """
        # ChauffeurNet perturbation
        if np.random.random() < self.perturb_prob:
            bev, traj_label = perturb_trajectory(
                bev, traj_label,
                max_pos=self.max_pos,
                max_heading=self.max_heading,
                resolution=self.resolution,
                v_max=self.v_max,
                w_max=self.w_max,
                dt=self.dt,
            )

        # Mirror augmentation
        if np.random.random() < self.mirror_prob:
            bev, traj_label = mirror_augment(bev, traj_label)

        return bev, traj_label


# ========================================================================
# POLAR SCAN AUGMENTATION (for ModularKinoFlowNet v2)
# ========================================================================

def mirror_scan(scan_ranges, scan_residuals, traj_label):
    """Mirror augmentation for polar scan data.

    Reverses the scan array (flips angular direction) and negates omega.
    Equivalent to mirror_augment for BEV but operates directly on polar scans.

    Args:
        scan_ranges: (N,) polar scan ranges
        scan_residuals: (T-1, N) temporal residual channels
        traj_label: (T, 2) trajectory [v, omega]

    Returns:
        scan_mirrored: (N,) reversed scan
        residuals_mirrored: (T-1, N) reversed residuals
        traj_mirrored: (T, 2) with negated omega
    """
    scan_mirrored = np.flip(scan_ranges).copy()
    residuals_mirrored = np.flip(scan_residuals, axis=-1).copy()
    traj_mirrored = traj_label.copy()
    if traj_mirrored.ndim == 1:
        traj_mirrored[1] = -traj_mirrored[1]
    else:
        traj_mirrored[:, 1] = -traj_mirrored[:, 1]
    return scan_mirrored, residuals_mirrored, traj_mirrored


def perturb_scan(scan_ranges, noise_std=0.02, dropout_prob=0.01):
    """Add noise and random dropout to polar scan for robustness.

    Args:
        scan_ranges: (N,) polar scan ranges
        noise_std: standard deviation of Gaussian noise (meters)
        dropout_prob: probability of dropping each ray (set to inf)

    Returns:
        scan_perturbed: (N,) noised scan
    """
    scan_perturbed = scan_ranges.copy()
    # Gaussian noise on valid ranges
    valid = np.isfinite(scan_perturbed) & (scan_perturbed > 0.1)
    noise = np.random.normal(0, noise_std, scan_perturbed.shape)
    scan_perturbed[valid] += noise[valid]
    scan_perturbed[valid] = np.maximum(scan_perturbed[valid], 0.1)
    # Random ray dropout
    dropout_mask = np.random.random(scan_perturbed.shape) < dropout_prob
    scan_perturbed[dropout_mask] = np.inf
    return scan_perturbed.astype(np.float32)


class ModularTrajectoryAugmentor:
    """Augmentation pipeline for ModularKinoFlowDataset.

    Applies polar scan augmentation (mirror + noise) and trajectory augmentation.
    """

    def __init__(self, mirror_prob=0.5, noise_prob=0.3,
                 noise_std=0.02, dropout_prob=0.01):
        self.mirror_prob = mirror_prob
        self.noise_prob = noise_prob
        self.noise_std = noise_std
        self.dropout_prob = dropout_prob

    def __call__(self, scan_current, scan_residuals, traj_label):
        """
        Args:
            scan_current: (N,) polar scan
            scan_residuals: (T-1, N) residuals
            traj_label: (T, 2) trajectory [v, omega]

        Returns:
            scan_current, scan_residuals, traj_label (augmented)
        """
        # Mirror augmentation
        if np.random.random() < self.mirror_prob:
            scan_current, scan_residuals, traj_label = mirror_scan(
                scan_current, scan_residuals, traj_label)

        # Scan noise
        if np.random.random() < self.noise_prob:
            scan_current = perturb_scan(
                scan_current, self.noise_std, self.dropout_prob)

        return scan_current, scan_residuals, traj_label
