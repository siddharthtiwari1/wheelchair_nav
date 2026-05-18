"""
BEV Grid Generator

Converts fused LaserScan (/scan_fused) to a 5-channel 200x200 BEV grid.
Used both at training time (offline, from rosbag) and inference time (real-time).

Grid specification:
    Size: 200 x 200 pixels
    Resolution: 0.05 m/pixel
    Physical coverage: 10m x 10m centered on wheelchair
    Coordinate frame: base_link (ego-centric, rotated with wheelchair)

Channels:
    0: Combined occupancy (LiDAR + camera fused obstacles)
    1: Temporal delta (|current - warped_previous| — shows MOVING obstacles)
    2: Goal direction (gaussian blob centered at relative goal position)
    3: Odometry trail (last 1s of ego-motion positions)
    4: Global route (planned path rendered as line in BEV)

Channel 1 temporal delta fixes a critical gap: without it, dynamic obstacles
are invisible to the model (just static pixels). With temporal warp, moving
objects show as bright pixels in Ch1, giving implicit velocity information.
This is the 3-line temporal BEV warp from SOTA research.
"""

import numpy as np
from scipy.ndimage import affine_transform as scipy_affine
from collections import deque

from wheelchair_e2e.models.dynamic_encoder import build_temporal_residuals


class BEVGenerator:
    """Convert fused LaserScan + route to 200x200 BEV grid with temporal delta."""

    def __init__(self, grid_size=200, resolution=0.05, temporal_decay=0.7):
        """
        Args:
            grid_size: Grid dimensions (grid_size x grid_size pixels)
            resolution: Meters per pixel (0.05 = 5cm/px)
            temporal_decay: Weight for warped previous frame (0.7 = 70% persistence)
        """
        self.grid_size = grid_size
        self.resolution = resolution
        self.center = grid_size // 2
        self.n_channels = 5
        self.temporal_decay = temporal_decay

        # State for temporal BEV warp (Flaw 4 fix)
        self.prev_occupancy = None  # (grid_size, grid_size) from previous frame
        self.prev_odom = None       # (x, y, θ) from previous frame

    def scan_to_bev(self, scan_ranges, angle_min, angle_max,
                    goal_dx, goal_dy, odom_history=None,
                    route_points=None, ego_odom=None):
        """
        Convert a LaserScan + route to a 5-channel BEV grid.

        Args:
            scan_ranges: array of range values from LaserScan
            angle_min: minimum scan angle (rad)
            angle_max: maximum scan angle (rad)
            goal_dx: goal x relative to base_link (meters)
            goal_dy: goal y relative to base_link (meters)
            odom_history: list of (x, y) relative positions from last 1s
            route_points: list of (x, y) in base_link frame from global planner
            ego_odom: (x, y, theta) current world-frame odom for temporal warp

        Returns:
            grid: (5, grid_size, grid_size) float32 numpy array
        """
        grid = np.zeros((self.n_channels, self.grid_size, self.grid_size),
                        dtype=np.float32)

        n_ranges = len(scan_ranges)
        if n_ranges == 0:
            return grid

        # --- Channel 0: Combined occupancy from fused scan ---
        angles = np.linspace(angle_min, angle_max, n_ranges)
        ranges = np.array(scan_ranges, dtype=np.float32)

        range_min = 0.1
        range_max = self.grid_size * self.resolution / 2  # 5m
        valid = (ranges > range_min) & (ranges < range_max) & np.isfinite(ranges)

        current_occ = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        if np.any(valid):
            x = ranges[valid] * np.cos(angles[valid])
            y = ranges[valid] * np.sin(angles[valid])

            px = (x / self.resolution + self.center).astype(int)
            py = (y / self.resolution + self.center).astype(int)

            mask = ((px >= 0) & (px < self.grid_size) &
                    (py >= 0) & (py < self.grid_size))

            current_occ[py[mask], px[mask]] = 1.0

        grid[0] = current_occ

        # --- Channel 1: Temporal delta (MOVING obstacle detection) ---
        # Warp previous occupancy to current frame, compute difference.
        # Moving obstacles appear as bright pixels (present in one, absent in other).
        grid[1] = self._compute_temporal_delta(current_occ, ego_odom)

        # --- Channel 2: Goal direction (gaussian blob) ---
        gx = int(goal_dx / self.resolution + self.center)
        gy = int(goal_dy / self.resolution + self.center)
        gx = np.clip(gx, 0, self.grid_size - 1)
        gy = np.clip(gy, 0, self.grid_size - 1)

        yy, xx = np.mgrid[0:self.grid_size, 0:self.grid_size]
        sigma = 10  # pixels (~0.5m spread)
        grid[2] = np.exp(-((xx - gx)**2 + (yy - gy)**2) / (2 * sigma**2))

        # --- Channel 3: Odometry trail (last 1s) ---
        if odom_history is not None:
            for ox, oy in odom_history[-10:]:
                opx = int(ox / self.resolution + self.center)
                opy = int(oy / self.resolution + self.center)
                if 0 <= opx < self.grid_size and 0 <= opy < self.grid_size:
                    grid[3, opy, opx] = 1.0

        # --- Channel 4: Global route ---
        if route_points is not None:
            self._render_route(grid[4], route_points)

        return grid

    def _compute_temporal_delta(self, current_occ, ego_odom):
        """
        Compute temporal delta channel: detects MOVING obstacles.

        Algorithm (3-line temporal BEV warp):
            1. Get odom delta (dx, dy, dθ) from EKF
            2. Affine warp previous occupancy to current ego frame
            3. Delta = |current - warped_previous|

        Moving objects show as bright pixels because they moved between frames.
        Static objects cancel out (present in both → delta ≈ 0).

        Args:
            current_occ: (grid_size, grid_size) current frame occupancy
            ego_odom: (x, y, theta) current world-frame pose, or None

        Returns:
            delta: (grid_size, grid_size) temporal delta map
        """
        if self.prev_occupancy is None or ego_odom is None or self.prev_odom is None:
            # First frame or no odom: store and return zeros
            self.prev_occupancy = current_occ.copy()
            if ego_odom is not None:
                self.prev_odom = ego_odom
            return np.zeros_like(current_occ)

        # Step 1: Compute ego-motion delta
        px, py, ptheta = self.prev_odom
        cx, cy, ctheta = ego_odom
        dx = cx - px
        dy = cy - py
        dtheta = ctheta - ptheta

        # Step 2: Affine warp previous occupancy to current frame
        cos_t = np.cos(-dtheta)
        sin_t = np.sin(-dtheta)
        dx_px = dx / self.resolution
        dy_px = dy / self.resolution

        # Rotation matrix (rotate previous frame to align with current)
        A = np.array([[cos_t, -sin_t],
                      [sin_t, cos_t]])

        # Offset: rotate around grid center, then translate
        c = self.center
        offset = np.array([c, c]) - A @ np.array([c + dy_px, c + dx_px])

        warped_prev = scipy_affine(
            self.prev_occupancy, A, offset=offset,
            order=1, mode='constant', cval=0.0
        )

        # Step 3: Temporal delta = |current - warped_previous|
        # Fuse: keep persistent obstacles via max
        fused = np.maximum(current_occ, self.temporal_decay * warped_prev)
        delta = np.abs(current_occ - warped_prev)

        # Update state for next frame
        self.prev_occupancy = current_occ.copy()
        self.prev_odom = ego_odom

        return delta

    def reset_temporal(self):
        """Reset temporal state (call on new goal or relocalization)."""
        self.prev_occupancy = None
        self.prev_odom = None

    def _render_route(self, channel, route_points):
        """
        Render a planned route as a line with width in a BEV channel.

        Uses Bresenham-style thick line drawing. Route points are in
        base_link frame (meters), converted to pixel coordinates.

        Args:
            channel: (grid_size, grid_size) array to draw on (modified in-place)
            route_points: list of (x, y) in base_link frame (meters)
        """
        if len(route_points) < 2:
            return

        line_width = 2  # pixels (~10cm wide)

        for i in range(len(route_points) - 1):
            x0_m, y0_m = route_points[i]
            x1_m, y1_m = route_points[i + 1]

            # Convert to pixel coords
            px0 = int(x0_m / self.resolution + self.center)
            py0 = int(y0_m / self.resolution + self.center)
            px1 = int(x1_m / self.resolution + self.center)
            py1 = int(y1_m / self.resolution + self.center)

            # Draw thick line between consecutive route points
            n_pts = max(abs(px1 - px0), abs(py1 - py0), 1) * 2
            for j in range(n_pts + 1):
                t = j / n_pts
                px = int(px0 + t * (px1 - px0))
                py = int(py0 + t * (py1 - py0))

                # Draw with width
                for dx in range(-line_width, line_width + 1):
                    for dy in range(-line_width, line_width + 1):
                        nx, ny = px + dx, py + dy
                        if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                            channel[ny, nx] = 1.0

    # ================================================================
    # Scan temporal stack for ModularKinoFlowNet (v2)
    # ================================================================

    def init_scan_buffer(self, temporal_frames=5):
        """Initialize scan temporal buffer for v2 modular architecture.

        Call once at startup or when resetting temporal state.
        """
        self.scan_buffer = deque(maxlen=temporal_frames)
        self.odom_delta_buffer = deque(maxlen=temporal_frames - 1)
        self.scan_angles = None
        self.temporal_frames = temporal_frames

    def update_scan_buffer(self, scan_ranges, angle_min, angle_max,
                           ego_odom=None):
        """Add a new scan frame to the temporal buffer.

        Args:
            scan_ranges: (N,) range values from LaserScan
            angle_min: minimum scan angle (rad)
            angle_max: maximum scan angle (rad)
            ego_odom: (x, y, theta) current world-frame pose for ego-compensation
        """
        if not hasattr(self, 'scan_buffer'):
            self.init_scan_buffer()

        ranges = np.array(scan_ranges, dtype=np.float32)

        # Store angles (same for all frames)
        if self.scan_angles is None or len(self.scan_angles) != len(ranges):
            self.scan_angles = np.linspace(
                angle_min, angle_max, len(ranges)).astype(np.float32)

        # Compute odom delta if we have a previous frame
        if ego_odom is not None and hasattr(self, '_prev_scan_odom') \
                and self._prev_scan_odom is not None:
            px, py, ptheta = self._prev_scan_odom
            cx, cy, ctheta = ego_odom
            delta = np.array([cx - px, cy - py, ctheta - ptheta],
                             dtype=np.float32)
            self.odom_delta_buffer.append(delta)

        self.scan_buffer.append(ranges)
        self._prev_scan_odom = ego_odom

    def get_scan_temporal_data(self):
        """Get current scan + temporal residuals for ModularKinoFlowNet.

        Returns:
            scan_current: (N,) current scan ranges (float32)
            scan_residuals: (T-1, N) temporal residuals, or zeros if buffer not full
            ready: bool, True if temporal buffer is full
        """
        if not hasattr(self, 'scan_buffer') or len(self.scan_buffer) == 0:
            return None, None, False

        scan_current = self.scan_buffer[-1].copy()
        N = len(scan_current)
        T = getattr(self, 'temporal_frames', 5)

        if len(self.scan_buffer) < T or len(self.odom_delta_buffer) < T - 1:
            # Buffer not full yet — return zeros for residuals
            residuals = np.zeros((T - 1, N), dtype=np.float32)
            return scan_current, residuals, len(self.scan_buffer) >= 2

        # Build temporal residuals via ego-compensated differencing
        scan_list = list(self.scan_buffer)
        odom_list = list(self.odom_delta_buffer)
        residuals = build_temporal_residuals(
            scan_list, odom_list, self.scan_angles
        )

        return scan_current, residuals, True

    def scan_msg_to_bev(self, scan_msg, goal_dx, goal_dy,
                        odom_history=None, route_points=None,
                        ego_odom=None):
        """
        Convenience wrapper for ROS2 LaserScan messages.

        Args:
            scan_msg: sensor_msgs/LaserScan message
            goal_dx: goal x relative to base_link (meters)
            goal_dy: goal y relative to base_link (meters)
            odom_history: list of (x, y) relative positions
            route_points: list of (x, y) from global planner in base_link frame
            ego_odom: (x, y, theta) world-frame odom for temporal warp

        Returns:
            grid: (5, 200, 200) float32 numpy array
        """
        return self.scan_to_bev(
            scan_ranges=scan_msg.ranges,
            angle_min=scan_msg.angle_min,
            angle_max=scan_msg.angle_max,
            goal_dx=goal_dx,
            goal_dy=goal_dy,
            odom_history=odom_history,
            route_points=route_points,
            ego_odom=ego_odom,
        )
