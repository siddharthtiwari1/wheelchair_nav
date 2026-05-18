#!/usr/bin/env python3
"""
Quantum-Inspired Trajectory Planner for Wheelchair Navigation.

Adapts the quantum-modulated Vicsek collective motion model to trajectory
selection. Instead of N agents with headings, we have N candidate trajectories
with (v, ω) commands. The density matrix creates interference patterns between
trajectory clusters — superposition states produce noise-robust decisions,
mimicking how humans hold multiple path options "in mind" simultaneously.

Mathematical mapping from swarm simulation:
    Agents (N=200)          → Candidate trajectories (N=64)
    Agent heading (θ)       → Trajectory direction (v, ω)
    Neighbor headings       → Nearby trajectory evaluations
    Density matrix ρ        → Decision state (superposition over strategies)
    Operator O(sensor)      → Cost-derived scoring matrix
    Tr(ρ ⊗ O)              → Quantum trajectory score
    Order parameter φ       → Decision confidence
    Phase transition        → Exploitation ↔ Exploration switch

Key insight from simulation: Superposition density matrices maintain coherent
decisions (φ≈0.98) even under noise η=2.0, while random/classical states
collapse at η≈0.7. For wheelchair navigation, this means robust trajectory
selection under sensor noise, occlusion, and dynamic obstacles.

Author: Sidd (adapted from quantum collective motion simulation)
"""

import numpy as np
from numpy.linalg import norm, eigvals
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import time


@dataclass
class TrajectoryCandidate:
    """A candidate (v, ω) trajectory with its evaluation."""
    v: float          # linear velocity (m/s)
    omega: float      # angular velocity (rad/s)
    cost: float = 0.0 # classical cost (lower = better)
    positions: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    headings: np.ndarray = field(default_factory=lambda: np.zeros(0))
    clearance: float = float('inf')  # min obstacle distance along path
    goal_progress: float = 0.0       # progress toward goal


@dataclass
class QuantumDecision:
    """Output of the quantum trajectory planner."""
    v: float                    # selected linear velocity
    omega: float                # selected angular velocity
    confidence: float           # order parameter φ ∈ [0, 1]
    quantum_scores: np.ndarray  # per-trajectory quantum scores
    mode: str                   # 'exploit' or 'explore'
    computation_time_ms: float  # wall clock time


class QuantumTrajectoryPlanner:
    """
    Quantum-inspired trajectory selection using density matrix interference.

    Instead of independently scoring N trajectories (classical MPPI), this
    planner constructs quantum operators from sensor data and uses density
    matrix traces to create interference between trajectory clusters:
      - Constructive interference: good trajectories near other good ones
        get amplified (human intuition: "this whole region looks safe")
      - Destructive interference: isolated good trajectories near bad ones
        get suppressed (human intuition: "too risky, narrow gap")

    The order parameter φ naturally provides a confidence metric:
      - φ > 0.7: trajectories agree → commit (exploitation)
      - φ < 0.4: trajectories disagree → sample more broadly (exploration)
    """

    def __init__(
        self,
        n_candidates: int = 64,
        nn: int = 3,            # neighbor count (2 or 3 qubits)
        use_superposition: bool = True,
        v_range: Tuple[float, float] = (0.0, 0.25),
        omega_range: Tuple[float, float] = (-0.6, 0.6),
        horizon: float = 3.0,   # seconds to simulate forward
        dt: float = 0.1,        # simulation timestep
        confidence_threshold: float = 0.6,
        wheel_separation: float = 0.565,
    ):
        self.n_candidates = n_candidates
        self.nn = nn
        self.use_superposition = use_superposition
        self.v_range = v_range
        self.omega_range = omega_range
        self.horizon = horizon
        self.dt = dt
        self.confidence_threshold = confidence_threshold
        self.wheel_sep = wheel_separation

        # Pre-compute density matrix (fixed for session, like in Vicsek model)
        self.dm = self._make_density_matrix(nn, use_superposition)

        # Pre-compute binary masks for basis states
        d = 2 ** nn
        self.masks = np.array([
            [int(c) for c in np.binary_repr(i).zfill(nn)]
            for i in range(d)
        ], dtype=np.float64)  # (d, nn)

        self.d = d

    def _make_density_matrix(self, nn: int, superposition: bool) -> np.ndarray:
        """
        Generate density matrix ρ = |ψ><ψ|.

        Superposition: |ψ> = (1/√d) Σ|i> — uniform over all basis states.
        This is the KEY insight from the simulation: uniform superposition
        creates maximal coherence, producing noise-robust decisions.
        """
        d = 2 ** nn
        if superposition:
            coeffs = np.ones(d)
        else:
            coeffs = np.random.rand(d)
        coeffs /= norm(coeffs)
        psi = coeffs.reshape(-1, 1)
        return psi @ psi.conj().T  # (d, d) density matrix

    def sample_trajectories(
        self,
        robot_x: float,
        robot_y: float,
        robot_theta: float,
        goal_x: float,
        goal_y: float,
    ) -> List[TrajectoryCandidate]:
        """
        Sample N candidate (v, ω) commands and simulate them forward.
        Uses stratified sampling: grid over (v, ω) space + jitter.
        """
        candidates = []
        n_v = int(np.sqrt(self.n_candidates))
        n_w = self.n_candidates // n_v

        vs = np.linspace(self.v_range[0], self.v_range[1], n_v)
        ws = np.linspace(self.omega_range[0], self.omega_range[1], n_w)

        n_steps = int(self.horizon / self.dt)

        for v in vs:
            for w in ws:
                # Add small jitter for diversity
                v_j = v + np.random.normal(0, 0.01)
                w_j = w + np.random.normal(0, 0.02)
                v_j = np.clip(v_j, *self.v_range)
                w_j = np.clip(w_j, *self.omega_range)

                # Forward-simulate differential drive
                positions = np.zeros((n_steps + 1, 2))
                headings = np.zeros(n_steps + 1)
                positions[0] = [robot_x, robot_y]
                headings[0] = robot_theta

                for t in range(n_steps):
                    headings[t + 1] = headings[t] + w_j * self.dt
                    positions[t + 1, 0] = positions[t, 0] + v_j * np.cos(headings[t + 1]) * self.dt
                    positions[t + 1, 1] = positions[t, 1] + v_j * np.sin(headings[t + 1]) * self.dt

                # Goal progress: how much closer do we get?
                dist_start = norm([goal_x - robot_x, goal_y - robot_y])
                dist_end = norm([goal_x - positions[-1, 0], goal_y - positions[-1, 1]])
                goal_progress = dist_start - dist_end  # positive = getting closer

                cand = TrajectoryCandidate(
                    v=v_j, omega=w_j,
                    positions=positions, headings=headings,
                    goal_progress=goal_progress,
                )
                candidates.append(cand)

        return candidates[:self.n_candidates]

    def evaluate_costs(
        self,
        candidates: List[TrajectoryCandidate],
        obstacle_points: np.ndarray,  # (M, 2) obstacle positions in robot frame
        goal_x: float,
        goal_y: float,
    ):
        """
        Compute classical costs for each trajectory.
        This feeds into the quantum operator construction.
        """
        for cand in candidates:
            # Obstacle clearance: min distance to any obstacle along path
            if len(obstacle_points) > 0:
                # Sample every 5th point for efficiency
                path_pts = cand.positions[::5]
                dists = np.linalg.norm(
                    obstacle_points[None, :, :] - path_pts[:, None, :], axis=2
                )
                cand.clearance = float(np.min(dists))
            else:
                cand.clearance = float('inf')

            # Combined cost: obstacle avoidance + goal progress + smoothness
            obstacle_cost = max(0, 0.5 - cand.clearance) * 10.0  # penalty if < 0.5m
            goal_cost = -cand.goal_progress * 2.0  # reward progress
            smoothness_cost = abs(cand.omega) * 0.3  # penalize sharp turns

            cand.cost = obstacle_cost + goal_cost + smoothness_cost

    def _find_nn_neighbors(
        self,
        candidates: List[TrajectoryCandidate],
    ) -> List[np.ndarray]:
        """
        For each trajectory, find its nn nearest neighbors in (v, ω) space.
        This mirrors the Vicsek cone-based neighbor search.
        """
        N = len(candidates)
        # Trajectory "positions" in parameter space
        params = np.array([[c.v, c.omega] for c in candidates])  # (N, 2)

        # Normalize to unit scale
        v_scale = max(self.v_range[1] - self.v_range[0], 1e-6)
        w_scale = max(self.omega_range[1] - self.omega_range[0], 1e-6)
        params_norm = params / [v_scale, w_scale]

        # Pairwise distances
        dists = np.linalg.norm(
            params_norm[:, None, :] - params_norm[None, :, :], axis=2
        )
        np.fill_diagonal(dists, np.inf)

        neighbors = []
        for i in range(N):
            nn_ids = np.argsort(dists[i])[:self.nn]
            neighbors.append(nn_ids)

        return neighbors

    def _build_quantum_operator(
        self,
        candidates: List[TrajectoryCandidate],
        nn_ids: np.ndarray,
        eta: float = 0.1,
    ) -> List[np.ndarray]:
        """
        Build quantum operator O for a trajectory using its neighbors' costs.

        This is the direct adaptation of the Vicsek model's operator construction:
          Off-diagonal: (1/nn) * (cost_score[i] + cost_score[j])
          Diagonal: η * random_noise

        Instead of neighbor orientations, we use COST SCORES (inverted cost
        so lower cost = higher score = better trajectory).

        The density matrix trace Tr(ρ ⊗ O) then creates interference:
        - Good trajectories near other good ones → constructive interference
        - Good trajectories near bad ones → destructive interference
        """
        nn = len(nn_ids)
        if nn != self.nn:
            return [np.zeros((self.d, self.d)) for _ in range(2)]

        # Convert costs to scores: high score = good trajectory
        # Use sigmoid-like transform: score = 1 / (1 + cost)
        nn_costs = np.array([candidates[j].cost for j in nn_ids])
        nn_scores = 1.0 / (1.0 + np.abs(nn_costs))  # (nn,)

        # Build operator for each "dimension" (v-dimension and ω-dimension)
        ops = []
        for dim in range(2):  # v-direction, ω-direction
            op = np.empty((self.d, self.d))

            # Project neighbor scores onto basis states via binary masks
            # masks: (d, nn), nn_scores: (nn,) → proj: (d,)
            proj = self.masks @ nn_scores  # (d,)

            unit_noise = np.random.normal()

            for i in range(self.d):
                for j in range(self.d):
                    if i != j:
                        op[i, j] = (1.0 / nn) * (proj[i] + proj[j])
                    else:
                        op[i, j] = eta * unit_noise

            ops.append(op)

        return ops

    def compute_quantum_scores(
        self,
        candidates: List[TrajectoryCandidate],
        sensor_noise: float = 0.1,
    ) -> Tuple[np.ndarray, float]:
        """
        Compute quantum scores for all trajectories using density matrix traces.

        Returns:
            scores: (N,) array of quantum scores per trajectory
            phi: order parameter (decision confidence)
        """
        N = len(candidates)
        neighbors = self._find_nn_neighbors(candidates)
        scores = np.zeros(N)

        # For each trajectory, build operator and compute trace
        all_directions = []
        for i in range(N):
            nn_ids = neighbors[i]
            ops = self._build_quantum_operator(candidates, nn_ids, eta=sensor_noise)

            # Quantum score: Tr(ρ ⊗ O) for each dimension
            score_dims = []
            for op in ops:
                trace_val = np.trace(self.dm @ op).real
                score_dims.append(trace_val)

            # Combined score across dimensions
            scores[i] = norm(score_dims)

            # Track direction for order parameter
            if norm(score_dims) > 1e-12:
                all_directions.append(np.array(score_dims) / norm(score_dims))
            else:
                all_directions.append(np.zeros(2))

        # Order parameter: φ = ||mean direction||
        # High φ = trajectories agree on direction = confident decision
        # Low φ = trajectories disagree = uncertain, explore more
        if len(all_directions) > 0:
            mean_dir = np.mean(all_directions, axis=0)
            phi = norm(mean_dir)
        else:
            phi = 0.0

        return scores, phi

    def select_trajectory(
        self,
        robot_x: float,
        robot_y: float,
        robot_theta: float,
        goal_x: float,
        goal_y: float,
        obstacle_points: np.ndarray,
        sensor_noise: float = 0.1,
    ) -> QuantumDecision:
        """
        Full quantum-inspired trajectory selection pipeline.

        1. Sample N candidate (v, ω) trajectories
        2. Evaluate classical costs (obstacles, goal, smoothness)
        3. Build quantum operators from costs
        4. Compute Tr(ρ ⊗ O) interference scores
        5. Check order parameter φ for confidence
        6. Select best trajectory or trigger exploration

        This mimics human navigation thinking:
        - Humans consider multiple paths simultaneously (superposition)
        - Paths near other good paths feel "safer" (constructive interference)
        - Isolated paths through narrow gaps feel "risky" (destructive interference)
        - Confident → commit; uncertain → slow down and look around
        """
        t0 = time.time()

        # Step 1: Sample trajectories
        candidates = self.sample_trajectories(
            robot_x, robot_y, robot_theta, goal_x, goal_y
        )

        # Step 2: Classical cost evaluation
        self.evaluate_costs(candidates, obstacle_points, goal_x, goal_y)

        # Step 3-4: Quantum scoring via density matrix traces
        q_scores, phi = self.compute_quantum_scores(candidates, sensor_noise)

        # Step 5: Decision based on confidence
        if phi > self.confidence_threshold:
            # EXPLOITATION: high confidence, commit to best quantum-scored trajectory
            mode = 'exploit'
            best_idx = np.argmax(q_scores)

            # Safety check: ensure clearance
            if candidates[best_idx].clearance < 0.3:
                # Override: stop if too close to obstacle
                v_out, omega_out = 0.0, 0.0
            else:
                v_out = candidates[best_idx].v
                omega_out = candidates[best_idx].omega
        else:
            # EXPLORATION: low confidence, slow down and use conservative trajectory
            mode = 'explore'
            # Select trajectory with best clearance (most cautious)
            clearances = np.array([c.clearance for c in candidates])
            safest_idx = np.argmax(clearances)
            v_out = candidates[safest_idx].v * 0.5  # half speed
            omega_out = candidates[safest_idx].omega * 0.5

        elapsed_ms = (time.time() - t0) * 1000.0

        return QuantumDecision(
            v=v_out,
            omega=omega_out,
            confidence=phi,
            quantum_scores=q_scores,
            mode=mode,
            computation_time_ms=elapsed_ms,
        )


class AdaptiveQuantumPlanner(QuantumTrajectoryPlanner):
    """
    Extended planner with adaptive quantum state selection.

    The simulation shows that:
    - 3-qubit superposition → most noise-robust (φ≈0.98 at η=2.0)
    - 2-qubit superposition → good but degrades at high noise
    - Random states → classical behavior (breaks at η≈0.7)

    This planner dynamically selects the quantum state based on
    estimated sensor noise level, mimicking how humans adjust their
    decision-making strategy based on uncertainty.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.noise_history = []
        self.confidence_history = []

    def estimate_noise(self, obstacle_points: np.ndarray) -> float:
        """
        Estimate sensor noise from observation consistency.
        More points with high variance → higher noise estimate.
        """
        if len(obstacle_points) < 3:
            return 0.5  # moderate default

        # Local density variation as noise proxy
        from scipy.spatial import KDTree
        try:
            tree = KDTree(obstacle_points)
            dists, _ = tree.query(obstacle_points, k=min(5, len(obstacle_points)))
            mean_spacing = np.mean(dists[:, 1:])
            spacing_var = np.var(dists[:, 1:])
            noise_estimate = spacing_var / (mean_spacing + 1e-6)
        except Exception:
            noise_estimate = 0.5

        return np.clip(noise_estimate, 0.01, 2.0)

    def adapt_quantum_state(self, noise_estimate: float):
        """
        Switch quantum strategy based on noise level.

        Low noise (η < 0.5):  2-qubit superposition (fast, sufficient)
        Med noise (η < 1.2):  3-qubit superposition (more robust)
        High noise (η > 1.2): 3-qubit superposition + more candidates
        """
        if noise_estimate < 0.5:
            if self.nn != 2:
                self.nn = 2
                self.dm = self._make_density_matrix(2, True)
                self.d = 4
                self.masks = np.array([
                    [int(c) for c in np.binary_repr(i).zfill(2)]
                    for i in range(4)
                ], dtype=np.float64)
        else:
            if self.nn != 3:
                self.nn = 3
                self.dm = self._make_density_matrix(3, True)
                self.d = 8
                self.masks = np.array([
                    [int(c) for c in np.binary_repr(i).zfill(3)]
                    for i in range(8)
                ], dtype=np.float64)

        if noise_estimate > 1.2:
            self.n_candidates = 128  # more samples under high noise
        else:
            self.n_candidates = 64

    def select_trajectory_adaptive(
        self,
        robot_x: float,
        robot_y: float,
        robot_theta: float,
        goal_x: float,
        goal_y: float,
        obstacle_points: np.ndarray,
    ) -> QuantumDecision:
        """Adaptive version: estimates noise and selects quantum strategy."""
        noise = self.estimate_noise(obstacle_points)
        self.adapt_quantum_state(noise)
        self.noise_history.append(noise)

        decision = self.select_trajectory(
            robot_x, robot_y, robot_theta,
            goal_x, goal_y, obstacle_points,
            sensor_noise=noise,
        )
        self.confidence_history.append(decision.confidence)
        return decision


# ── Standalone demo ──────────────────────────────────────────

def demo():
    """Demonstrate quantum trajectory selection on a simple scenario."""
    print("=" * 60)
    print("Quantum-Inspired Trajectory Planner — Demo")
    print("Adapted from quantum collective motion simulation")
    print("=" * 60)

    # Create planner with 3-qubit superposition (most robust)
    planner = QuantumTrajectoryPlanner(
        n_candidates=64,
        nn=3,
        use_superposition=True,
    )

    # Scenario: robot at origin, goal ahead, obstacles on sides
    robot_x, robot_y, robot_theta = 0.0, 0.0, 0.0
    goal_x, goal_y = 3.0, 0.0

    # Corridor-like obstacles
    obstacles = np.vstack([
        np.column_stack([np.linspace(0.5, 2.5, 20), np.full(20, 0.8)]),   # wall above
        np.column_stack([np.linspace(0.5, 2.5, 20), np.full(20, -0.8)]),  # wall below
    ])

    print(f"\nRobot: ({robot_x}, {robot_y}, θ={robot_theta:.1f})")
    print(f"Goal:  ({goal_x}, {goal_y})")
    print(f"Obstacles: {len(obstacles)} points (corridor walls)")
    print(f"Quantum state: {'superposition' if planner.use_superposition else 'random'} "
          f"{planner.nn}-qubit (d={planner.d})")

    # Test under increasing noise
    print(f"\n{'Noise η':>8} | {'v':>5} {'ω':>6} | {'φ':>5} | {'Mode':>8} | {'Time':>6}")
    print("-" * 55)

    for noise in [0.0, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0]:
        decision = planner.select_trajectory(
            robot_x, robot_y, robot_theta,
            goal_x, goal_y, obstacles,
            sensor_noise=noise,
        )
        print(f"{noise:8.1f} | {decision.v:5.2f} {decision.omega:6.3f} | "
              f"{decision.confidence:5.3f} | {decision.mode:>8} | "
              f"{decision.computation_time_ms:5.1f}ms")

    # Compare superposition vs random (the key finding!)
    print(f"\n{'=' * 60}")
    print("Comparison: Superposition vs Random density matrix")
    print("(This reproduces the simulation's phase transition result)")
    print(f"{'=' * 60}")

    for dm_type, use_sup in [("Superposition 3q", True), ("Random 3q", False)]:
        planner_test = QuantumTrajectoryPlanner(
            n_candidates=64, nn=3, use_superposition=use_sup,
        )
        print(f"\n{dm_type}:")
        print(f"  {'η':>4} | {'φ':>6} | {'mode':>8}")
        print(f"  {'-' * 25}")
        for noise in [0.0, 0.5, 1.0, 1.5, 2.0]:
            dec = planner_test.select_trajectory(
                robot_x, robot_y, robot_theta,
                goal_x, goal_y, obstacles,
                sensor_noise=noise,
            )
            print(f"  {noise:4.1f} | {dec.confidence:6.3f} | {dec.mode:>8}")


if __name__ == "__main__":
    demo()
