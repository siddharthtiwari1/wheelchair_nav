"""
Classical Baselines for Quantum Navigation Ablation.

All baselines share the same trajectory sampling and cost evaluation as the
quantum planner, differing ONLY in how they score/select trajectories. This
isolates the effect of the quantum formalism.

Baselines:
  1. ClassicalBestCost  — argmin(cost), confidence = 1/(1+cost_var)
  2. KNNMeanScore       — same neighbors, simple mean (THE CRITICAL ABLATION)
  3. WeightedKNN        — distance-weighted neighbor scores
  4. MPPIPlanner        — exp(-cost/lambda) weighting, entropy confidence
  5. IdentityScoring    — no neighbors, raw cost only
  6. RPPGroundTruth     — from /cmd_vel (requires rosbag data, not used offline)
"""

import numpy as np
from numpy.linalg import norm
from typing import List, Dict, Tuple
from dataclasses import dataclass

from wheelchair_e2e.quantum_nav.quantum_trajectory_planner import (
    TrajectoryCandidate, QuantumTrajectoryPlanner
)


class BasePlanner:
    """Base class for all baseline planners.

    Shares trajectory sampling and cost evaluation with the quantum planner
    to ensure a fair comparison.
    """

    def __init__(
        self,
        n_candidates: int = 64,
        v_range: Tuple[float, float] = (0.0, 0.25),
        omega_range: Tuple[float, float] = (-0.6, 0.6),
        horizon: float = 3.0,
        dt: float = 0.1,
        confidence_threshold: float = 0.6,
        nn: int = 3,
    ):
        self.n_candidates = n_candidates
        self.v_range = v_range
        self.omega_range = omega_range
        self.horizon = horizon
        self.dt = dt
        self.confidence_threshold = confidence_threshold
        self.nn = nn

        # Use the quantum planner's sampling/evaluation machinery
        self._sampler = QuantumTrajectoryPlanner(
            n_candidates=n_candidates,
            nn=nn,
            v_range=v_range,
            omega_range=omega_range,
            horizon=horizon,
            dt=dt,
        )

    def _sample_and_evaluate(self, robot_x, robot_y, robot_theta,
                              goal_x, goal_y, obstacle_points):
        """Sample trajectories and evaluate costs (shared across baselines)."""
        candidates = self._sampler.sample_trajectories(
            robot_x, robot_y, robot_theta, goal_x, goal_y)
        self._sampler.evaluate_costs(candidates, obstacle_points, goal_x, goal_y)
        return candidates

    def _find_nn_neighbors(self, candidates):
        """Find nearest neighbors in (v, omega) parameter space."""
        return self._sampler._find_nn_neighbors(candidates)

    def _select_from_scores(self, candidates, scores, confidence):
        """Select trajectory from scores using exploit/explore logic."""
        if confidence > self.confidence_threshold:
            mode = 'exploit'
            best_idx = np.argmax(scores)
            if candidates[best_idx].clearance < 0.3:
                v_out, omega_out = 0.0, 0.0
            else:
                v_out = candidates[best_idx].v
                omega_out = candidates[best_idx].omega
        else:
            mode = 'explore'
            clearances = np.array([c.clearance for c in candidates])
            safest_idx = np.argmax(clearances)
            v_out = candidates[safest_idx].v * 0.5
            omega_out = candidates[safest_idx].omega * 0.5

        return {
            'v': v_out,
            'omega': omega_out,
            'confidence': confidence,
            'mode': mode,
            'scores': scores,
        }

    def plan(self, robot_x, robot_y, robot_theta, goal_x, goal_y,
             obstacle_points) -> Dict:
        """Override in subclasses."""
        raise NotImplementedError


class ClassicalBestCost(BasePlanner):
    """Baseline 1: Argmin(cost), confidence = 1/(1+cost_var).

    The simplest possible planner: pick the lowest-cost trajectory.
    No neighbor information, no quantum formalism.
    """

    def plan(self, robot_x, robot_y, robot_theta, goal_x, goal_y,
             obstacle_points):
        candidates = self._sample_and_evaluate(
            robot_x, robot_y, robot_theta, goal_x, goal_y, obstacle_points)

        costs = np.array([c.cost for c in candidates])
        scores = 1.0 / (1.0 + np.abs(costs))  # invert costs to scores

        cost_var = np.var(costs)
        confidence = 1.0 / (1.0 + cost_var)

        return self._select_from_scores(candidates, scores, confidence)


class KNNMeanScore(BasePlanner):
    """Baseline 2: k-NN Mean Score — THE CRITICAL ABLATION.

    Uses the SAME k-NN neighbors as the quantum planner, but scores
    each trajectory as the simple mean of neighbor scores (no density
    matrix, no operator, no trace).

    If quantum scores correlate > 0.95 with these scores, then the
    quantum formalism is cosmetic for uniform superposition rho.
    """

    def plan(self, robot_x, robot_y, robot_theta, goal_x, goal_y,
             obstacle_points):
        candidates = self._sample_and_evaluate(
            robot_x, robot_y, robot_theta, goal_x, goal_y, obstacle_points)

        neighbors = self._find_nn_neighbors(candidates)
        costs = np.array([c.cost for c in candidates])
        # Same score transform as quantum planner
        raw_scores = 1.0 / (1.0 + np.abs(costs))

        N = len(candidates)
        scores = np.zeros(N)

        for i in range(N):
            nn_ids = neighbors[i]
            nn_scores = raw_scores[nn_ids]
            scores[i] = np.mean(nn_scores)

        # Confidence: variance of score directions (analogous to phi)
        if np.std(scores) > 1e-12:
            score_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-12)
            confidence = 1.0 - np.std(score_norm)
        else:
            confidence = 0.5

        return self._select_from_scores(candidates, scores, confidence)


class WeightedKNN(BasePlanner):
    """Baseline 3: Distance-Weighted k-NN.

    Like KNNMeanScore but weights neighbors by inverse distance
    in (v, omega) parameter space. Closer neighbors count more.
    """

    def plan(self, robot_x, robot_y, robot_theta, goal_x, goal_y,
             obstacle_points):
        candidates = self._sample_and_evaluate(
            robot_x, robot_y, robot_theta, goal_x, goal_y, obstacle_points)

        neighbors = self._find_nn_neighbors(candidates)
        costs = np.array([c.cost for c in candidates])
        raw_scores = 1.0 / (1.0 + np.abs(costs))

        # Compute pairwise distances for weighting
        params = np.array([[c.v, c.omega] for c in candidates])
        v_scale = max(self.v_range[1] - self.v_range[0], 1e-6)
        w_scale = max(self.omega_range[1] - self.omega_range[0], 1e-6)
        params_norm = params / [v_scale, w_scale]

        N = len(candidates)
        scores = np.zeros(N)

        for i in range(N):
            nn_ids = neighbors[i]
            nn_scores = raw_scores[nn_ids]
            # Inverse distance weights
            dists = norm(params_norm[nn_ids] - params_norm[i], axis=1)
            weights = 1.0 / (dists + 1e-6)
            weights /= weights.sum()
            scores[i] = np.sum(weights * nn_scores)

        if np.std(scores) > 1e-12:
            score_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-12)
            confidence = 1.0 - np.std(score_norm)
        else:
            confidence = 0.5

        return self._select_from_scores(candidates, scores, confidence)


class MPPIPlanner(BasePlanner):
    """Baseline 4: Model Predictive Path Integral (MPPI).

    Proper MPPI: scores = exp(-cost/lambda), weighted average for selection,
    entropy of the weight distribution as confidence.
    """

    def __init__(self, lambda_: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.lambda_ = lambda_

    def plan(self, robot_x, robot_y, robot_theta, goal_x, goal_y,
             obstacle_points):
        candidates = self._sample_and_evaluate(
            robot_x, robot_y, robot_theta, goal_x, goal_y, obstacle_points)

        costs = np.array([c.cost for c in candidates])

        # MPPI weights: exp(-cost/lambda)
        # Shift costs for numerical stability
        costs_shifted = costs - np.min(costs)
        weights = np.exp(-costs_shifted / self.lambda_)
        weights /= (weights.sum() + 1e-12)

        # Weighted average (v, omega)
        vs = np.array([c.v for c in candidates])
        omegas = np.array([c.omega for c in candidates])
        v_out = np.sum(weights * vs)
        omega_out = np.sum(weights * omegas)

        # Confidence from entropy of weight distribution
        # Low entropy = weights concentrated = confident
        entropy = -np.sum(weights * np.log(weights + 1e-12))
        max_entropy = np.log(len(candidates))
        confidence = 1.0 - entropy / max_entropy

        # Safety check
        best_idx = np.argmax(weights)
        if candidates[best_idx].clearance < 0.3:
            v_out *= 0.0
            omega_out *= 0.0

        if confidence < self.confidence_threshold:
            mode = 'explore'
            v_out *= 0.5
            omega_out *= 0.5
        else:
            mode = 'exploit'

        scores = weights  # use weights as "scores" for correlation analysis

        return {
            'v': v_out,
            'omega': omega_out,
            'confidence': confidence,
            'mode': mode,
            'scores': scores,
        }


class IdentityScoring(BasePlanner):
    """Baseline 6: Identity scoring — no neighbors, raw cost only.

    Each trajectory scored purely by its own cost, no neighborhood
    information. Tests whether the k-NN component adds value.
    """

    def plan(self, robot_x, robot_y, robot_theta, goal_x, goal_y,
             obstacle_points):
        candidates = self._sample_and_evaluate(
            robot_x, robot_y, robot_theta, goal_x, goal_y, obstacle_points)

        costs = np.array([c.cost for c in candidates])
        scores = 1.0 / (1.0 + np.abs(costs))

        cost_var = np.var(costs)
        confidence = 1.0 / (1.0 + cost_var)

        return self._select_from_scores(candidates, scores, confidence)


class RPPGroundTruth(BasePlanner):
    """Baseline 7: RPP ground truth from /cmd_vel.

    Not a planner — just wraps the ground truth commanded velocities
    from Nav2's RegulatedPurePursuit controller. Used for comparison.
    Requires cmd_v and cmd_omega to be passed separately.
    """

    def __init__(self, **kwargs):
        # Don't init sampler — we don't need it
        self.confidence_threshold = kwargs.get('confidence_threshold', 0.6)

    def plan(self, robot_x, robot_y, robot_theta, goal_x, goal_y,
             obstacle_points, cmd_v=0.0, cmd_omega=0.0):
        return {
            'v': cmd_v,
            'omega': cmd_omega,
            'confidence': 1.0,  # RPP is always "confident"
            'mode': 'exploit',
            'scores': np.array([1.0]),
        }
