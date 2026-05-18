"""
Metrics for Quantum Navigation Evaluation.

Provides:
  - Spearman rank correlation (key ablation metric)
  - Trajectory quality metrics (clearance, smoothness, goal progress)
  - Confidence calibration metrics
  - Cross-planner comparison utilities
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Tuple


def spearman_rank_correlation(x, y):
    """Compute Spearman rank correlation between two arrays.

    The KEY metric for the ablation: if quantum scores and k-NN mean scores
    have rho > 0.95, the quantum formalism is cosmetic for uniform rho.
    """
    if len(x) < 3 or len(y) < 3:
        return 0.0
    rho, pvalue = stats.spearmanr(x, y)
    return float(rho) if np.isfinite(rho) else 0.0


def pearson_correlation(x, y):
    """Pearson linear correlation."""
    if len(x) < 3 or len(y) < 3:
        return 0.0
    r, pvalue = stats.pearsonr(x, y)
    return float(r) if np.isfinite(r) else 0.0


def kendall_tau(x, y):
    """Kendall's tau rank correlation — more robust than Spearman for ties."""
    if len(x) < 3 or len(y) < 3:
        return 0.0
    tau, pvalue = stats.kendalltau(x, y)
    return float(tau) if np.isfinite(tau) else 0.0


# ── Trajectory Quality Metrics ──────────────────────────────────────

def velocity_tracking_error(v_planned, omega_planned, v_actual, omega_actual):
    """RMS error between planned and actual velocities.

    Lower = planner produces commands closer to what was actually executed.
    """
    v_err = np.sqrt(np.mean((v_planned - v_actual) ** 2))
    omega_err = np.sqrt(np.mean((omega_planned - omega_actual) ** 2))
    return {'v_rmse': float(v_err), 'omega_rmse': float(omega_err)}


def smoothness(v_seq, omega_seq, dt=0.1):
    """Trajectory smoothness: mean squared jerk.

    Lower = smoother trajectory. Computed as second derivative of velocity.
    """
    if len(v_seq) < 3:
        return {'v_jerk': 0.0, 'omega_jerk': 0.0}

    v_accel = np.diff(v_seq) / dt
    v_jerk = np.diff(v_accel) / dt
    omega_accel = np.diff(omega_seq) / dt
    omega_jerk = np.diff(omega_accel) / dt

    return {
        'v_jerk': float(np.mean(v_jerk ** 2)),
        'omega_jerk': float(np.mean(omega_jerk ** 2)),
    }


def clearance_stats(clearance_seq):
    """Statistics on obstacle clearance over trajectory."""
    if len(clearance_seq) == 0:
        return {'clearance_min': 0.0, 'clearance_mean': 0.0, 'clearance_p10': 0.0}
    return {
        'clearance_min': float(np.min(clearance_seq)),
        'clearance_mean': float(np.mean(clearance_seq)),
        'clearance_p10': float(np.percentile(clearance_seq, 10)),
    }


def goal_progress_rate(goal_dist_seq, dt=0.1):
    """Rate of progress toward goal (m/s).

    Positive = getting closer, negative = moving away.
    """
    if len(goal_dist_seq) < 2:
        return 0.0
    progress = -np.diff(goal_dist_seq) / dt  # negative diff = getting closer
    return float(np.mean(progress))


# ── Confidence Calibration ──────────────────────────────────────────

def confidence_calibration(confidence_seq, success_seq, n_bins=10):
    """How well does confidence predict success?

    Bins confidence into intervals, computes actual success rate per bin.
    Perfect calibration: predicted confidence = actual success rate.

    Args:
        confidence_seq: planner confidence (phi) values
        success_seq: binary success (1 = obstacle-free, 0 = collision risk)
        n_bins: number of calibration bins

    Returns:
        Dict with calibration error and per-bin data
    """
    bins = np.linspace(0, 1, n_bins + 1)
    bin_confidences = []
    bin_success_rates = []

    for i in range(n_bins):
        mask = (confidence_seq >= bins[i]) & (confidence_seq < bins[i+1])
        if np.sum(mask) > 0:
            bin_confidences.append(np.mean(confidence_seq[mask]))
            bin_success_rates.append(np.mean(success_seq[mask]))

    if not bin_confidences:
        return {'calibration_error': 1.0, 'bins': [], 'rates': []}

    bc = np.array(bin_confidences)
    br = np.array(bin_success_rates)
    ece = float(np.mean(np.abs(bc - br)))  # Expected Calibration Error

    return {
        'calibration_error': ece,
        'bin_confidences': bc.tolist(),
        'bin_success_rates': br.tolist(),
    }


def mode_transition_analysis(mode_seq):
    """Analyze exploit/explore mode transitions.

    Returns statistics on how often and how quickly the planner switches modes.
    """
    if len(mode_seq) == 0:
        return {}

    n_exploit = np.sum(np.array(mode_seq) == 'exploit')
    n_explore = np.sum(np.array(mode_seq) == 'explore')
    total = len(mode_seq)

    # Count transitions
    transitions = 0
    exploit_streaks = []
    explore_streaks = []
    current_streak = 1

    for i in range(1, len(mode_seq)):
        if mode_seq[i] != mode_seq[i-1]:
            transitions += 1
            if mode_seq[i-1] == 'exploit':
                exploit_streaks.append(current_streak)
            else:
                explore_streaks.append(current_streak)
            current_streak = 1
        else:
            current_streak += 1

    # Final streak
    if mode_seq[-1] == 'exploit':
        exploit_streaks.append(current_streak)
    else:
        explore_streaks.append(current_streak)

    return {
        'exploit_fraction': float(n_exploit / total) if total > 0 else 0.0,
        'explore_fraction': float(n_explore / total) if total > 0 else 0.0,
        'n_transitions': transitions,
        'transition_rate': float(transitions / total) if total > 0 else 0.0,
        'mean_exploit_streak': float(np.mean(exploit_streaks)) if exploit_streaks else 0.0,
        'mean_explore_streak': float(np.mean(explore_streaks)) if explore_streaks else 0.0,
    }


# ── Score Correlation Analysis ──────────────────────────────────────

def score_correlation_matrix(score_dict):
    """Compute pairwise Spearman correlations between all planners' scores.

    Args:
        score_dict: {planner_name: (N_timesteps, N_candidates) scores}

    Returns:
        (n_planners, n_planners) correlation matrix, planner names
    """
    names = sorted(score_dict.keys())
    n = len(names)
    corr = np.eye(n)

    for i in range(n):
        for j in range(i+1, n):
            scores_i = score_dict[names[i]]
            scores_j = score_dict[names[j]]

            # Compare score rankings per timestep
            rhos = []
            for t in range(min(len(scores_i), len(scores_j))):
                si = scores_i[t]
                sj = scores_j[t]
                if len(si) == len(sj) and len(si) > 3:
                    rho = spearman_rank_correlation(si, sj)
                    if np.isfinite(rho):
                        rhos.append(rho)
            if rhos:
                corr[i, j] = np.mean(rhos)
                corr[j, i] = corr[i, j]

    return corr, names


def per_timestep_rank_correlation(scores_a, scores_b):
    """Compute Spearman correlation between two planners' scores at each timestep.

    Returns array of (n_timesteps,) correlations — the distribution
    matters more than the mean for the GO/NO-GO decision.
    """
    n = min(len(scores_a), len(scores_b))
    rhos = np.zeros(n)

    for t in range(n):
        sa = scores_a[t]
        sb = scores_b[t]
        if len(sa) == len(sb) and len(sa) > 3:
            rhos[t] = spearman_rank_correlation(sa, sb)
        else:
            rhos[t] = np.nan

    return rhos


# ── Aggregate Metrics ───────────────────────────────────────────────

def compute_all_metrics(results_dict):
    """Compute comprehensive metrics from evaluation results.

    Args:
        results_dict: output from evaluate_session() in quantum_rosbag_eval.py

    Returns:
        Dict of planner_name -> metrics_dict
    """
    planner_names = results_dict['planner_names']
    n_planners = len(planner_names)
    n_ts = results_dict['n_timesteps']

    gt_v = results_dict['gt_v']
    gt_omega = results_dict['gt_omega']
    gt_clearance = results_dict['gt_clearance']
    goal_dist = results_dict['goal_dist']

    all_metrics = {}

    for pi, name in enumerate(planner_names):
        v_planned = results_dict['v'][pi]
        omega_planned = results_dict['omega'][pi]
        confidence = results_dict['confidence'][pi]
        mode = results_dict['mode'][pi]
        time_ms = results_dict['time_ms'][pi]

        metrics = {}

        # Velocity tracking
        metrics.update(velocity_tracking_error(
            v_planned, omega_planned, gt_v, gt_omega))

        # Smoothness
        metrics.update(smoothness(v_planned, omega_planned))

        # Confidence stats
        metrics['confidence_mean'] = float(np.mean(confidence))
        metrics['confidence_std'] = float(np.std(confidence))
        metrics['confidence_min'] = float(np.min(confidence))
        metrics['confidence_max'] = float(np.max(confidence))

        # Mode analysis
        metrics.update(mode_transition_analysis(mode))

        # Timing
        metrics['time_ms_mean'] = float(np.mean(time_ms))
        metrics['time_ms_p99'] = float(np.percentile(time_ms, 99))

        # Goal progress
        metrics['goal_progress_rate'] = goal_progress_rate(goal_dist)

        # Success proxy: clearance > 0.3m
        success = gt_clearance > 0.3
        metrics.update(confidence_calibration(confidence, success.astype(float)))

        all_metrics[name] = metrics

    return all_metrics
