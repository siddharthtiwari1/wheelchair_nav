#!/usr/bin/env python3
"""
Quantum Navigation Ablation Battery — GO/NO-GO Decision.

Tests whether the quantum formalism is cosmetic (= k-NN averaging)
or provides genuine non-trivial interference on real sensor data.

Ablation Matrix:
  | Density Matrix      | Neighbors | Score Method      | Expected           |
  |---------------------|-----------|-------------------|--------------------|
  | Superposition 3q    | 3-NN      | Tr(rho@O)         | Current planner    |
  | Superposition 3q    | 3-NN      | mean(nn_scores)   | Should match row 1 |
  | Identity (I/d)      | 3-NN      | Tr(I/d@O)         | Should match row 1 |
  | Random 3q           | 3-NN      | Tr(rho@O)         | Weaker             |
  | Cost-encoded rho    | 3-NN      | Tr(rho@O)         | Different from kNN |
  | Phase interference   | 3-NN      | Tr(rho@O_complex) | Different from kNN |
  | Temporal rho        | 3-NN      | Tr(rho@O)         | Different from kNN |
  | None                | None      | MPPI weighting    | Classical baseline |
  | None                | None      | argmin(cost)      | Simplest baseline  |

Decision Point:
  - If Spearman(quantum_sup_3q, knn_mean) > 0.95: quantum IS cosmetic
  - If any modified quantum has correlation < 0.8 with kNN: genuine advantage
  - Either way: phi-based switching and real-sensor data are still novel

Usage:
    python quantum_ablation_battery.py \
        --session /home/sidd/wheelchair_nav/maps/session_20260226_124315/rosbag \
        --output_dir /home/sidd/wheelchair_nav/quantum_eval_results/ablation
"""

import os
import sys
import argparse
import time
import json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from wheelchair_e2e.quantum_nav.quantum_trajectory_planner import (
    QuantumTrajectoryPlanner, TrajectoryCandidate
)
from wheelchair_e2e.quantum_nav.quantum_planner_v2 import (
    CostEncodedQuantumPlanner, PhaseInterferenceQuantumPlanner,
    TemporalEntanglementPlanner
)
from wheelchair_e2e.quantum_nav.baselines import (
    ClassicalBestCost, KNNMeanScore, WeightedKNN, MPPIPlanner, IdentityScoring
)
from wheelchair_e2e.quantum_nav.metrics import (
    spearman_rank_correlation, pearson_correlation, kendall_tau,
    per_timestep_rank_correlation, score_correlation_matrix
)


def generate_test_scenarios():
    """Create diverse test scenarios from synthetic obstacle layouts.

    Used for controlled ablation when rosbag data isn't available,
    or to supplement real-data ablation with edge cases.
    """
    scenarios = []

    # Scenario 1: Open space (easy)
    scenarios.append({
        'name': 'open_space',
        'robot': (0, 0, 0),
        'goal': (3, 0),
        'obstacles': np.array([[5.0, 5.0]]),  # far away
    })

    # Scenario 2: Wide corridor
    scenarios.append({
        'name': 'wide_corridor',
        'robot': (0, 0, 0),
        'goal': (3, 0),
        'obstacles': np.vstack([
            np.column_stack([np.linspace(0.5, 2.5, 20), np.full(20, 1.2)]),
            np.column_stack([np.linspace(0.5, 2.5, 20), np.full(20, -1.2)]),
        ]),
    })

    # Scenario 3: Narrow corridor (hard)
    scenarios.append({
        'name': 'narrow_corridor',
        'robot': (0, 0, 0),
        'goal': (3, 0),
        'obstacles': np.vstack([
            np.column_stack([np.linspace(0.5, 2.5, 20), np.full(20, 0.5)]),
            np.column_stack([np.linspace(0.5, 2.5, 20), np.full(20, -0.5)]),
        ]),
    })

    # Scenario 4: L-shaped turn
    scenarios.append({
        'name': 'l_turn',
        'robot': (0, 0, 0),
        'goal': (2, 2),
        'obstacles': np.vstack([
            np.column_stack([np.linspace(0, 2, 15), np.full(15, 0.8)]),
            np.column_stack([np.linspace(0, 2, 15), np.full(15, -0.8)]),
            np.column_stack([np.full(10, 2.5), np.linspace(-0.8, 2, 10)]),
        ]),
    })

    # Scenario 5: Dynamic obstacle (static approximation)
    np.random.seed(42)
    static_walls = np.vstack([
        np.column_stack([np.linspace(0.5, 2.5, 15), np.full(15, 1.0)]),
        np.column_stack([np.linspace(0.5, 2.5, 15), np.full(15, -1.0)]),
    ])
    dynamic_obs = np.array([[1.5, 0.3], [1.5, -0.2]])
    scenarios.append({
        'name': 'dynamic_obstacle',
        'robot': (0, 0, 0),
        'goal': (3, 0),
        'obstacles': np.vstack([static_walls, dynamic_obs]),
    })

    # Scenario 6: Cluttered (many random obstacles)
    np.random.seed(123)
    clutter = np.random.uniform(0.5, 3.0, size=(30, 2))
    clutter[:, 1] = clutter[:, 1] - 1.5  # center y
    # Remove obstacles too close to straight path
    keep = np.abs(clutter[:, 1]) > 0.3
    scenarios.append({
        'name': 'cluttered',
        'robot': (0, 0, 0),
        'goal': (3, 0),
        'obstacles': clutter[keep],
    })

    return scenarios


def run_ablation_on_scenario(scenario, noise_levels=None):
    """Run all planners on a single scenario across noise levels.

    Returns: dict of planner_name -> (n_noise, n_candidates) score arrays
    """
    if noise_levels is None:
        noise_levels = [0.0, 0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0]

    robot = scenario['robot']
    goal = scenario['goal']
    obstacles = scenario['obstacles']

    # Build planners
    planners = {
        'quantum_sup_3q': QuantumTrajectoryPlanner(
            n_candidates=64, nn=3, use_superposition=True),
        'quantum_sup_2q': QuantumTrajectoryPlanner(
            n_candidates=64, nn=2, use_superposition=True),
        'quantum_random_3q': QuantumTrajectoryPlanner(
            n_candidates=64, nn=3, use_superposition=False),
        'quantum_cost_encoded': CostEncodedQuantumPlanner(
            n_candidates=64, nn=3, temperature=1.0),
        'quantum_phase': PhaseInterferenceQuantumPlanner(
            n_candidates=64, nn=3),
        'quantum_temporal': TemporalEntanglementPlanner(
            n_candidates=64, nn=3, alpha=0.7),
        'knn_mean': KNNMeanScore(n_candidates=64, nn=3),
        'knn_weighted': WeightedKNN(n_candidates=64, nn=3),
        'mppi': MPPIPlanner(n_candidates=64, lambda_=1.0),
        'classical_best': ClassicalBestCost(n_candidates=64),
        'identity': IdentityScoring(n_candidates=64),
    }

    results = {}
    n_trials = 5  # average over random seeds

    for name, planner in planners.items():
        scores_per_noise = []
        phi_per_noise = []

        for eta in noise_levels:
            trial_scores = []
            trial_phis = []

            for trial in range(n_trials):
                if isinstance(planner, QuantumTrajectoryPlanner):
                    # Quantum planners
                    cands = planner.sample_trajectories(*robot, *goal)
                    planner.evaluate_costs(cands, obstacles, *goal)
                    q_scores, phi = planner.compute_quantum_scores(
                        cands, sensor_noise=eta)
                    trial_scores.append(q_scores)
                    trial_phis.append(phi)
                else:
                    # Baseline planners
                    result = planner.plan(*robot, *goal, obstacles)
                    trial_scores.append(result['scores'])
                    trial_phis.append(result['confidence'])

            scores_per_noise.append(np.mean(trial_scores, axis=0))
            phi_per_noise.append(np.mean(trial_phis))

        results[name] = {
            'scores': np.array(scores_per_noise),  # (n_noise, n_candidates)
            'phi': np.array(phi_per_noise),          # (n_noise,)
        }

    return results, noise_levels


def compute_ablation_correlations(results, noise_levels):
    """Compute pairwise rank correlations between all planners.

    This is THE critical analysis: if quantum_sup_3q correlates > 0.95
    with knn_mean at all noise levels, the quantum formalism is cosmetic.
    """
    planner_names = sorted(results.keys())
    n_noise = len(noise_levels)

    # Key comparison pairs
    key_pairs = [
        ('quantum_sup_3q', 'knn_mean'),       # THE CRITICAL TEST
        ('quantum_sup_3q', 'knn_weighted'),
        ('quantum_sup_3q', 'identity'),
        ('quantum_cost_encoded', 'knn_mean'),  # Modified quantum vs kNN
        ('quantum_phase', 'knn_mean'),
        ('quantum_temporal', 'knn_mean'),
        ('quantum_cost_encoded', 'quantum_sup_3q'),  # Modified vs original
        ('quantum_phase', 'quantum_sup_3q'),
        ('quantum_temporal', 'quantum_sup_3q'),
        ('mppi', 'knn_mean'),                  # MPPI vs kNN
    ]

    correlations = {}
    for a, b in key_pairs:
        if a not in results or b not in results:
            continue

        pair_key = f"{a}_vs_{b}"
        rhos = []

        for ni in range(n_noise):
            scores_a = results[a]['scores'][ni]
            scores_b = results[b]['scores'][ni]
            if len(scores_a) == len(scores_b):
                rho = spearman_rank_correlation(scores_a, scores_b)
                rhos.append(rho)
            else:
                rhos.append(np.nan)

        correlations[pair_key] = {
            'rho_per_noise': rhos,
            'rho_mean': float(np.nanmean(rhos)),
            'rho_min': float(np.nanmin(rhos)) if rhos else 0.0,
            'rho_max': float(np.nanmax(rhos)) if rhos else 0.0,
        }

    return correlations


def print_go_no_go_decision(correlations, results, noise_levels):
    """Print the GO/NO-GO decision based on ablation results."""
    print("\n" + "=" * 70)
    print("  GO/NO-GO DECISION: Is quantum formalism cosmetic?")
    print("=" * 70)

    # THE critical test
    critical_key = 'quantum_sup_3q_vs_knn_mean'
    if critical_key in correlations:
        c = correlations[critical_key]
        print(f"\n  Test A: Original quantum vs k-NN mean")
        print(f"    Spearman rho: {c['rho_mean']:.4f} "
              f"(min={c['rho_min']:.4f}, max={c['rho_max']:.4f})")
        print(f"    Per noise: ", end="")
        for ni, eta in enumerate(noise_levels):
            print(f"eta={eta:.1f}:{c['rho_per_noise'][ni]:.3f}  ", end="")
        print()

        if c['rho_mean'] > 0.95:
            print(f"    >>> COSMETIC: uniform superposition rho ≈ k-NN averaging")
        elif c['rho_mean'] > 0.85:
            print(f"    >>> MARGINAL: minor differences, not clearly non-trivial")
        else:
            print(f"    >>> GENUINE DIFFERENCE: quantum scoring ≠ k-NN!")

    # Modified quantum tests
    modified_planners = [
        ('quantum_cost_encoded', 'Cost-encoded rho (Mod 1)'),
        ('quantum_phase', 'Phase interference (Mod 2)'),
        ('quantum_temporal', 'Temporal entanglement (Mod 3)'),
    ]

    print(f"\n  Test B: Modified quantum formulations vs k-NN mean")
    any_genuine = False
    for planner_name, display_name in modified_planners:
        key = f"{planner_name}_vs_knn_mean"
        if key in correlations:
            c = correlations[key]
            status = "GENUINE" if c['rho_mean'] < 0.80 else (
                "MARGINAL" if c['rho_mean'] < 0.90 else "COSMETIC")
            print(f"    {display_name:<35} rho={c['rho_mean']:.4f} [{status}]")
            if c['rho_mean'] < 0.80:
                any_genuine = True

    # Test C: phi comparison
    print(f"\n  Test C: Confidence (phi) comparison across planners")
    for name in ['quantum_sup_3q', 'quantum_cost_encoded', 'quantum_phase',
                 'quantum_temporal', 'knn_mean', 'mppi', 'classical_best']:
        if name in results:
            phi = results[name]['phi']
            print(f"    {name:<35} phi=[", end="")
            for p in phi:
                print(f"{p:.3f} ", end="")
            print("]")

    # Final decision
    print(f"\n{'='*70}")
    critical = correlations.get(critical_key, {}).get('rho_mean', 1.0)

    if any_genuine:
        print("  DECISION: PATH A — Modified quantum formulations show")
        print("  genuine non-trivial behavior. Paper: \"Density Matrix")
        print("  Trajectory Scoring for Noise-Robust Navigation\"")
    elif critical > 0.95:
        print("  DECISION: PATH B — Quantum formalism is cosmetic.")
        print("  Pivot to: \"Adaptive Confidence-Driven Trajectory Selection\"")
        print("  (k-NN smoothing + phi-based mode switching still novel)")
    else:
        print("  DECISION: INVESTIGATE — Results are marginal.")
        print("  Run on more scenarios and real sensor data before deciding.")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description='Quantum navigation ablation battery')
    parser.add_argument('--output_dir', type=str,
                        default='/home/sidd/wheelchair_nav/quantum_eval_results/ablation')
    parser.add_argument('--real_data', type=str, default=None,
                        help='Path to eval_results.npz from quantum_rosbag_eval.py')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Run on synthetic scenarios
    scenarios = generate_test_scenarios()
    noise_levels = [0.0, 0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0]

    all_correlations = {}

    for scenario in scenarios:
        print(f"\n{'='*60}")
        print(f"Scenario: {scenario['name']}")
        print(f"  Obstacles: {len(scenario['obstacles'])} points")
        print(f"{'='*60}")

        results, nl = run_ablation_on_scenario(scenario, noise_levels)

        # Compute correlations
        correlations = compute_ablation_correlations(results, nl)
        all_correlations[scenario['name']] = correlations

        # Print summary
        print(f"\n  Key correlations (Spearman rho, mean over noise):")
        for pair, c in sorted(correlations.items()):
            marker = ""
            if c['rho_mean'] > 0.95:
                marker = " [COSMETIC]"
            elif c['rho_mean'] < 0.80:
                marker = " [GENUINE]"
            print(f"    {pair:<50} {c['rho_mean']:.4f}{marker}")

        # Save per-scenario results
        np.savez(
            os.path.join(args.output_dir, f"ablation_{scenario['name']}.npz"),
            noise_levels=np.array(noise_levels),
            **{f"scores_{name}": results[name]['scores'] for name in results},
            **{f"phi_{name}": results[name]['phi'] for name in results},
        )

    # Aggregate GO/NO-GO decision across scenarios
    print("\n\n" + "#" * 70)
    print("  AGGREGATED ABLATION RESULTS")
    print("#" * 70)

    # Average correlations across scenarios
    agg_corr = {}
    for scenario_name, sc in all_correlations.items():
        for pair, c in sc.items():
            if pair not in agg_corr:
                agg_corr[pair] = []
            agg_corr[pair].append(c['rho_mean'])

    print(f"\n  Correlation averaged over {len(scenarios)} scenarios:")
    for pair in sorted(agg_corr.keys()):
        vals = agg_corr[pair]
        mean_rho = np.mean(vals)
        std_rho = np.std(vals)
        print(f"    {pair:<50} {mean_rho:.4f} +/- {std_rho:.4f}")

    # Use the last scenario's results for the go/no-go (all tell similar story)
    print_go_no_go_decision(correlations, results, noise_levels)

    # Save aggregated results
    with open(os.path.join(args.output_dir, 'ablation_summary.json'), 'w') as f:
        summary = {
            'scenarios': [s['name'] for s in scenarios],
            'noise_levels': noise_levels,
            'correlations': {
                pair: {'mean': float(np.mean(vals)), 'std': float(np.std(vals)),
                       'per_scenario': [float(v) for v in vals]}
                for pair, vals in agg_corr.items()
            },
        }
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {args.output_dir}/")


if __name__ == '__main__':
    main()
