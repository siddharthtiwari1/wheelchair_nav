#!/usr/bin/env python3
"""
Results Visualization for Quantum Navigation Experiments.

Generates paper-ready figures from evaluation results:
  1. Real-data phase diagram (main figure)
  2. Ablation heatmap (quantum vs kNN correlation)
  3. Score correlation scatter plots
  4. Confidence calibration curves
  5. Mode transition timeline
  6. Computation time comparison

Usage:
    python quantum_plot_results.py \
        --results_dir /home/sidd/wheelchair_nav/quantum_eval_results \
        --output_dir /home/sidd/wheelchair_nav/docs/quantum_nav_figures
"""

import os
import sys
import argparse
import glob
import json
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap

# Paper-quality settings
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 200,
    'savefig.dpi': 300,
})

# Color palette (colorblind-friendly)
COLORS = {
    'quantum_sup_3q': '#2ca02c',        # green
    'quantum_sup_2q': '#17becf',         # cyan
    'quantum_random_3q': '#ff7f0e',      # orange
    'quantum_cost_encoded': '#d62728',   # red
    'quantum_phase': '#9467bd',          # purple
    'quantum_temporal': '#8c564b',       # brown
    'quantum_adaptive': '#e377c2',       # pink
    'knn_mean': '#1f77b4',              # blue
    'knn_weighted': '#7f7f7f',           # gray
    'mppi': '#bcbd22',                   # olive
    'classical_best': '#aec7e8',         # light blue
    'identity': '#c7c7c7',              # light gray
}

DISPLAY_NAMES = {
    'quantum_sup_3q': 'Quantum 3q (sup)',
    'quantum_sup_2q': 'Quantum 2q (sup)',
    'quantum_random_3q': 'Quantum 3q (rand)',
    'quantum_cost_encoded': 'Cost-encoded rho',
    'quantum_phase': 'Phase interference',
    'quantum_temporal': 'Temporal entanglement',
    'quantum_adaptive': 'Adaptive quantum',
    'knn_mean': 'k-NN mean',
    'knn_weighted': 'Weighted k-NN',
    'mppi': 'MPPI',
    'classical_best': 'Classical best-cost',
    'identity': 'Identity (raw cost)',
}


# ── Figure 1: Real-Data Phase Diagram ──────────────────────────────

def fig_phase_diagram(noise_dir, output_dir):
    """Plot phi vs calibrated noise for all planners.

    THE paper's main figure.
    """
    phase_file = os.path.join(noise_dir, 'phase_data_gaussian.npz')
    if not os.path.exists(phase_file):
        print(f"  Skipping phase diagram: {phase_file} not found")
        return

    data = np.load(phase_file, allow_pickle=True)
    levels = data['levels']
    planner_names = list(data['planner_names'])

    fig, ax = plt.subplots(figsize=(10, 6))

    for name in planner_names:
        key = f'phi_{name}'
        if key not in data:
            continue
        phi = data[key]  # (n_levels, n_timesteps)
        phi_mean = np.mean(phi, axis=1)
        phi_std = np.std(phi, axis=1)

        color = COLORS.get(name, '#333333')
        label = DISPLAY_NAMES.get(name, name)

        # Determine line style
        ls = '-'
        if 'random' in name or 'classical' in name or 'identity' in name:
            ls = '--'
        elif 'mppi' in name:
            ls = '-.'

        ax.errorbar(levels, phi_mean, yerr=phi_std,
                     label=label, color=color, lw=2.5, capsize=3,
                     marker='o', markersize=5, ls=ls)

    # Exploit/explore threshold
    ax.axhline(y=0.6, color='gray', ls=':', alpha=0.5, lw=1.5)
    ax.text(levels[-1] * 0.85, 0.62, 'Exploit/Explore\nthreshold',
            fontsize=8, color='gray', ha='center')

    # Sensor reference lines
    sensor_noise = {
        'RPLidar': 0.01, 'D455@3m': 0.036, 'D435i@3m': 0.068,
    }
    for sensor, sigma in sensor_noise.items():
        if sigma <= levels[-1]:
            ax.axvline(x=sigma, color='lightblue', ls=':', alpha=0.4)
            ax.text(sigma, 1.02, sensor, fontsize=7, rotation=45,
                    ha='left', va='bottom', color='steelblue')

    ax.set_xlabel('Range Noise $\\sigma$ (meters)', fontsize=13)
    ax.set_ylabel('Decision Confidence $\\phi$', fontsize=13)
    ax.set_title('Real-Data Phase Diagram: Planner Confidence vs Sensor Noise',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, loc='lower left', ncol=2)
    ax.grid(True, alpha=0.2)
    ax.set_xlim(-0.02, levels[-1] * 1.05)
    ax.set_ylim(-0.05, 1.1)

    path = os.path.join(output_dir, 'real_phase_diagram.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Figure 2: Ablation Heatmap ─────────────────────────────────────

def fig_ablation_heatmap(ablation_dir, output_dir):
    """Heatmap of pairwise Spearman correlations between planners."""
    summary_file = os.path.join(ablation_dir, 'ablation_summary.json')
    if not os.path.exists(summary_file):
        print(f"  Skipping ablation heatmap: {summary_file} not found")
        return

    with open(summary_file) as f:
        summary = json.load(f)

    correlations = summary['correlations']

    # Extract unique planner names
    all_planners = set()
    for pair_key in correlations:
        parts = pair_key.split('_vs_')
        all_planners.add(parts[0])
        all_planners.add(parts[1])

    planner_list = sorted(all_planners)
    n = len(planner_list)
    corr_matrix = np.eye(n)

    for pair_key, vals in correlations.items():
        parts = pair_key.split('_vs_')
        if len(parts) == 2:
            i = planner_list.index(parts[0]) if parts[0] in planner_list else -1
            j = planner_list.index(parts[1]) if parts[1] in planner_list else -1
            if i >= 0 and j >= 0:
                corr_matrix[i, j] = vals['mean']
                corr_matrix[j, i] = vals['mean']

    fig, ax = plt.subplots(figsize=(10, 8))

    # Custom colormap: red (different) -> white (uncorrelated) -> green (similar)
    cmap = LinearSegmentedColormap.from_list(
        'rg', ['#d62728', '#ffffff', '#2ca02c'], N=256)

    im = ax.imshow(corr_matrix, cmap=cmap, vmin=-1, vmax=1, aspect='auto')
    plt.colorbar(im, ax=ax, label='Spearman $\\rho$', shrink=0.8)

    # Labels
    labels = [DISPLAY_NAMES.get(p, p) for p in planner_list]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)

    # Annotate cells
    for i in range(n):
        for j in range(n):
            val = corr_matrix[i, j]
            if i != j:
                color = 'black' if abs(val) < 0.7 else 'white'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                        fontsize=7, color=color)

    ax.set_title('Ablation: Pairwise Score Rank Correlations\n'
                 '(High = similar ranking, red = genuinely different)',
                 fontsize=13, fontweight='bold')

    path = os.path.join(output_dir, 'ablation_heatmap.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Figure 3: Score Correlation Scatter ─────────────────────────────

def fig_score_scatter(ablation_dir, output_dir):
    """Scatter plots of quantum vs kNN scores for key scenarios."""
    # Find ablation result files
    ablation_files = sorted(glob.glob(
        os.path.join(ablation_dir, 'ablation_*.npz')))

    if not ablation_files:
        print(f"  Skipping score scatter: no ablation files found")
        return

    # Pick up to 4 scenarios
    files = ablation_files[:4]
    n_plots = len(files)

    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4.5))
    if n_plots == 1:
        axes = [axes]

    for idx, f in enumerate(files):
        ax = axes[idx]
        data = np.load(f, allow_pickle=True)
        scenario = os.path.basename(f).replace('ablation_', '').replace('.npz', '')

        # Get scores at noise=0.3 (moderate)
        for noise_idx in [2]:  # index 2 = eta=0.3
            q_key = 'scores_quantum_sup_3q'
            k_key = 'scores_knn_mean'
            ce_key = 'scores_quantum_cost_encoded'

            if q_key in data and k_key in data:
                q_scores = data[q_key][noise_idx]
                k_scores = data[k_key][noise_idx]

                # Normalize for comparison
                q_norm = (q_scores - q_scores.min()) / (q_scores.max() - q_scores.min() + 1e-12)
                k_norm = (k_scores - k_scores.min()) / (k_scores.max() - k_scores.min() + 1e-12)

                ax.scatter(k_norm, q_norm, alpha=0.5, s=15, c='#2ca02c',
                           label='Quantum sup 3q')

                # Perfect correlation line
                ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, lw=1)

                # Cost-encoded if available
                if ce_key in data:
                    ce_scores = data[ce_key][noise_idx]
                    ce_norm = (ce_scores - ce_scores.min()) / (ce_scores.max() - ce_scores.min() + 1e-12)
                    ax.scatter(k_norm, ce_norm, alpha=0.5, s=15, c='#d62728',
                               marker='^', label='Cost-encoded')

        ax.set_xlabel('k-NN Mean Score (normalized)')
        if idx == 0:
            ax.set_ylabel('Quantum Score (normalized)')
        ax.set_title(scenario.replace('_', ' ').title(), fontsize=11)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.2)
        if idx == 0:
            ax.legend(fontsize=8, loc='upper left')

    fig.suptitle('Score Correlation: Quantum vs k-NN Mean\n'
                 '(Points on diagonal = equivalent scoring)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()

    path = os.path.join(output_dir, 'score_correlation_scatter.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Figure 4: Confidence Timeline ──────────────────────────────────

def fig_confidence_timeline(eval_dir, output_dir):
    """Time-series of confidence (phi) for all planners on a session."""
    # Find eval results
    eval_files = sorted(glob.glob(
        os.path.join(eval_dir, 'session_*/eval_results.npz')))

    if not eval_files:
        print(f"  Skipping confidence timeline: no eval results found")
        return

    # Use first session
    data = np.load(eval_files[0], allow_pickle=True)
    planner_names = list(data['planner_names'])
    n_ts = int(data['n_timesteps']) if 'n_timesteps' in data else \
        data['confidence'].shape[1]

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Top: confidence over time
    ax = axes[0]
    t = np.arange(n_ts) / 10.0  # seconds at 10Hz

    for pi, name in enumerate(planner_names[:7]):  # max 7 for readability
        phi = data['confidence'][pi]
        color = COLORS.get(name, f'C{pi}')
        label = DISPLAY_NAMES.get(name, name)
        ax.plot(t, phi, color=color, lw=1.2, alpha=0.8, label=label)

    ax.axhline(y=0.6, color='gray', ls=':', alpha=0.5, lw=1.5)
    ax.set_ylabel('Confidence $\\phi$')
    ax.set_title('Planner Confidence Over Time (Real Sensor Data)', fontweight='bold')
    ax.legend(fontsize=8, ncol=3, loc='lower left')
    ax.grid(True, alpha=0.2)
    ax.set_ylim(-0.05, 1.05)

    # Bottom: mode (exploit/explore) for quantum planner
    ax = axes[1]
    for pi, name in enumerate(planner_names[:3]):
        mode = data['mode'][pi]
        is_exploit = (mode == 'exploit').astype(float)
        color = COLORS.get(name, f'C{pi}')
        label = DISPLAY_NAMES.get(name, name)
        ax.fill_between(t, pi + is_exploit * 0.8, pi,
                         alpha=0.5, color=color, label=label)
        ax.text(-0.5, pi + 0.4, label, fontsize=8, va='center')

    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Exploit (filled) / Explore (empty)')
    ax.set_yticks([])
    ax.grid(True, alpha=0.2, axis='x')

    plt.tight_layout()

    path = os.path.join(output_dir, 'confidence_timeline.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Figure 5: Computation Time Comparison ───────────────────────────

def fig_computation_time(eval_dir, output_dir):
    """Box plot of computation time per planner."""
    eval_files = sorted(glob.glob(
        os.path.join(eval_dir, 'session_*/eval_results.npz')))

    if not eval_files:
        print(f"  Skipping computation time: no eval results found")
        return

    data = np.load(eval_files[0], allow_pickle=True)
    planner_names = list(data['planner_names'])

    fig, ax = plt.subplots(figsize=(12, 5))

    times_list = []
    labels = []
    colors = []

    for pi, name in enumerate(planner_names):
        t_ms = data['time_ms'][pi]
        times_list.append(t_ms[t_ms > 0])  # filter zeros
        labels.append(DISPLAY_NAMES.get(name, name))
        colors.append(COLORS.get(name, f'C{pi}'))

    bp = ax.boxplot(times_list, labels=labels, patch_artist=True,
                     showfliers=False, widths=0.6)

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_ylabel('Computation Time (ms)')
    ax.set_title('Per-Timestep Computation Time by Planner', fontweight='bold')
    ax.axhline(y=100, color='red', ls='--', alpha=0.5, lw=1)
    ax.text(len(labels) + 0.3, 100, '10Hz budget\n(100ms)', fontsize=8, color='red')
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.grid(True, alpha=0.2, axis='y')

    plt.tight_layout()

    path = os.path.join(output_dir, 'computation_time.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Figure 6: Dropout Phase Diagram ────────────────────────────────

def fig_dropout_phase(noise_dir, output_dir):
    """Phase diagram under sensor dropout (simulates camera failure)."""
    phase_file = os.path.join(noise_dir, 'phase_data_dropout.npz')
    if not os.path.exists(phase_file):
        print(f"  Skipping dropout phase: {phase_file} not found")
        return

    data = np.load(phase_file, allow_pickle=True)
    levels = data['levels']
    planner_names = list(data['planner_names'])

    fig, ax = plt.subplots(figsize=(8, 5))

    for name in planner_names:
        key = f'phi_{name}'
        if key not in data:
            continue
        phi = data[key]
        phi_mean = np.mean(phi, axis=1)
        phi_std = np.std(phi, axis=1)

        color = COLORS.get(name, '#333333')
        label = DISPLAY_NAMES.get(name, name)

        ax.errorbar(levels * 100, phi_mean, yerr=phi_std,
                     label=label, color=color, lw=2, capsize=3,
                     marker='s', markersize=5)

    ax.axhline(y=0.6, color='gray', ls=':', alpha=0.5, lw=1.5)
    ax.set_xlabel('Sensor Dropout (%)', fontsize=12)
    ax.set_ylabel('Decision Confidence $\\phi$', fontsize=12)
    ax.set_title('Confidence Under Sensor Dropout\n'
                 '(Simulates camera failure / occlusion)', fontweight='bold')
    ax.legend(fontsize=9, loc='lower left')
    ax.grid(True, alpha=0.2)

    path = os.path.join(output_dir, 'dropout_phase_diagram.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Main ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Generate quantum nav result figures')
    parser.add_argument('--results_dir', type=str,
                        default='/home/sidd/wheelchair_nav/quantum_eval_results')
    parser.add_argument('--output_dir', type=str,
                        default='/home/sidd/wheelchair_nav/docs/quantum_nav_figures')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    noise_dir = os.path.join(args.results_dir, 'noise')
    ablation_dir = os.path.join(args.results_dir, 'ablation')

    print("Generating quantum navigation figures...")
    print(f"  Results: {args.results_dir}")
    print(f"  Output:  {args.output_dir}\n")

    fig_phase_diagram(noise_dir, args.output_dir)
    fig_ablation_heatmap(ablation_dir, args.output_dir)
    fig_score_scatter(ablation_dir, args.output_dir)
    fig_confidence_timeline(args.results_dir, args.output_dir)
    fig_computation_time(args.results_dir, args.output_dir)
    fig_dropout_phase(noise_dir, args.output_dir)

    print(f"\nAll figures saved to {args.output_dir}/")


if __name__ == '__main__':
    main()
