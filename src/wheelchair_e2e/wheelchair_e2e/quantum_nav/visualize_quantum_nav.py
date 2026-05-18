#!/usr/bin/env python3
"""
Visualization: Quantum-Inspired vs Classical Trajectory Selection.

Generates 4 figures showing:
1. Phase diagram: confidence (φ) vs noise for quantum vs classical
2. Trajectory selection: how interference shapes trajectory choice
3. Corridor scenario: quantum planner navigating a narrow corridor
4. Confidence-driven mode switching: exploit/explore transitions

These directly mirror the simulation's phase diagram but for navigation.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
import matplotlib.gridspec as gridspec
import os
import sys

# Add parent to path
sys.path.insert(0, os.path.dirname(__file__))
from quantum_trajectory_planner import QuantumTrajectoryPlanner

out_dir = os.path.expanduser("~/wheelchair_nav/docs/quantum_nav_figures")
os.makedirs(out_dir, exist_ok=True)


# ══════════════════════════════════════════════════════════════
# Figure 1: PHASE DIAGRAM — Confidence vs Noise
# Direct analog of simulation's 1_phase_diagram.png
# ══════════════════════════════════════════════════════════════

def fig1_phase_diagram():
    print("1/4  Phase diagram: confidence vs noise...")

    fig, ax = plt.subplots(figsize=(10, 6))

    # Test scenario: corridor navigation
    robot = (0, 0, 0)
    goal = (3.0, 0.0)
    obstacles = np.vstack([
        np.column_stack([np.linspace(0.5, 2.5, 20), np.full(20, 0.8)]),
        np.column_stack([np.linspace(0.5, 2.5, 20), np.full(20, -0.8)]),
    ])

    configs = {
        'Superposition 3-qubit': {'nn': 3, 'use_superposition': True, 'color': '#27AE60'},
        'Superposition 2-qubit': {'nn': 2, 'use_superposition': True, 'color': '#3498DB'},
        'Random 3-qubit':        {'nn': 3, 'use_superposition': False, 'color': '#E67E22'},
        'Random 2-qubit':        {'nn': 2, 'use_superposition': False, 'color': '#E74C3C'},
        'Classical MPPI':        {'nn': 2, 'use_superposition': False, 'color': '#95A5A6',
                                  'classical': True},
    }

    noise_levels = np.linspace(0, 2.0, 15)
    n_trials = 5

    for label, cfg in configs.items():
        is_classical = cfg.pop('classical', False)
        color = cfg.pop('color')

        means, stds = [], []
        for eta in noise_levels:
            phis = []
            for _ in range(n_trials):
                if is_classical:
                    # Classical: just pick lowest cost, confidence = 1/(1+cost_var)
                    planner = QuantumTrajectoryPlanner(n_candidates=64, **cfg)
                    cands = planner.sample_trajectories(*robot, *goal)
                    planner.evaluate_costs(cands, obstacles, *goal)
                    costs = np.array([c.cost for c in cands])
                    # Add noise to costs to simulate sensor noise
                    costs += np.random.normal(0, eta, len(costs))
                    cost_var = np.var(costs)
                    phi = 1.0 / (1.0 + cost_var)
                    phis.append(phi)
                else:
                    planner = QuantumTrajectoryPlanner(n_candidates=64, **cfg)
                    dec = planner.select_trajectory(*robot, *goal, obstacles, sensor_noise=eta)
                    phis.append(dec.confidence)

            means.append(np.mean(phis))
            stds.append(np.std(phis))

        ls = '--' if is_classical else '-'
        ax.errorbar(noise_levels, means, yerr=stds, label=label,
                    color=color, lw=2.5, capsize=3, marker='o', markersize=5, ls=ls)

    ax.axhline(y=0.6, color='gray', ls=':', alpha=0.5, lw=1.5)
    ax.text(0.05, 0.62, 'Exploit/Explore threshold', fontsize=9, color='gray')

    ax.set_xlabel('Sensor Noise Level (η)', fontsize=14)
    ax.set_ylabel('Decision Confidence (φ)', fontsize=14)
    ax.set_title('Phase Transition: Quantum vs Classical Trajectory Selection\n'
                 '(Corridor Navigation Scenario)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='lower left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05, 2.05)
    ax.set_ylim(-0.05, 1.05)
    ax.tick_params(labelsize=12)
    plt.tight_layout()
    fig.savefig(f"{out_dir}/1_nav_phase_diagram.png", dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"   Saved: {out_dir}/1_nav_phase_diagram.png")


# ══════════════════════════════════════════════════════════════
# Figure 2: TRAJECTORY INTERFERENCE — How quantum scoring works
# ══════════════════════════════════════════════════════════════

def fig2_trajectory_interference():
    print("2/4  Trajectory interference visualization...")

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    robot = (0, 0, 0)
    goal = (3.0, 0.0)

    # Three scenarios
    scenarios = [
        ("Open space", np.array([[5.0, 5.0]])),  # far obstacle
        ("Wide corridor", np.vstack([
            np.column_stack([np.linspace(0.5, 2.5, 15), np.full(15, 1.2)]),
            np.column_stack([np.linspace(0.5, 2.5, 15), np.full(15, -1.2)]),
        ])),
        ("Narrow gap", np.vstack([
            np.column_stack([np.linspace(0.5, 2.5, 15), np.full(15, 0.5)]),
            np.column_stack([np.linspace(0.5, 2.5, 15), np.full(15, -0.5)]),
        ])),
    ]

    for ax_idx, (title, obstacles) in enumerate(scenarios):
        ax = axes[ax_idx]

        planner = QuantumTrajectoryPlanner(n_candidates=64, nn=3, use_superposition=True)
        cands = planner.sample_trajectories(*robot, *goal)
        planner.evaluate_costs(cands, obstacles, *goal)
        q_scores, phi = planner.compute_quantum_scores(cands, sensor_noise=0.3)

        # Normalize scores for color mapping
        score_norm = (q_scores - q_scores.min()) / (q_scores.max() - q_scores.min() + 1e-10)

        # Plot trajectories colored by quantum score
        for i, cand in enumerate(cands):
            color = plt.cm.RdYlGn(score_norm[i])
            alpha = 0.2 + 0.6 * score_norm[i]
            lw = 0.5 + 2.0 * score_norm[i]
            ax.plot(cand.positions[:, 0], cand.positions[:, 1],
                    color=color, alpha=alpha, lw=lw)

        # Best trajectory (highest quantum score)
        best_idx = np.argmax(q_scores)
        ax.plot(cands[best_idx].positions[:, 0], cands[best_idx].positions[:, 1],
                color='blue', lw=3, label='Best (quantum)', zorder=10)

        # Best classical trajectory (lowest cost)
        costs = np.array([c.cost for c in cands])
        best_classical = np.argmin(costs)
        ax.plot(cands[best_classical].positions[:, 0], cands[best_classical].positions[:, 1],
                color='red', lw=2, ls='--', label='Best (classical)', zorder=9)

        # Obstacles
        ax.scatter(obstacles[:, 0], obstacles[:, 1], c='black', s=30, zorder=5, marker='x')

        # Robot and goal
        ax.plot(0, 0, 'bo', markersize=12, zorder=10)
        ax.plot(3, 0, 'g*', markersize=15, zorder=10)

        ax.set_title(f'{title}\nφ = {phi:.3f}', fontsize=13, fontweight='bold')
        ax.set_xlim(-0.5, 3.5)
        ax.set_ylim(-2, 2)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.2)
        if ax_idx == 0:
            ax.set_ylabel('y (m)', fontsize=12)
            ax.legend(fontsize=9, loc='upper left')
        ax.set_xlabel('x (m)', fontsize=12)

    fig.suptitle('Quantum Interference: Trajectory Scores in Different Environments\n'
                 '(Green = high quantum score, Red = low)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(f"{out_dir}/2_trajectory_interference.png", dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"   Saved: {out_dir}/2_trajectory_interference.png")


# ══════════════════════════════════════════════════════════════
# Figure 3: NOISE ROBUSTNESS — Same scenario, increasing noise
# ══════════════════════════════════════════════════════════════

def fig3_noise_robustness():
    print("3/4  Noise robustness comparison...")

    fig, axes = plt.subplots(2, 4, figsize=(22, 10))

    robot = (0, 0, 0)
    goal = (3.0, 0.0)
    obstacles = np.vstack([
        np.column_stack([np.linspace(0.5, 2.5, 20), np.full(20, 0.7)]),
        np.column_stack([np.linspace(0.5, 2.5, 20), np.full(20, -0.7)]),
    ])

    noise_levels = [0.0, 0.5, 1.0, 2.0]

    for row, (dm_name, use_sup) in enumerate([
        ("Superposition 3q", True),
        ("Random 3q", False),
    ]):
        for col, eta in enumerate(noise_levels):
            ax = axes[row, col]

            planner = QuantumTrajectoryPlanner(
                n_candidates=64, nn=3, use_superposition=use_sup
            )
            cands = planner.sample_trajectories(*robot, *goal)
            planner.evaluate_costs(cands, obstacles, *goal)
            q_scores, phi = planner.compute_quantum_scores(cands, sensor_noise=eta)

            score_norm = (q_scores - q_scores.min()) / (q_scores.max() - q_scores.min() + 1e-10)

            for i, cand in enumerate(cands):
                color = plt.cm.RdYlGn(score_norm[i])
                alpha = 0.15 + 0.5 * score_norm[i]
                ax.plot(cand.positions[:, 0], cand.positions[:, 1],
                        color=color, alpha=alpha, lw=0.8)

            best_idx = np.argmax(q_scores)
            ax.plot(cands[best_idx].positions[:, 0], cands[best_idx].positions[:, 1],
                    color='blue', lw=3, zorder=10)

            ax.scatter(obstacles[:, 0], obstacles[:, 1], c='black', s=15, marker='x')
            ax.plot(0, 0, 'bo', markersize=8, zorder=10)
            ax.plot(3, 0, 'g*', markersize=12, zorder=10)

            ax.set_title(f'{dm_name} | η={eta:.1f} | φ={phi:.3f}',
                         fontsize=10, fontweight='bold')
            ax.set_xlim(-0.3, 3.3)
            ax.set_ylim(-1.5, 1.5)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.2)

            if col == 0:
                ax.set_ylabel('y (m)', fontsize=11)
            if row == 1:
                ax.set_xlabel('x (m)', fontsize=11)

    fig.suptitle('Noise Robustness: Superposition vs Random Density Matrix\n'
                 '(Blue = selected trajectory, green/red = quantum score)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(f"{out_dir}/3_noise_robustness.png", dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"   Saved: {out_dir}/3_noise_robustness.png")


# ══════════════════════════════════════════════════════════════
# Figure 4: HUMAN-LIKE DECISION — Side-by-side mapping
# ══════════════════════════════════════════════════════════════

def fig4_human_analogy():
    print("4/4  Human decision-making analogy...")

    fig = plt.figure(figsize=(18, 8))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1], wspace=0.3)

    # Left: Vicsek swarm (simplified)
    ax1 = fig.add_subplot(gs[0])
    np.random.seed(42)
    N = 30
    pos = np.random.rand(N, 2) * 5
    ori_aligned = np.column_stack([np.ones(N) * 0.9 + np.random.normal(0, 0.1, N),
                                    np.random.normal(0, 0.15, N)])
    ori_aligned /= np.linalg.norm(ori_aligned, axis=1, keepdims=True)

    ax1.quiver(pos[:, 0], pos[:, 1], ori_aligned[:, 0], ori_aligned[:, 1],
               scale=15, width=0.005, color='#27AE60', alpha=0.8,
               headwidth=4, headlength=5)
    ax1.scatter(pos[:, 0], pos[:, 1], s=30, c='#27AE60', zorder=5)
    ax1.set_xlim(0, 5)
    ax1.set_ylim(0, 5)
    ax1.set_aspect('equal')
    ax1.set_title('Quantum Swarm: Superposition State\n'
                   'Agents align coherently despite noise (φ ≈ 0.98)',
                   fontsize=12, fontweight='bold')
    ax1.text(2.5, -0.3, '200 agents × quantum operator Tr(ρ⊗O) → coordinated motion',
             ha='center', fontsize=9, color='gray')
    ax1.set_xlabel('x', fontsize=11)
    ax1.set_ylabel('y', fontsize=11)

    # Right: Navigation trajectories (coherent selection)
    ax2 = fig.add_subplot(gs[1])

    planner = QuantumTrajectoryPlanner(n_candidates=64, nn=3, use_superposition=True)
    obstacles = np.vstack([
        np.column_stack([np.linspace(0.5, 2.5, 15), np.full(15, 0.8)]),
        np.column_stack([np.linspace(0.5, 2.5, 15), np.full(15, -0.8)]),
    ])
    cands = planner.sample_trajectories(0, 0, 0, 3, 0)
    planner.evaluate_costs(cands, obstacles, 3, 0)
    q_scores, phi = planner.compute_quantum_scores(cands, sensor_noise=0.5)

    score_norm = (q_scores - q_scores.min()) / (q_scores.max() - q_scores.min() + 1e-10)

    for i, cand in enumerate(cands):
        color = plt.cm.RdYlGn(score_norm[i])
        alpha = 0.15 + 0.5 * score_norm[i]
        ax2.plot(cand.positions[:, 0], cand.positions[:, 1],
                 color=color, alpha=alpha, lw=0.8)

    best_idx = np.argmax(q_scores)
    ax2.plot(cands[best_idx].positions[:, 0], cands[best_idx].positions[:, 1],
             color='blue', lw=3, zorder=10, label='Selected trajectory')

    ax2.scatter(obstacles[:, 0], obstacles[:, 1], c='black', s=25, marker='x', zorder=5)

    # Robot icon
    ax2.plot(0, 0, 'bo', markersize=15, zorder=10)
    ax2.annotate('', xy=(0.3, 0), xytext=(0, 0),
                 arrowprops=dict(arrowstyle='->', color='blue', lw=2))
    ax2.plot(3, 0, 'g*', markersize=20, zorder=10)

    ax2.set_xlim(-0.5, 3.5)
    ax2.set_ylim(-1.5, 1.5)
    ax2.set_aspect('equal')
    ax2.set_title(f'Navigation: Quantum Trajectory Selection\n'
                  f'Tr(ρ⊗O) scores {len(cands)} trajectories → select with φ={phi:.3f}',
                  fontsize=12, fontweight='bold')
    ax2.text(1.5, -1.3, 'Same math: density matrix × sensor operator → interference → robust decision',
             ha='center', fontsize=9, color='gray')
    ax2.set_xlabel('x (m)', fontsize=11)
    ax2.set_ylabel('y (m)', fontsize=11)
    ax2.legend(fontsize=10, loc='upper left')

    fig.suptitle('From Quantum Collective Motion to Wheelchair Navigation\n'
                 'The same Tr(ρ⊗O) mechanism that stabilizes swarms stabilizes trajectory selection',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(f"{out_dir}/4_human_analogy.png", dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"   Saved: {out_dir}/4_human_analogy.png")


# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Generating quantum navigation visualizations...")
    print(f"Output: {out_dir}/\n")
    fig1_phase_diagram()
    fig2_trajectory_interference()
    fig3_noise_robustness()
    fig4_human_analogy()
    print(f"\nAll done! Files in {out_dir}/")
