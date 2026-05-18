#!/usr/bin/env python3
"""
Smoke Test for KinoFlow v2 Modular Architecture.

Verifies:
    1. All tensor shapes through full forward pass
    2. Parameter count (~1.3M model, ~1.8M with scorer)
    3. Backward pass (gradients flow through all modules)
    4. Single inference latency
    5. Interface compatibility (encode, generate, generate_multi_sample)
    6. Ablation flags work (each disables expected component)

Usage:
    python scripts/smoke_test.py
"""

import time
import sys
import torch
import numpy as np

# Add parent to path
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from wheelchair_e2e.models.kinoflow_net import ModularKinoFlowNet, KinoFlowNet
from wheelchair_e2e.models.scoring_network import DualSpaceScoringTransformer
from wheelchair_e2e.models.flow_matching import flow_matching_loss
from wheelchair_e2e.models.scan_encoder import StaticSceneEncoder
from wheelchair_e2e.models.dynamic_encoder import DynamicObstacleEncoder
from wheelchair_e2e.models.goal_encoder import GoalEncoder, compute_goal_features
from wheelchair_e2e.models.velocity_encoder import VelocityContextEncoder
from wheelchair_e2e.models.fusion import TransformerFusion
from wheelchair_e2e.models.trajectory_transformer import TrajectoryTransformerVectorField


def test_individual_encoders():
    """Test each encoder module independently."""
    print("\n=== Testing Individual Encoders ===")
    B = 4

    # E1: Static Scene
    e1 = StaticSceneEncoder(scan_points=720, output_dim=128)
    x1 = torch.randn(B, 720)
    out1 = e1(x1)
    assert out1.shape == (B, 128), f"E1 shape: {out1.shape}"
    p1 = sum(p.numel() for p in e1.parameters())
    print(f"  E1 StaticSceneEncoder: {out1.shape}, {p1:,} params")

    # E2: Dynamic Obstacles
    e2 = DynamicObstacleEncoder(scan_points=720, temporal_frames=5, output_dim=128)
    x2 = torch.randn(B, 4, 720)  # 4 residual channels
    out2 = e2(x2)
    assert out2.shape == (B, 128), f"E2 shape: {out2.shape}"
    p2 = sum(p.numel() for p in e2.parameters())
    print(f"  E2 DynamicObstacleEncoder: {out2.shape}, {p2:,} params")

    # E3: Goal
    e3 = GoalEncoder(output_dim=64)
    x3 = torch.randn(B, 4)
    out3 = e3(x3)
    assert out3.shape == (B, 64), f"E3 shape: {out3.shape}"
    p3 = sum(p.numel() for p in e3.parameters())
    print(f"  E3 GoalEncoder: {out3.shape}, {p3:,} params")

    # Test compute_goal_features
    gf = compute_goal_features(3.0, 2.0)
    assert len(gf) == 4
    print(f"  Goal features (3.0, 2.0): {[f'{x:.3f}' for x in gf]}")

    # E4: Velocity Context
    e4 = VelocityContextEncoder(input_dim=30, output_dim=64)
    x4 = torch.randn(B, 30)
    out4 = e4(x4)
    assert out4.shape == (B, 64), f"E4 shape: {out4.shape}"
    p4 = sum(p.numel() for p in e4.parameters())
    print(f"  E4 VelocityContextEncoder: {out4.shape}, {p4:,} params")

    total_enc = p1 + p2 + p3 + p4
    print(f"  Total encoder params: {total_enc:,}")
    return True


def test_fusion():
    """Test Transformer fusion module."""
    print("\n=== Testing Fusion ===")
    B = 4

    fusion = TransformerFusion(
        scene_dim=128, dynamic_dim=128, goal_dim=64, odom_dim=64,
        d_model=128, cond_dim=256, nhead=4, num_layers=2)

    scene = torch.randn(B, 128)
    dynamic = torch.randn(B, 128)
    goal = torch.randn(B, 64)
    odom = torch.randn(B, 64)

    out = fusion(scene, dynamic, goal, odom)
    assert out.shape == (B, 256), f"Fusion shape: {out.shape}"
    p = sum(p.numel() for p in fusion.parameters())
    print(f"  TransformerFusion: {out.shape}, {p:,} params")

    # Test concat ablation
    fusion_concat = TransformerFusion(
        scene_dim=128, dynamic_dim=128, goal_dim=64, odom_dim=64,
        d_model=128, cond_dim=256, concat_fusion=True)
    out_concat = fusion_concat(scene, dynamic, goal, odom)
    assert out_concat.shape == (B, 256)
    print(f"  Concat fusion ablation: OK")
    return True


def test_trajectory_transformer():
    """Test Trajectory Transformer vector field."""
    print("\n=== Testing Trajectory Transformer ===")
    B = 4
    H = 10

    vf = TrajectoryTransformerVectorField(
        horizon=H, cond_dim=256, d_model=128, nhead=4, num_layers=2)

    x_t = torch.randn(B, H * 2)
    t = torch.rand(B)
    cond = torch.randn(B, 256)

    out = vf(x_t, t, cond)
    assert out.shape == (B, H * 2), f"VF shape: {out.shape}"
    p = sum(p.numel() for p in vf.parameters())
    print(f"  TrajectoryTransformerVectorField: {out.shape}, {p:,} params")
    return True


def test_full_model():
    """Test full ModularKinoFlowNet."""
    print("\n=== Testing Full ModularKinoFlowNet ===")
    B = 4
    device = 'cpu'

    model = ModularKinoFlowNet(
        v_max=0.25, w_max=1.0, horizon=10, n_euler_steps=3,
        n_samples=8, scan_points=720, temporal_frames=5,
    ).to(device)

    scan = torch.randn(B, 720, device=device)
    residuals = torch.randn(B, 4, 720, device=device)
    goal = torch.randn(B, 4, device=device)
    odom = torch.randn(B, 30, device=device)

    # Test encode
    cond, hidden = model.encode(scan, residuals, goal, odom)
    assert cond.shape == (B, 256), f"Cond shape: {cond.shape}"
    print(f"  encode(): cond={cond.shape}, hidden={hidden.shape}")

    # Test forward
    cond2, hidden2 = model(scan, residuals, goal, odom)
    assert cond2.shape == (B, 256)
    print(f"  forward(): OK")

    # Test generate
    vel_traj, poses, hidden3 = model.generate(scan, residuals, goal, odom)
    assert vel_traj.shape == (B, 10, 2), f"vel_traj: {vel_traj.shape}"
    assert poses.shape == (B, 10, 3), f"poses: {poses.shape}"
    print(f"  generate(): vel={vel_traj.shape}, poses={poses.shape}")

    # Test generate_multi_sample
    bev_occ = torch.zeros(1, 200, 200, device=device)
    (best_vel, best_poses, all_vel, all_poses, best_idx,
     scores, hidden4) = model.generate_multi_sample(
        scan[:1], residuals[:1], goal[:1], odom[:1],
        bev_occupancy=bev_occ, goal_dx=2.0, goal_dy=1.0)
    assert best_vel.shape == (1, 10, 2)
    assert all_vel.shape == (8, 10, 2)
    assert scores.shape == (8,)
    print(f"  generate_multi_sample(): K=8, best_idx={best_idx}, "
          f"scores={[f'{s:.2f}' for s in scores.tolist()]}")

    # Param count
    pc = model.get_param_count()
    print(f"\n  Parameter breakdown:")
    for k, v in pc.items():
        print(f"    {k}: {v:,}")
    print(f"    ---")
    print(f"    Total: {pc['total']:,} ({pc['total']/1e6:.2f}M)")
    return True


def test_backward():
    """Test that gradients flow through the full model."""
    print("\n=== Testing Backward Pass ===")
    B = 4
    H = 10

    model = ModularKinoFlowNet(horizon=H)

    scan = torch.randn(B, 720)
    residuals = torch.randn(B, 4, 720)
    goal = torch.randn(B, 4)
    odom = torch.randn(B, 30)

    cond, _ = model(scan, residuals, goal, odom)

    # Flow matching loss
    x_1 = torch.randn(B, H * 2)  # target
    x_0 = torch.randn(B, H * 2)  # noise
    t = torch.rand(B)

    loss = flow_matching_loss(model.vector_field, x_0, x_1, t, cond)
    loss.backward()

    # Check all parameters have gradients
    n_with_grad = 0
    n_total = 0
    for name, param in model.named_parameters():
        n_total += 1
        if param.grad is not None:
            n_with_grad += 1

    print(f"  Loss: {loss.item():.4f}")
    print(f"  Params with gradients: {n_with_grad}/{n_total}")
    assert n_with_grad == n_total, \
        f"Missing gradients in {n_total - n_with_grad} params"
    print(f"  All gradients flowing correctly!")
    return True


def test_inference_latency():
    """Time inference on CPU."""
    print("\n=== Testing Inference Latency ===")
    model = ModularKinoFlowNet(horizon=10, n_samples=8, n_euler_steps=3)
    model.eval()

    scan = torch.randn(1, 720)
    residuals = torch.randn(1, 4, 720)
    goal = torch.randn(1, 4)
    odom = torch.randn(1, 30)
    bev_occ = torch.zeros(1, 200, 200)

    # Warm up
    for _ in range(3):
        with torch.no_grad():
            model.generate_multi_sample(
                scan, residuals, goal, odom,
                bev_occupancy=bev_occ, goal_dx=2.0, goal_dy=1.0)

    # Time 20 iterations
    N = 20
    t0 = time.time()
    for _ in range(N):
        with torch.no_grad():
            model.generate_multi_sample(
                scan, residuals, goal, odom,
                bev_occupancy=bev_occ, goal_dx=2.0, goal_dy=1.0)
    elapsed = (time.time() - t0) / N * 1000

    print(f"  CPU inference (K=8, 3 Euler steps): {elapsed:.1f}ms")
    print(f"  Estimated rate: {1000/elapsed:.0f} Hz")
    return True


def test_ablations():
    """Test that ablation flags work."""
    print("\n=== Testing Ablation Flags ===")
    B = 2

    scan = torch.randn(B, 720)
    residuals = torch.randn(B, 4, 720)
    goal = torch.randn(B, 4)
    odom = torch.randn(B, 30)

    ablations = [
        ('no_dynamic', {'no_dynamic': True}),
        ('no_temporal', {'no_temporal': True}),
        ('no_goal', {'no_goal': True}),
        ('no_velocity', {'no_velocity': True}),
        ('mlp_vectorfield', {'mlp_vectorfield': True}),
        ('concat_fusion', {'concat_fusion': True}),
    ]

    base_model = ModularKinoFlowNet()
    base_params = base_model.get_param_count()['total']

    for name, kwargs in ablations:
        m = ModularKinoFlowNet(**kwargs)
        pc = m.get_param_count()['total']
        with torch.no_grad():
            vel, poses, _ = m.generate(scan, residuals, goal, odom)
        assert vel.shape == (B, 10, 2)
        diff = pc - base_params
        print(f"  --{name}: OK, {pc:,} params "
              f"({'+' if diff >= 0 else ''}{diff:,})")

    print(f"  All ablation flags working!")
    return True


def test_scorer_integration():
    """Test scorer works with modular model (backbone_dim=128)."""
    print("\n=== Testing Scorer Integration ===")

    model = ModularKinoFlowNet(horizon=10, n_samples=4)
    scorer = DualSpaceScoringTransformer(
        embed_dim=128, backbone_dim=128, horizon=10)
    model.set_learned_scorer(scorer)

    scan = torch.randn(1, 720)
    residuals = torch.randn(1, 4, 720)
    goal = torch.randn(1, 4)
    odom = torch.randn(1, 30)
    bev_occ = torch.zeros(1, 200, 200)

    with torch.no_grad():
        (best_vel, _, _, _, best_idx, scores, _) = \
            model.generate_multi_sample(
                scan, residuals, goal, odom,
                bev_occupancy=bev_occ, goal_dx=2.0, goal_dy=1.0)

    assert best_vel.shape == (1, 10, 2)
    assert scores.shape == (4,)

    scorer_params = sum(p.numel() for p in scorer.parameters())
    model_params = model.get_param_count()['total']
    total = model_params + scorer_params
    print(f"  Model: {model_params:,}, Scorer: {scorer_params:,}, "
          f"Total: {total:,} ({total/1e6:.2f}M)")
    print(f"  Learned scorer integration: OK")
    return True


def main():
    print("=" * 60)
    print("KinoFlow v2 Smoke Test")
    print("=" * 60)

    tests = [
        ("Individual Encoders", test_individual_encoders),
        ("Fusion", test_fusion),
        ("Trajectory Transformer", test_trajectory_transformer),
        ("Full Model", test_full_model),
        ("Backward Pass", test_backward),
        ("Inference Latency", test_inference_latency),
        ("Ablation Flags", test_ablations),
        ("Scorer Integration", test_scorer_integration),
    ]

    passed = 0
    failed = 0
    for name, test_fn in tests:
        try:
            result = test_fn()
            if result:
                passed += 1
            else:
                failed += 1
                print(f"  FAILED: {name}")
        except Exception as e:
            failed += 1
            print(f"  FAILED: {name} — {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'=' * 60}")
    print(f"Results: {passed}/{passed + failed} passed")
    if failed == 0:
        print("ALL TESTS PASSED!")
    else:
        print(f"{failed} TESTS FAILED")
        sys.exit(1)


if __name__ == '__main__':
    main()
