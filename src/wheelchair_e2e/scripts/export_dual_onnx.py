#!/usr/bin/env python3
"""
Export KinoFlow v2 as dual ONNX models for C++ Nav2 controller plugin.

Exports TWO separate ONNX models from a single checkpoint:
  1. encoder.onnx (~500KB) — 4 encoders + Transformer fusion + GRU (stateless)
     Inputs: scan_current(1,720), scan_residuals(1,4,720),
             goal_features(1,4), odom_history(1,30)
     Output: conditioning(1,256)

  2. vector_field.onnx (~25KB) — TrajectoryTransformerVectorField
     Inputs: x_t(K,20), t(K,), cond(K,256)
     Output: v_field(K,20)

The C++ plugin runs:
  1. encoder.onnx once per frame → cond(1,256)
  2. Replicates cond to (K,256)
  3. Runs vector_field.onnx 3× (Euler ODE) → denoised trajectories
  4. scale + integrate + score in pure C++

Usage:
    python export_dual_onnx.py \
        --checkpoint training_data/best_model.pth \
        --output_dir models/

    # Creates: models/encoder.onnx, models/vector_field.onnx
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def parse_args():
    parser = argparse.ArgumentParser(
        description='Export KinoFlow v2 as dual ONNX for C++ plugin')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (.pth)')
    parser.add_argument('--output_dir', type=str, default='models/',
                        help='Output directory for ONNX files')
    parser.add_argument('--opset', type=int, default=17)
    parser.add_argument('--v_max', type=float, default=0.25)
    parser.add_argument('--w_max', type=float, default=1.0)
    parser.add_argument('--scan_points', type=int, default=720)
    parser.add_argument('--temporal_frames', type=int, default=5)
    parser.add_argument('--horizon', type=int, default=10)
    parser.add_argument('--n_samples', type=int, default=8,
                        help='K samples for vector field batch verification')
    parser.add_argument('--verify', action='store_true', default=True,
                        help='Verify ONNX outputs match PyTorch')
    parser.add_argument('--no-verify', dest='verify', action='store_false')
    return parser.parse_args()


class StatelessEncoderWrapper(torch.nn.Module):
    """Wraps ModularKinoFlowNet.encode() with hidden=None (stateless)."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, scan_current, scan_residuals, goal_features,
                odom_history):
        cond, _ = self.model.encode(
            scan_current, scan_residuals, goal_features, odom_history,
            hidden=None)
        return cond


class VectorFieldWrapper(torch.nn.Module):
    """Wraps the vector field forward() for ONNX export."""

    def __init__(self, vector_field):
        super().__init__()
        self.vf = vector_field

    def forward(self, x_t, t, cond):
        return self.vf(x_t, t, cond)


def verify_onnx(onnx_path, wrapper, ort_inputs, torch_inputs, name):
    """Verify ONNX output matches PyTorch within tolerance."""
    import onnxruntime as ort

    sess = ort.InferenceSession(onnx_path)
    onnx_out = sess.run(None, ort_inputs)

    with torch.no_grad():
        torch_out = wrapper(*torch_inputs).numpy()

    diff = np.abs(torch_out - onnx_out[0]).max()
    size_kb = os.path.getsize(onnx_path) / 1024
    print(f"  [{name}] Max diff: {diff:.6f}, Size: {size_kb:.1f} KB")
    assert diff < 0.05, f"Verification failed for {name}: diff={diff}"
    print(f"  [{name}] Verification PASSED")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    from wheelchair_e2e.models.kinoflow_net import ModularKinoFlowNet

    # Load model
    model = ModularKinoFlowNet(
        v_max=args.v_max, w_max=args.w_max,
        horizon=args.horizon,
        scan_points=args.scan_points,
        temporal_frames=args.temporal_frames,
    )
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    pc = model.get_param_count()
    print(f"Model loaded: {args.checkpoint}")
    print(f"Parameters: {pc['total']:,} ({pc['total']/1e6:.2f}M)")

    n_residual = args.temporal_frames - 1
    traj_dim = args.horizon * 2
    K = args.n_samples

    # ===================================================================
    # Export 1: encoder.onnx
    # ===================================================================
    print(f"\n--- Exporting encoder.onnx (opset {args.opset}) ---")

    encoder_wrapper = StatelessEncoderWrapper(model)
    encoder_wrapper.eval()

    scan_current = torch.randn(1, args.scan_points)
    scan_residuals = torch.randn(1, n_residual, args.scan_points)
    goal_features = torch.randn(1, 4)
    odom_history = torch.randn(1, 30)

    encoder_path = os.path.join(args.output_dir, 'encoder.onnx')
    torch.onnx.export(
        encoder_wrapper,
        (scan_current, scan_residuals, goal_features, odom_history),
        encoder_path,
        opset_version=args.opset,
        input_names=['scan_current', 'scan_residuals',
                     'goal_features', 'odom_history'],
        output_names=['conditioning'],
        dynamic_axes=None,
    )
    print(f"  Saved: {encoder_path}")

    if args.verify:
        verify_onnx(
            encoder_path, encoder_wrapper,
            {'scan_current': scan_current.numpy(),
             'scan_residuals': scan_residuals.numpy(),
             'goal_features': goal_features.numpy(),
             'odom_history': odom_history.numpy()},
            (scan_current, scan_residuals, goal_features, odom_history),
            'encoder')

    # ===================================================================
    # Export 2: vector_field.onnx
    # ===================================================================
    print(f"\n--- Exporting vector_field.onnx (opset {args.opset}) ---")

    vf_wrapper = VectorFieldWrapper(model.vector_field)
    vf_wrapper.eval()

    # Use K samples to match inference batch size
    x_t = torch.randn(K, traj_dim)
    t = torch.full((K,), 0.333)
    cond = torch.randn(K, model.cond_dim)

    vf_path = os.path.join(args.output_dir, 'vector_field.onnx')
    torch.onnx.export(
        vf_wrapper,
        (x_t, t, cond),
        vf_path,
        opset_version=args.opset,
        input_names=['x_t', 't', 'cond'],
        output_names=['v_field'],
        dynamic_axes={
            'x_t': {0: 'batch'},
            't': {0: 'batch'},
            'cond': {0: 'batch'},
            'v_field': {0: 'batch'},
        },
    )
    print(f"  Saved: {vf_path}")

    if args.verify:
        verify_onnx(
            vf_path, vf_wrapper,
            {'x_t': x_t.numpy(),
             't': t.numpy(),
             'cond': cond.numpy()},
            (x_t, t, cond),
            'vector_field')

    # ===================================================================
    # Summary
    # ===================================================================
    enc_size = os.path.getsize(encoder_path) / 1024
    vf_size = os.path.getsize(vf_path) / 1024
    print(f"\n{'='*60}")
    print(f"  DUAL ONNX EXPORT COMPLETE")
    print(f"  encoder.onnx:      {enc_size:.1f} KB")
    print(f"  vector_field.onnx: {vf_size:.1f} KB")
    print(f"  Total:             {(enc_size + vf_size):.1f} KB")
    print(f"{'='*60}")
    print(f"\nC++ plugin loads:")
    print(f"  encoder:      scan(1,720) + res(1,4,720) + goal(1,4) + odom(1,30) → cond(1,256)")
    print(f"  vector_field: x_t(K,{traj_dim}) + t(K,) + cond(K,256) → v(K,{traj_dim})")
    print(f"  K={K} samples, {args.horizon} horizon, 3 Euler steps")


if __name__ == '__main__':
    main()
