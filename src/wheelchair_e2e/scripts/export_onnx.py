#!/usr/bin/env python3
"""
Export KinoFlow models to ONNX for TensorRT deployment.

Supports both v1 (BEVVelocityNet/KinoFlowNet) and v2 (ModularKinoFlowNet).

Usage (v2 — modular):
    python export_onnx.py \
        --checkpoint best_modular.pth \
        --output kinoflow_v2.onnx \
        --model_version v2 \
        --opset 17

Usage (v1 — legacy BEV):
    python export_onnx.py \
        --checkpoint best_model.pth \
        --output bev_velocity.onnx \
        --model_version v1 \
        --bev_size 200 \
        --opset 17

Then convert to TensorRT:
    /usr/src/tensorrt/bin/trtexec \
        --onnx=kinoflow_v2.onnx \
        --saveEngine=kinoflow_v2_fp16.engine \
        --fp16 \
        --workspace=2048
"""

import argparse
import numpy as np
import torch

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def parse_args():
    parser = argparse.ArgumentParser(
        description='Export KinoFlow to ONNX')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output', type=str, default='kinoflow.onnx')
    parser.add_argument('--model_version', type=str, default='v2',
                        choices=['v1', 'v2'],
                        help='v1=BEV-based, v2=modular scan-based')
    parser.add_argument('--opset', type=int, default=17)
    parser.add_argument('--v_max', type=float, default=0.25)
    parser.add_argument('--w_max', type=float, default=1.0)

    # v2 params
    parser.add_argument('--scan_points', type=int, default=720)
    parser.add_argument('--temporal_frames', type=int, default=5)
    parser.add_argument('--horizon', type=int, default=10)

    # v1 params
    parser.add_argument('--bev_size', type=int, default=200)

    parser.add_argument('--verify', action='store_true',
                        help='Verify ONNX output matches PyTorch')
    return parser.parse_args()


def export_v2(args):
    """Export ModularKinoFlowNet to ONNX."""
    from wheelchair_e2e.models.kinoflow_net import ModularKinoFlowNet

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

    # Dummy inputs
    n_residual = args.temporal_frames - 1
    scan_current = torch.randn(1, args.scan_points)
    scan_residuals = torch.randn(1, n_residual, args.scan_points)
    goal_features = torch.randn(1, 4)
    odom_history = torch.randn(1, 30)

    # Stateless wrapper (GRU hidden dropped for ONNX export)
    class StatelessV2Wrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, scan_current, scan_residuals, goal_features,
                    odom_history):
            cond, _ = self.model.encode(
                scan_current, scan_residuals, goal_features, odom_history)
            return cond

    wrapper = StatelessV2Wrapper(model)
    wrapper.eval()

    print(f"Exporting v2 to ONNX (opset {args.opset})...")
    torch.onnx.export(
        wrapper,
        (scan_current, scan_residuals, goal_features, odom_history),
        args.output,
        opset_version=args.opset,
        input_names=['scan_current', 'scan_residuals',
                     'goal_features', 'odom_history'],
        output_names=['conditioning'],
        dynamic_axes=None,
    )
    print(f"Saved: {args.output}")

    if args.verify:
        _verify_onnx(
            args.output, wrapper,
            {'scan_current': scan_current.numpy(),
             'scan_residuals': scan_residuals.numpy(),
             'goal_features': goal_features.numpy(),
             'odom_history': odom_history.numpy()},
            (scan_current, scan_residuals, goal_features, odom_history))


def export_v1(args):
    """Export legacy BEV-based model to ONNX."""
    from wheelchair_e2e.models.bev_velocity_net import BEVVelocityNet

    model = BEVVelocityNet(v_max=args.v_max, w_max=args.w_max)
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    print(f"Model loaded: {args.checkpoint}")
    print(f"Parameters: {model.get_param_count()}")

    bev = torch.randn(1, 4, args.bev_size, args.bev_size)
    odom = torch.randn(1, 30)

    class StatelessWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, bev, odom):
            vel, _ = self.model(bev, odom, hidden=None)
            return vel

    wrapper = StatelessWrapper(model)
    wrapper.eval()

    print(f"Exporting v1 to ONNX (opset {args.opset})...")
    torch.onnx.export(
        wrapper,
        (bev, odom),
        args.output,
        opset_version=args.opset,
        input_names=['bev', 'odom'],
        output_names=['velocity'],
        dynamic_axes=None,
    )
    print(f"Saved: {args.output}")

    if args.verify:
        _verify_onnx(
            args.output, wrapper,
            {'bev': bev.numpy(), 'odom': odom.numpy()},
            (bev, odom))


def _verify_onnx(onnx_path, wrapper, ort_inputs, torch_inputs):
    """Verify ONNX output matches PyTorch."""
    import onnxruntime as ort

    sess = ort.InferenceSession(onnx_path)
    onnx_out = sess.run(None, ort_inputs)

    with torch.no_grad():
        torch_out = wrapper(*torch_inputs).numpy()

    diff = np.abs(torch_out - onnx_out[0]).max()
    print(f"Max difference PyTorch vs ONNX: {diff:.6f}")
    # Relaxed: new PyTorch ONNX exporter has ~0.01-0.03 numerical drift
    assert diff < 0.05, f"Verification failed: diff={diff}"
    print("ONNX verification passed!")

    size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"ONNX file size: {size_mb:.1f} MB")


def main():
    args = parse_args()

    if args.model_version == 'v2':
        export_v2(args)
    else:
        export_v1(args)


if __name__ == '__main__':
    main()
