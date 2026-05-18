#!/usr/bin/env python3
"""Export fine-tuned Depth Anything V2 to ONNX for TensorRT deployment.

Usage:
    python3 export_onnx.py \
        --checkpoint /path/to/best_model.pth \
        --encoder vits \
        --output mono_depth_vits.onnx \
        --verify

Then convert to TensorRT on Jetson:
    /usr/src/tensorrt/bin/trtexec \
        --onnx=mono_depth_vits.onnx \
        --saveEngine=mono_depth_vits_fp16.engine \
        --fp16 \
        --workspace=2048 \
        --minShapes=image:1x3x518x518 \
        --optShapes=image:1x3x518x518 \
        --maxShapes=image:1x3x518x518
"""

import argparse
import os
import sys

import numpy as np
import torch

# Add parent to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from wheelchair_mono_depth.training.train import load_model, MODEL_CONFIGS


class DepthAnythingWrapper(torch.nn.Module):
    """Wrapper that takes a single image tensor and returns depth tensor."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image):
        depth = self.model(image)
        if depth.dim() == 3:
            depth = depth.unsqueeze(1)
        return depth


def export_onnx(checkpoint_path, encoder, output_path, input_size=518, verify=False):
    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    max_depth = ckpt.get('max_depth', 10.0)

    model = load_model(encoder, max_depth)
    state_dict = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    wrapper = DepthAnythingWrapper(model)

    # Dummy input
    dummy = torch.randn(1, 3, input_size, input_size)

    # Export
    print(f'Exporting to {output_path}...')
    torch.onnx.export(
        wrapper,
        dummy,
        output_path,
        opset_version=17,
        input_names=['image'],
        output_names=['depth'],
        dynamic_axes=None,  # Fixed shape for TensorRT optimization
    )
    print(f'ONNX model saved: {output_path} '
          f'({os.path.getsize(output_path) / 1e6:.1f} MB)')

    # Verify
    if verify:
        try:
            import onnxruntime as ort
        except ImportError:
            print('onnxruntime not installed, skipping verification')
            return

        session = ort.InferenceSession(output_path)
        onnx_input = dummy.numpy()
        onnx_output = session.run(None, {'image': onnx_input})[0]

        with torch.no_grad():
            torch_output = wrapper(dummy).numpy()

        diff = np.abs(onnx_output - torch_output)
        print(f'Verification: max diff={diff.max():.6f}, '
              f'mean diff={diff.mean():.6f}')
        if diff.max() < 0.01:
            print('PASS: ONNX output matches PyTorch')
        else:
            print('WARNING: Large difference between ONNX and PyTorch outputs')


def main():
    parser = argparse.ArgumentParser(description='Export Depth Anything V2 to ONNX')
    parser.add_argument('--checkpoint', required=True, help='Path to .pth checkpoint')
    parser.add_argument('--encoder', default='vits', choices=['vits', 'vitb', 'vitl'])
    parser.add_argument('--output', default='mono_depth.onnx', help='Output ONNX path')
    parser.add_argument('--input_size', type=int, default=518)
    parser.add_argument('--verify', action='store_true', help='Verify with onnxruntime')
    args = parser.parse_args()

    export_onnx(args.checkpoint, args.encoder, args.output, args.input_size, args.verify)


if __name__ == '__main__':
    main()
