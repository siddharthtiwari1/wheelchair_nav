#!/usr/bin/env python3
"""Export Depth Anything V3 to ONNX for future TensorRT deployment.

This script exports the DA3 model to ONNX format, which can then be
converted to TensorRT for low-latency inference on Jetson Orin Nano.

Usage:
    python3 export_da3_onnx.py
    python3 export_da3_onnx.py --model DA3METRIC-SMALL --input-size 518 \
        --output /home/sidd/wheelchair_nav/models/da3_small.onnx

Requires:
    pip install depth-anything-3 onnx onnxruntime
"""

import argparse
import os

import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description='Export DA3 model to ONNX format')
    parser.add_argument(
        '--model', default='da3metric-large',
        help='DA3 model name (da3metric-large, da3-large, da3-small)')
    parser.add_argument(
        '--input-size', type=int, default=518,
        help='Model input resolution (default: 518)')
    parser.add_argument(
        '--output', default=None,
        help='Output ONNX path (default: models/<model>.onnx)')
    parser.add_argument(
        '--opset', type=int, default=17,
        help='ONNX opset version (default: 17)')
    parser.add_argument(
        '--verify', action='store_true',
        help='Verify ONNX output matches PyTorch output')
    args = parser.parse_args()

    import torch

    if args.output is None:
        model_dir = '/home/sidd/wheelchair_nav/models'
        os.makedirs(model_dir, exist_ok=True)
        safe_name = args.model.lower().replace('-', '_')
        args.output = os.path.join(model_dir, f'{safe_name}.onnx')

    print(f'Loading DA3 model: {args.model}')
    from depth_anything_3.api import DepthAnything3
    model = DepthAnything3(args.model).to('cuda')
    model.eval()

    # Create dummy input
    dummy_input = torch.randn(
        1, 3, args.input_size, args.input_size,
        device='cuda', dtype=torch.float32)

    print(f'Exporting to ONNX: {args.output}')
    print(f'  Input shape: {dummy_input.shape}')
    print(f'  Opset: {args.opset}')

    torch.onnx.export(
        model,
        dummy_input,
        args.output,
        opset_version=args.opset,
        input_names=['input'],
        output_names=['depth'],
        dynamic_axes={
            'input': {0: 'batch', 2: 'height', 3: 'width'},
            'depth': {0: 'batch', 2: 'height', 3: 'width'},
        },
    )

    file_size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f'Exported: {args.output} ({file_size_mb:.1f} MB)')

    if args.verify:
        print('Verifying ONNX output...')
        import onnxruntime as ort

        ort_session = ort.InferenceSession(args.output)
        ort_input = dummy_input.cpu().numpy()

        with torch.no_grad():
            torch_out = model(dummy_input).cpu().numpy()

        ort_out = ort_session.run(None, {'input': ort_input})[0]
        max_diff = np.max(np.abs(torch_out - ort_out))
        print(f'  Max absolute difference: {max_diff:.6f}')
        if max_diff < 0.01:
            print('  PASS: ONNX output matches PyTorch')
        else:
            print('  WARN: Large difference detected')

    print('\nNext steps:')
    print('  1. Install TensorRT: sudo apt install tensorrt')
    print(f'  2. Convert: trtexec --onnx={args.output} '
          f'--saveEngine={args.output.replace(".onnx", ".engine")} '
          '--fp16')
    print('  3. Use in da3_depth_node.py with TensorRT backend')


if __name__ == '__main__':
    main()
