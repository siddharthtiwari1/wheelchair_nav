"""Setup for wheelchair_mono_depth package - installed via ament_cmake_python."""
from setuptools import setup, find_packages

setup(
    name='wheelchair_mono_depth',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'torch',
        'torchvision',
        'numpy',
        'opencv-python',
    ],
    entry_points={
        'console_scripts': [
            'data_collection_node = wheelchair_mono_depth.data_collection_node:main',
            'mono_depth_node = wheelchair_mono_depth.inference.mono_depth_node:main',
            'mono_depth_multi_node = wheelchair_mono_depth.inference.mono_depth_multi_node:main',
            'depth_eval_node = wheelchair_mono_depth.inference.depth_eval_node:main',
            'da3_depth_node = wheelchair_mono_depth.inference.da3_depth_node:main',
            'depth_scan_benchmark_node = wheelchair_mono_depth.inference.depth_scan_benchmark_node:main',
            'da3_lidar_anchor_node = wheelchair_mono_depth.inference.da3_lidar_anchor_node:main',
            'da3_multi_depth_node = wheelchair_mono_depth.inference.da3_multi_depth_node:main',
        ],
    },
)
