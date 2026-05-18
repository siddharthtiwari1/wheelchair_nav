"""Setup for wheelchair_e2e package - installed via ament_cmake_python."""
from setuptools import setup, find_packages

setup(
    name='wheelchair_e2e',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'torch',
        'torchvision',
        'numpy',
        'opencv-python',
    ],
)
