from setuptools import setup

setup(
    name='ground_truth_labeling',
    version='0.1.0',
    packages=['ground_truth_labeling'],
    install_requires=[
        'numpy',
        'sensor_msgs_py',
        'opencv-python',
        'cv_bridge',
        'matplotlib',
    ],
    entry_points={
        'console_scripts': [
            'rosbag_frame_extractor=ground_truth_labeling.rosbag_frame_extractor:main',
            'analysis_tools=ground_truth_labeling.analysis_tools:main',
        ],
    },
)
