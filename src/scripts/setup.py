from setuptools import setup, find_packages

package_name = 'scripts'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='sidd',
    maintainer_email='s24035@students.iitmandi.ac.in',
    description='Wheelchair utility scripts',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'twist_stamped_teleop = scripts.twist_stamped_teleop:main',
            'imu_out_to_imu = scripts.imu_out_to_imu:main',
            'sim_odom_bias = scripts.sim_odom_bias:main',
            'topic_data_logger = scripts.topic_data_logger:main',
            'imu_diagnostic = scripts.imu_diagnostic:main',
            'scan_data_logger = scripts.scan_data_logger:main',
            'slam_session_manager = scripts.slam_session_manager:main',
            'rgb_depth_saver = scripts.rgb_depth_saver:main',
            'dataset_recorder = scripts.dataset_recorder:main',
            'rgb_velocity_recorder = scripts.rgb_velocity_recorder:main',
        ],
    },
)
