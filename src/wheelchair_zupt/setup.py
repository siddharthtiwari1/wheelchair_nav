from setuptools import setup, find_packages

package_name = 'wheelchair_zupt'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'numpy', 'scipy'],
    zip_safe=True,
    maintainer='sidd',
    maintainer_email='s24035@students.iitmandi.ac.in',
    description='ZUPT-Enhanced Odometry for Wheelchair',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'zupt_node = wheelchair_zupt.zupt_node:main',
            'ekf_fusion_node = wheelchair_zupt.ekf_fusion_node:main',
            'robust_ekf_zupt_node = wheelchair_zupt.robust_ekf_zupt_node:main',
            'improved_ekf_node = wheelchair_zupt.improved_ekf_node:main',
        ],
    },
)
