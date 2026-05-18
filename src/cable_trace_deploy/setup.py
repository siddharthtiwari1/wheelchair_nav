from setuptools import setup

package_name = 'cable_trace_deploy'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    package_data={package_name: ['weights.zip']},
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    entry_points={
        'console_scripts': [
            'cable_trace_node = cable_trace_deploy.cable_trace_node:main',
        ],
    },
)
