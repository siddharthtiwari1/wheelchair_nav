# Wheelchair Autonomous Navigation

ROS2 Jazzy workspace for autonomous wheelchair navigation with SLAM, localization, and Nav2.

## Quick Start

```bash
# Clone the repository
git clone <repo-url> wheelchair_nav
cd wheelchair_nav

# Setup and build (first time)
source setup.bash

# Run autonomous navigation
run_nav

# Or run SLAM mapping
run_slam
```

## Requirements

- ROS2 Jazzy (or Humble)
- Nav2 navigation stack
- robot_localization
- slam_toolbox
- laser_filters
- RPLidar ROS2 driver
- RealSense ROS2 driver

### Install Dependencies

```bash
sudo apt update
sudo apt install ros-jazzy-nav2-bringup ros-jazzy-robot-localization \
    ros-jazzy-slam-toolbox ros-jazzy-laser-filters \
    ros-jazzy-realsense2-camera ros-jazzy-imu-filter-madgwick
```

## Hardware Setup

- **Wheelchair Base**: Custom differential drive with Arduino interface
- **LIDAR**: RPLidar S3 (360 deg scan on /scan)
- **Camera**: RealSense D455 (depth + RGB + IMU)
- **Controller**: Arduino via USB serial

### USB Device Permissions

```bash
# Add udev rules for USB devices
sudo cp scripts/99-wheelchair-usb.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules
sudo udevadm trigger
```

## Workspace Structure

```
wheelchair_nav/
├── setup.bash                    # Main setup script
├── maps/                         # Saved maps
│   └── my_map_final_cleaned.yaml # Pre-built map
├── src/
│   ├── wheelchair_bringup/       # Launch files
│   ├── wheelchair_navigation/    # Nav2 configs
│   ├── wheelchair_localization/  # EKF, AMCL configs
│   ├── wheelchair_description/   # URDF, RViz configs
│   ├── wheelchair_firmware/      # Hardware interface
│   ├── wheelchair_mapping/       # SLAM configs
│   ├── wheelchair_zupt/          # ZUPT odometry
│   ├── wc_control/               # ros2_control configs
│   ├── rplidar_ros/              # RPLidar driver
│   └── scripts/                  # Utility scripts
└── README.md
```

## Usage

### Autonomous Navigation

```bash
source setup.bash
run_nav
```

This launches:
- Hardware interface (Arduino, sensors)
- EKF sensor fusion (wheel odom + IMU)
- AMCL global localization
- Nav2 navigation stack
- RViz visualization

Use RViz "2D Goal Pose" button to send navigation goals.

### SLAM Mapping

```bash
source setup.bash
run_slam
```

Drive the wheelchair around to build a map. Press Ctrl+C to save.

### Custom Map

```bash
run_nav map_name:=/path/to/your/map.yaml
```

## Launch Arguments

### Navigation Launch

| Argument | Default | Description |
|----------|---------|-------------|
| map_name | maps/my_map_final_cleaned.yaml | Path to map YAML |
| use_rviz | true | Launch RViz |
| use_sim_time | false | Use simulation time |
| nav2_params | nav2_params_robust.yaml | Nav2 parameters |

### Localization Launch

| Argument | Default | Description |
|----------|---------|-------------|
| use_zupt | true | Use ZUPT-enhanced odometry |
| use_fused_scan | true | Fuse depth + LIDAR for AMCL |
| use_depth | true | Enable depth camera |

## Troubleshooting

### "Device busy" errors
Kill existing ROS2 processes first:
```bash
pkill -9 -f ros2; pkill -9 -f rviz; sleep 3
```

### AMCL not converging
- Check that map is loaded: `ros2 topic echo /map --once`
- Verify LIDAR scan: `ros2 topic hz /scan_filtered`
- Re-initialize pose in RViz: "2D Pose Estimate"

### Navigation not moving
- Check costmaps: `ros2 topic hz /local_costmap/costmap`
- Verify controller: `ros2 lifecycle get /controller_server`

## Author

Siddharth Tiwari (s24035@students.iitmandi.ac.in)
