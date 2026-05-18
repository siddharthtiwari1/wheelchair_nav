<div align="center">

# 🦽 Autonomous Wheelchair Navigation

**ROS 2 navigation stack for a self-driving powered wheelchair.
LiDAR and depth-camera fusion, EKF odometry, SLAM mapping, and Nav2.**

[![ROS 2 Jazzy](https://img.shields.io/badge/ROS%202-Jazzy-22314E?logo=ros&logoColor=white)](https://docs.ros.org/en/jazzy/)
[![Ubuntu 24.04](https://img.shields.io/badge/Ubuntu-24.04-E95420?logo=ubuntu&logoColor=white)](https://releases.ubuntu.com/24.04/)
[![Nav2](https://img.shields.io/badge/Navigation-Nav2-2C7BB6)](https://navigation.ros.org/)
[![C++ | Python](https://img.shields.io/badge/Lang-C%2B%2B%20%7C%20Python-3776AB)](#)
[![Platform: Jetson Orin](https://img.shields.io/badge/Edge-Jetson%20Orin-76B900?logo=nvidia&logoColor=white)](#)

</div>

---

## Overview

This is the onboard navigation software for an autonomous powered wheelchair. The
chair carries a person, so every design choice favors safe and predictable motion
over speed. It is built and tested for indoor use: homes, hospital wards, and
corridors.

The wheelchair sees obstacles with a 2D LiDAR and three RealSense depth cameras.
Wheel encoders and an IMU feed an EKF for odometry, because the encoders on their own
drift and underestimate distance by roughly 20%. Mapping runs on SLAM Toolbox,
localization on AMCL, and Nav2 handles planning and control. The recovery behaviors
back the chair up before anything else, since spinning a loaded wheelchair in place
is neither safe nor comfortable for the passenger.

## Key Features

- RPLidar S3 and three RealSense depth cameras fused into one scan for the Nav2 costmap.
- A 6-state EKF `[x, y, θ, v, ω, gyro_bias]` with zero-velocity updates and ongoing gyro-bias correction. It also compensates the ~20% wheel-encoder underestimation.
- SLAM Toolbox mapping. Lidar-only by default, because camera noise degrades scan matching. Fused and hospital-corridor modes are available as launch options.
- Velocity-limited control with acceleration smoothing on `/cmd_vel`, and a recovery tree that backs up before it spins.
- Runs on an NVIDIA Jetson Orin. Cameras start staggered so they do not saturate the USB bus.

## System Architecture

<div align="center">

![System architecture](assets/architecture.png)

</div>

Data flows bottom to top. Sensors and the Arduino sit at the base. The perception
layer filters the scan and fuses LiDAR with the three depth cameras into
`/scan_fused`, while wheel odometry and the Madgwick-filtered IMU go into the EKF.
SLAM Toolbox and AMCL handle localization, the planning layer maintains the global
and local costmaps and runs the SMAC 2D A* planner, and the navigation and control
layers turn the path into smoothed, speed-limited velocity commands at
`/wc_control/cmd_vel`.

The TF tree is `map → odom → base_link → wheelchair_main → lidar → laser`, with the
camera frames branching off `base_link`. The `odom→base_link` transform is published
by the EKF/ZUPT node. The diff-drive controller's own TF publishing is left off on
purpose so the two do not fight over the same transform.

## Repository Structure

| Package | Build | Responsibility |
|---|---|---|
| `wheelchair_bringup` | ament_cmake | Top-level launch files (navigation, SLAM, RTAB-Map, ablations) |
| `wheelchair_navigation` | ament_cmake | Nav2 parameter sets + behavior trees |
| `wheelchair_localization` | ament_python | Scan-fusion node, odom corrector, SLAM/AMCL/laser configs |
| `wheelchair_zupt` | ament_python | EKF + zero-velocity-update odometry (primary odom source) |
| `wheelchair_description` | ament_cmake | URDF/xacro, meshes, RViz configs |
| `wheelchair_firmware` | ament_cmake | C++ `ros2_control` Arduino hardware-interface plugin |
| `wheelchair_mapping` | ament_cmake | SLAM Toolbox launch/config helpers |
| `wc_control` | ament_cmake | `ros2_control` diff-drive config + IMU pipeline nodes |
| `scripts` | ament_python | Teleop bridge, data loggers, diagnostics |
| `rplidar_ros` | ament_cmake | RPLidar S3 driver (vendored) |

## Hardware

| Component | Model | Topic / Interface |
|---|---|---|
| Base | Custom differential drive | Arduino @ `/dev/ttyACM0` |
| LiDAR | RPLidar S3 (10 Hz) | `/scan` @ `/dev/ttyUSB0` |
| Front camera | RealSense D455 | depth + RGB + IMU |
| Side cameras | RealSense D455 (left), D435i (right) | depth |
| Odometry | Wheel encoders + RealSense IMU | `/wc_control/odom`, `/camera/imu` |

| Parameter | Value |
|---|---|
| Wheel radius | 0.1524 m |
| Wheel separation | 0.565 m |
| Max velocity | 0.25 m/s |

## Installation

**Prerequisites:** Ubuntu 24.04 + [ROS 2 Jazzy](https://docs.ros.org/en/jazzy/Installation.html).

```bash
# 1. Clone into a workspace
git clone https://github.com/siddharthtiwari1/wheelchair_nav.git
cd wheelchair_nav

# 2. Install dependencies
sudo apt update
sudo apt install \
    ros-jazzy-navigation2 ros-jazzy-nav2-bringup \
    ros-jazzy-robot-localization ros-jazzy-slam-toolbox \
    ros-jazzy-laser-filters ros-jazzy-realsense2-camera \
    ros-jazzy-realsense2-description ros-jazzy-imu-filter-madgwick \
    ros-jazzy-ros2-control ros-jazzy-ros2-controllers \
    ros-jazzy-controller-manager ros-jazzy-xacro

# 3. Build
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install
source install/setup.bash
```

`source setup.bash` configures the environment and exposes the convenience commands
below.

### USB device permissions

```bash
sudo cp scripts/99-wheelchair-usb.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules && sudo udevadm trigger
```

## Usage

| Command | Equivalent launch | Purpose |
|---|---|---|
| `run_nav` | `wheelchair_bringup wheelchair_fusion_nav.launch.py` | Full autonomous navigation (needs a map) |
| `run_slam` | `wheelchair_bringup wheelchair_slam_mapping.launch.py` | Build a map (lidar-only by default) |
| `run_localization` | `wheelchair_bringup wheelchair_fusion_localization.launch.py` | Localization only, no Nav2 |

```bash
# Autonomous navigation with the default map
source setup.bash
run_nav

# Use a custom map / params
run_nav map_name:=/path/to/map.yaml \
        nav2_params:=src/wheelchair_navigation/config/nav2_params_3cam_v29.yaml

# Mapping (drive around, Ctrl+C to save)
run_slam                       # lidar-only (default)
run_slam use_fused_slam:=true  # lidar + camera occupancy
run_slam hospital_mode:=true   # 30 m range tuned for corridors

# Alternative SLAM with visual loop closure
ros2 launch wheelchair_bringup wheelchair_rtabmap_mapping.launch.py
```

Send goals with the RViz **2D Goal Pose** tool, or via the action interface:

```bash
ros2 action send_goal /navigate_to_pose nav2_msgs/action/NavigateToPose \
  "{pose: {header: {frame_id: 'map'}, pose: {position: {x: 1.0, y: 0.5}, orientation: {w: 1.0}}}}"
```

## Configuration

| File | Controls |
|---|---|
| `wheelchair_navigation/config/nav2_params_3cam_v29.yaml` | Active Nav2 stack (3-cam STVL, safe RPP) |
| `wheelchair_navigation/behavior_tree/wheelchair_robust_nav_v3.xml` | Active behavior tree (backup-first recovery) |
| `wheelchair_localization/config/slam_toolbox_motion_compensated_v2.yaml` | Active lidar-only SLAM |
| `wheelchair_localization/config/amcl_fusion.yaml` | AMCL particle filter |
| `wheelchair_localization/config/laser_filter_robust.yaml` | Laser filter chain |
| `wc_control/config/wc_control_safe_v2.yaml` | Diff-drive controller velocity limits |

> **Config versioning rule.** Configuration files (`*.yaml`, behavior-tree `*.xml`,
> RViz configs) are **never edited in place**. Each change creates a new versioned
> file (`<base>_<variant>[_vN].yaml`), the launch default is repointed to it, and the
> old file is kept for rollback. Source code (`.py`, `.cpp`) is edited normally.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Device busy` on launch | Kill stale processes: `pkill -9 -f ros; pkill -9 -f rviz; pkill -9 -f realsense; sleep 2` |
| AMCL not converging | Verify map (`ros2 topic echo /map --once`) and scan (`ros2 topic hz /scan_filtered`); re-set pose with RViz **2D Pose Estimate** |
| Robot not moving | Check `ros2 lifecycle get /controller_server` and `ros2 topic hz /local_costmap/costmap` |
| TF errors | `ros2 run tf2_tools view_frames` to inspect the tree |

### Known issues

- **Laser filter crash** — angular/shadow/temporal/box filters segfault on ROS 2 Jazzy; only range + speckle filters are enabled in `laser_filter_robust.yaml`.
- **Odometry underestimation** — keep `position_feedback: false` in the diff-drive config to avoid ~35 % underestimation from position differentiation.
- **TF ownership** — the diff-drive controller's `publish_odom_tf` must stay `false`; the EKF/ZUPT node owns `odom→base_link`.
- **AMCL jumping** — `recovery_alpha_fast: 0.0` prevents aggressive re-localization jumps.

## Author

**Siddharth Tiwari** — IIT Mandi · [s24035@students.iitmandi.ac.in](mailto:s24035@students.iitmandi.ac.in)

## License

Released for academic and research use. Please contact the author regarding other use
or collaboration.
