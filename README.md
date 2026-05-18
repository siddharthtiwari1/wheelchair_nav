<div align="center">

# 🦽 Autonomous Wheelchair Navigation

**ROS 2 navigation stack for a self-driving powered wheelchair.
LiDAR and depth-camera fusion, ZUPT/EKF odometry, SLAM mapping, and Nav2.**

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

The wheelchair has a differential-drive base driven by an Arduino, a 2D RPLidar, and
three Intel RealSense depth cameras (the IMU lives inside the front camera). The
software turns those sensor streams into one of two things: a map, by driving the
chair manually while SLAM Toolbox builds an occupancy grid, or autonomous motion, by
giving a goal pose and letting Nav2 plan and drive to it on a saved map. The two
pipelines share about 80% of their nodes; they differ only at the top, SLAM Toolbox
for mapping versus map_server, AMCL, and Nav2 for navigation.

## Highlights

- A 2D LiDAR and three depth cameras fused into a single scan (`scan_fusion_v9`, "nearest obstacle wins") so the chair avoids tables, shelves, and overhangs a flat LiDAR slice would miss.
- Drift-tolerant odometry: a complementary ZUPT filter on the navigation path and a 6-state EKF `[x, y, θ, v, ω, gyro_bias]` on the mapping path, both with zero-velocity updates and gyro-bias tracking.
- SLAM Toolbox mapping with lidar-only, fused, and hospital-corridor configurations selectable from the launch file.
- Safety in depth: velocity ceilings enforced at four layers (smoother, controller, hardware interface, per-wheel clamp) and a recovery tree that backs up before it spins.
- Tested on an NVIDIA Jetson Orin. The three cameras start staggered so they do not saturate the USB bus.

## System Architecture

<div align="center">

![System architecture](assets/architecture.png)

</div>

The stack is organized in five layers. Data flows strictly upward; the only thing
that travels back down is the velocity command to the motors.

| Layer | Responsibility |
|---|---|
| **1. Hardware & drivers** | Arduino motors and encoders, RPLidar S3, 3× RealSense, camera IMU |
| **2. Sensor conditioning** | Laser filter chain, IMU calibrate/bias/Madgwick/republish, scan fusion |
| **3. State estimation** | ZUPT/EKF fuses wheel odometry and IMU into `/odometry/filtered` and the `odom→base_link` transform |
| **4. World frame** | SLAM Toolbox (mapping) or map_server + AMCL (navigation), producing `map→odom` |
| **5. Behavior** | Navigation only: Nav2 behavior tree, planner, controller, smoothed `/cmd_vel` |

**TF tree:** `map → odom → base_link → wheelchair_main → lidar → laser`, with the
camera frames branching off `base_link`. `map→odom` is the periodic correction onto
the map (SLAM while mapping, AMCL while navigating). `odom→base_link` is the smooth
dead-reckoning estimate published by the ZUPT/EKF node. Everything below `base_link`
is rigid geometry from the URDF. The diff-drive controller's own TF publishing is
disabled on purpose so the two estimators never fight over the same transform.

## Repository Structure

| Package | Build | Responsibility |
|---|---|---|
| `wheelchair_bringup` | ament_cmake | Top-level launch files (navigation, SLAM, localization, RTAB-Map, ablations) |
| `wheelchair_navigation` | ament_cmake | Nav2 parameter sets and behavior trees |
| `wheelchair_localization` | ament_python | `scan_fusion_v9`, odom corrector, SLAM / AMCL / laser-filter configs |
| `wheelchair_zupt` | ament_python | `zupt_node` (navigation) and `robust_ekf_zupt_node` (mapping) odometry |
| `wheelchair_description` | ament_cmake | URDF / xacro, meshes, RViz configs |
| `wheelchair_firmware` | ament_cmake | C++ `ros2_control` Arduino hardware-interface plugin |
| `wheelchair_mapping` | ament_cmake | SLAM Toolbox launch and config helpers |
| `wc_control` | ament_cmake | `ros2_control` diff-drive config and the 4-node IMU pipeline |
| `scripts` | ament_python | Teleop bridge, `cmd_vel` bridge, data loggers, diagnostics |
| `rplidar_ros` | ament_cmake | RPLidar S3 driver (vendored) |

## Hardware

| Component | Model | Interface |
|---|---|---|
| Base | Custom differential drive | Arduino @ `/dev/ttyACM0`, 50 Hz wheel odometry |
| LiDAR | RPLidar S3, 360° 2D, ~10 Hz | `/scan` @ `/dev/ttyUSB0` |
| Front camera | RealSense D455, depth 424×240 @ 6 Hz | depth + the only IMU source (`/camera/imu`) |
| Left camera | RealSense D455, depth-only, faces +90° | USB |
| Right camera | RealSense D435i, depth-only, faces −90° | USB |

| Parameter | Value |
|---|---|
| Wheel radius | 0.1524 m |
| Wheel separation | 0.565 m |
| Footprint | 0.45 × 0.35 m + 0.03 m padding |
| Velocity ceiling (smoother) | 0.25 m/s, 0.35 rad/s |

Velocity is capped at four independent layers (velocity smoother, diff-drive
controller, hardware interface, per-wheel ±3.5 rad/s clamp). The smoother caps first,
so the lower layers act as backstops.

## Build & Install

**Prerequisites:** Ubuntu 24.04 with [ROS 2 Jazzy](https://docs.ros.org/en/jazzy/Installation.html).

```bash
# 1. Clone
git clone https://github.com/siddharthtiwari1/wheelchair_nav.git
cd wheelchair_nav

# 2. Dependencies
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

The workspace is built with `--symlink-install`, so editing a YAML, behavior-tree
XML, or RViz config under `src/` takes effect on the next launch with no rebuild.

`source setup.bash` configures the environment and defines the `run_nav`,
`run_slam`, and `run_localization` helpers.

### USB device permissions

```bash
sudo cp scripts/99-wheelchair-usb.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules && sudo udevadm trigger
```

## Usage

| Helper | Launch file | Purpose |
|---|---|---|
| `run_nav` | `wheelchair_fusion_nav.launch.py` | Autonomous navigation on a saved map |
| `run_slam` | `wheelchair_slam_mapping.launch.py` | Build a map by driving manually |
| `run_localization` | `wheelchair_fusion_localization.launch.py` | Sensors + odometry + AMCL, no Nav2 |

### Autonomous navigation

```bash
source setup.bash
run_nav
```

Defaults: `map_name:=maps/h2.yaml`,
`nav2_params:=nav2_params_3cam_v29.yaml`,
`bt_xml:=wheelchair_robust_nav_v3.xml`. Override any of them:

```bash
run_nav map_name:=/abs/path/to/your_map.yaml \
        nav2_params:=src/wheelchair_navigation/config/nav2_params_3cam_v29.yaml
```

Send a goal with the RViz **2D Goal Pose** tool, or through the action interface:

```bash
ros2 action send_goal /navigate_to_pose nav2_msgs/action/NavigateToPose \
  "{pose: {header: {frame_id: 'map'}, pose: {position: {x: 1.0, y: 0.5}, orientation: {w: 1.0}}}}"
```

### Mapping

```bash
run_slam                        # default: fused SLAM (slam_toolbox_fused_v21.yaml, /scan_fused)
run_slam use_fused_slam:=false  # lidar-only (slam_toolbox_motion_compensated_v2.yaml)
run_slam hospital_mode:=true    # long-range config tuned for corridors
```

> **Note.** The `use_fused_slam` launch argument defaults to `true`, so a bare
> `run_slam` builds a fused (LiDAR + camera) map. Pass `use_fused_slam:=false` for
> the lidar-only map. To save, drive the area and press `Ctrl+C`: the session
> manager catches the interrupt and writes the map plus a rosbag automatically.

An alternative SLAM with visual loop closure is available:

```bash
ros2 launch wheelchair_bringup wheelchair_rtabmap_mapping.launch.py
```

## Software Pipeline

**Perception (Layer 2).** The raw `/scan` passes through a `laser_filters` chain
(range and speckle filters; a 0.15 m range floor also removes the chair's own
self-reflection) to `/scan_filtered`. A four-node IMU pipeline turns the raw,
biased `/camera/imu` into a clean, base-frame `/imu`: a startup calibrator measures
gyro bias from the first ~3 s of stillness, a bias corrector subtracts it, a
Madgwick filter estimates orientation, and a republisher rotates it into the robot
frame. `scan_fusion_v9` then overlays the three depth cameras onto the filtered
scan: per angular bin the nearest valid return wins, within a 0.10–1.80 m height
window, with per-camera range caps (front 3.5 m, sides 3.0 m) and a minimum of two
points per bin to reject outliers. Its output, `/scan_fused`, feeds the costmaps.

**Odometry (Layer 3).** Both pipelines fuse `/wc_control/odom` (wheels) with `/imu`
and publish `/odometry/filtered` plus `odom→base_link`. Navigation runs `zupt_node`,
a complementary filter weighting the gyro over the encoders with zero-velocity
updates when the chair is still. Mapping runs `robust_ekf_zupt_node`, a 6-state EKF
`[x, y, θ, v, ω, gyro_bias]` with zero-velocity updates and continuous gyro-bias
recalibration.

**Localization (Layer 4).** While mapping, SLAM Toolbox (sync mode) provides `/map`
and `map→odom`. While navigating, `map_server` serves the saved map and AMCL
(`amcl_fusion.yaml`, 3000–10000 particles, running on `/scan_filtered`) provides
`map→odom`.

**Navigation (Layer 5).** Nav2 runs seven lifecycle servers. `SmacPlanner2D` plans
a global path; `RegulatedPurePursuitController` follows it at 20 Hz toward a
0.25 m/s target; `bt_navigator` ticks the behavior tree; `behavior_server` provides
recovery primitives; the velocity smoother acceleration-limits the result onto
`/cmd_vel`, which a bridge converts to the `TwistStamped` the diff-drive controller
expects. The local costmap is a rolling 5×5 m window (3-camera STVL voxels +
`/scan_fused` obstacle layer + 0.25 m inflation); the global costmap adds the static
map and a 0.28 m inflation.

**Recovery behavior tree.** A `RecoveryNode` with seven retries wraps a plan and
follow pipeline. On failure it cycles a backup-first sequence, because a loaded
wheelchair backs up more safely than it spins: back up 0.15 m, clear costmaps,
wait 2 s, spin +30°, back up 0.20 m, spin −30°, then a 0.25 m backup as a last
resort. A new goal short-circuits recovery immediately.

## Configuration

| File | Loaded by | Notes |
|---|---|---|
| `wc_control/config/wc_control_safe_v2.yaml` | `controller_manager` | Diff-drive limits (both pipelines) |
| `wheelchair_localization/config/laser_filter_robust.yaml` | laser filter | Range + speckle filters |
| `wheelchair_localization/config/slam_toolbox_fused_v21.yaml` | `slam_toolbox` | Mapping, default fused branch |
| `wheelchair_localization/config/slam_toolbox_motion_compensated_v2.yaml` | `slam_toolbox` | Mapping, lidar-only branch |
| `wheelchair_localization/config/amcl_fusion.yaml` | `amcl` | The AMCL config used at run time |
| `wheelchair_navigation/config/nav2_params_3cam_v29.yaml` | 7 Nav2 servers | Default Nav2 stack |
| `wheelchair_navigation/behavior_tree/wheelchair_robust_nav_v3.xml` | `bt_navigator` | Default recovery behavior tree |

> **Config versioning rule.** Configuration files (`*.yaml`, behavior-tree `*.xml`,
> RViz configs) are never edited in place. Each change creates a new versioned file
> (`<base>_<variant>[_vN].yaml`), the launch default is repointed to it, and the old
> file is kept for rollback. Source code (`.py`, `.cpp`) is edited normally.

## Operating Notes

- **Keep the chair still for the first ~3 s** after launch so the IMU calibrator can
  measure gyro bias. Otherwise hard-coded bias defaults are used.
- The three RealSense cameras start staggered (front first, then left, then right)
  to avoid USB bandwidth contention. Scan fusion waits until all three are up.
- Save a map with `Ctrl+C`; the session manager writes the map and a rosbag. SLAM
  Toolbox logs a harmless "bond timeout" from the Nav2 lifecycle manager.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Device busy` on launch | `pkill -9 -f ros; pkill -9 -f rviz; pkill -9 -f realsense; sleep 2` |
| AMCL not converging | Verify map (`ros2 topic echo /map --once`) and scan (`ros2 topic hz /scan_filtered`); re-set the pose with RViz **2D Pose Estimate** |
| Robot not moving | Check `ros2 lifecycle get /controller_server` and `ros2 topic hz /local_costmap/costmap` |
| Pose "teleports" in RViz | A second publisher of `odom→base_link`; the diff-drive `publish_odom_tf` must stay `false` |
| TF errors | `ros2 run tf2_tools view_frames` to inspect the tree |

### Known issues

- **Laser filter.** Angular, shadow, temporal-median, and box filters segfault on
  ROS 2 Jazzy and are disabled; only range and speckle are active. Self-filtering is
  done by the 0.15 m range floor.
- **Wheel-encoder underestimation.** `position_feedback` must stay `false` in the
  diff-drive config; differentiating position against slower Arduino data
  under-reports speed by about 35%.

## Author

**Siddharth Tiwari**, IIT Mandi · [s24035@students.iitmandi.ac.in](mailto:s24035@students.iitmandi.ac.in)

## License

Released for academic and research use. Please contact the author regarding other
use or collaboration.
