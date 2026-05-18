# Autonomous Wheelchair Navigation

ROS 2 Jazzy navigation stack for a self-driving powered wheelchair: LiDAR and
depth-camera fusion, ZUPT/EKF odometry, SLAM Toolbox mapping, and Nav2.

The chair has a differential-drive base driven by an Arduino, a 2D RPLidar, and
three Intel RealSense depth cameras (the IMU is inside the front camera). It carries
a passenger, so the stack prioritises safe, predictable motion over speed. There are
two pipelines, sharing about 80% of their nodes: drive manually while SLAM Toolbox
builds a map, or give a goal pose and let Nav2 drive there on a saved map.

## Architecture

![System architecture](assets/architecture.png)

Five layers. Data flows upward; only the velocity command flows back down to the
motors.

1. **Hardware & drivers**: Arduino motors/encoders, RPLidar S3, 3× RealSense, camera IMU.
2. **Sensor conditioning**: laser filter chain, IMU calibrate/bias/Madgwick/republish, scan fusion.
3. **State estimation**: ZUPT/EKF fuses wheel odometry and IMU into `/odometry/filtered` and `odom→base_link`.
4. **World frame**: SLAM Toolbox (mapping) or map_server + AMCL (navigation), producing `map→odom`.
5. **Behaviour**: navigation only: Nav2 behaviour tree, planner, controller, smoothed `/cmd_vel`.

TF tree: `map → odom → base_link → wheelchair_main → lidar → laser`, cameras
branching off `base_link`. `map→odom` is the periodic map correction (SLAM or AMCL),
`odom→base_link` the smooth dead-reckoning estimate from the ZUPT/EKF node. The
diff-drive controller's own odom TF is disabled so the two estimators do not collide.

## Packages

| Package | Responsibility |
|---|---|
| `wheelchair_bringup` | Top-level launch files |
| `wheelchair_navigation` | Nav2 parameter sets and behaviour trees |
| `wheelchair_localization` | `scan_fusion_v9`, AMCL / laser / SLAM configs |
| `wheelchair_zupt` | `zupt_node` (nav) and `robust_ekf_zupt_node` (slam) odometry |
| `wheelchair_description` | URDF/xacro, meshes, RViz configs |
| `wheelchair_firmware` | C++ `ros2_control` Arduino hardware interface |
| `wheelchair_mapping` | SLAM Toolbox launch/config helpers |
| `wc_control` | Diff-drive config and the 4-node IMU pipeline |
| `scripts` | Teleop, `cmd_vel` bridge, loggers, diagnostics |
| `rplidar_ros` | RPLidar S3 driver (vendored) |

## Hardware

| Item | Detail |
|---|---|
| Base | Differential drive, Arduino `/dev/ttyACM0`, 50 Hz wheel odometry |
| LiDAR | RPLidar S3, 360° 2D, ~10 Hz, `/scan` on `/dev/ttyUSB0` |
| Front camera | RealSense D455, depth 424×240 @ 6 Hz, sole IMU source |
| Side cameras | RealSense D455 (left, +90°), D435i (right, −90°), depth only |
| Geometry | Wheel radius 0.1524 m, separation 0.565 m, footprint 0.45 × 0.35 m |
| Velocity ceiling | 0.25 m/s, 0.35 rad/s at the smoother (capped again at 3 lower layers) |

## Build

Requires Ubuntu 24.04 with [ROS 2 Jazzy](https://docs.ros.org/en/jazzy/Installation.html).

```bash
git clone https://github.com/siddharthtiwari1/wheelchair_nav.git
cd wheelchair_nav

sudo apt update
sudo apt install \
    ros-jazzy-navigation2 ros-jazzy-nav2-bringup \
    ros-jazzy-robot-localization ros-jazzy-slam-toolbox \
    ros-jazzy-laser-filters ros-jazzy-realsense2-camera \
    ros-jazzy-realsense2-description ros-jazzy-imu-filter-madgwick \
    ros-jazzy-ros2-control ros-jazzy-ros2-controllers \
    ros-jazzy-controller-manager ros-jazzy-xacro

source /opt/ros/jazzy/setup.bash
colcon build --symlink-install
source install/setup.bash
```

The build uses `--symlink-install`, so editing a YAML, behaviour-tree XML, or RViz
config under `src/` takes effect on the next launch with no rebuild. `source
setup.bash` defines the `run_nav`, `run_slam`, and `run_localization` helpers.

USB device rules (one time):

```bash
sudo cp scripts/99-wheelchair-usb.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules && sudo udevadm trigger
```

## Usage

| Helper | Launch | Purpose |
|---|---|---|
| `run_nav` | `wheelchair_fusion_nav.launch.py` | Autonomous navigation on a saved map |
| `run_slam` | `wheelchair_slam_mapping.launch.py` | Build a map by driving manually |
| `run_localization` | `wheelchair_fusion_localization.launch.py` | Sensors + odometry + AMCL, no Nav2 |

**Navigation.** Defaults: `map_name:=maps/h2.yaml`,
`nav2_params:=nav2_params_3cam_v29.yaml`, `bt_xml:=wheelchair_robust_nav_v3.xml`.

```bash
source setup.bash
run_nav
run_nav map_name:=/abs/path/to/your_map.yaml      # override any default
```

Send a goal with the RViz **2D Goal Pose** tool or the action interface:

```bash
ros2 action send_goal /navigate_to_pose nav2_msgs/action/NavigateToPose \
  "{pose: {header: {frame_id: 'map'}, pose: {position: {x: 1.0, y: 0.5}, orientation: {w: 1.0}}}}"
```

**Mapping.** `use_fused_slam` defaults to `true`, so a bare `run_slam` builds a
LiDAR + camera map. Press `Ctrl+C` to save; the session manager writes the map and
a rosbag.

```bash
run_slam                        # fused (slam_toolbox_fused_v21.yaml, /scan_fused)
run_slam use_fused_slam:=false  # lidar-only (slam_toolbox_motion_compensated_v2.yaml)
run_slam hospital_mode:=true    # long-range corridor config
```

## Pipeline

**Perception.** `/scan` passes a `laser_filters` chain (range + speckle; the 0.15 m
range floor also removes the chair's self-reflection) to `/scan_filtered`. A
four-node IMU pipeline turns the raw `/camera/imu` into a clean base-frame `/imu`:
startup calibrator (measures gyro bias from the first ~3 s of stillness), bias
corrector, Madgwick filter, base-frame republisher. `scan_fusion_v9` overlays the
three depth cameras onto the filtered scan, nearest valid return per angular bin,
0.10–1.80 m height window, range caps front 3.5 m / sides 3.0 m, ≥2 points/bin. Its
`/scan_fused` output feeds the costmaps.

**Odometry.** Both pipelines fuse `/wc_control/odom` and `/imu` into
`/odometry/filtered` and `odom→base_link`. Navigation uses `zupt_node`, a
complementary filter favouring the gyro with zero-velocity updates when still.
Mapping uses `robust_ekf_zupt_node`, a 6-state EKF `[x, y, θ, v, ω, gyro_bias]` with
zero-velocity updates and continuous gyro-bias recalibration.

**Localisation.** Mapping: SLAM Toolbox (sync) provides `/map` and `map→odom`.
Navigation: `map_server` serves the saved map and AMCL (`amcl_fusion.yaml`,
3000–10000 particles, on `/scan_filtered`) provides `map→odom`.

**Navigation.** Seven Nav2 servers: `SmacPlanner2D` global path,
`RegulatedPurePursuitController` at 20 Hz toward 0.25 m/s, `bt_navigator`,
`behavior_server`, smoother, velocity smoother, waypoint follower. Local costmap is
a rolling 5×5 m window (3-camera STVL + `/scan_fused` obstacle + 0.25 m inflation);
the global costmap adds the static map and 0.28 m inflation.

**Recovery.** A `RecoveryNode` (7 retries) wraps the plan/follow pipeline. On
failure it runs a backup-first sequence, since a loaded chair backs up more safely
than it spins: back up 0.15 m, clear costmaps, wait 2 s, spin +30°, back up 0.20 m,
spin −30°, then a 0.25 m backup as last resort. A new goal cancels recovery.

## Configuration

| File | Loaded by |
|---|---|
| `wc_control/config/wc_control_safe_v2.yaml` | `controller_manager` (diff-drive limits) |
| `wheelchair_localization/config/laser_filter_robust.yaml` | laser filter |
| `wheelchair_localization/config/slam_toolbox_fused_v21.yaml` | `slam_toolbox` (default fused) |
| `wheelchair_localization/config/slam_toolbox_motion_compensated_v2.yaml` | `slam_toolbox` (lidar-only) |
| `wheelchair_localization/config/amcl_fusion.yaml` | `amcl` |
| `wheelchair_navigation/config/nav2_params_3cam_v29.yaml` | the 7 Nav2 servers |
| `wheelchair_navigation/behavior_tree/wheelchair_robust_nav_v3.xml` | `bt_navigator` |

Config files (`*.yaml`, behaviour-tree `*.xml`, RViz configs) are never edited in
place. Each change creates a new versioned file, the launch default is repointed to
it, and the old file is kept for rollback. Source code is edited normally.

## Notes

- Keep the chair still for the first ~3 s after launch so the IMU calibrator can
  measure gyro bias; otherwise hard-coded defaults are used.
- The three cameras start staggered to avoid USB bandwidth contention; scan fusion
  waits until all three are up.
- Laser filter: only range and speckle are active. The angular, shadow,
  temporal-median, and box filters segfault on ROS 2 Jazzy and are disabled.
- `position_feedback` must stay `false` in the diff-drive config; differentiating
  position against slower Arduino data under-reports speed by ~35%.

| Symptom | Check |
|---|---|
| `Device busy` on launch | `pkill -9 -f ros; pkill -9 -f rviz; pkill -9 -f realsense; sleep 2` |
| AMCL not converging | `ros2 topic echo /map --once`, `ros2 topic hz /scan_filtered`, reset pose in RViz |
| Robot not moving | `ros2 lifecycle get /controller_server`, `ros2 topic hz /local_costmap/costmap` |
| Pose teleports in RViz | A second `odom→base_link` publisher; diff-drive `publish_odom_tf` must be `false` |

## Author

Siddharth Tiwari, IIT Mandi. s24035@students.iitmandi.ac.in

Released for academic and research use.
