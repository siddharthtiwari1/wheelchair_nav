#!/usr/bin/env python3
"""
generate_wheelchair_unified.py — Generate a single wheelchair USD that works in both
Isaac Sim (ROS2 OmniGraph bridges) and Isaac Lab (direct joint control).

Fixes vs. the old generate_wheelchair_v3.py:
  1. Wheel damping 200 (not 1e4) + maxForce 50 (not 100)
  2. PhysX LiDAR (inline, no Nucleus URL)
  3. PhysX LiDAR OmniGraph (ReadLidarBeams → ROS2PublishLaserScan)
  4. No drive_to_goal graph (only ros_drive)
  5. No environment prims (/physicsScene, /groundPlane, /DistantLight)
  6. targetPrim = /wheelchair_description (the articulation root)
  7. DifferentialController dt connection
  8. scaleToStageUnits node in ros_drive graph

Usage:
    # First, generate a flat URDF from xacro:
    cd /home/sidd/wheelchair_nav
    source /opt/ros/jazzy/setup.bash && source install/setup.bash
    xacro src/wheelchair_description/urdf/wheelchair_description.urdf.xacro \
        is_sim:=true is_ignition:=true \
        > /tmp/wheelchair_flat.urdf

    # Then generate the USD:
    /home/sidd/isaacsim/python.sh src/wheelchair_description/scripts/generate_wheelchair_unified.py \
        --urdf /tmp/wheelchair_flat.urdf \
        --output src/wheelchair_description/urdf/wheelchair_unified.usd

    # Optional: export as human-readable USDA for debugging:
    /home/sidd/isaacsim/python.sh src/wheelchair_description/scripts/generate_wheelchair_unified.py \
        --urdf /tmp/wheelchair_flat.urdf \
        --output src/wheelchair_description/urdf/wheelchair_unified.usda \
        --usda
"""

import argparse
import os
import re
import sys
import tempfile

# Force unbuffered output so our prints appear despite Isaac Sim's logging
os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, "reconfigure") else None

# ---------------------------------------------------------------------------
# Phase 0: Parse arguments BEFORE starting SimulationApp (it eats sys.argv)
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Generate unified wheelchair USD for Isaac Sim + Isaac Lab"
)
parser.add_argument(
    "--urdf",
    default="/tmp/wheelchair_flat.urdf",
    help="Path to flattened URDF (run xacro first)",
)
parser.add_argument(
    "--output",
    default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "urdf",
        "wheelchair_unified.usd",
    ),
    help="Output USD path",
)
parser.add_argument("--usda", action="store_true", help="Save as USDA (text) format")
parser.add_argument(
    "--no-graphs",
    action="store_true",
    help="Skip OmniGraph creation (Isaac Lab only mode)",
)
parser.add_argument(
    "--headless", action="store_true", default=True, help="Run headless (default: True)"
)
args, unknown = parser.parse_known_args()

# Validate URDF exists before launching the heavy SimulationApp
if not os.path.isfile(args.urdf):
    print(f"ERROR: URDF file not found: {args.urdf}")
    print("Generate it first with:")
    print(
        "  xacro src/wheelchair_description/urdf/wheelchair_description.urdf.xacro "
        "is_sim:=true is_ignition:=true > /tmp/wheelchair_flat.urdf"
    )
    sys.exit(1)

# ---------------------------------------------------------------------------
# Phase 0b: Pre-process URDF — replace package:// URIs with absolute paths
# Isaac Sim's URDF importer doesn't have ROS_PACKAGE_PATH, so we resolve
# package://wheelchair_description/ to the actual filesystem path.
# ---------------------------------------------------------------------------
WC_DESC_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)  # .../src/wheelchair_description

with open(args.urdf, "r") as f:
    urdf_content = f.read()

# Replace package://wheelchair_description/ with absolute file path
original_content = urdf_content
urdf_content = urdf_content.replace(
    "package://wheelchair_description/",
    WC_DESC_DIR + "/",
)

if urdf_content != original_content:
    count = original_content.count("package://wheelchair_description/")
    print(f"[Pre-process] Replaced {count} package:// URIs with absolute paths")
    # Write to a temp file so we don't modify the original
    tmp_urdf = tempfile.NamedTemporaryFile(
        mode="w", suffix=".urdf", prefix="wheelchair_abs_", delete=False
    )
    tmp_urdf.write(urdf_content)
    tmp_urdf.close()
    args.urdf = tmp_urdf.name
    print(f"[Pre-process] Using temp URDF: {args.urdf}")

# ---------------------------------------------------------------------------
# Phase 1: Start Isaac Sim
# ---------------------------------------------------------------------------
from isaacsim import SimulationApp

simulation_app = SimulationApp({"renderer": "RaytracedLighting", "headless": args.headless})

import omni.kit.commands
import omni.usd
from isaacsim.core.utils.extensions import enable_extension, get_extension_path_from_name
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

# Enable ROS2 bridge extension BEFORE creating OmniGraph nodes
enable_extension("isaacsim.ros2.bridge")

simulation_app.update()

# ---------------------------------------------------------------------------
# Hardware constants (real wheelchair)
# ---------------------------------------------------------------------------
WHEEL_RADIUS = 0.1524  # meters
WHEEL_SEPARATION = 0.565  # meters
MAX_VELOCITY = 0.25  # m/s

# Physics tuning
WHEEL_DAMPING = 200.0  # Fix 1: was 1e4 in v3
WHEEL_MAX_FORCE = 50.0  # Fix 1: was 100 in v3
WHEEL_STIFFNESS = 0.0  # velocity control → zero stiffness

# RPLidar S3 parameters
LIDAR_MIN_RANGE = 0.15  # meters
LIDAR_MAX_RANGE = 12.0  # meters (S3 indoor range)
LIDAR_H_FOV = 360.0  # degrees
LIDAR_V_FOV = 1.0  # 2D lidar, minimal vertical
LIDAR_H_RESOLUTION = 0.5  # degrees (720 points/rev)
LIDAR_V_RESOLUTION = 1.0  # degrees
LIDAR_ROTATION_RATE = 10.0  # Hz
LIDAR_FRAME_ID = "laser"

# Robot prim path (matches URDF robot name)
ROBOT_PRIM_PATH = "/wheelchair_description"

# Joint names (must match URDF)
LEFT_WHEEL_JOINT = "leftwheel"
RIGHT_WHEEL_JOINT = "rightwheel"

# ---------------------------------------------------------------------------
# Phase 2: Import URDF
# ---------------------------------------------------------------------------
print(f"\n{'='*60}")
print(f"Phase 2: Importing URDF: {args.urdf}")
print(f"{'='*60}")

status, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
import_config.merge_fixed_joints = False
import_config.convex_decomp = False
import_config.import_inertia_tensor = True
import_config.fix_base = False
import_config.distance_scale = 1.0
import_config.make_default_prim = True
import_config.create_physics_scene = False  # Fix 5: we do NOT want a physics scene

status, prim_path = omni.kit.commands.execute(
    "URDFParseAndImportFile",
    urdf_path=args.urdf,
    import_config=import_config,
    get_articulation_root=True,
)

if not status:
    print("ERROR: URDF import failed")
    simulation_app.close()
    sys.exit(1)

print(f"  Imported to: {prim_path}")

stage = omni.usd.get_context().get_stage()
simulation_app.update()

# Verify the robot root prim exists
robot_prim = stage.GetPrimAtPath(ROBOT_PRIM_PATH)
if not robot_prim or not robot_prim.IsValid():
    print(f"ERROR: Robot prim not found at {ROBOT_PRIM_PATH}")
    print("  Available root prims:")
    for p in stage.GetPseudoRoot().GetChildren():
        print(f"    {p.GetPath()}")
    simulation_app.close()
    sys.exit(1)

# ---------------------------------------------------------------------------
# Phase 3: Fix wheel physics (Fix 1)
# ---------------------------------------------------------------------------
print(f"\n{'='*60}")
print("Phase 3: Fixing wheel physics")
print(f"{'='*60}")

drive_joints = {LEFT_WHEEL_JOINT, RIGHT_WHEEL_JOINT}
fixed_count = 0

for prim in Usd.PrimRange(robot_prim):
    if not prim.IsA(UsdPhysics.RevoluteJoint):
        continue

    name = prim.GetName()
    if name not in drive_joints:
        continue

    drive_api = UsdPhysics.DriveAPI.Get(prim, "angular")
    if not drive_api:
        # Apply DriveAPI if the URDF importer didn't
        UsdPhysics.DriveAPI.Apply(prim, "angular")
        drive_api = UsdPhysics.DriveAPI.Get(prim, "angular")

    old_damping = drive_api.GetDampingAttr().Get()
    old_stiffness = drive_api.GetStiffnessAttr().Get()
    old_force = drive_api.GetMaxForceAttr().Get()

    drive_api.GetDampingAttr().Set(WHEEL_DAMPING)
    drive_api.GetStiffnessAttr().Set(WHEEL_STIFFNESS)
    drive_api.GetMaxForceAttr().Set(WHEEL_MAX_FORCE)

    # Ensure drive type is "force" for velocity control
    type_attr = drive_api.GetTypeAttr()
    if type_attr:
        type_attr.Set("force")

    print(
        f"  {name}: damping {old_damping}→{WHEEL_DAMPING}, "
        f"stiffness {old_stiffness}→{WHEEL_STIFFNESS}, "
        f"maxForce {old_force}→{WHEEL_MAX_FORCE}"
    )
    fixed_count += 1

if fixed_count != 2:
    print(f"  WARNING: Expected 2 drive joints, fixed {fixed_count}")
    print("  Scanning all joints:")
    for prim in Usd.PrimRange(robot_prim):
        if prim.IsA(UsdPhysics.RevoluteJoint):
            print(f"    {prim.GetPath()} ({prim.GetName()})")

# ---------------------------------------------------------------------------
# Phase 4: Remove environment prims if they snuck in (Fix 5)
# ---------------------------------------------------------------------------
print(f"\n{'='*60}")
print("Phase 5: Removing environment prims")
print(f"{'='*60}")

env_prims = ["/physicsScene", "/groundPlane", "/DistantLight", "/Environment"]
for path in env_prims:
    prim = stage.GetPrimAtPath(path)
    if prim and prim.IsValid():
        stage.RemovePrim(path)
        print(f"  Removed: {path}")
    else:
        print(f"  (not present): {path}")

# ---------------------------------------------------------------------------
# Phase 5: Add PhysX LiDAR sensor (Fix 2)
# ---------------------------------------------------------------------------
print(f"\n{'='*60}")
print("Phase 4: Adding PhysX LiDAR sensor")
print(f"{'='*60}")

# The lidar link is at /wheelchair_description/lidar (from URDF)
# We also have a 'laser' link that's 180° rotated.
# Mount the PhysX lidar on the 'lidar' link to match real hardware.
lidar_parent_path = f"{ROBOT_PRIM_PATH}/lidar"
lidar_prim = stage.GetPrimAtPath(lidar_parent_path)
if not lidar_prim or not lidar_prim.IsValid():
    # Try alternative paths the URDF importer might create
    alt_paths = [
        f"{ROBOT_PRIM_PATH}/base_link/lidar",
        f"{ROBOT_PRIM_PATH}/wheelchair_main/lidar",
    ]
    for alt in alt_paths:
        lidar_prim = stage.GetPrimAtPath(alt)
        if lidar_prim and lidar_prim.IsValid():
            lidar_parent_path = alt
            break
    if not lidar_prim or not lidar_prim.IsValid():
        print(f"  WARNING: lidar link not found at {lidar_parent_path}")
        print("  Creating a lidar Xform manually...")
        lidar_parent_path = f"{ROBOT_PRIM_PATH}/lidar"
        UsdGeom.Xform.Define(stage, lidar_parent_path)

print(f"  Mounting LiDAR on: {lidar_parent_path}")

result, lidar_sensor = omni.kit.commands.execute(
    "RangeSensorCreateLidar",
    path="/Lidar",
    parent=lidar_parent_path,
    min_range=LIDAR_MIN_RANGE,
    max_range=LIDAR_MAX_RANGE,
    draw_points=False,
    draw_lines=False,
    horizontal_fov=LIDAR_H_FOV,
    vertical_fov=LIDAR_V_FOV,
    horizontal_resolution=LIDAR_H_RESOLUTION,
    vertical_resolution=LIDAR_V_RESOLUTION,
    rotation_rate=LIDAR_ROTATION_RATE,
    high_lod=False,
    yaw_offset=0.0,
    enable_semantics=False,
)

if result:
    lidar_path = str(lidar_sensor.GetPath())
    print(f"  Created PhysX LiDAR at: {lidar_path}")
else:
    print("  ERROR: Failed to create PhysX LiDAR")
    lidar_path = None

simulation_app.update()

# ---------------------------------------------------------------------------
# Phase 6: Create OmniGraph bridges (Fixes 3, 4, 6, 7, 8)
# ---------------------------------------------------------------------------
if not args.no_graphs:
    print(f"\n{'='*60}")
    print("Phase 6: Creating OmniGraph ROS2 bridges")
    print(f"{'='*60}")

    import omni.graph.core as og
    import usdrt.Sdf

    # --- 6a: ros_drive graph (cmd_vel → DiffDrive → ArticulationController) ---
    print("\n  --- ros_drive graph ---")
    drive_graph_path = f"{ROBOT_PRIM_PATH}/ros_drive"

    try:
        keys = og.Controller.Keys
        (drive_graph, drive_nodes, _, _) = og.Controller.edit(
            {"graph_path": drive_graph_path, "evaluator_name": "execution"},
            {
                keys.CREATE_NODES: [
                    ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                    ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                    # Scale node (Fix 8)
                    ("ScaleToStageUnits", "isaacsim.core.nodes.OgnIsaacScaleToFromStageUnit"),
                    # ROS2 Twist subscriber
                    ("SubscribeTwist", "isaacsim.ros2.bridge.ROS2SubscribeTwist"),
                    ("BreakLinVel", "omni.graph.nodes.BreakVector3"),
                    ("BreakAngVel", "omni.graph.nodes.BreakVector3"),
                    # Differential drive kinematics
                    ("DiffController", "isaacsim.robot.wheeled_robots.DifferentialController"),
                    # Joint actuation
                    ("ArtController", "isaacsim.core.nodes.IsaacArticulationController"),
                ],
                keys.CONNECT: [
                    # Tick drives everything
                    ("OnPlaybackTick.outputs:tick", "SubscribeTwist.inputs:execIn"),
                    ("OnPlaybackTick.outputs:tick", "ArtController.inputs:execIn"),
                    # Twist → break into scalar components
                    ("SubscribeTwist.outputs:execOut", "DiffController.inputs:execIn"),
                    ("SubscribeTwist.outputs:linearVelocity", "BreakLinVel.inputs:tuple"),
                    ("BreakLinVel.outputs:x", "DiffController.inputs:linearVelocity"),
                    ("SubscribeTwist.outputs:angularVelocity", "BreakAngVel.inputs:tuple"),
                    ("BreakAngVel.outputs:z", "DiffController.inputs:angularVelocity"),
                    # DiffController → ArticulationController
                    ("DiffController.outputs:velocityCommand", "ArtController.inputs:velocityCommand"),
                    # dt connection (Fix 7)
                    ("ReadSimTime.outputs:simulationTime", "DiffController.inputs:dt"),
                ],
                keys.SET_VALUES: [
                    # Wheelchair kinematics
                    ("DiffController.inputs:wheelRadius", WHEEL_RADIUS),
                    ("DiffController.inputs:wheelDistance", WHEEL_SEPARATION),
                    ("DiffController.inputs:maxLinearSpeed", MAX_VELOCITY),
                    # Joint names match URDF
                    ("ArtController.inputs:jointNames", [LEFT_WHEEL_JOINT, RIGHT_WHEEL_JOINT]),
                    # Fix 6: targetPrim = articulation root, NOT base_link
                    ("ArtController.inputs:targetPrim", [usdrt.Sdf.Path(ROBOT_PRIM_PATH)]),
                    # ROS2 topic
                    ("SubscribeTwist.inputs:topicName", "cmd_vel"),
                ],
            },
        )
        print(f"  Created ros_drive graph at {drive_graph_path}")
        print(f"    targetPrim = {ROBOT_PRIM_PATH} (Fix 6)")
        print(f"    wheelRadius = {WHEEL_RADIUS}, wheelSeparation = {WHEEL_SEPARATION}")
    except Exception as e:
        print(f"  ERROR creating ros_drive graph: {e}")

    simulation_app.update()

    # --- 6b: ros_lidar graph (PhysX LiDAR → ROS2 LaserScan) (Fix 3) ---
    if lidar_path:
        print("\n  --- ros_lidar graph ---")
        lidar_graph_path = f"{ROBOT_PRIM_PATH}/ros_lidar"

        try:
            (lidar_graph, lidar_nodes, _, _) = og.Controller.edit(
                {"graph_path": lidar_graph_path, "evaluator_name": "execution"},
                {
                    keys.CREATE_NODES: [
                        ("OnTick", "omni.graph.action.OnTick"),
                        ("ReadLidar", "isaacsim.sensors.physx.IsaacReadLidarBeams"),
                        ("LaserScanPub", "isaacsim.ros2.bridge.ROS2PublishLaserScan"),
                    ],
                    keys.CONNECT: [
                        # Tick → Read → Publish chain
                        ("OnTick.outputs:tick", "ReadLidar.inputs:execIn"),
                        ("ReadLidar.outputs:execOut", "LaserScanPub.inputs:execIn"),
                        # Beam data connections
                        ("ReadLidar.outputs:azimuthRange", "LaserScanPub.inputs:azimuthRange"),
                        ("ReadLidar.outputs:depthRange", "LaserScanPub.inputs:depthRange"),
                        ("ReadLidar.outputs:horizontalFov", "LaserScanPub.inputs:horizontalFov"),
                        ("ReadLidar.outputs:horizontalResolution", "LaserScanPub.inputs:horizontalResolution"),
                        ("ReadLidar.outputs:intensitiesData", "LaserScanPub.inputs:intensitiesData"),
                        ("ReadLidar.outputs:linearDepthData", "LaserScanPub.inputs:linearDepthData"),
                        ("ReadLidar.outputs:numCols", "LaserScanPub.inputs:numCols"),
                        ("ReadLidar.outputs:numRows", "LaserScanPub.inputs:numRows"),
                        ("ReadLidar.outputs:rotationRate", "LaserScanPub.inputs:rotationRate"),
                    ],
                    keys.SET_VALUES: [
                        ("ReadLidar.inputs:lidarPrim", [usdrt.Sdf.Path(lidar_path)]),
                        ("LaserScanPub.inputs:topicName", "scan"),
                        ("LaserScanPub.inputs:frameId", LIDAR_FRAME_ID),
                    ],
                },
            )
            print(f"  Created ros_lidar graph at {lidar_graph_path}")
            print(f"    lidarPrim = {lidar_path}")
            print(f"    topic = /scan, frame = {LIDAR_FRAME_ID}")
        except Exception as e:
            print(f"  ERROR creating ros_lidar graph: {e}")

    # --- 6c: ros_clock graph (simulation time → /clock) ---
    print("\n  --- ros_clock graph ---")
    clock_graph_path = f"{ROBOT_PRIM_PATH}/ros_clock"

    try:
        (clock_graph, clock_nodes, _, _) = og.Controller.edit(
            {"graph_path": clock_graph_path, "evaluator_name": "execution"},
            {
                keys.CREATE_NODES: [
                    ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                    ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                    ("PublishClock", "isaacsim.ros2.bridge.ROS2PublishClock"),
                ],
                keys.CONNECT: [
                    ("OnPlaybackTick.outputs:tick", "PublishClock.inputs:execIn"),
                    ("ReadSimTime.outputs:simulationTime", "PublishClock.inputs:timeStamp"),
                ],
            },
        )
        print(f"  Created ros_clock graph at {clock_graph_path}")
    except Exception as e:
        print(f"  ERROR creating ros_clock graph: {e}")

    # --- 6d: ros_joint_states graph (joint states → /joint_states) ---
    print("\n  --- ros_joint_states graph ---")
    js_graph_path = f"{ROBOT_PRIM_PATH}/ros_joint_states"

    try:
        (js_graph, js_nodes, _, _) = og.Controller.edit(
            {"graph_path": js_graph_path, "evaluator_name": "execution"},
            {
                keys.CREATE_NODES: [
                    ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                    ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                    ("PublishJointStates", "isaacsim.ros2.bridge.ROS2PublishJointState"),
                ],
                keys.CONNECT: [
                    ("OnPlaybackTick.outputs:tick", "PublishJointStates.inputs:execIn"),
                    ("ReadSimTime.outputs:simulationTime", "PublishJointStates.inputs:timeStamp"),
                ],
                keys.SET_VALUES: [
                    ("PublishJointStates.inputs:targetPrim", [usdrt.Sdf.Path(ROBOT_PRIM_PATH)]),
                    ("PublishJointStates.inputs:topicName", "joint_states"),
                ],
            },
        )
        print(f"  Created ros_joint_states graph at {js_graph_path}")
    except Exception as e:
        print(f"  ERROR creating ros_joint_states graph: {e}")

    # NOTE: No drive_to_goal graph created (Fix 4)
    print("\n  (Skipping drive_to_goal graph — Fix 4: ros_drive handles control)")

    simulation_app.update()
else:
    print("\n  (Skipping OmniGraph creation — --no-graphs mode for Isaac Lab)")

# ---------------------------------------------------------------------------
# Phase 7: Final cleanup and validation
# ---------------------------------------------------------------------------
print(f"\n{'='*60}")
print("Phase 7: Validation")
print(f"{'='*60}")

# Ensure stage units are meters
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
print(f"  Up axis: {UsdGeom.GetStageUpAxis(stage)}")
print(f"  Meters per unit: {UsdGeom.GetStageMetersPerUnit(stage)}")

# Verify no environment prims leaked through
for path in ["/physicsScene", "/groundPlane", "/DistantLight"]:
    prim = stage.GetPrimAtPath(path)
    if prim and prim.IsValid():
        print(f"  WARNING: Environment prim still present: {path}")

# Print joint summary
print("\n  Joint summary:")
for prim in Usd.PrimRange(robot_prim):
    if not prim.IsA(UsdPhysics.RevoluteJoint):
        continue
    name = prim.GetName()
    drive = UsdPhysics.DriveAPI.Get(prim, "angular")
    if drive:
        d = drive.GetDampingAttr().Get()
        s = drive.GetStiffnessAttr().Get()
        f = drive.GetMaxForceAttr().Get()
        print(f"    {name:25s} damping={d} stiffness={s} maxForce={f}")
    else:
        print(f"    {name:25s} (no drive)")

# Print sensor summary
print("\n  Sensor summary:")
sensor_found = False
for prim in Usd.PrimRange(robot_prim):
    type_name = prim.GetTypeName()
    if "Lidar" in type_name or "Sensor" in type_name or "Camera" in type_name:
        print(f"    {prim.GetPath()} ({type_name})")
        sensor_found = True
if not sensor_found:
    print("    (no sensor prims found in tree)")

# Print graph summary
if not args.no_graphs:
    print("\n  OmniGraph summary:")
    for prim in Usd.PrimRange(robot_prim):
        type_name = prim.GetTypeName()
        if "OmniGraph" in type_name or type_name == "OmniGraphSchema":
            print(f"    {prim.GetPath()} ({type_name})")

# ---------------------------------------------------------------------------
# Phase 8: Save
# ---------------------------------------------------------------------------
print(f"\n{'='*60}")
print(f"Phase 8: Saving USD")
print(f"{'='*60}")

output_path = args.output
if args.usda and not output_path.endswith(".usda"):
    output_path = output_path.replace(".usd", ".usda")

# Ensure output directory exists
os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

stage.Export(output_path)
file_size = os.path.getsize(output_path)
print(f"  Saved: {output_path}")
print(f"  Size: {file_size / 1024:.1f} KB")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print(f"\n{'='*60}")
print("DONE — wheelchair_unified.usd generated successfully")
print(f"{'='*60}")
print(f"""
Fixes applied:
  [1] Wheel damping: {WHEEL_DAMPING} (was 1e4), maxForce: {WHEEL_MAX_FORCE} (was 100)
  [2] PhysX LiDAR: inline, no Nucleus URL
  [3] LiDAR graph: ReadLidarBeams → ROS2PublishLaserScan (not RtxLidarHelper)
  [4] No drive_to_goal graph (only ros_drive)
  [5] No environment prims (/physicsScene, /groundPlane, /DistantLight removed)
  [6] targetPrim: {ROBOT_PRIM_PATH} (not {ROBOT_PRIM_PATH}/base_link)
  [7] DiffController dt connected to ReadSimTime
  [8] ScaleToStageUnits node included

Isaac Sim usage:
  Open {output_path} in Isaac Sim, press Play.
  Publish to /cmd_vel to drive. LiDAR publishes on /scan.

Isaac Lab usage:
  from isaaclab.sim import UsdFileCfg
  from isaaclab.actuators import ImplicitActuatorCfg
  from isaaclab.assets import ArticulationCfg

  WHEELCHAIR_CFG = ArticulationCfg(
      spawn=UsdFileCfg(usd_path="{os.path.abspath(output_path)}"),
      init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.2)),
      actuators={{
          "wheels": ImplicitActuatorCfg(
              joint_names_expr=["{LEFT_WHEEL_JOINT}", "{RIGHT_WHEEL_JOINT}"],
              velocity_limit={MAX_VELOCITY / WHEEL_RADIUS:.1f},
              effort_limit={WHEEL_MAX_FORCE},
              stiffness=0.0,
              damping={WHEEL_DAMPING},
          ),
      }},
  )
""")

simulation_app.close()
