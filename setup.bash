#!/bin/bash
# ============================================================================
# WHEELCHAIR NAVIGATION WORKSPACE SETUP
# ============================================================================
# This script sets up the wheelchair navigation ROS2 workspace.
#
# Usage:
#   source setup.bash        # First time: builds and sources workspace
#   source setup.bash --skip # Skip build, just source existing install
#
# After sourcing, you can run:
#   run_nav                  # Launch full autonomous navigation
#   run_slam                 # Launch SLAM mapping
# ============================================================================

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export WHEELCHAIR_WS="$SCRIPT_DIR"

echo "============================================================================"
echo "  WHEELCHAIR NAVIGATION WORKSPACE SETUP"
echo "============================================================================"
echo "  Workspace: $WHEELCHAIR_WS"
echo "============================================================================"

# Source ROS2
if [ -f /opt/ros/jazzy/setup.bash ]; then
    source /opt/ros/jazzy/setup.bash
    echo "[OK] Sourced ROS2 Jazzy"
elif [ -f /opt/ros/humble/setup.bash ]; then
    source /opt/ros/humble/setup.bash
    echo "[OK] Sourced ROS2 Humble"
else
    echo "[ERROR] ROS2 not found! Please install ROS2 Jazzy or Humble."
    return 1 2>/dev/null || exit 1
fi

# Check if we should skip building
if [ "$1" != "--skip" ]; then
    # Build the workspace
    echo ""
    echo "[BUILD] Building workspace..."
    cd "$WHEELCHAIR_WS"

    # Clean build if requested
    if [ "$1" == "--clean" ]; then
        echo "[BUILD] Cleaning previous build..."
        rm -rf build install log
    fi

    colcon build --symlink-install 2>&1 | tail -20

    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo "[OK] Build successful"
    else
        echo "[ERROR] Build failed! Check errors above."
        return 1 2>/dev/null || exit 1
    fi
fi

# Source the workspace
if [ -f "$WHEELCHAIR_WS/install/setup.bash" ]; then
    source "$WHEELCHAIR_WS/install/setup.bash"
    echo "[OK] Sourced workspace install"
else
    echo "[ERROR] Workspace not built! Run 'source setup.bash' without --skip first."
    return 1 2>/dev/null || exit 1
fi

# Create convenience functions
run_nav() {
    echo "============================================================================"
    echo "  LAUNCHING AUTONOMOUS NAVIGATION"
    echo "============================================================================"
    echo "  Press Ctrl+C to stop"
    echo "============================================================================"

    # Kill any existing ROS2 processes
    pkill -9 -f ros2 2>/dev/null
    pkill -9 -f rviz 2>/dev/null
    sleep 2

    ros2 launch wheelchair_bringup wheelchair_fusion_nav.launch.py "$@"
}

run_slam() {
    echo "============================================================================"
    echo "  LAUNCHING SLAM MAPPING"
    echo "============================================================================"
    echo "  Drive the wheelchair around to build a map."
    echo "  Press Ctrl+C to stop and save the map."
    echo "============================================================================"

    # Kill any existing ROS2 processes
    pkill -9 -f ros2 2>/dev/null
    pkill -9 -f rviz 2>/dev/null
    sleep 2

    ros2 launch wheelchair_bringup wheelchair_slam_mapping.launch.py "$@"
}

run_localization() {
    echo "============================================================================"
    echo "  LAUNCHING LOCALIZATION ONLY"
    echo "============================================================================"

    # Kill any existing ROS2 processes
    pkill -9 -f ros2 2>/dev/null
    pkill -9 -f rviz 2>/dev/null
    sleep 2

    ros2 launch wheelchair_bringup wheelchair_fusion_localization.launch.py "$@"
}

run_slam_session() {
    echo "============================================================================"
    echo "  LAUNCHING SLAM SESSION (auto-save + analysis)"
    echo "============================================================================"
    echo "  Drive the wheelchair around to build a map."
    echo "  Press Ctrl+C to save map and run instant analysis."
    echo "============================================================================"

    bash "$WHEELCHAIR_WS/run_slam_session.bash" "$@"
}

export -f run_nav
export -f run_slam
export -f run_localization
export -f run_slam_session

echo ""
echo "============================================================================"
echo "  SETUP COMPLETE"
echo "============================================================================"
echo "  Available commands:"
echo "    run_nav             - Launch autonomous navigation"
echo "    run_slam            - Launch SLAM mapping (manual save)"
echo "    run_slam_session    - Launch SLAM with auto-save + analysis"
echo "    run_localization    - Launch localization only"
echo ""
echo "  Example:"
echo "    run_slam_session slam_config:=/path/to/config.yaml"
echo "============================================================================"

cd "$WHEELCHAIR_WS"
