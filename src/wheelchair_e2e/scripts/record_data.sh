#!/bin/bash
# =============================================================
# Data Collection for E2E Wheelchair Navigation
# =============================================================
#
# MINIMAL 2-topic recording for E2E-2 (BEV-Velocity CNN):
#
#   /scan_fused          → BEV occupancy grid (channels 0, 1)
#   /odometry/filtered   → velocity labels (twist.linear.x = v,
#                           twist.angular.z = ω, from EKF-fused
#                           wheel encoders + IMU)
#                         → odom history input (v, ω, θ × 10 steps)
#                         → ego-motion trail (BEV channel 3)
#                         → goal computation (5s stop = segment endpoint)
#
# Labels are ACTUAL EXECUTED velocities from EKF (not commanded /cmd_vel).
# This captures real human driving dynamics including motor response,
# acceleration curves, and physical constraints.
#
# Goal detection: stop for 5+ seconds → end of segment → that position
# becomes the goal for all preceding timesteps in that segment.
# Relative goal computed from odom pose, no map/AMCL needed.
#
# Prerequisites:
#   1. Sensor stack running (LiDAR + cameras + scan_depth_fusion + EKF)
#   2. Teleop controller (joystick or keyboard)
#   NO map, NO AMCL, NO Nav2 needed.
#
# Usage:
#   ./record_data.sh                    # default name with timestamp
#   ./record_data.sh my_corridor_run    # custom name
# =============================================================

OUTPUT_NAME=${1:-"e2e_training_$(date +%Y%m%d_%H%M%S)"}

echo "============================================"
echo "E2E-2 Data Collection (2 topics)"
echo "============================================"
echo "Output: $OUTPUT_NAME"
echo ""
echo "Recording topics:"
echo "  /scan_fused           (BEV occupancy input)"
echo "  /odometry/filtered    (velocity labels + odom + goal)"
echo ""
echo "Labels: actual executed velocity from EKF"
echo "  twist.linear.x  = v  (wheel encoders, filtered)"
echo "  twist.angular.z = ω  (IMU yaw rate, filtered)"
echo ""
echo "Goal: stop 5s = segment endpoint"
echo "Estimated storage: ~2 MB/min"
echo "Press Ctrl+C to stop"
echo "============================================"

ros2 bag record \
    /scan_fused \
    /odometry/filtered \
    -o "$OUTPUT_NAME"
