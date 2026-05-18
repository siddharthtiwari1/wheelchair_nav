// KinoFlow v2 Nav2 Controller Plugin — ONNX Runtime C++ implementation
//
// Two-ONNX-model architecture:
//   encoder.onnx: scan + residuals + goal + odom → conditioning (256-d)
//   vector_field.onnx: x_t + t + cond → vector field (called 3× for Euler ODE)
//
// Pure C++ math for: noise generation, Euler integration, trajectory scaling,
// unicycle kinematics integration, costmap scoring, EMA smoothing.

#include "kinoflow_controller/kinoflow_controller.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <numeric>

#include "nav2_core/controller_exceptions.hpp"
#include "nav2_util/node_utils.hpp"
#include "pluginlib/class_list_macros.hpp"
#include "tf2/utils.h"

namespace kinoflow_controller
{

void KinoFlowController::configure(
  const rclcpp_lifecycle::LifecycleNode::WeakPtr & parent,
  std::string name,
  std::shared_ptr<tf2_ros::Buffer> tf,
  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros)
{
  node_ = parent;
  plugin_name_ = name;
  tf_ = tf;
  costmap_ros_ = costmap_ros;

  auto node = node_.lock();
  if (!node) {
    throw std::runtime_error("Failed to lock node in configure()");
  }
  logger_ = node->get_logger();
  clock_ = node->get_clock();

  // Declare and read parameters
  nav2_util::declare_parameter_if_not_declared(
    node, name + ".encoder_model_path",
    rclcpp::ParameterValue(std::string("")));
  nav2_util::declare_parameter_if_not_declared(
    node, name + ".vector_field_model_path",
    rclcpp::ParameterValue(std::string("")));
  nav2_util::declare_parameter_if_not_declared(
    node, name + ".v_max", rclcpp::ParameterValue(0.25));
  nav2_util::declare_parameter_if_not_declared(
    node, name + ".w_max", rclcpp::ParameterValue(1.0));
  nav2_util::declare_parameter_if_not_declared(
    node, name + ".dt", rclcpp::ParameterValue(0.1));
  nav2_util::declare_parameter_if_not_declared(
    node, name + ".horizon", rclcpp::ParameterValue(10));
  nav2_util::declare_parameter_if_not_declared(
    node, name + ".n_samples", rclcpp::ParameterValue(8));
  nav2_util::declare_parameter_if_not_declared(
    node, name + ".n_euler_steps", rclcpp::ParameterValue(3));
  nav2_util::declare_parameter_if_not_declared(
    node, name + ".scan_points", rclcpp::ParameterValue(720));
  nav2_util::declare_parameter_if_not_declared(
    node, name + ".temporal_frames", rclcpp::ParameterValue(5));
  nav2_util::declare_parameter_if_not_declared(
    node, name + ".safety_min_range", rclcpp::ParameterValue(0.3));
  nav2_util::declare_parameter_if_not_declared(
    node, name + ".safety_slow_range", rclcpp::ParameterValue(0.6));

  node->get_parameter(name + ".encoder_model_path", encoder_model_path_);
  node->get_parameter(name + ".vector_field_model_path", vf_model_path_);
  node->get_parameter(name + ".v_max", v_max_);
  node->get_parameter(name + ".w_max", w_max_);
  node->get_parameter(name + ".dt", dt_);
  node->get_parameter(name + ".horizon", horizon_);
  node->get_parameter(name + ".n_samples", n_samples_);
  node->get_parameter(name + ".n_euler_steps", n_euler_steps_);
  node->get_parameter(name + ".scan_points", scan_points_);
  node->get_parameter(name + ".temporal_frames", temporal_frames_);
  node->get_parameter(name + ".safety_min_range", safety_min_range_);
  node->get_parameter(name + ".safety_slow_range", safety_slow_range_);

  // Initialize ONNX Runtime
  ort_env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "KinoFlow");

  Ort::SessionOptions opts;
  opts.SetIntraOpNumThreads(1);
  opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

  // Load ONNX models (graceful fallback to dummy mode if files missing)
  auto load_model = [&](const std::string & path, const std::string & name)
    -> std::unique_ptr<Ort::Session> {
    if (path.empty()) {
      RCLCPP_WARN(logger_, "%s path is empty — skipping", name.c_str());
      return nullptr;
    }
    std::ifstream f(path);
    if (!f.good()) {
      RCLCPP_WARN(logger_, "%s file not found: %s — skipping",
        name.c_str(), path.c_str());
      return nullptr;
    }
    auto session = std::make_unique<Ort::Session>(*ort_env_, path.c_str(), opts);
    RCLCPP_INFO(logger_, "Loaded %s: %s", name.c_str(), path.c_str());
    return session;
  };

  encoder_session_ = load_model(encoder_model_path_, "encoder");
  vf_session_ = load_model(vf_model_path_, "vector_field");

  if (!encoder_session_ || !vf_session_) {
    RCLCPP_WARN(logger_,
      "ONNX models not loaded — running in DUMMY mode (zero velocity)");
  }

  // Initialize footprint collision checker (RPP/MPPI pattern)
  footprint_collision_checker_ =
    std::make_unique<nav2_costmap_2d::FootprintCollisionChecker<
      nav2_costmap_2d::Costmap2D *>>(costmap_ros_->getCostmap());

  RCLCPP_INFO(logger_,
    "KinoFlow configured: v_max=%.2f, w_max=%.2f, K=%d, H=%d, euler=%d",
    v_max_, w_max_, n_samples_, horizon_, n_euler_steps_);
}

void KinoFlowController::cleanup()
{
  RCLCPP_INFO(logger_, "Cleaning up KinoFlow controller");
  scan_sub_.reset();
  odom_sub_.reset();
  encoder_session_.reset();
  vf_session_.reset();
  ort_env_.reset();
}

void KinoFlowController::activate()
{
  RCLCPP_INFO(logger_, "Activating KinoFlow controller");
  auto node = node_.lock();
  if (!node) {return;}

  scan_sub_ = node->create_subscription<sensor_msgs::msg::LaserScan>(
    "/scan_fused", rclcpp::SensorDataQoS(),
    std::bind(&KinoFlowController::scanCallback, this, std::placeholders::_1));

  odom_sub_ = node->create_subscription<nav_msgs::msg::Odometry>(
    "/odometry/filtered", rclcpp::SensorDataQoS(),
    std::bind(&KinoFlowController::odomCallback, this, std::placeholders::_1));

  // Safety filter init
  last_valid_inference_time_ = clock_->now();
  last_scan_time_ = clock_->now();
  last_odom_time_ = clock_->now();
  consecutive_sanity_fails_ = 0;
  crawl_mode_ = false;
}

void KinoFlowController::deactivate()
{
  RCLCPP_INFO(logger_, "Deactivating KinoFlow controller");
  scan_sub_.reset();
  odom_sub_.reset();
}

void KinoFlowController::setPlan(const nav_msgs::msg::Path & path)
{
  current_path_ = path;
}

void KinoFlowController::setSpeedLimit(
  const double & speed_limit, const bool & percentage)
{
  speed_limit_ = static_cast<float>(speed_limit);
  speed_limit_is_pct_ = percentage;
}

// ==========================================================================
// Sensor callbacks
// ==========================================================================

void KinoFlowController::scanCallback(
  const sensor_msgs::msg::LaserScan::SharedPtr msg)
{
  std::lock_guard<std::mutex> lock(scan_mutex_);

  std::vector<float> scan(scan_points_, 0.0f);
  int n = std::min(static_cast<int>(msg->ranges.size()), scan_points_);
  for (int i = 0; i < n; ++i) {
    float r = msg->ranges[i];
    scan[i] = (std::isfinite(r) && r > msg->range_min) ? r : 0.0f;
  }

  scan_buffer_.push_back(std::move(scan));
  while (static_cast<int>(scan_buffer_.size()) > temporal_frames_) {
    scan_buffer_.pop_front();
  }

  last_scan_time_ = clock_->now();
}

void KinoFlowController::odomCallback(
  const nav_msgs::msg::Odometry::SharedPtr msg)
{
  std::lock_guard<std::mutex> lock(odom_mutex_);

  float v = static_cast<float>(msg->twist.twist.linear.x);
  float w = static_cast<float>(msg->twist.twist.angular.z);

  // Extract yaw from quaternion
  double yaw = tf2::getYaw(msg->pose.pose.orientation);
  float theta = static_cast<float>(yaw);

  odom_buffer_.push_back({v, w, theta});
  while (odom_buffer_.size() > 10) {
    odom_buffer_.pop_front();
  }

  last_odom_time_ = clock_->now();
}

// ==========================================================================
// Main inference loop
// ==========================================================================

geometry_msgs::msg::TwistStamped KinoFlowController::computeVelocityCommands(
  const geometry_msgs::msg::PoseStamped & pose,
  const geometry_msgs::msg::Twist & /*velocity*/,
  nav2_core::GoalChecker * /*goal_checker*/)
{
  geometry_msgs::msg::TwistStamped cmd;
  cmd.header.stamp = clock_->now();
  cmd.header.frame_id = pose.header.frame_id;

  // Dummy mode: no models loaded → zero velocity
  if (!encoder_session_ || !vf_session_) {
    return cmd;
  }

  // --- Sensor staleness check (production pattern) ---
  // If sensors are stale, zero velocity — don't trust old data
  {
    double scan_age = (clock_->now() - last_scan_time_).seconds();
    double odom_age = (clock_->now() - last_odom_time_).seconds();
    if (scan_age > SENSOR_STALE_TIMEOUT_S) {
      RCLCPP_WARN_THROTTLE(logger_, *clock_, 2000,
        "SAFETY: Scan data stale (%.1fs old) — zero velocity", scan_age);
      return cmd;
    }
    if (odom_age > SENSOR_STALE_TIMEOUT_S) {
      RCLCPP_WARN_THROTTLE(logger_, *clock_, 2000,
        "SAFETY: Odom data stale (%.1fs old) — zero velocity", odom_age);
      return cmd;
    }
  }

  // Check we have enough sensor data
  bool have_scan = false;
  bool have_odom = false;
  {
    std::lock_guard<std::mutex> lock(scan_mutex_);
    have_scan = (static_cast<int>(scan_buffer_.size()) >= temporal_frames_);
  }
  {
    std::lock_guard<std::mutex> lock(odom_mutex_);
    have_odom = (odom_buffer_.size() >= 10);
  }

  if (!have_scan || !have_odom) {
    return cmd;  // Not enough data yet — zero velocity
  }

  if (current_path_.poses.empty()) {
    return cmd;
  }

  // --- Step 1: Goal features ---
  auto goal_feats = computeGoalFeatures(pose);

  // --- Step 2: Current scan ---
  std::vector<float> scan_current;
  {
    std::lock_guard<std::mutex> lock(scan_mutex_);
    scan_current = scan_buffer_.back();
  }

  // --- Step 3: Scan residuals ---
  auto scan_residuals = computeScanResiduals();

  // --- Step 4: Odom history ---
  auto odom_history = computeOdomHistory();

  // --- Step 5: Run encoder ONNX ---
  // Input shapes: scan(1,720), residuals(1,4,720), goal(1,4), odom(1,30)
  int n_residual = temporal_frames_ - 1;

  std::array<int64_t, 2> scan_shape = {1, static_cast<int64_t>(scan_points_)};
  std::array<int64_t, 3> res_shape = {
    1, static_cast<int64_t>(n_residual), static_cast<int64_t>(scan_points_)};
  std::array<int64_t, 2> goal_shape = {1, 4};
  std::array<int64_t, 2> odom_shape = {1, 30};

  auto scan_tensor = Ort::Value::CreateTensor<float>(
    mem_info_, scan_current.data(), scan_current.size(),
    scan_shape.data(), scan_shape.size());
  auto res_tensor = Ort::Value::CreateTensor<float>(
    mem_info_, scan_residuals.data(), scan_residuals.size(),
    res_shape.data(), res_shape.size());
  auto goal_tensor = Ort::Value::CreateTensor<float>(
    mem_info_, goal_feats.data(), goal_feats.size(),
    goal_shape.data(), goal_shape.size());
  auto odom_tensor = Ort::Value::CreateTensor<float>(
    mem_info_, odom_history.data(), odom_history.size(),
    odom_shape.data(), odom_shape.size());

  std::array<const char*, 4> enc_input_names = {
    "scan_current", "scan_residuals", "goal_features", "odom_history"};
  std::array<const char*, 1> enc_output_names = {"conditioning"};

  std::array<Ort::Value, 4> enc_inputs{
    std::move(scan_tensor), std::move(res_tensor),
    std::move(goal_tensor), std::move(odom_tensor)};

  auto enc_outputs = encoder_session_->Run(
    Ort::RunOptions{nullptr},
    enc_input_names.data(), enc_inputs.data(), enc_inputs.size(),
    enc_output_names.data(), enc_output_names.size());

  const float* cond_data = enc_outputs[0].GetTensorData<float>();
  constexpr int COND_DIM = 256;

  // NaN/Inf guard on encoder output (MPPI pattern)
  if (hasNanInf(cond_data, COND_DIM)) {
    RCLCPP_WARN_THROTTLE(logger_, *clock_, 1000,
      "SAFETY: Encoder output contains NaN/Inf — zero velocity");
    return cmd;
  }

  // --- Step 6: Replicate conditioning for K samples ---
  int K = n_samples_;
  int traj_dim = horizon_ * 2;

  std::vector<float> cond_k(K * COND_DIM);
  for (int k = 0; k < K; ++k) {
    std::memcpy(cond_k.data() + k * COND_DIM, cond_data, COND_DIM * sizeof(float));
  }

  // --- Step 7: Generate noise z0 ~ N(0,1) ---
  std::vector<float> z(K * traj_dim);
  for (auto & val : z) {
    val = normal_dist_(rng_);
  }

  // --- Step 8: Euler ODE loop (3 steps) ---
  float euler_dt = 1.0f / static_cast<float>(n_euler_steps_);

  for (int step = 0; step < n_euler_steps_; ++step) {
    float t_val = step * euler_dt;

    // Time tensor: (K,) all same value
    std::vector<float> t_vec(K, t_val);

    std::array<int64_t, 2> z_shape = {
      static_cast<int64_t>(K), static_cast<int64_t>(traj_dim)};
    std::array<int64_t, 1> t_shape = {static_cast<int64_t>(K)};
    std::array<int64_t, 2> c_shape = {
      static_cast<int64_t>(K), static_cast<int64_t>(COND_DIM)};

    auto z_tensor = Ort::Value::CreateTensor<float>(
      mem_info_, z.data(), z.size(), z_shape.data(), z_shape.size());
    auto t_tensor = Ort::Value::CreateTensor<float>(
      mem_info_, t_vec.data(), t_vec.size(), t_shape.data(), t_shape.size());
    auto c_tensor = Ort::Value::CreateTensor<float>(
      mem_info_, cond_k.data(), cond_k.size(), c_shape.data(), c_shape.size());

    std::array<const char*, 3> vf_input_names = {"x_t", "t", "cond"};
    std::array<const char*, 1> vf_output_names = {"v_field"};

    std::array<Ort::Value, 3> vf_inputs{
      std::move(z_tensor), std::move(t_tensor), std::move(c_tensor)};

    auto vf_outputs = vf_session_->Run(
      Ort::RunOptions{nullptr},
      vf_input_names.data(), vf_inputs.data(), vf_inputs.size(),
      vf_output_names.data(), vf_output_names.size());

    const float* dx = vf_outputs[0].GetTensorData<float>();

    // z += dx * euler_dt
    for (int i = 0; i < K * traj_dim; ++i) {
      z[i] += dx[i] * euler_dt;
    }
  }

  // NaN/Inf guard on ODE output
  if (hasNanInf(z.data(), static_cast<int>(z.size()))) {
    RCLCPP_WARN_THROTTLE(logger_, *clock_, 1000,
      "SAFETY: ODE output contains NaN/Inf — zero velocity");
    return cmd;
  }

  // --- Step 9: Scale trajectories (tanh → physical) ---
  // z is (K, traj_dim) = (K, H*2), interpret as (K, H, 2)
  // vel_trajs[k][h] = {v, omega}
  std::vector<std::vector<std::array<float, 2>>> vel_trajs(K);
  for (int k = 0; k < K; ++k) {
    vel_trajs[k].resize(horizon_);
    for (int h = 0; h < horizon_; ++h) {
      float raw_v = z[k * traj_dim + h * 2 + 0];
      float raw_w = z[k * traj_dim + h * 2 + 1];
      vel_trajs[k][h][0] = (std::tanh(raw_v) + 1.0f) / 2.0f * v_max_;
      vel_trajs[k][h][1] = std::tanh(raw_w) * w_max_;
    }
  }

  // --- Step 10: Integrate unicycle kinematics → poses ---
  // poses[k] = flat [x0,y0,θ0, x1,y1,θ1, ...]
  std::vector<std::vector<float>> poses(K);
  for (int k = 0; k < K; ++k) {
    poses[k].resize(horizon_ * 3);
    float x = 0.0f, y = 0.0f, theta = 0.0f;
    for (int h = 0; h < horizon_; ++h) {
      float v = vel_trajs[k][h][0];
      float omega = vel_trajs[k][h][1];
      theta += omega * dt_;
      x += v * std::cos(theta) * dt_;
      y += v * std::sin(theta) * dt_;
      poses[k][h * 3 + 0] = x;
      poses[k][h * 3 + 1] = y;
      poses[k][h * 3 + 2] = theta;
    }
  }

  // --- Step 11: Score against Nav2 local costmap ---
  // Goal relative to robot
  auto & goal_pose = current_path_.poses.back().pose;
  double robot_yaw = tf2::getYaw(pose.pose.orientation);
  double dx_world = goal_pose.position.x - pose.pose.position.x;
  double dy_world = goal_pose.position.y - pose.pose.position.y;
  float goal_dx = static_cast<float>(
    dx_world * std::cos(-robot_yaw) - dy_world * std::sin(-robot_yaw));
  float goal_dy = static_cast<float>(
    dx_world * std::sin(-robot_yaw) + dy_world * std::cos(-robot_yaw));

  auto * costmap = costmap_ros_->getCostmap();
  std::vector<float> scores(K);
  for (int k = 0; k < K; ++k) {
    scores[k] = scoreTrajectory(poses[k], horizon_, costmap, goal_dx, goal_dy);
  }

  // --- Step 12: Select best trajectory ---
  int best_k = static_cast<int>(
    std::max_element(scores.begin(), scores.end()) - scores.begin());
  float best_v = vel_trajs[best_k][0][0];
  float best_w = vel_trajs[best_k][0][1];

  // ================================================================
  // SAFETY FILTER — Minimal intervention for learned E2E controller
  // ================================================================
  //
  // Design philosophy (CBF/Simplex/CARE patterns):
  //   "Find the closest safe command to what the learned policy wanted.
  //    When the command IS safe, pass it through UNCHANGED."
  //
  // We ONLY handle what the downstream pipeline CANNOT:
  //   1. Inference sanity (only we see the K=8 trajectory samples)
  //   2. Costmap collision (only we can project the footprint)
  //   3. Scan-based E-STOP (fastest possible reaction, no pipeline delay)
  //   4. Per-wheel kinematics (only we know diff-drive geometry)
  //
  // We DO NOT duplicate what velocity_smoother already does:
  //   - Acceleration limiting → velocity_smoother (0.15 m/s², 0.6 rad/s²)
  //   - Velocity deadband → velocity_smoother (0.02 m/s)
  //   - EMA smoothing → velocity_smoother (40Hz interpolation)
  //   - Hard velocity clamp → velocity_smoother ([-0.05,0.25], [-0.35,0.35])
  //
  // Pipeline: KinoFlow → /cmd_vel_nav → velocity_smoother → /cmd_vel
  //           → DiffDriveController → WheelchairInterface → Arduino
  // ================================================================

  // --- Filter 1: Trajectory coherence (unique to learned controllers) ---
  // If the K=8 samples wildly disagree, the model is outputting noise.
  // Minimal intervention: cap to crawl speed instead of zeroing.
  // The best-scored trajectory is still our best guess from the policy.
  {
    float mean_v = 0.0f, mean_w = 0.0f;
    for (int k = 0; k < K; ++k) {
      mean_v += vel_trajs[k][0][0];
      mean_w += vel_trajs[k][0][1];
    }
    mean_v /= K;
    mean_w /= K;

    float var_v = 0.0f, var_w = 0.0f;
    for (int k = 0; k < K; ++k) {
      float dv = vel_trajs[k][0][0] - mean_v;
      float dw = vel_trajs[k][0][1] - mean_w;
      var_v += dv * dv;
      var_w += dw * dw;
    }
    var_v /= K;
    var_w /= K;

    float std_v = std::sqrt(var_v);
    float std_w = std::sqrt(var_w);

    // Update watchdog — inference ran successfully (catches ONNX crashes)
    last_valid_inference_time_ = clock_->now();

    if (std_v > v_max_ * 0.5f || std_w > w_max_ * 0.5f) {
      // Incoherent: cap to crawl speed (minimal intervention, not zero)
      consecutive_sanity_fails_++;
      best_v = std::min(best_v, CRAWL_V);
      best_w = std::clamp(best_w, -CRAWL_W_MAX, CRAWL_W_MAX);
      RCLCPP_WARN_THROTTLE(logger_, *clock_, 2000,
        "FILTER: Incoherent (std_v=%.3f, std_w=%.3f) → crawl cap v=%.3f w=%.3f [%d fails]",
        std_v, std_w, best_v, best_w, consecutive_sanity_fails_);

      if (consecutive_sanity_fails_ >= CRAWL_MODE_THRESHOLD && !crawl_mode_) {
        crawl_mode_ = true;
        RCLCPP_WARN(logger_,
          "FILTER: %d consecutive failures → CRAWL MODE (v≤%.2f, |w|≤%.2f)",
          consecutive_sanity_fails_, CRAWL_V, CRAWL_W_MAX);
      }
    } else {
      // Coherent inference — reset failure tracking
      consecutive_sanity_fails_ = 0;
      if (crawl_mode_) {
        crawl_mode_ = false;
        RCLCPP_INFO(logger_, "FILTER: Coherent inference restored — exiting crawl mode");
      }
    }
  }

  // --- Filter 2: Watchdog timeout ---
  // If no coherent inference for > WATCHDOG_TIMEOUT_S, zero.
  // This catches ONNX crashes, frozen threads, sustained garbage.
  {
    double since_valid = (clock_->now() - last_valid_inference_time_).seconds();
    if (since_valid > WATCHDOG_TIMEOUT_S) {
      RCLCPP_WARN_THROTTLE(logger_, *clock_, 2000,
        "FILTER: Watchdog timeout (%.1fs no valid inference) → zero", since_valid);
      cmd.twist.linear.x = 0.0;
      cmd.twist.angular.z = 0.0;
      return cmd;
    }
  }

  // --- Filter 3: Crawl mode cap ---
  // After sustained incoherence, limit velocities even when inference
  // becomes coherent again (trust must be re-earned).
  if (crawl_mode_) {
    best_v = std::min(best_v, CRAWL_V);
    best_w = std::clamp(best_w, -CRAWL_W_MAX, CRAWL_W_MAX);
  }

  // --- Filter 4: Per-wheel velocity sanity (diff-drive kinematics) ---
  // The velocity_smoother doesn't know about diff-drive geometry.
  // Ensure neither wheel exceeds MAX_WHEEL_SPEED to prevent single-wheel spin.
  // This is the ONLY velocity modification the smoother cannot do.
  {
    float wheel_l = best_v + best_w * WHEEL_HALF_SEP;
    float wheel_r = best_v - best_w * WHEEL_HALF_SEP;
    float max_wheel = std::max(std::abs(wheel_l), std::abs(wheel_r));

    if (max_wheel > MAX_WHEEL_SPEED && max_wheel > 1e-6f) {
      float scale = MAX_WHEEL_SPEED / max_wheel;
      best_v *= scale;
      best_w *= scale;
      RCLCPP_DEBUG_THROTTLE(logger_, *clock_, 2000,
        "FILTER: Per-wheel scale %.2f (wL=%.3f wR=%.3f)",
        scale, wheel_l, wheel_r);
    }
  }

  // --- Filter 5: Scan-based E-STOP (fastest possible reaction) ---
  // This bypasses the full costmap pipeline for emergency stopping.
  // Costmap updates at 5Hz. Scan arrives at 10Hz. We react immediately.
  // Only two zones: STOP and SLOW. No over-engineering.
  float min_range = std::numeric_limits<float>::max();
  {
    std::lock_guard<std::mutex> lock(scan_mutex_);
    if (!scan_buffer_.empty()) {
      for (float r : scan_buffer_.back()) {
        if (r > 0.01f && r < min_range) {
          min_range = r;
        }
      }
    }
  }

  if (min_range < safety_min_range_) {
    // E-STOP: obstacle inside safety envelope
    RCLCPP_WARN_THROTTLE(logger_, *clock_, 1000,
      "FILTER: E-STOP min_range=%.2fm < %.2fm", min_range, safety_min_range_);
    cmd.twist.linear.x = 0.0;
    cmd.twist.angular.z = 0.0;
    return cmd;
  }
  if (min_range < safety_slow_range_) {
    // Scale down proportionally — let velocity_smoother handle the ramp
    float scale = (min_range - safety_min_range_) /
                  (safety_slow_range_ - safety_min_range_);
    best_v *= scale;
    best_w *= scale;  // Reduce turning too — don't swing into obstacle
  }

  // --- Filter 6: Footprint collision projection ---
  // Project (v,w) forward and check the FULL robot footprint against costmap.
  // Minimal intervention: cascade of fallback trajectories before zeroing.
  // This is crucial for learned controllers — model might want to turn into wall,
  // but straight ahead might be clear.
  if (best_v > 0.01f || std::abs(best_w) > 0.01f) {
    if (isCollisionImminent(pose, best_v, best_w)) {
      // Fallback 1: keep speed, halve angular (model direction, less turn)
      float fb1_v = best_v;
      float fb1_w = best_w * 0.5f;
      if (!isCollisionImminent(pose, fb1_v, fb1_w)) {
        RCLCPP_INFO_THROTTLE(logger_, *clock_, 2000,
          "FILTER: Collision → reduced turn w=%.3f→%.3f", best_w, fb1_w);
        best_w = fb1_w;
      }
      // Fallback 2: go straight at crawl speed (ignore model's turn)
      else if (!isCollisionImminent(pose, CRAWL_V, 0.0f)) {
        RCLCPP_INFO_THROTTLE(logger_, *clock_, 2000,
          "FILTER: Collision → straight crawl v=%.3f", CRAWL_V);
        best_v = CRAWL_V;
        best_w = 0.0f;
      }
      // Fallback 3: zero (truly stuck — BT will handle recovery)
      else {
        RCLCPP_WARN_THROTTLE(logger_, *clock_, 2000,
          "FILTER: Collision in all directions → zero");
        best_v = 0.0f;
        best_w = 0.0f;
      }
    }
  }

  // --- Filter 7: Speed limit (Nav2 dynamic speed limit interface) ---
  float effective_v_max = v_max_;
  if (speed_limit_is_pct_) {
    effective_v_max = v_max_ * speed_limit_;
  } else {
    effective_v_max = speed_limit_;
  }
  best_v = std::clamp(best_v, 0.0f, effective_v_max);
  best_w = std::clamp(best_w, -w_max_, w_max_);

  // --- Output directly to /cmd_vel_nav ---
  // velocity_smoother handles: acceleration limiting, deadband, interpolation.
  // DiffDriveController handles: closed-loop wheel control.
  // WheelchairInterface handles: final hardware safety check.
  // We trust the pipeline — just like RPP does.

  cmd.twist.linear.x = static_cast<double>(best_v);
  cmd.twist.angular.z = static_cast<double>(best_w);

  return cmd;
}

// ==========================================================================
// Helper methods
// ==========================================================================

std::array<float, 4> KinoFlowController::computeGoalFeatures(
  const geometry_msgs::msg::PoseStamped & pose) const
{
  if (current_path_.poses.empty()) {
    return {0.0f, 0.0f, 1.0f, 0.0f};
  }

  auto & goal = current_path_.poses.back().pose;
  double dx = goal.position.x - pose.pose.position.x;
  double dy = goal.position.y - pose.pose.position.y;
  double dist = std::hypot(dx, dy);
  double robot_yaw = tf2::getYaw(pose.pose.orientation);
  double bearing = std::atan2(dy, dx) - robot_yaw;

  // Normalize bearing to [-pi, pi]
  while (bearing > M_PI) {bearing -= 2 * M_PI;}
  while (bearing < -M_PI) {bearing += 2 * M_PI;}

  float norm_dist = static_cast<float>(std::min(dist / 5.0, 1.0));
  float norm_bearing = static_cast<float>(bearing / M_PI);

  return {
    norm_dist,
    norm_bearing,
    static_cast<float>(std::cos(bearing)),
    static_cast<float>(std::sin(bearing))
  };
}

std::vector<float> KinoFlowController::computeScanResiduals() const
{
  int n_residual = temporal_frames_ - 1;
  std::vector<float> residuals(n_residual * scan_points_, 0.0f);

  std::lock_guard<std::mutex> lock(scan_mutex_);
  if (static_cast<int>(scan_buffer_.size()) < temporal_frames_) {
    return residuals;
  }

  // residuals[i] = scan_buffer_[i+1] - scan_buffer_[0]  (temporal diff)
  const auto & base = scan_buffer_[scan_buffer_.size() - temporal_frames_];
  for (int i = 0; i < n_residual; ++i) {
    const auto & scan_i = scan_buffer_[scan_buffer_.size() - temporal_frames_ + 1 + i];
    for (int j = 0; j < scan_points_; ++j) {
      residuals[i * scan_points_ + j] = scan_i[j] - base[j];
    }
  }

  return residuals;
}

std::vector<float> KinoFlowController::computeOdomHistory() const
{
  std::vector<float> history(30, 0.0f);

  std::lock_guard<std::mutex> lock(odom_mutex_);
  int n = std::min(static_cast<int>(odom_buffer_.size()), 10);
  int offset = static_cast<int>(odom_buffer_.size()) - n;
  for (int i = 0; i < n; ++i) {
    history[i * 3 + 0] = odom_buffer_[offset + i][0];  // v
    history[i * 3 + 1] = odom_buffer_[offset + i][1];  // ω
    history[i * 3 + 2] = odom_buffer_[offset + i][2];  // θ
  }

  return history;
}

float KinoFlowController::scoreTrajectory(
  const std::vector<float> & poses, int horizon,
  nav2_costmap_2d::Costmap2D * costmap,
  float goal_dx, float goal_dy) const
{
  float score = 0.0f;
  float goal_dist_init = std::hypot(goal_dx, goal_dy);
  float collision_penalty = 0.0f;
  unsigned int mx, my;

  for (int h = 0; h < horizon; ++h) {
    float x = poses[h * 3 + 0];
    float y = poses[h * 3 + 1];

    // Convert base_link-relative pose to world costmap cell
    // The costmap is in the odom frame, poses are in base_link
    // For local costmap (rolling window centered on robot), robot is at origin
    double wx = costmap->getOriginX() + costmap->getSizeInMetersX() / 2.0 + x;
    double wy = costmap->getOriginY() + costmap->getSizeInMetersY() / 2.0 + y;

    if (costmap->worldToMap(wx, wy, mx, my)) {
      unsigned char cost = costmap->getCost(mx, my);
      if (cost >= nav2_costmap_2d::INSCRIBED_INFLATED_OBSTACLE) {
        collision_penalty += 1.0f;
      } else if (cost > nav2_costmap_2d::FREE_SPACE) {
        collision_penalty += static_cast<float>(cost) / 252.0f;
      }
    } else {
      // Out of costmap bounds
      collision_penalty += 0.5f;
    }
  }
  score -= 20.0f * collision_penalty / static_cast<float>(horizon);

  // Goal progress
  float end_x = poses[(horizon - 1) * 3 + 0];
  float end_y = poses[(horizon - 1) * 3 + 1];
  float endpoint_dist = std::hypot(end_x - goal_dx, end_y - goal_dy);
  float progress = (goal_dist_init - endpoint_dist) /
                   std::max(goal_dist_init, 0.1f);
  score += 5.0f * progress;

  // Forward progress (mean v)
  // We don't have vel_trajs here, approximate from pose differences
  float total_dist = std::hypot(end_x, end_y);
  float avg_speed = total_dist / (horizon * dt_);
  score += 1.0f * avg_speed / v_max_;

  // Smoothness (approximate from pose changes)
  if (horizon >= 3) {
    float jerk = 0.0f;
    for (int h = 2; h < horizon; ++h) {
      float dx1 = poses[h * 3] - poses[(h - 1) * 3];
      float dy1 = poses[h * 3 + 1] - poses[(h - 1) * 3 + 1];
      float dx0 = poses[(h - 1) * 3] - poses[(h - 2) * 3];
      float dy0 = poses[(h - 1) * 3 + 1] - poses[(h - 2) * 3 + 1];
      float ddx = dx1 - dx0;
      float ddy = dy1 - dy0;
      jerk += ddx * ddx + ddy * ddy;
    }
    float max_jerk = jerk + 1e-6f;  // Normalize per-trajectory
    score += 2.0f * (1.0f - jerk / max_jerk);
  }

  return score;
}

// ==========================================================================
// Safety helpers (RPP/MPPI patterns)
// ==========================================================================

bool KinoFlowController::hasNanInf(const float * data, int size) const
{
  for (int i = 0; i < size; ++i) {
    if (!std::isfinite(data[i])) {
      return true;
    }
  }
  return false;
}

float KinoFlowController::getGoalDistance(
  const geometry_msgs::msg::PoseStamped & pose) const
{
  if (current_path_.poses.empty()) {
    return std::numeric_limits<float>::max();
  }
  auto & goal = current_path_.poses.back().pose;
  double dx = goal.position.x - pose.pose.position.x;
  double dy = goal.position.y - pose.pose.position.y;
  return static_cast<float>(std::hypot(dx, dy));
}

bool KinoFlowController::isCollisionImminent(
  const geometry_msgs::msg::PoseStamped & robot_pose,
  float linear_vel, float angular_vel)
{
  // RPP pattern: project trajectory forward up to max_time_to_collision_,
  // check the full robot footprint at each step using Nav2's collision checker.
  auto * costmap = costmap_ros_->getCostmap();
  footprint_collision_checker_->setCostmap(costmap);
  auto footprint = costmap_ros_->getRobotFootprint();

  double robot_yaw = tf2::getYaw(robot_pose.pose.orientation);
  double x = robot_pose.pose.position.x;
  double y = robot_pose.pose.position.y;
  double theta = robot_yaw;

  float speed = std::hypot(linear_vel, angular_vel * WHEEL_HALF_SEP);
  if (speed < 0.01f) {
    return false;  // Not moving — no collision risk
  }

  // Step along trajectory at collision_check_step_ intervals
  float total_dist = 0.0f;
  float max_dist = speed * max_time_to_collision_;
  float step = collision_check_step_;

  while (total_dist < max_dist) {
    // Advance pose using unicycle kinematics
    theta += angular_vel * (step / speed);
    x += linear_vel * std::cos(theta) * (step / speed);
    y += linear_vel * std::sin(theta) * (step / speed);
    total_dist += step;

    // Check footprint at this projected pose
    double cost = footprint_collision_checker_->footprintCostAtPose(
      x, y, theta, footprint);

    if (cost >= static_cast<double>(nav2_costmap_2d::LETHAL_OBSTACLE)) {
      return true;  // Collision detected
    }
    if (cost < 0.0) {
      return true;  // Off-map — treat as collision
    }
  }

  return false;
}

}  // namespace kinoflow_controller

PLUGINLIB_EXPORT_CLASS(
  kinoflow_controller::KinoFlowController,
  nav2_core::Controller)
