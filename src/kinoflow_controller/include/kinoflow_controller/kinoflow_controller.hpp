#ifndef KINOFLOW_CONTROLLER__KINOFLOW_CONTROLLER_HPP_
#define KINOFLOW_CONTROLLER__KINOFLOW_CONTROLLER_HPP_

#include <deque>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <vector>

#include "nav2_core/controller.hpp"
#include "nav2_costmap_2d/costmap_2d_ros.hpp"
#include "nav2_costmap_2d/footprint_collision_checker.hpp"
#include "nav2_util/lifecycle_node.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "nav_msgs/msg/path.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/twist_stamped.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"
#include "tf2_ros/buffer.h"
#include "rclcpp/rclcpp.hpp"

#include <onnxruntime_cxx_api.h>

namespace kinoflow_controller
{

class KinoFlowController : public nav2_core::Controller
{
public:
  KinoFlowController() = default;
  ~KinoFlowController() override = default;

  void configure(
    const rclcpp_lifecycle::LifecycleNode::WeakPtr & parent,
    std::string name,
    std::shared_ptr<tf2_ros::Buffer> tf,
    std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros) override;

  void cleanup() override;
  void activate() override;
  void deactivate() override;
  void setPlan(const nav_msgs::msg::Path & path) override;

  geometry_msgs::msg::TwistStamped computeVelocityCommands(
    const geometry_msgs::msg::PoseStamped & pose,
    const geometry_msgs::msg::Twist & velocity,
    nav2_core::GoalChecker * goal_checker) override;

  void setSpeedLimit(const double & speed_limit, const bool & percentage) override;

private:
  // Sensor callbacks
  void scanCallback(const sensor_msgs::msg::LaserScan::SharedPtr msg);
  void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg);

  // Inference helpers
  std::array<float, 4> computeGoalFeatures(
    const geometry_msgs::msg::PoseStamped & pose) const;
  std::vector<float> computeScanResiduals() const;
  std::vector<float> computeOdomHistory() const;
  float scoreTrajectory(
    const std::vector<float> & poses, int horizon,
    nav2_costmap_2d::Costmap2D * costmap,
    float goal_dx, float goal_dy) const;

  // Safety helpers (RPP/MPPI patterns)
  bool isCollisionImminent(
    const geometry_msgs::msg::PoseStamped & robot_pose,
    float linear_vel, float angular_vel);
  bool hasNanInf(const float * data, int size) const;
  float getGoalDistance(const geometry_msgs::msg::PoseStamped & pose) const;

  // Node and TF
  rclcpp_lifecycle::LifecycleNode::WeakPtr node_;
  std::string plugin_name_;
  std::shared_ptr<tf2_ros::Buffer> tf_;
  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros_;
  rclcpp::Logger logger_{rclcpp::get_logger("KinoFlowController")};
  rclcpp::Clock::SharedPtr clock_;

  // ONNX Runtime
  std::unique_ptr<Ort::Env> ort_env_;
  std::unique_ptr<Ort::Session> encoder_session_;
  std::unique_ptr<Ort::Session> vf_session_;
  Ort::MemoryInfo mem_info_{Ort::MemoryInfo::CreateCpu(
    OrtAllocatorType::OrtArenaAllocator, OrtMemTypeDefault)};

  // Footprint collision checker (RPP/MPPI pattern)
  std::unique_ptr<
    nav2_costmap_2d::FootprintCollisionChecker<nav2_costmap_2d::Costmap2D *>>
    footprint_collision_checker_;

  // Sensor subscriptions
  rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;

  // Sensor buffers (thread-safe)
  std::deque<std::vector<float>> scan_buffer_;       // last T scans (720 each)
  std::deque<std::array<float, 3>> odom_buffer_;     // last 10 [v, ω, θ]
  mutable std::mutex scan_mutex_;
  mutable std::mutex odom_mutex_;

  // Sensor staleness tracking
  rclcpp::Time last_scan_time_;
  rclcpp::Time last_odom_time_;
  static constexpr double SENSOR_STALE_TIMEOUT_S = 1.0;

  // State
  nav_msgs::msg::Path current_path_;

  // Parameters
  std::string encoder_model_path_;
  std::string vf_model_path_;
  float v_max_{0.25f};
  float w_max_{1.0f};
  float dt_{0.1f};
  int horizon_{10};
  int n_samples_{8};
  int n_euler_steps_{3};
  int scan_points_{720};
  int temporal_frames_{5};
  float safety_min_range_{0.3f};
  float safety_slow_range_{0.6f};

  // Speed limit (Nav2 dynamic speed limit interface)
  float speed_limit_{1.0f};
  bool speed_limit_is_pct_{true};

  // ================================================================
  // SAFETY FILTER STATE
  // ================================================================
  // Only what the downstream pipeline (velocity_smoother, DiffDrive,
  // WheelchairInterface) CANNOT handle. No duplication.

  // Wheelchair differential drive constants
  static constexpr float WHEEL_HALF_SEP = 0.2825f;  // 0.565m / 2
  static constexpr float MAX_WHEEL_SPEED = 0.30f;    // per-wheel limit (m/s)

  // Watchdog timeout (catches ONNX crashes, frozen threads)
  rclcpp::Time last_valid_inference_time_;
  static constexpr double WATCHDOG_TIMEOUT_S = 0.5;

  // Graceful degradation (sustained incoherent inference)
  int consecutive_sanity_fails_{0};
  static constexpr int CRAWL_MODE_THRESHOLD = 5;
  static constexpr float CRAWL_V = 0.05f;
  static constexpr float CRAWL_W_MAX = 0.15f;
  bool crawl_mode_{false};

  // Collision projection (footprint-based, costmap-aware)
  float max_time_to_collision_{1.0f};
  float collision_check_step_{0.05f};

  // RNG for noise generation
  std::mt19937 rng_{std::random_device{}()};
  std::normal_distribution<float> normal_dist_{0.0f, 1.0f};
};

}  // namespace kinoflow_controller

#endif  // KINOFLOW_CONTROLLER__KINOFLOW_CONTROLLER_HPP_
