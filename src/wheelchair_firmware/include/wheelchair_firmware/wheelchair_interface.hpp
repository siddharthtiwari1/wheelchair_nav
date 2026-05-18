#ifndef WHEELCHAIR_INTERFACE_HPP
#define WHEELCHAIR_INTERFACE_HPP

#include <rclcpp/rclcpp.hpp>
#include <hardware_interface/system_interface.hpp>
#include <libserial/SerialPort.h>
#include <rclcpp_lifecycle/state.hpp>
#include <rclcpp_lifecycle/node_interfaces/lifecycle_node_interface.hpp>

#include <vector>
#include <string>
#include <memory>

namespace wheelchair_firmware
{

using CallbackReturn = rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;

class WheelchairInterface : public hardware_interface::SystemInterface
{
public:
  WheelchairInterface();
  virtual ~WheelchairInterface();

  // Lifecycle callbacks
  CallbackReturn on_activate(const rclcpp_lifecycle::State &previous_state) override;
  CallbackReturn on_deactivate(const rclcpp_lifecycle::State &previous_state) override;

  // Hardware interface callbacks
  CallbackReturn on_init(const hardware_interface::HardwareInfo &hardware_info) override;
  CallbackReturn on_cleanup(const rclcpp_lifecycle::State &previous_state) override;
  CallbackReturn on_shutdown(const rclcpp_lifecycle::State &previous_state) override;
  CallbackReturn on_configure(const rclcpp_lifecycle::State &previous_state) override;
  CallbackReturn on_error(const rclcpp_lifecycle::State &previous_state) override;

  // Interface exports
  std::vector<hardware_interface::StateInterface> export_state_interfaces() override;
  std::vector<hardware_interface::CommandInterface> export_command_interfaces() override;

  // Read/Write operations
  hardware_interface::return_type read(const rclcpp::Time &time, const rclcpp::Duration &period) override;
  hardware_interface::return_type write(const rclcpp::Time &time, const rclcpp::Duration &period) override;

private:
  // Serial communication
  std::unique_ptr<LibSerial::SerialPort> motor_controller_;
  std::string port_;
  int baud_rate_;

  // Wheel states and commands
  std::vector<double> wheel_velocity_commands_;
  std::vector<double> wheel_position_states_;
  std::vector<double> wheel_velocity_states_;
  std::vector<double> wheel_effort_states_;

  // Safety features
  double max_linear_velocity_;
  double max_angular_velocity_;
  double emergency_stop_deceleration_;
  bool emergency_stop_active_;

  // Battery monitoring
  double battery_voltage_;
  double battery_current_;
  double battery_percentage_;

  // IMU data
  double imu_orientation_[4];  // Quaternion [x, y, z, w]
  double imu_angular_velocity_[3];  // [x, y, z]
  double imu_linear_acceleration_[3];  // [x, y, z]

  // Timing
  rclcpp::Time last_command_time_;
  rclcpp::Duration command_timeout_;
  rclcpp::Clock clock_; // <-- THIS LINE WAS MISSING AND NEEDS TO BE ADDED

  // Wheelchair parameters
  double wheel_base_;
  double wheel_radius_;
  double left_wheel_radius_correction_;

  // Helper methods
  bool sendCommand(const std::string &command);
  bool readResponse(std::string &response);
  void checkSafetyLimits();
  void applyEmergencyStop();
  bool isCommandStale(const rclcpp::Time &current_time);
  
  // Arduino communication protocol parsers
  void parseArduinoData(const std::string &data);
  void parseLegacyFormat(const std::string &data);
  void parseWheelVelocitiesSimple(const std::string &data);
};

}  // namespace wheelchair_firmware

#endif  // WHEELCHAIR_INTERFACE_HPP