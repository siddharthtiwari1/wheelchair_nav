#include "wheelchair_firmware/wheelchair_interface.hpp"
#include <hardware_interface/types/hardware_interface_type_values.hpp>
#include <pluginlib/class_list_macros.hpp>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <regex>
#include <thread>
#include <chrono>

namespace wheelchair_firmware
{

WheelchairInterface::WheelchairInterface()
  : command_timeout_(0, 0),
    clock_(RCL_SYSTEM_TIME)
{
}

WheelchairInterface::~WheelchairInterface()
{
  if (motor_controller_ && motor_controller_->IsOpen())
  {
    try
    {
      motor_controller_->Close();
    }
    catch (...)
    {
      RCLCPP_FATAL_STREAM(rclcpp::get_logger("WheelchairInterface"),
                          "Something went wrong while closing connection with port " << port_);
    }
  }
}

CallbackReturn WheelchairInterface::on_init(const hardware_interface::HardwareInfo &hardware_info)
{
  CallbackReturn result = hardware_interface::SystemInterface::on_init(hardware_info);
  if (result != CallbackReturn::SUCCESS)
  {
    return result;
  }

  // Get hardware parameters with defaults matching your wheelchair
  try
  {
    port_ = info_.hardware_parameters.count("port") ? 
            info_.hardware_parameters.at("port") : "/dev/ttyACM0";
    
    baud_rate_ = info_.hardware_parameters.count("baud_rate") ?
                 std::stoi(info_.hardware_parameters.at("baud_rate")) : 115200;
    
    wheel_base_ = info_.hardware_parameters.count("wheel_base") ?
                  std::stod(info_.hardware_parameters.at("wheel_base")) : 0.565;
    
    wheel_radius_ = info_.hardware_parameters.count("wheel_radius") ?
                    std::stod(info_.hardware_parameters.at("wheel_radius")) : 0.1524;

    left_wheel_radius_correction_ = info_.hardware_parameters.count("left_wheel_radius_correction") ?
                                   std::stod(info_.hardware_parameters.at("left_wheel_radius_correction")) : 1.0;
    
    
    max_linear_velocity_ = info_.hardware_parameters.count("max_linear_velocity") ?
                           std::stod(info_.hardware_parameters.at("max_linear_velocity")) : 0.40;

    max_angular_velocity_ = info_.hardware_parameters.count("max_angular_velocity") ?
                            std::stod(info_.hardware_parameters.at("max_angular_velocity")) : 1.0;
    
    command_timeout_ = rclcpp::Duration::from_seconds(
        info_.hardware_parameters.count("command_timeout") ?
        std::stod(info_.hardware_parameters.at("command_timeout")) : 0.5);
    
    emergency_stop_deceleration_ = info_.hardware_parameters.count("emergency_stop_deceleration") ?
                                   std::stod(info_.hardware_parameters.at("emergency_stop_deceleration")) : 5.0;
  }
  catch (const std::out_of_range &e)
  {
    RCLCPP_FATAL(rclcpp::get_logger("WheelchairInterface"),
                 "Missing required parameter: %s", e.what());
    return CallbackReturn::FAILURE;
  }

  // Initialize vectors
  // FIXED: Renamed variables to match header declarations.
  wheel_velocity_commands_.resize(info_.joints.size(), 0.0);
  wheel_position_states_.resize(info_.joints.size(), 0.0);
  wheel_velocity_states_.resize(info_.joints.size(), 0.0);
  wheel_effort_states_.resize(info_.joints.size(), 0.0);

  // Initialize IMU data
  for (int i = 0; i < 4; i++) imu_orientation_[i] = (i == 3) ? 1.0 : 0.0; // Identity quaternion
  for (int i = 0; i < 3; i++) {
    imu_angular_velocity_[i] = 0.0;
    imu_linear_acceleration_[i] = 0.0;
  }

  // Initialize battery monitoring
  battery_voltage_ = 0.0;
  battery_current_ = 0.0;
  battery_percentage_ = 100.0;

  // Initialize safety features
  emergency_stop_active_ = false;
  last_command_time_ = clock_.now(); // MODIFIED: Use member clock

  motor_controller_ = std::make_unique<LibSerial::SerialPort>();

  return CallbackReturn::SUCCESS;
}

std::vector<hardware_interface::StateInterface> WheelchairInterface::export_state_interfaces()
{
  std::vector<hardware_interface::StateInterface> state_interfaces;

  // Wheel interfaces
  for (size_t i = 0; i < info_.joints.size(); i++)
  {
    state_interfaces.emplace_back(hardware_interface::StateInterface(
        info_.joints[i].name, hardware_interface::HW_IF_POSITION, &wheel_position_states_[i])); // FIXED
    state_interfaces.emplace_back(hardware_interface::StateInterface(
        info_.joints[i].name, hardware_interface::HW_IF_VELOCITY, &wheel_velocity_states_[i])); // FIXED
    state_interfaces.emplace_back(hardware_interface::StateInterface(
        info_.joints[i].name, hardware_interface::HW_IF_EFFORT, &wheel_effort_states_[i]));
  }

  // IMU interfaces
  state_interfaces.emplace_back(hardware_interface::StateInterface(
      "camera_imu", "orientation.x", &imu_orientation_[0]));
  state_interfaces.emplace_back(hardware_interface::StateInterface(
      "camera_imu", "orientation.y", &imu_orientation_[1]));
  state_interfaces.emplace_back(hardware_interface::StateInterface(
      "camera_imu", "orientation.z", &imu_orientation_[2]));
  state_interfaces.emplace_back(hardware_interface::StateInterface(
      "camera_imu", "orientation.w", &imu_orientation_[3]));

  state_interfaces.emplace_back(hardware_interface::StateInterface(
      "camera_imu", "angular_velocity.x", &imu_angular_velocity_[0]));
  state_interfaces.emplace_back(hardware_interface::StateInterface(
      "camera_imu", "angular_velocity.y", &imu_angular_velocity_[1]));
  state_interfaces.emplace_back(hardware_interface::StateInterface(
      "camera_imu", "angular_velocity.z", &imu_angular_velocity_[2]));

  state_interfaces.emplace_back(hardware_interface::StateInterface(
      "camera_imu", "linear_acceleration.x", &imu_linear_acceleration_[0]));
  state_interfaces.emplace_back(hardware_interface::StateInterface(
      "camera_imu", "linear_acceleration.y", &imu_linear_acceleration_[1]));
  state_interfaces.emplace_back(hardware_interface::StateInterface(
      "camera_imu", "linear_acceleration.z", &imu_linear_acceleration_[2]));

  // Battery interfaces
  state_interfaces.emplace_back(hardware_interface::StateInterface(
      "battery", "voltage", &battery_voltage_));
  state_interfaces.emplace_back(hardware_interface::StateInterface(
      "battery", "current", &battery_current_));
  state_interfaces.emplace_back(hardware_interface::StateInterface(
      "battery", "percentage", &battery_percentage_));

  return state_interfaces;
}

std::vector<hardware_interface::CommandInterface> WheelchairInterface::export_command_interfaces()
{
  std::vector<hardware_interface::CommandInterface> command_interfaces;

  // Velocity command interfaces for wheels
  for (size_t i = 0; i < info_.joints.size(); i++)
  {
    command_interfaces.emplace_back(hardware_interface::CommandInterface(
        info_.joints[i].name, hardware_interface::HW_IF_VELOCITY, &wheel_velocity_commands_[i])); // FIXED
  }

  return command_interfaces;
}

CallbackReturn WheelchairInterface::on_configure(const rclcpp_lifecycle::State &)
{
  RCLCPP_INFO(rclcpp::get_logger("WheelchairInterface"), "Configuring wheelchair hardware...");
  return CallbackReturn::SUCCESS;
}

CallbackReturn WheelchairInterface::on_activate(const rclcpp_lifecycle::State &)
{
  RCLCPP_INFO(rclcpp::get_logger("WheelchairInterface"), "Starting wheelchair hardware...");

  // Reset all states and commands
  std::fill(wheel_velocity_commands_.begin(), wheel_velocity_commands_.end(), 0.0); // FIXED
  std::fill(wheel_position_states_.begin(), wheel_position_states_.end(), 0.0);   // FIXED
  std::fill(wheel_velocity_states_.begin(), wheel_velocity_states_.end(), 0.0);   // FIXED
  std::fill(wheel_effort_states_.begin(), wheel_effort_states_.end(), 0.0);

  emergency_stop_active_ = false;
  last_command_time_ = clock_.now(); // MODIFIED: Use member clock

  try
  {
    RCLCPP_INFO(rclcpp::get_logger("WheelchairInterface"),
                "Attempting to open serial port: %s", port_.c_str());
    
    motor_controller_->Open(port_);
    
    RCLCPP_INFO(rclcpp::get_logger("WheelchairInterface"),
                "Successfully opened port %s, configuring...", port_.c_str());
    
    motor_controller_->SetBaudRate(LibSerial::BaudRate::BAUD_115200);
    motor_controller_->SetFlowControl(LibSerial::FlowControl::FLOW_CONTROL_NONE);
    motor_controller_->SetParity(LibSerial::Parity::PARITY_NONE);
    motor_controller_->SetStopBits(LibSerial::StopBits::STOP_BITS_1);
    motor_controller_->SetCharacterSize(LibSerial::CharacterSize::CHAR_SIZE_8);

    // Wait for Arduino to initialize (matches your Arduino 5 second relay delay)
    RCLCPP_INFO(rclcpp::get_logger("WheelchairInterface"),
                "Port configured successfully. Waiting for Arduino initialization...");
    std::this_thread::sleep_for(std::chrono::milliseconds(2000));
    
    // Send initial stop command to ensure wheelchair is stopped
    sendCommand("rp0.0,lp0.0,");
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  catch (const std::exception& e)
  {
    RCLCPP_FATAL(rclcpp::get_logger("WheelchairInterface"),
                 "Failed to open serial port %s: %s", port_.c_str(), e.what());
    return CallbackReturn::FAILURE;
  }
  catch (...)
  {
    RCLCPP_FATAL(rclcpp::get_logger("WheelchairInterface"),
                 "Unknown error opening serial port %s", port_.c_str());
    return CallbackReturn::FAILURE;
  }

  RCLCPP_INFO(rclcpp::get_logger("WheelchairInterface"),
              "Hardware started, ready to take commands");
  return CallbackReturn::SUCCESS;
}

CallbackReturn WheelchairInterface::on_deactivate(const rclcpp_lifecycle::State &)
{
  RCLCPP_INFO(rclcpp::get_logger("WheelchairInterface"), "Stopping wheelchair hardware...");

  // Send stop command before closing (Arduino format)
  if (motor_controller_ && motor_controller_->IsOpen())
  {
    try
    {
      // Send multiple stop commands to ensure Arduino receives them
      for (int i = 0; i < 3; i++)
      {
        sendCommand("rp0.0,lp0.0,");
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
      }
      
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
      motor_controller_->Close();
    }
    catch (...)
    {
      RCLCPP_ERROR(rclcpp::get_logger("WheelchairInterface"),
                   "Error during deactivation");
    }
  }

  RCLCPP_INFO(rclcpp::get_logger("WheelchairInterface"), "Hardware stopped");
  return CallbackReturn::SUCCESS;
}

CallbackReturn WheelchairInterface::on_cleanup(const rclcpp_lifecycle::State &)
{
  RCLCPP_INFO(rclcpp::get_logger("WheelchairInterface"), "Cleaning up wheelchair hardware...");
  return CallbackReturn::SUCCESS;
}

CallbackReturn WheelchairInterface::on_shutdown(const rclcpp_lifecycle::State &)
{
  RCLCPP_INFO(rclcpp::get_logger("WheelchairInterface"), "Shutting down wheelchair hardware...");
  return on_deactivate(rclcpp_lifecycle::State());
}

CallbackReturn WheelchairInterface::on_error(const rclcpp_lifecycle::State &)
{
  RCLCPP_ERROR(rclcpp::get_logger("WheelchairInterface"), "Hardware error occurred");
  applyEmergencyStop();
  return CallbackReturn::SUCCESS;
}

hardware_interface::return_type WheelchairInterface::read(const rclcpp::Time &time,
                                                         const rclcpp::Duration &)
{
  if (!motor_controller_->IsOpen())
  {
    return hardware_interface::return_type::ERROR;
  }

  // NOTE: Command timeout check removed - bumperbot doesn't have this and it causes
  // false emergency stops. The diff_drive_controller handles its own cmd_vel_timeout.
  // If needed, the timeout should be checked in write() AFTER commands are processed,
  // not in read() before write() has a chance to update last_command_time_.

  // Read data from Arduino using final_control protocol
  if (motor_controller_->IsDataAvailable())
  {
    std::string response;
    if (readResponse(response))
    {
      parseArduinoData(response);
    }
  }

  return hardware_interface::return_type::OK;
}

void WheelchairInterface::parseArduinoData(const std::string &data)
{
  try
  {
    RCLCPP_DEBUG(rclcpp::get_logger("WheelchairInterface"), 
                "Received: %s", data.c_str());
    
    // Handle Arduino status/initialization messages (same as python script)
    if (data.find("PPM signal lost") != std::string::npos ||
        data.find("stopping") != std::string::npos)
    {
      RCLCPP_WARN(rclcpp::get_logger("WheelchairInterface"), 
                  "Arduino status: %s", data.c_str());
      return;
    }
    
    if (data.find("Wheelchair Controller") != std::string::npos ||
        data.find("Ready") != std::string::npos ||
        data.find("Commands:") != std::string::npos ||
        data.find("PPM Support") != std::string::npos)
    {
      RCLCPP_INFO(rclcpp::get_logger("WheelchairInterface"), 
                  "Arduino initialization: %s", data.c_str());
      return;
    }

    // Use the same robust parsing as arduino_receiver.py parse_legacy_format()
    parseLegacyFormat(data);
  }
  catch (const std::exception &e)
  {
    RCLCPP_ERROR(rclcpp::get_logger("WheelchairInterface"),
                 "Error parsing Arduino data \"%s\": %s", data.c_str(), e.what());
  }
}

void WheelchairInterface::parseLegacyFormat(const std::string &data)
{
  try
  {
    RCLCPP_INFO(rclcpp::get_logger("WheelchairInterface"),
                "Processing Arduino data: \"%s\"", data.c_str());
    
    // Extract mode information for debugging (like Python script)
    std::vector<std::string> mode_info;
    if (data.find("[PPM]") != std::string::npos) {
      mode_info.push_back("PPM_ACTIVE");
    }
    if (data.find("[WHEEL]") != std::string::npos) {
      mode_info.push_back("WHEEL_MODE");
    }
    if (data.find("[CMD_VEL]") != std::string::npos) {
      mode_info.push_back("CMDVEL_MODE");
    }
    if (data.find("[PIVOT]") != std::string::npos) {
      mode_info.push_back("PIVOT_MODE");
    }
    
    if (!mode_info.empty()) {
      std::string modes = "";
      for (const auto& mode : mode_info) {
        modes += mode + " ";
      }
      RCLCPP_INFO(rclcpp::get_logger("WheelchairInterface"),
                  "Arduino modes detected: %s", modes.c_str());
    }

    // Remove all square bracket indicators and extra spaces (like Python script)
    std::string cleaned_data = data;
    std::regex bracket_cleaner(R"(\[.*?\])");
    cleaned_data = std::regex_replace(cleaned_data, bracket_cleaner, "");
    
    // Remove leading/trailing whitespace
    cleaned_data.erase(0, cleaned_data.find_first_not_of(" \t\r\n"));
    cleaned_data.erase(cleaned_data.find_last_not_of(" \t\r\n") + 1);

    // Look for pattern starting with 'r' or 'l' followed by 'p' or 'n' and numbers
    std::regex wheel_match(R"(([rl][pn]\d+\.?\d*.*))");
    std::smatch match;
    if (std::regex_search(cleaned_data, match, wheel_match)) {
      cleaned_data = match[1].str();
    }

    RCLCPP_INFO(rclcpp::get_logger("WheelchairInterface"),
                "Cleaned for parsing: \"%s\"", cleaned_data.c_str());

    // Parse if we have wheel data (like Python script check)
    if (cleaned_data.find('r') != std::string::npos && cleaned_data.find('l') != std::string::npos) {
      parseWheelVelocitiesSimple(cleaned_data);
    } else {
      RCLCPP_WARN(rclcpp::get_logger("WheelchairInterface"),
                  "No wheel velocity data found in: \"%s\"", data.c_str());
    }
  }
  catch (const std::exception &e)
  {
    RCLCPP_ERROR(rclcpp::get_logger("WheelchairInterface"),
                 "Error in legacy parsing for \"%s\": %s", data.c_str(), e.what());
  }
}

void WheelchairInterface::parseWheelVelocitiesSimple(const std::string &data)
{
  try
  {
    double right_vel = 0.0;
    double left_vel = 0.0;
    bool right_found = false;
    bool left_found = false;

    RCLCPP_INFO(rclcpp::get_logger("WheelchairInterface"),
                "Parsing wheel velocities from: \"%s\"", data.c_str());

    // Use the same robust regex as arduino_receiver.py (lines 184-185)
    std::regex right_pattern(R"(r([pn])-?(\d*\.?\d+))");
    std::regex left_pattern(R"(l([pn])-?(\d*\.?\d+))");

    std::smatch match;

    // Find right wheel velocity (same logic as Python script)
    if (std::regex_search(data, match, right_pattern)) {
      try {
        std::string sign_char = match[1].str();
        std::string value_str = match[2].str();
        double value = std::stod(value_str);
        right_vel = (sign_char == "p") ? value : -value;
        right_found = true;
        RCLCPP_INFO(rclcpp::get_logger("WheelchairInterface"),
                    "Right wheel parsed: r%s%s -> %.3f rad/s",
                    sign_char.c_str(), value_str.c_str(), right_vel);
      } catch (const std::exception &e) {
        RCLCPP_WARN(rclcpp::get_logger("WheelchairInterface"),
                   "Failed to parse right wheel value: %s", e.what());
      }
    }

    // Find left wheel velocity (same logic as Python script)
    if (std::regex_search(data, match, left_pattern)) {
      try {
        std::string sign_char = match[1].str();
        std::string value_str = match[2].str();
        double value = std::stod(value_str);
        // FIXED: Parse left wheel same as right wheel - Arduino sends correct signs
        left_vel = (sign_char == "p") ? value : -value;
        left_found = true;
        RCLCPP_INFO(rclcpp::get_logger("WheelchairInterface"),
                    "Left wheel parsed: l%s%s -> %.3f rad/s",
                    sign_char.c_str(), value_str.c_str(), left_vel);
      } catch (const std::exception &e) {
        RCLCPP_WARN(rclcpp::get_logger("WheelchairInterface"),
                   "Failed to parse left wheel value: %s", e.what());
      }
    }

    // Update wheel states if we have at least one valid wheel velocity
    if (right_found || left_found) {
      if (right_found) {
        wheel_velocity_states_[1] = right_vel;  // rightwheel -> index 1
      }
      if (left_found) {
        wheel_velocity_states_[0] = left_vel;   // leftwheel -> index 0
      }

      // Integrate velocity to get position
      static auto last_time = clock_.now();
      auto current_time = clock_.now();
      double dt = (current_time - last_time).seconds();

      if (dt > 0.0 && dt < 1.0) {  // Sanity check
        if (right_found) {
          wheel_position_states_[1] += right_vel * dt;  // rightwheel
        }
        if (left_found) {
          wheel_position_states_[0] += left_vel * dt;   // leftwheel
        }
      }
      last_time = current_time;

      // Convert to linear velocities for logging (same as Python script)
      double right_linear = wheel_velocity_states_[1] * wheel_radius_;
      double left_linear = wheel_velocity_states_[0] * wheel_radius_;

      RCLCPP_INFO(rclcpp::get_logger("WheelchairInterface"),
                  "Wheel velocities - Right: %.3f rad/s (%.3f m/s), Left: %.3f rad/s (%.3f m/s)",
                  wheel_velocity_states_[1], right_linear,
                  wheel_velocity_states_[0], left_linear);
    } else {
      RCLCPP_WARN(rclcpp::get_logger("WheelchairInterface"),
                  "No valid wheel velocity patterns found in: \"%s\"", data.c_str());
    }
  }
  catch (const std::exception &e)
  {
    RCLCPP_ERROR(rclcpp::get_logger("WheelchairInterface"),
                 "Error parsing wheel velocities from \"%s\": %s", data.c_str(), e.what());
  }
}

hardware_interface::return_type WheelchairInterface::write(const rclcpp::Time &time,
                                                          const rclcpp::Duration &)
{
  if (!motor_controller_->IsOpen())
  {
    return hardware_interface::return_type::ERROR;
  }

  // FIXED: Match bumperbot's simple approach - no emergency stop logic here
  // The diff_drive_controller handles cmd_vel_timeout on its own

  // SAFETY: Enforce velocity limits before sending to Arduino.
  // This is the LAST software defense — clamps wheel commands to safe
  // linear (0.40 m/s) and angular (1.0 rad/s) robot velocities.
  checkSafetyLimits();

  // wheel_velocity_commands_[0] = leftwheel (from diff_drive_controller)
  // wheel_velocity_commands_[1] = rightwheel (from diff_drive_controller)
  double left_cmd = wheel_velocity_commands_[0];   // leftwheel
  double right_cmd = wheel_velocity_commands_[1];  // rightwheel

  // Format command using bumperbot's exact format (with zero padding for values < 10)
  // Arduino expects: rp05.00,lp03.00,
  std::stringstream cmd_stream;

  char right_wheel_sign = right_cmd >= 0 ? 'p' : 'n';
  char left_wheel_sign = left_cmd >= 0 ? 'p' : 'n';

  // Add leading zero for values < 10 (matches bumperbot)
  std::string compensate_zeros_right = std::abs(right_cmd) < 10.0 ? "0" : "";
  std::string compensate_zeros_left = std::abs(left_cmd) < 10.0 ? "0" : "";

  cmd_stream << std::fixed << std::setprecision(2)
             << "r" << right_wheel_sign << compensate_zeros_right << std::abs(right_cmd)
             << ",l" << left_wheel_sign << compensate_zeros_left << std::abs(left_cmd) << ",";

  try
  {
    motor_controller_->Write(cmd_stream.str());
    // Note: bumperbot doesn't call DrainWriteBuffer, trying without it
  }
  catch (...)
  {
    RCLCPP_ERROR_STREAM(rclcpp::get_logger("WheelchairInterface"),
                        "Something went wrong while sending the message "
                            << cmd_stream.str() << " to the port " << port_);
    return hardware_interface::return_type::ERROR;
  }

  RCLCPP_DEBUG(rclcpp::get_logger("WheelchairInterface"),
              "Sent command: %s", cmd_stream.str().c_str());

  return hardware_interface::return_type::OK;
}

bool WheelchairInterface::sendCommand(const std::string &command)
{
  try
  {
    motor_controller_->Write(command + "\n");
    motor_controller_->DrainWriteBuffer();
    return true;
  }
  catch (...)
  {
    RCLCPP_ERROR_STREAM(rclcpp::get_logger("WheelchairInterface"),
                        "Failed to send command: " << command);
    return false;
  }
}

bool WheelchairInterface::readResponse(std::string &response)
{
  try
  {
    // Use timeout for non-blocking read like your Python script
    motor_controller_->ReadLine(response, '\n', 100);  // 100ms timeout
    
    // Remove newline and carriage return characters
    response.erase(std::remove(response.begin(), response.end(), '\n'), response.end());
    response.erase(std::remove(response.begin(), response.end(), '\r'), response.end());
    
    return !response.empty();
  }
  catch (const LibSerial::ReadTimeout&)
  {
    // Timeout is normal, not an error
    return false;
  }
  catch (...)
  {
    RCLCPP_DEBUG(rclcpp::get_logger("WheelchairInterface"),
                "Serial read error occurred");
    return false;
  }
}

void WheelchairInterface::checkSafetyLimits()
{
  // FIXED: Correct differential drive kinematics with proper joint mapping
  // wheel_velocity_commands_[0] = leftwheel, wheel_velocity_commands_[1] = rightwheel
  double left_vel = wheel_velocity_commands_[0];
  double right_vel = wheel_velocity_commands_[1];

  // Calculate robot linear and angular velocities
  double linear_vel = (left_vel + right_vel) * wheel_radius_ / 2.0;
  double angular_vel = (right_vel - left_vel) * wheel_radius_ / wheel_base_;

  // Apply limits
  if (std::abs(linear_vel) > max_linear_velocity_)
  {
    double scale = max_linear_velocity_ / std::abs(linear_vel);
    wheel_velocity_commands_[0] *= scale; // leftwheel
    wheel_velocity_commands_[1] *= scale; // rightwheel
    RCLCPP_WARN(rclcpp::get_logger("WheelchairInterface"),
                "Linear velocity limited to %.2f m/s", max_linear_velocity_);
  }

  if (std::abs(angular_vel) > max_angular_velocity_)
  {
    double scale = max_angular_velocity_ / std::abs(angular_vel);
    wheel_velocity_commands_[0] *= scale; // leftwheel
    wheel_velocity_commands_[1] *= scale; // rightwheel
    RCLCPP_WARN(rclcpp::get_logger("WheelchairInterface"),
                "Angular velocity limited to %.2f rad/s", max_angular_velocity_);
  }

  // Check battery level for safety
  if (battery_percentage_ < 10.0)
  {
    // MODIFIED: Use the member variable clock_ instead of a temporary rclcpp::Clock()
    RCLCPP_WARN_THROTTLE(rclcpp::get_logger("WheelchairInterface"),
                         clock_, 5000,
                         "Low battery warning: %.1f%%", battery_percentage_);
  }
}

void WheelchairInterface::applyEmergencyStop()
{
  emergency_stop_active_ = true;
  
  // Send immediate stop command in Arduino format
  if (motor_controller_ && motor_controller_->IsOpen())
  {
    sendCommand("rp0.0,lp0.0,");
  }
  
  RCLCPP_ERROR(rclcpp::get_logger("WheelchairInterface"),
               "EMERGENCY STOP ACTIVATED");
}

bool WheelchairInterface::isCommandStale(const rclcpp::Time &current_time)
{
  // Use the same clock source for both times
  auto current_clock_time = clock_.now();
  return (current_clock_time - last_command_time_) > command_timeout_;
}

}  // namespace wheelchair_firmware

PLUGINLIB_EXPORT_CLASS(wheelchair_firmware::WheelchairInterface, hardware_interface::SystemInterface)