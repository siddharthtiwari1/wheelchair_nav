cmake_minimum_required(VERSION 3.8)
project(wc_control)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED) 

install(
  DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)

install(
  FILES
    # Only keep the actively used sensor launch file
    ${PROJECT_NAME}/wheelchair_sensors.launch.py
  DESTINATION share/${PROJECT_NAME}/launch
)
# Removed redundant launch files: realsense_d435i_color, imu, rtabmap, realsense_rtabmap_full,
# wheelchair_vio*, all_sensors, wheelchair_complete_system

ament_python_install_package(${PROJECT_NAME})

install(PROGRAMS
  # Only keep the actively used IMU republisher
  ${PROJECT_NAME}/imu_wheelchair_republisher.py
  # IMU bias corrector (applies bias correction before Madgwick filter)
  ${PROJECT_NAME}/imu_bias_corrector.py
  # IMU startup calibrator (measures gyro bias at startup)
  ${PROJECT_NAME}/imu_startup_calibrator.py
  DESTINATION lib/${PROJECT_NAME}
)
# Removed redundant scripts: simple_serial_transmitter, simple_serial_receiver,
# imu_base_link_republisher, odom_camera_republisher

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)

  set(ament_cmake_copyright_FOUND TRUE)

  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
