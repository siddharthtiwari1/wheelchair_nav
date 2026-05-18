cmake_minimum_required(VERSION 3.8)
project(kinoflow_controller)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 17)

# ROS2 dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(nav2_core REQUIRED)
find_package(nav2_costmap_2d REQUIRED)
find_package(nav2_util REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(pluginlib REQUIRED)

# ONNX Runtime (custom find module)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
find_package(OnnxRuntime REQUIRED)

set(dependencies
  rclcpp
  rclcpp_lifecycle
  nav2_core
  nav2_costmap_2d
  nav2_util
  nav_msgs
  geometry_msgs
  sensor_msgs
  tf2_ros
  tf2_geometry_msgs
  pluginlib
)

# Shared library
add_library(${PROJECT_NAME} SHARED
  src/kinoflow_controller.cpp
)

target_include_directories(${PROJECT_NAME} PRIVATE
  include
  ${OnnxRuntime_INCLUDE_DIRS}
)

target_link_libraries(${PROJECT_NAME}
  OnnxRuntime::OnnxRuntime
)

ament_target_dependencies(${PROJECT_NAME} ${dependencies})

# Register plugin with nav2_core
pluginlib_export_plugin_description_file(nav2_core kinoflow_controller_plugin.xml)

# Install
install(TARGETS ${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include
)

# Export
ament_export_include_directories(include)
ament_export_libraries(${PROJECT_NAME})
ament_export_dependencies(${dependencies})

ament_package()
