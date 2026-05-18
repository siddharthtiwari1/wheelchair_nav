cmake_minimum_required(VERSION 3.8)
project(wheelchair_e2e)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)

# Install Python modules
ament_python_install_package(${PROJECT_NAME})

# Install launch files
install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

# Install config files
install(DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config
)

# Install Python executables (ROS2 nodes)
install(PROGRAMS
  wheelchair_e2e/inference/kinoflow_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME kinoflow_node
)
install(PROGRAMS
  wheelchair_e2e/inference/e2e_velocity_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME e2e_velocity_node
)

# Install scripts
install(DIRECTORY scripts/
  DESTINATION share/${PROJECT_NAME}/scripts
  USE_SOURCE_PERMISSIONS
)

# Install resource directory
install(DIRECTORY resource/
  DESTINATION share/${PROJECT_NAME}/resource
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
