cmake_minimum_required(VERSION 3.8)
project(wheelchair_mono_depth)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)

# Install Python modules
ament_python_install_package(${PROJECT_NAME})

# Install Python node entry points as executables
# (ament_cmake_python does not process setup.py entry_points)
install(PROGRAMS
  wheelchair_mono_depth/data_collection_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME data_collection_node
)
install(PROGRAMS
  wheelchair_mono_depth/inference/mono_depth_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME mono_depth_node
)
install(PROGRAMS
  wheelchair_mono_depth/inference/mono_depth_multi_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME mono_depth_multi_node
)
install(PROGRAMS
  wheelchair_mono_depth/inference/depth_eval_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME depth_eval_node
)
install(PROGRAMS
  wheelchair_mono_depth/inference/da3_depth_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME da3_depth_node
)
install(PROGRAMS
  wheelchair_mono_depth/inference/depth_scan_benchmark_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME depth_scan_benchmark_node
)
install(PROGRAMS
  wheelchair_mono_depth/inference/da3_lidar_anchor_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME da3_lidar_anchor_node
)
install(PROGRAMS
  wheelchair_mono_depth/inference/da3_multi_depth_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME da3_multi_depth_node
)

# Install launch files
install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

# Install config files
install(DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config
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
