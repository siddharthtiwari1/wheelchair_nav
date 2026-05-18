# FindOnnxRuntime.cmake
# Finds ONNX Runtime C++ SDK
#
# Sets:
#   OnnxRuntime_FOUND
#   OnnxRuntime_INCLUDE_DIRS
#   OnnxRuntime_LIBRARIES
#   OnnxRuntime::OnnxRuntime (imported target)

# Search paths: env var, workspace third_party, /opt
set(_ORT_SEARCH_PATHS
  $ENV{ONNXRUNTIME_ROOT}
  ${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/onnxruntime
  /opt/onnxruntime
  /usr/local
  /usr
)

find_path(OnnxRuntime_INCLUDE_DIR
  NAMES onnxruntime_cxx_api.h
  PATHS ${_ORT_SEARCH_PATHS}
  PATH_SUFFIXES include
)

find_library(OnnxRuntime_LIBRARY
  NAMES onnxruntime
  PATHS ${_ORT_SEARCH_PATHS}
  PATH_SUFFIXES lib lib64
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(OnnxRuntime
  REQUIRED_VARS OnnxRuntime_LIBRARY OnnxRuntime_INCLUDE_DIR
)

if(OnnxRuntime_FOUND)
  set(OnnxRuntime_INCLUDE_DIRS ${OnnxRuntime_INCLUDE_DIR})
  set(OnnxRuntime_LIBRARIES ${OnnxRuntime_LIBRARY})

  if(NOT TARGET OnnxRuntime::OnnxRuntime)
    add_library(OnnxRuntime::OnnxRuntime SHARED IMPORTED)
    set_target_properties(OnnxRuntime::OnnxRuntime PROPERTIES
      IMPORTED_LOCATION ${OnnxRuntime_LIBRARY}
      INTERFACE_INCLUDE_DIRECTORIES ${OnnxRuntime_INCLUDE_DIR}
    )
  endif()
endif()
