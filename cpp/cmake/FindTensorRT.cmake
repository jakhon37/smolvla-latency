# Finds the TensorRT runtime library and headers (NvInferRuntime.h, libnvinfer).
#
#   TENSORRT_ROOT  an extracted TensorRT tarball or install prefix; also read from the environment. The Debian
#                  packages and JetPack install into the system paths, which are searched after it.
#
# Defines TensorRT_FOUND, TensorRT_INCLUDE_DIR, TensorRT_LIBRARY, TensorRT_VERSION and the imported target
# TensorRT::nvinfer. TensorRT_INCLUDE_DIR is set whenever the headers are found, even without the library, so
# the backend can be compile-checked on a machine without TensorRT.

set(_trt_hints ${TENSORRT_ROOT} $ENV{TENSORRT_ROOT})
# find_path and find_library cache their results; drop them when the root changes so a reconfigure takes it.
if(NOT "${TENSORRT_ROOT}|$ENV{TENSORRT_ROOT}" STREQUAL "${_TensorRT_SEARCHED_ROOT}")
  unset(TensorRT_INCLUDE_DIR CACHE)
  unset(TensorRT_LIBRARY CACHE)
  set(_TensorRT_SEARCHED_ROOT "${TENSORRT_ROOT}|$ENV{TENSORRT_ROOT}" CACHE INTERNAL "")
endif()
find_path(TensorRT_INCLUDE_DIR NvInferRuntime.h
  HINTS ${_trt_hints}
  PATH_SUFFIXES include include/x86_64-linux-gnu include/aarch64-linux-gnu)
find_library(TensorRT_LIBRARY nvinfer
  HINTS ${_trt_hints}
  PATH_SUFFIXES lib lib64 lib/x86_64-linux-gnu lib/aarch64-linux-gnu)

if(TensorRT_INCLUDE_DIR AND EXISTS "${TensorRT_INCLUDE_DIR}/NvInferVersion.h")
  # TensorRT 10.x defines either NV_TENSORRT_MAJOR 10 or TRT_MAJOR_ENTERPRISE 10 with NV_TENSORRT_MAJOR aliasing it.
  file(READ "${TensorRT_INCLUDE_DIR}/NvInferVersion.h" _trt_version_h)
  foreach(_part MAJOR MINOR PATCH)
    string(REGEX MATCH "#define (NV_TENSORRT|TRT)_${_part}(_ENTERPRISE)? +([0-9]+)" _match "${_trt_version_h}")
    set(_trt_${_part} "${CMAKE_MATCH_3}")
  endforeach()
  if(_trt_MAJOR)
    set(TensorRT_VERSION "${_trt_MAJOR}.${_trt_MINOR}.${_trt_PATCH}")
  endif()
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(TensorRT
  REQUIRED_VARS TensorRT_LIBRARY TensorRT_INCLUDE_DIR
  VERSION_VAR TensorRT_VERSION)

if(TensorRT_FOUND AND NOT TARGET TensorRT::nvinfer)
  add_library(TensorRT::nvinfer UNKNOWN IMPORTED)
  set_target_properties(TensorRT::nvinfer PROPERTIES
    IMPORTED_LOCATION "${TensorRT_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${TensorRT_INCLUDE_DIR}")
endif()
mark_as_advanced(TensorRT_INCLUDE_DIR TensorRT_LIBRARY)
