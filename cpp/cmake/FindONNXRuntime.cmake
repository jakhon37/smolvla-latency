# Finds the ONNX Runtime C/C++ SDK: onnxruntime_cxx_api.h and libonnxruntime.
#
#   ONNXRUNTIME_ROOT  an extracted release (onnxruntime-linux-x64-<version>.tgz, or what
#                     scripts/fetch_onnxruntime.sh writes) or an install prefix; also read from the environment.
#                     Searched before the system paths.
#
# Defines ONNXRuntime_FOUND, ONNXRuntime_INCLUDE_DIR, ONNXRuntime_LIBRARY, ONNXRuntime_VERSION (1.<C API version>)
# and the imported target ONNXRuntime::ONNXRuntime.

set(_ort_hints ${ONNXRUNTIME_ROOT} $ENV{ONNXRUNTIME_ROOT})
# find_path and find_library cache their results; drop them when the root changes so a reconfigure takes it.
if(NOT "${ONNXRUNTIME_ROOT}|$ENV{ONNXRUNTIME_ROOT}" STREQUAL "${_ONNXRuntime_SEARCHED_ROOT}")
  unset(ONNXRuntime_INCLUDE_DIR CACHE)
  unset(ONNXRuntime_LIBRARY CACHE)
  set(_ONNXRuntime_SEARCHED_ROOT "${ONNXRUNTIME_ROOT}|$ENV{ONNXRUNTIME_ROOT}" CACHE INTERNAL "")
endif()
find_path(ONNXRuntime_INCLUDE_DIR onnxruntime_cxx_api.h
  HINTS ${_ort_hints}
  PATH_SUFFIXES include include/onnxruntime include/onnxruntime/core/session)
find_library(ONNXRuntime_LIBRARY onnxruntime HINTS ${_ort_hints} PATH_SUFFIXES lib lib64)

if(ONNXRuntime_INCLUDE_DIR AND EXISTS "${ONNXRuntime_INCLUDE_DIR}/onnxruntime_c_api.h")
  # The C API version is the release's minor version (ORT_API_VERSION 31 is ONNX Runtime 1.31).
  file(STRINGS "${ONNXRuntime_INCLUDE_DIR}/onnxruntime_c_api.h" _ort_api REGEX "^#define ORT_API_VERSION [0-9]+")
  string(REGEX REPLACE "^#define ORT_API_VERSION ([0-9]+).*" "1.\\1" ONNXRuntime_VERSION "${_ort_api}")
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ONNXRuntime
  REQUIRED_VARS ONNXRuntime_LIBRARY ONNXRuntime_INCLUDE_DIR
  VERSION_VAR ONNXRuntime_VERSION)

if(ONNXRuntime_FOUND AND NOT TARGET ONNXRuntime::ONNXRuntime)
  add_library(ONNXRuntime::ONNXRuntime UNKNOWN IMPORTED)
  set_target_properties(ONNXRuntime::ONNXRuntime PROPERTIES
    IMPORTED_LOCATION "${ONNXRuntime_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${ONNXRuntime_INCLUDE_DIR}")
endif()
mark_as_advanced(ONNXRuntime_INCLUDE_DIR ONNXRuntime_LIBRARY)
