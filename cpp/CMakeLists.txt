cmake_minimum_required(VERSION 3.18)
foreach(_policy CMP0135 CMP0144)  # URL downloads get extraction-time timestamps; ONNXRUNTIME_ROOT is a find root
  if(POLICY ${_policy})
    cmake_policy(SET ${_policy} NEW)
  endif()
endforeach()
project(smolvla_rt VERSION 0.1.0 DESCRIPTION "C++ runtime for the split SmolVLA graphs" LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)  # the Python module links the static library
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

option(SMOLVLA_RT_WITH_ORT "Build the ONNX Runtime backend (needs the ONNX Runtime C/C++ SDK, see ONNXRUNTIME_ROOT)" ON)
set(SMOLVLA_RT_WITH_TENSORRT AUTO CACHE STRING
  "Build the TensorRT backend: ON, OFF, or AUTO to build it when TensorRT (TENSORRT_ROOT) and CUDA are found")
set_property(CACHE SMOLVLA_RT_WITH_TENSORRT PROPERTY STRINGS AUTO ON OFF)
option(SMOLVLA_RT_TRT_COMPILE_CHECK
  "Compile the TensorRT backend against its headers without linking it, where TensorRT is not installed" OFF)
set(SMOLVLA_RT_CUDA_INCLUDE_DIR "" CACHE PATH "cuda_runtime_api.h directory for SMOLVLA_RT_TRT_COMPILE_CHECK")
option(SMOLVLA_RT_BUILD_TESTS "Build the GoogleTest tests" ON)
option(SMOLVLA_RT_BUILD_PYTHON "Build the pybind11 module smolvla_rt_py" OFF)
option(SMOLVLA_RT_WERROR "Treat compiler warnings as errors" OFF)

include(FetchContent)
find_package(Threads REQUIRED)

# Header-only JSON parser for split_manifest.json and bundle headers: the system copy (nlohmann-json3-dev), else
# the 110 kB release archive. Offline, point FETCHCONTENT_SOURCE_DIR_NLOHMANN_JSON at an unpacked copy.
find_package(nlohmann_json 3.10 QUIET)
if(NOT nlohmann_json_FOUND)
  FetchContent_Declare(nlohmann_json
    URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
    URL_HASH SHA256=d6c65aca6b1ed68e7a182f4757257b107ae403032760ed6ef121c9d55e81757d)
  FetchContent_MakeAvailable(nlohmann_json)
endif()

if(SMOLVLA_RT_WITH_ORT)
  if(NOT ONNXRUNTIME_ROOT AND NOT DEFINED ENV{ONNXRUNTIME_ROOT}
     AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime/include")
    set(ONNXRUNTIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime")  # scripts/fetch_onnxruntime.sh
  endif()
  find_package(ONNXRuntime REQUIRED)
endif()

string(TOUPPER "${SMOLVLA_RT_WITH_TENSORRT}" _trt_mode)
set(SMOLVLA_RT_WITH_TENSORRT_BUILT OFF)
if(_trt_mode MATCHES "^(ON|TRUE|YES|1|AUTO)$")
  find_package(TensorRT QUIET)
  find_package(CUDAToolkit QUIET)
  if(TensorRT_FOUND AND CUDAToolkit_FOUND)
    set(SMOLVLA_RT_WITH_TENSORRT_BUILT ON)
  elseif(NOT _trt_mode STREQUAL "AUTO")
    message(FATAL_ERROR "SMOLVLA_RT_WITH_TENSORRT=${SMOLVLA_RT_WITH_TENSORRT}, but TensorRT (set TENSORRT_ROOT) "
                        "or the CUDA toolkit was not found")
  endif()
endif()

# Common flags. -ffp-contract=off keeps the Euler update a separate multiply and add, as PyTorch computes it;
# GCC would otherwise fuse them into an FMA wherever the target has one (e.g. -march=native).
add_library(smolvla_rt_flags INTERFACE)
target_compile_options(smolvla_rt_flags INTERFACE
  $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra -Wpedantic -Wshadow -ffp-contract=off>
  $<$<AND:$<CXX_COMPILER_ID:GNU,Clang,AppleClang>,$<BOOL:${SMOLVLA_RT_WERROR}>>:-Werror>)

set(SMOLVLA_RT_HAS_ORT 0)
set(SMOLVLA_RT_HAS_TENSORRT 0)
if(SMOLVLA_RT_WITH_ORT)
  set(SMOLVLA_RT_HAS_ORT 1)
endif()
if(SMOLVLA_RT_WITH_TENSORRT_BUILT)
  set(SMOLVLA_RT_HAS_TENSORRT 1)
endif()
configure_file(include/smolvla_rt/build_config.h.in "${CMAKE_CURRENT_BINARY_DIR}/include/smolvla_rt/build_config.h")

add_library(smolvla_rt
  src/bundle.cpp
  src/contract.cpp
  src/control_loop.cpp
  src/cuda_device.cpp  # needs no CUDA headers: the CUDA runtime comes in through CudaApi
  src/graph.cpp
  src/json_util.cpp
  src/loader.cpp
  src/manifest.cpp
  src/memory.cpp
  src/runtime.cpp
  src/stats.cpp
  src/tensor.cpp)
target_include_directories(smolvla_rt PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_link_libraries(smolvla_rt PUBLIC Threads::Threads PRIVATE smolvla_rt_flags nlohmann_json::nlohmann_json)
if(SMOLVLA_RT_WITH_ORT)
  target_sources(smolvla_rt PRIVATE src/ort_graph.cpp)
  target_link_libraries(smolvla_rt PRIVATE ONNXRuntime::ONNXRuntime)
endif()
if(SMOLVLA_RT_WITH_TENSORRT_BUILT)
  target_sources(smolvla_rt PRIVATE src/trt_graph.cpp src/cuda_api.cpp)
  target_link_libraries(smolvla_rt PRIVATE TensorRT::nvinfer CUDA::cudart)
endif()

if(SMOLVLA_RT_TRT_COMPILE_CHECK AND NOT SMOLVLA_RT_WITH_TENSORRT_BUILT)
  # Objects only: proves the TensorRT backend compiles against the given headers. Nothing links them.
  find_path(TensorRT_INCLUDE_DIR NvInferRuntime.h HINTS ${TENSORRT_ROOT} $ENV{TENSORRT_ROOT} PATH_SUFFIXES include)
  find_path(_cuda_runtime_include cuda_runtime_api.h HINTS ${SMOLVLA_RT_CUDA_INCLUDE_DIR} ${CUDAToolkit_INCLUDE_DIRS})
  if(NOT TensorRT_INCLUDE_DIR OR NOT _cuda_runtime_include)
    message(FATAL_ERROR "SMOLVLA_RT_TRT_COMPILE_CHECK needs TENSORRT_ROOT (NvInferRuntime.h) and "
                        "SMOLVLA_RT_CUDA_INCLUDE_DIR (cuda_runtime_api.h)")
  endif()
  # The loader is compiled again with a build_config.h that has the TensorRT backend, for its TensorRT branch.
  set(SMOLVLA_RT_HAS_TENSORRT 1)
  configure_file(include/smolvla_rt/build_config.h.in
                 "${CMAKE_CURRENT_BINARY_DIR}/trt_compile_check/include/smolvla_rt/build_config.h")
  set(SMOLVLA_RT_HAS_TENSORRT 0)
  add_library(smolvla_rt_trt_compile_check OBJECT src/trt_graph.cpp src/cuda_api.cpp src/loader.cpp)
  target_include_directories(smolvla_rt_trt_compile_check PRIVATE
    "${CMAKE_CURRENT_BINARY_DIR}/trt_compile_check/include" include)
  # SYSTEM, as for the imported targets of a real build: TensorRT's own headers use deprecated declarations.
  target_include_directories(smolvla_rt_trt_compile_check SYSTEM PRIVATE
    "${TensorRT_INCLUDE_DIR}" "${_cuda_runtime_include}")
  target_link_libraries(smolvla_rt_trt_compile_check PRIVATE smolvla_rt_flags)
  message(STATUS "TensorRT backend: compile check only, against ${TensorRT_INCLUDE_DIR}")
endif()

add_executable(smolvla_rt_bench tools/bench.cpp)
target_link_libraries(smolvla_rt_bench PRIVATE smolvla_rt smolvla_rt_flags nlohmann_json::nlohmann_json)

if(SMOLVLA_RT_BUILD_PYTHON)
  # The module is "smolvla_rt"; pybind11 comes from CMAKE_PREFIX_PATH or from the Python it builds for.
  find_package(Python 3.9 REQUIRED COMPONENTS Interpreter Development.Module)
  find_package(pybind11 CONFIG QUIET)
  if(NOT pybind11_FOUND)
    execute_process(COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
                    OUTPUT_VARIABLE _pybind11_dir OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
    find_package(pybind11 CONFIG REQUIRED HINTS "${_pybind11_dir}")
  endif()
  pybind11_add_module(smolvla_rt_py python/bindings.cpp)
  set_target_properties(smolvla_rt_py PROPERTIES
    OUTPUT_NAME smolvla_rt
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python")
  target_link_libraries(smolvla_rt_py PRIVATE smolvla_rt smolvla_rt_flags)
endif()

if(SMOLVLA_RT_BUILD_TESTS)
  enable_testing()
  find_package(GTest 1.11 CONFIG QUIET)
  if(NOT GTest_FOUND)
    FetchContent_Declare(googletest
      GIT_REPOSITORY https://github.com/google/googletest.git
      GIT_TAG v1.15.2
      GIT_SHALLOW TRUE)
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
  endif()
  add_subdirectory(tests)
endif()

include(GNUInstallDirs)
install(TARGETS smolvla_rt smolvla_rt_bench)
install(DIRECTORY include/smolvla_rt DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} FILES_MATCHING PATTERN "*.h")
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/include/smolvla_rt/build_config.h"
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/smolvla_rt)

message(STATUS "smolvla_rt: ONNX Runtime ${SMOLVLA_RT_WITH_ORT} ${ONNXRuntime_VERSION}, "
               "TensorRT ${SMOLVLA_RT_WITH_TENSORRT_BUILT} ${TensorRT_VERSION}")
