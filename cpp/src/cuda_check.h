// Internal: CUDA runtime error handling for the CUDA and TensorRT sources.
#pragma once

#include <cuda_runtime_api.h>

#include <stdexcept>
#include <string>

namespace smolvla_rt::detail {

inline void check_cuda(cudaError_t status, const char* what) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(what) + " failed: " + cudaGetErrorName(status) + " (" +
                             cudaGetErrorString(status) + ")");
  }
}

}  // namespace smolvla_rt::detail
