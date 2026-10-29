// The CUDA runtime behind CudaDevice: built with the TensorRT backend, which links libcudart.
#include <cuda_runtime_api.h>

#include <string>

#include "smolvla_rt/cuda_device.h"

namespace smolvla_rt {

namespace {

int get_device_count(int* count) { return static_cast<int>(cudaGetDeviceCount(count)); }
int set_device(int device) { return static_cast<int>(cudaSetDevice(device)); }
int malloc_device(void** ptr, size_t bytes) { return static_cast<int>(cudaMalloc(ptr, bytes)); }
int free_device(void* ptr) { return static_cast<int>(cudaFree(ptr)); }
int memset_device(void* ptr, int value, size_t bytes) { return static_cast<int>(cudaMemset(ptr, value, bytes)); }

int copy_to_device(void* dst, const void* src, size_t bytes) {
  return static_cast<int>(cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice));
}

int copy_to_host(void* dst, const void* src, size_t bytes) {
  return static_cast<int>(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost));
}

std::string describe(int status) {
  const auto error = static_cast<cudaError_t>(status);
  return std::string(cudaGetErrorName(error)) + " (" + cudaGetErrorString(error) + ")";
}

}  // namespace

const CudaApi& cuda_runtime_api() {
  static const CudaApi api{get_device_count, set_device,     malloc_device, free_device,
                           memset_device,    copy_to_device, copy_to_host,  describe};
  return api;
}

}  // namespace smolvla_rt
