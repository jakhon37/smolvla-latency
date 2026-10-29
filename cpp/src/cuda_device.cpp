#include "smolvla_rt/cuda_device.h"

#include <stdexcept>
#include <string>

namespace smolvla_rt {

CudaDevice::CudaDevice(int device, const CudaApi& api) : device_(device), api_(&api) {
  int count = 0;
  check(api_->get_device_count(&count), "cudaGetDeviceCount");
  if (device < 0 || device >= count) {
    throw std::runtime_error("CUDA device " + std::to_string(device) + " requested, " + std::to_string(count) +
                             " available");
  }
  select();
}

void CudaDevice::select() const { check(api_->set_device(device_), "cudaSetDevice"); }

void CudaDevice::check(int status, const char* what) const {
  if (status != 0) throw std::runtime_error(std::string(what) + " failed: " + api_->describe(status));
}

Buffer CudaDevice::allocate(size_t bytes) {
  select();
  void* p = nullptr;
  check(api_->malloc_device(&p, bytes ? bytes : 1), "cudaMalloc");
  // Freed on this device whichever thread drops the buffer; a destructor has no way to report a failure.
  Buffer buffer(p, bytes, MemoryKind::kCuda, [api = api_, device = device_](void* q) {
    if (api->set_device(device) == 0) api->free_device(q);
  });
  check(api_->memset_device(p, 0, bytes), "cudaMemset");
  return buffer;
}

void CudaDevice::copy_from_host(void* dst, const void* src, size_t bytes) {
  select();
  check(api_->copy_to_device(dst, src, bytes), "cudaMemcpy host to device");
}

void CudaDevice::copy_to_host(void* dst, const void* src, size_t bytes) {
  select();
  check(api_->copy_to_host(dst, src, bytes), "cudaMemcpy device to host");
}

}  // namespace smolvla_rt
