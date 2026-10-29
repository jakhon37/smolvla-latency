// CUDA device memory for graphs that bind device buffers (TrtGraph).
#pragma once

#include <cstddef>
#include <string>

#include "smolvla_rt/memory.h"

namespace smolvla_rt {

// The CUDA runtime calls CudaDevice makes. They are function pointers so that the device-selection logic below
// can be tested on a machine without CUDA. Each returns 0 (cudaSuccess) or a CUDA error code.
struct CudaApi {
  int (*get_device_count)(int* count);
  int (*set_device)(int device);  // for the calling thread
  int (*malloc_device)(void** ptr, size_t bytes);
  int (*free_device)(void* ptr);
  int (*memset_device)(void* ptr, int value, size_t bytes);
  int (*copy_to_device)(void* dst, const void* src, size_t bytes);  // synchronous, legacy default stream
  int (*copy_to_host)(void* dst, const void* src, size_t bytes);    // synchronous, legacy default stream
  std::string (*describe)(int status);  // e.g. "cudaErrorNoDevice (no CUDA-capable device is detected)"
};

// The CUDA runtime library. Defined in builds with the TensorRT backend only (src/cuda_api.cpp).
const CudaApi& cuda_runtime_api();

// The current CUDA device is a per-thread setting, and the runtime's calls can come from several threads: the
// control loop's inference thread, or Python threads through the module. So every method selects this device on
// the calling thread before its CUDA calls, and a buffer selects it again when it is freed.
class CudaDevice final : public Device {
 public:
  // Throws std::runtime_error if CUDA has no device with this index. `api` must outlive the device and every
  // buffer it allocates.
  explicit CudaDevice(int device = 0, const CudaApi& api = cuda_runtime_api());

  int index() const { return device_; }
  MemoryKind kind() const override { return MemoryKind::kCuda; }
  Buffer allocate(size_t bytes) override;
  // Synchronous copies on the legacy default stream; see TrtGraph for how its stream orders against them.
  void copy_from_host(void* dst, const void* src, size_t bytes) override;
  void copy_to_host(void* dst, const void* src, size_t bytes) override;

 private:
  void select() const;
  void check(int status, const char* what) const;

  int device_;
  const CudaApi* api_;
};

}  // namespace smolvla_rt
