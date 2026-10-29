// CudaDevice's device selection, on a stand-in for the CUDA runtime: no GPU or CUDA headers needed.
#include <gtest/gtest.h>

#include <cstdlib>
#include <cstring>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "smolvla_rt/cuda_device.h"

namespace smolvla_rt {
namespace {

// Two devices whose memory is host memory. As in CUDA, the current device belongs to the calling thread and is 0
// on a new thread. Every memory call records its name and the device current on its thread.
constexpr int kDevices = 2;
thread_local int current_device = 0;
std::mutex calls_mutex;
std::vector<std::pair<std::string, int>> calls;
int device_count_status = 0;

void record(const char* name) {
  std::lock_guard<std::mutex> lock(calls_mutex);
  calls.emplace_back(name, current_device);
}

int get_device_count(int* count) {
  *count = kDevices;
  return device_count_status;
}

int set_device(int device) {
  if (device < 0 || device >= kDevices) return 101;  // cudaErrorInvalidDevice
  current_device = device;
  return 0;
}

int malloc_device(void** ptr, size_t bytes) {
  record("malloc");
  *ptr = std::malloc(bytes);
  return *ptr ? 0 : 2;  // cudaErrorMemoryAllocation
}

int free_device(void* ptr) {
  record("free");
  std::free(ptr);
  return 0;
}

int memset_device(void* ptr, int value, size_t bytes) {
  record("memset");
  std::memset(ptr, value, bytes);
  return 0;
}

int copy_to_device(void* dst, const void* src, size_t bytes) {
  record("copy_to_device");
  std::memcpy(dst, src, bytes);
  return 0;
}

int copy_to_host(void* dst, const void* src, size_t bytes) {
  record("copy_to_host");
  std::memcpy(dst, src, bytes);
  return 0;
}

std::string describe(int status) { return "fake CUDA error " + std::to_string(status); }

const CudaApi kFakeCuda{get_device_count, set_device,     malloc_device, free_device,
                        memset_device,    copy_to_device, copy_to_host,  describe};

void on_new_thread(const std::function<void()>& work) {
  std::thread thread(work);
  thread.join();
}

// Each operation runs on a thread of its own, where device 0 is current until CudaDevice selects its device, as
// the control loop's inference thread and Python threads call into a runtime built elsewhere.
TEST(CudaDevice, SelectsItsDeviceOnEveryCallingThread) {
  calls.clear();
  CudaDevice device(1, kFakeCuda);
  EXPECT_EQ(device.index(), 1);
  EXPECT_EQ(device.kind(), MemoryKind::kCuda);

  Buffer buffer;
  on_new_thread([&] { buffer = device.allocate(4 * sizeof(float)); });
  const float in[4] = {1.0f, -2.0f, 3.5f, 0.25f};
  float out[4] = {};
  on_new_thread([&] { device.copy_from_host(buffer.data(), in, sizeof in); });
  on_new_thread([&] { device.copy_to_host(out, buffer.data(), sizeof out); });
  on_new_thread([&] { buffer.reset(); });

  EXPECT_EQ(std::memcmp(in, out, sizeof in), 0);
  const std::vector<std::pair<std::string, int>> expected = {
      {"malloc", 1}, {"memset", 1}, {"copy_to_device", 1}, {"copy_to_host", 1}, {"free", 1}};
  EXPECT_EQ(calls, expected);
}

TEST(CudaDevice, RejectsMissingDevicesAndReportsCudaErrors) {
  try {
    CudaDevice device(kDevices, kFakeCuda);
    ADD_FAILURE() << "device " << kDevices << " was accepted";
  } catch (const std::runtime_error& e) {
    EXPECT_EQ(std::string(e.what()), "CUDA device 2 requested, 2 available");
  }
  EXPECT_THROW(CudaDevice(-1, kFakeCuda), std::runtime_error);

  device_count_status = 35;  // cudaErrorInsufficientDriver, as on a machine without a GPU driver
  try {
    CudaDevice device(0, kFakeCuda);
    ADD_FAILURE() << "a failed cudaGetDeviceCount was ignored";
  } catch (const std::runtime_error& e) {
    EXPECT_EQ(std::string(e.what()), "cudaGetDeviceCount failed: fake CUDA error 35");
  }
  device_count_status = 0;
}

}  // namespace
}  // namespace smolvla_rt
