// Buffers and the memory they live in: host memory for ONNX Runtime, CUDA device memory for TensorRT.
#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <utility>

namespace smolvla_rt {

enum class MemoryKind { kHost, kCuda };

const char* to_string(MemoryKind kind);

// An owning, move-only block of memory. The deleter matches the allocator that produced it, and may carry what
// it needs, such as the CUDA device to free on.
class Buffer {
 public:
  using Deleter = std::function<void(void*)>;

  Buffer() = default;
  Buffer(void* data, size_t bytes, MemoryKind kind, Deleter deleter)
      : data_(data), bytes_(bytes), kind_(kind), deleter_(std::move(deleter)) {}
  Buffer(Buffer&& other) noexcept { *this = std::move(other); }
  Buffer& operator=(Buffer&& other) noexcept;
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
  ~Buffer() { reset(); }

  void* data() const { return data_; }
  template <typename T>
  T* as() const {
    return static_cast<T*>(data_);
  }
  size_t bytes() const { return bytes_; }
  MemoryKind kind() const { return kind_; }
  explicit operator bool() const { return data_ != nullptr; }

  void reset();

 private:
  void* data_ = nullptr;
  size_t bytes_ = 0;
  MemoryKind kind_ = MemoryKind::kHost;
  Deleter deleter_ = nullptr;
};

// Where a runtime keeps the buffers its graphs read and write, and how host data gets in and out of them.
class Device {
 public:
  virtual ~Device() = default;

  virtual MemoryKind kind() const = 0;
  // Zero-filled, aligned for any element type.
  virtual Buffer allocate(size_t bytes) = 0;
  // `dst` is memory of this device; returns when the copy is complete.
  virtual void copy_from_host(void* dst, const void* src, size_t bytes) = 0;
  // `src` is memory of this device; returns when the copy is complete.
  virtual void copy_to_host(void* dst, const void* src, size_t bytes) = 0;
};

// Plain host memory, 64-byte aligned like the bundle's tensors.
class HostDevice final : public Device {
 public:
  MemoryKind kind() const override { return MemoryKind::kHost; }
  Buffer allocate(size_t bytes) override;
  void copy_from_host(void* dst, const void* src, size_t bytes) override;
  void copy_to_host(void* dst, const void* src, size_t bytes) override;
};

Buffer allocate_host(size_t bytes);

}  // namespace smolvla_rt
