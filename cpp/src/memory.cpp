#include "smolvla_rt/memory.h"

#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>
#include <utility>

namespace smolvla_rt {

namespace {

constexpr size_t kAlignment = 64;

void free_host(void* p) { std::free(p); }

}  // namespace

const char* to_string(MemoryKind kind) { return kind == MemoryKind::kHost ? "host" : "cuda"; }

Buffer& Buffer::operator=(Buffer&& other) noexcept {
  if (this != &other) {
    reset();
    data_ = other.data_;
    bytes_ = other.bytes_;
    kind_ = other.kind_;
    deleter_ = std::move(other.deleter_);
    other.data_ = nullptr;
    other.bytes_ = 0;
    other.deleter_ = nullptr;
  }
  return *this;
}

void Buffer::reset() {
  if (data_ && deleter_) deleter_(data_);
  data_ = nullptr;
  bytes_ = 0;
  deleter_ = nullptr;
}

Buffer allocate_host(size_t bytes) {
  if (bytes > std::numeric_limits<size_t>::max() - kAlignment) throw std::bad_alloc();  // padding would wrap
  // aligned_alloc needs a size that is a multiple of the alignment, and a nonzero one to return a pointer.
  const size_t padded = (bytes + kAlignment - 1) / kAlignment * kAlignment;
  void* p = std::aligned_alloc(kAlignment, padded ? padded : kAlignment);
  if (!p) throw std::bad_alloc();
  std::memset(p, 0, padded ? padded : kAlignment);
  return Buffer(p, bytes, MemoryKind::kHost, free_host);
}

Buffer HostDevice::allocate(size_t bytes) { return allocate_host(bytes); }

void HostDevice::copy_from_host(void* dst, const void* src, size_t bytes) {
  if (bytes) std::memcpy(dst, src, bytes);
}

void HostDevice::copy_to_host(void* dst, const void* src, size_t bytes) {
  if (bytes) std::memcpy(dst, src, bytes);
}

}  // namespace smolvla_rt
