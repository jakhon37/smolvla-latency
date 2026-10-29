// Test doubles: graphs computed by a C++ function, and devices that count what the runtime does with memory.
#pragma once

#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

#include "smolvla_rt/contract.h"
#include "smolvla_rt/graph.h"
#include "smolvla_rt/memory.h"

namespace smolvla_rt::testing {

class FakeGraph final : public Graph {
 public:
  using Compute = std::function<void(FakeGraph&)>;

  FakeGraph(std::string name, std::vector<TensorSpec> inputs, std::vector<TensorSpec> outputs, Compute compute,
            MemoryKind kind = MemoryKind::kHost)
      : name_(std::move(name)),
        inputs_(std::move(inputs)),
        outputs_(std::move(outputs)),
        compute_(std::move(compute)),
        kind_(kind) {}

  const std::vector<TensorSpec>& inputs() const override { return inputs_; }
  const std::vector<TensorSpec>& outputs() const override { return outputs_; }
  MemoryKind memory_kind() const override { return kind_; }
  std::string backend() const override { return "fake"; }

  void bind(const std::string& name, void* data) override {
    find_spec(name);
    bound_[name] = data;
    ++binds;
  }

  void run() override {
    for (const auto* specs : {&inputs_, &outputs_}) {
      for (const auto& s : *specs) {
        if (!bound_.count(s.name)) throw std::runtime_error(name_ + ": " + s.name + " not bound");
      }
    }
    bound_at_run.push_back(bound_);
    compute_(*this);
  }

  template <typename T>
  T* data(const std::string& name) {
    return static_cast<T*>(bound_.at(name));
  }
  size_t count(const std::string& name) const { return static_cast<size_t>(element_count(find_spec(name).shape)); }

  int binds = 0;
  std::vector<std::map<std::string, void*>> bound_at_run;  // what was bound at each run()

 private:
  const TensorSpec& find_spec(const std::string& name) const {
    for (const auto* specs : {&inputs_, &outputs_}) {
      for (const auto& s : *specs) {
        if (s.name == name) return s;
      }
    }
    throw std::invalid_argument(name_ + " has no tensor " + name);
  }

  std::string name_;
  std::vector<TensorSpec> inputs_, outputs_;
  Compute compute_;
  MemoryKind kind_;
  std::map<std::string, void*> bound_;
};

// Host memory that the runtime has to treat as device memory: it reports kCuda, so the runtime stages through
// host copies exactly as it would for a GPU, and the copies are counted.
class CountingDevice final : public Device {
 public:
  explicit CountingDevice(MemoryKind kind) : kind_(kind) {}
  MemoryKind kind() const override { return kind_; }
  Buffer allocate(size_t bytes) override {
    ++allocations;
    const size_t padded = (bytes / 64 + 1) * 64;
    void* p = std::aligned_alloc(64, padded);
    if (!p) throw std::bad_alloc();
    std::memset(p, 0, padded);
    return Buffer(p, bytes, kind_, [](void* q) { std::free(q); });
  }
  void copy_from_host(void* dst, const void* src, size_t bytes) override {
    ++uploads;
    std::memcpy(dst, src, bytes);
  }
  void copy_to_host(void* dst, const void* src, size_t bytes) override {
    ++downloads;
    std::memcpy(dst, src, bytes);
  }

  int allocations = 0, uploads = 0, downloads = 0;

 private:
  MemoryKind kind_;
};

}  // namespace smolvla_rt::testing
