// Graph backend on the ONNX Runtime C++ API: the CPU execution provider, or CUDA when the library has it.
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "smolvla_rt/graph.h"

namespace smolvla_rt {

struct OrtOptions {
  // Intra-op threads per session; 0 lets ONNX Runtime choose (one per physical core).
  int intra_op_threads = 0;
  // ONNX Runtime's workers spin while waiting for work by default. Turning it off saves CPU for other threads,
  // e.g. a control loop, at some latency cost.
  bool allow_spinning = true;
  // "cpu", "cuda", or "auto" for CUDA when the ONNX Runtime library has it. With CUDA the buffers stay in host
  // memory and ONNX Runtime copies them to and from the GPU on every run, prefix K/V included; TrtGraph keeps
  // them on the device instead. TF32 is turned off, so fp32 graphs compute in fp32 like the PyTorch reference.
  std::string provider = "cpu";
  int cuda_device = 0;
};

class OrtGraph final : public Graph {
 public:
  // Loads an ONNX file (its .onnx.data weights next to it). Throws std::runtime_error with ONNX Runtime's
  // message on failure, or if an input or output has a dynamic shape or an unsupported element type.
  OrtGraph(const std::string& onnx_path, const OrtOptions& options);
  ~OrtGraph() override;

  const std::vector<TensorSpec>& inputs() const override;
  const std::vector<TensorSpec>& outputs() const override;
  MemoryKind memory_kind() const override { return MemoryKind::kHost; }
  void bind(const std::string& name, void* data) override;
  void run() override;
  std::string backend() const override;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// The ONNX Runtime library version, e.g. "1.31.0".
std::string ort_version();
// Whether the loaded ONNX Runtime library has the CUDA execution provider.
bool ort_cuda_available();

}  // namespace smolvla_rt
