// Graph backend on the TensorRT 10 runtime API: a serialized engine, CUDA device buffers, one stream.
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "smolvla_rt/graph.h"

namespace smolvla_rt {

class TrtGraph final : public Graph {
 public:
  // Deserializes an engine built from one of the split graphs, for example
  //   trtexec --onnx=prefix_fp16.onnx --stronglyTyped --saveEngine=prefix_fp16.engine
  // on CUDA device `device`. Throws std::runtime_error if the file, the engine or the CUDA setup fails, or if
  // the engine has dynamic shapes. run() and the destructor select `device` on the calling thread first.
  explicit TrtGraph(const std::string& engine_path, int device = 0);
  ~TrtGraph() override;

  const std::vector<TensorSpec>& inputs() const override;
  const std::vector<TensorSpec>& outputs() const override;
  MemoryKind memory_kind() const override { return MemoryKind::kCuda; }
  // Sets the tensor's device address in the execution context.
  void bind(const std::string& name, void* data) override;
  // enqueueV3 on the graph's stream, then waits for it.
  void run() override;
  std::string backend() const override;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// The loaded TensorRT library's version, e.g. "10.16.0".
std::string tensorrt_version();

}  // namespace smolvla_rt
