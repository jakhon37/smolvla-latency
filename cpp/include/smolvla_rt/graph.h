// The interface every backend implements for one of the three split graphs.
#pragma once

#include <string>
#include <vector>

#include "smolvla_rt/memory.h"
#include "smolvla_rt/tensor.h"

namespace smolvla_rt {

// One ONNX graph or TensorRT engine with named, statically shaped inputs and outputs.
//
// The caller owns every input and output buffer and binds each one by name before the first run(). Bindings
// stay in place across runs, so a runtime binds its pre-allocated buffers once and a graph's outputs can be
// bound directly as another graph's inputs (prefix -> expert_step). Not thread-safe: one thread drives a graph.
class Graph {
 public:
  virtual ~Graph() = default;

  // In the graph's declared order, which is the contract's order (docs/SPLIT_GRAPH.md).
  virtual const std::vector<TensorSpec>& inputs() const = 0;
  virtual const std::vector<TensorSpec>& outputs() const = 0;

  // The memory bound buffers must live in.
  virtual MemoryKind memory_kind() const = 0;

  // Points the named input or output at `data`: spec.bytes() bytes of memory_kind() memory that must stay valid
  // until the name is bound again or the graph is destroyed. Throws std::invalid_argument for an unknown name.
  virtual void bind(const std::string& name, void* data) = 0;

  // Runs on the bound buffers. The outputs are complete when it returns. Throws if a tensor is not bound.
  virtual void run() = 0;

  // What executes the graph, for reports, e.g. "onnxruntime 1.31.0 CPUExecutionProvider".
  virtual std::string backend() const = 0;
};

// The spec called `name`; throws std::invalid_argument naming `what` if there is none.
const TensorSpec& find_spec(const std::vector<TensorSpec>& specs, const std::string& name, const std::string& what);

}  // namespace smolvla_rt
