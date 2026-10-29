// SmolVLA action-chunk inference over the three split graphs: vision per camera, prefix once, and the Euler
// flow-matching loop over expert_step.
#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "smolvla_rt/contract.h"
#include "smolvla_rt/graph.h"
#include "smolvla_rt/memory.h"

namespace smolvla_rt {

// One preprocessed observation, laid out as docs/SPLIT_GRAPH.md "Inputs" describes. The pointers only need
// to stay valid during predict_action_chunk.
struct Observation {
  std::vector<const float*> pixel_values;  // per camera, [1, 3, H, W] float32 in [-1, 1]
  const int64_t* lang_tokens = nullptr;    // [1, L], right-padded to L
  const uint8_t* lang_mask = nullptr;      // [1, L], 1 for real tokens, 0 for padding
  const float* state = nullptr;            // [1, S], normalised, zero-padded
};

// Milliseconds spent in each part of one predict_action_chunk call.
struct ChunkTimings {
  double inputs = 0;           // copying the observation into graph buffers, with the fp16 cast if any
  std::vector<double> vision;  // one per camera
  double prefix = 0;
  std::vector<double> expert_step;  // one per Euler step, the graph alone
  double euler = 0;                 // all steps' updates and their x_t, t, v_t copies
  double total = 0;
};

class SmolVLARuntime {
 public:
  // Called after expert step k with x_t (its input) and v_t, both [1, C, A] in host memory.
  using StepObserver = std::function<void(int64_t step, const float* x_t, const float* v_t)>;

  // Checks the graphs against the contract for `config` (names, order, dtypes, shapes, and expert_step taking
  // exactly the prefix outputs), then allocates every buffer on `device` and binds it once. The graphs'
  // memory_kind() must be device->kind(). Throws std::invalid_argument describing the first mismatch.
  SmolVLARuntime(SplitConfig config, std::unique_ptr<Graph> vision, std::unique_ptr<Graph> prefix,
                 std::unique_ptr<Graph> expert_step, std::shared_ptr<Device> device);

  const SplitConfig& config() const { return config_; }
  const EulerSchedule& schedule() const { return schedule_; }
  // The dtype the vision graph takes pixels in; observations are always float32 and cast if needed.
  DType pixel_dtype() const;
  // The graphs' backend, e.g. "onnxruntime 1.31.0 CPUExecutionProvider".
  std::string backend() const { return expert_step_->backend(); }

  // One action chunk from `noise` ([1, C, A] float32, the first x_t) into `actions` ([1, C, action_dim]
  // float32, normalised: what predict_action_chunk returns before the postprocessor). Same observation and
  // noise give the same actions as ort_flow.SplitRuntime. Reuses the buffers allocated at construction, so
  // calls must not overlap: one thread at a time.
  void predict_action_chunk(const Observation& obs, const float* noise, float* actions);

  const ChunkTimings& last_timings() const { return timings_; }

  // For checking a run against reference values; an empty observer turns it off.
  void set_step_observer(StepObserver observer) { step_observer_ = std::move(observer); }
  // image_tokens_<c> and the prefix outputs of the last chunk.
  const TensorSpec& tensor_spec(const std::string& name) const;
  void copy_to_host(const std::string& name, void* dst) const;

 private:
  struct Slot {
    TensorSpec spec;
    Buffer device;  // what the graphs are bound to
    Buffer host;    // staging copy, only when device memory is not host memory
    void* host_data() const { return host ? host.data() : device.data(); }
  };

  // `staged` adds a host copy when the device is not the host, for tensors the runtime reads or writes itself.
  Slot make_slot(const TensorSpec& spec, bool staged) const;
  void upload(const Slot& slot) const;
  void download(const Slot& slot) const;
  const Slot& slot(const std::string& name) const;

  SplitConfig config_;
  EulerSchedule schedule_;
  std::unique_ptr<Graph> vision_;
  std::unique_ptr<Graph> prefix_;
  std::unique_ptr<Graph> expert_step_;
  std::shared_ptr<Device> device_;

  std::vector<Slot> pixels_;        // vision input, per camera
  std::vector<Slot> image_tokens_;  // vision output per camera = prefix input image_tokens_<c>
  Slot lang_tokens_, lang_mask_, state_;
  std::vector<Slot> prefix_out_;  // prefix outputs = expert_step inputs after x_t and t
  Slot x_t_, t_, v_t_;

  ChunkTimings timings_;
  StepObserver step_observer_;
};

}  // namespace smolvla_rt
