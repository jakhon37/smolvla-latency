// Sizes, tensor names and the Euler schedule of the split graphs: smolvla_latency/split_contract.py in C++.
// docs/SPLIT_GRAPH.md is the full contract.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace smolvla_rt {

constexpr const char* kManifestFile = "split_manifest.json";
constexpr const char* kVision = "vision";
constexpr const char* kPrefix = "prefix";
constexpr const char* kExpertStep = "expert_step";

// The "config" object of split_manifest.json and of a bundle's meta: everything a runtime needs to size its
// buffers. The VLM hidden size is not in it; it comes from the graphs' image_tokens shape.
struct SplitConfig {
  std::vector<std::string> camera_keys;  // camera c feeds pixel_values_<c> and image_tokens_<c>
  int64_t image_height = 0;
  int64_t image_width = 0;
  int64_t image_tokens = 0;  // per camera
  int64_t lang_len = 0;
  int64_t state_dim = 0;
  int64_t max_state_dim = 0;
  int64_t action_dim = 0;
  int64_t max_action_dim = 0;
  int64_t chunk_size = 0;
  int64_t num_steps = 0;
  int64_t num_layers = 0;
  int64_t expert_layers = 0;  // action expert layers; the split graphs need one per VLM layer (num_layers)
  int64_t kv_heads = 0;
  int64_t head_dim = 0;
  int64_t prefix_len = 0;

  int64_t num_cameras() const { return static_cast<int64_t>(camera_keys.size()); }
  // Throws std::invalid_argument if a size is not positive or above 2^31 - 1, if expert_layers is not num_layers,
  // or if prefix_len is not cameras * T + L + 1. The message names the SmolVLA setting a mismatch points to.
  void validate() const;
};

bool operator==(const SplitConfig& a, const SplitConfig& b);
inline bool operator!=(const SplitConfig& a, const SplitConfig& b) { return !(a == b); }

// past_key_0, past_value_0, past_key_1, ...: interleaved per layer.
std::vector<std::string> kv_names(int64_t num_layers);
// image_tokens_<c> for each camera, then lang_tokens, lang_mask, state.
std::vector<std::string> prefix_input_names(int64_t num_cameras);
// prefix_pad_mask, then kv_names.
std::vector<std::string> prefix_output_names(int64_t num_layers);
// x_t, t, then the prefix outputs in the same order.
std::vector<std::string> expert_input_names(int64_t num_layers);

// LeRobot's euler_integrate schedule: dt = -1 / num_steps and t_k = 1 + k * dt, both computed in double and
// rounded once to float, exactly as split_contract.euler_schedule does.
struct EulerSchedule {
  float dt = 0.0f;
  std::vector<float> t;
};
EulerSchedule euler_schedule(int64_t num_steps);

// x <- x + dt * v elementwise: a float32 multiply, then a float32 add, each rounded, as LeRobot's update
// `x_t + dt * v_t` does. The library is compiled without floating-point contraction so this never becomes a
// fused multiply-add, which would round once and stop matching PyTorch bit for bit.
void euler_update(float* x, const float* v, float dt, size_t n);

}  // namespace smolvla_rt
