// A fake SmolVLA for runtime tests: graphs with the contract's names and shapes for a small config, computed by
// simple functions that make the data flow visible and the Euler trajectory exact in float32.
#pragma once

#include <chrono>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "fake_graph.h"
#include "smolvla_rt/contract.h"
#include "smolvla_rt/runtime.h"

namespace smolvla_rt::testing {

inline constexpr int64_t kHidden = 2;

inline SplitConfig small_config() {
  SplitConfig c;
  c.camera_keys = {"cam0", "cam1"};
  c.image_height = 4;
  c.image_width = 4;
  c.image_tokens = 3;
  c.lang_len = 5;
  c.state_dim = 3;
  c.max_state_dim = 6;
  c.action_dim = 2;
  c.max_action_dim = 4;
  c.chunk_size = 3;
  c.num_steps = 4;  // dt = -1/4 and t = 1, 3/4, 1/2, 1/4: everything below is exact in float32
  c.num_layers = 2;
  c.expert_layers = 2;
  c.kv_heads = 1;
  c.head_dim = 2;
  c.prefix_len = 2 * 3 + 5 + 1;
  return c;
}

inline float read(FakeGraph& g, const TensorSpec& spec, size_t i) {
  switch (spec.dtype) {
    case DType::kFloat32:
      return g.data<float>(spec.name)[i];
    case DType::kFloat16:
      return half_to_float(g.data<uint16_t>(spec.name)[i]);
    case DType::kInt64:
      return static_cast<float>(g.data<int64_t>(spec.name)[i]);
    case DType::kBool:
      return g.data<uint8_t>(spec.name)[i];
  }
  return 0;
}

inline void fill(FakeGraph& g, const TensorSpec& spec, float value) {
  for (size_t i = 0; i < static_cast<size_t>(element_count(spec.shape)); ++i) {
    switch (spec.dtype) {
      case DType::kFloat32:
        g.data<float>(spec.name)[i] = value;
        break;
      case DType::kFloat16:
        g.data<uint16_t>(spec.name)[i] = float_to_half(value);
        break;
      case DType::kInt64:
        g.data<int64_t>(spec.name)[i] = static_cast<int64_t>(value);
        break;
      case DType::kBool:
        g.data<uint8_t>(spec.name)[i] = value != 0;
        break;
    }
  }
}

// Graphs with the contract's names and shapes for small_config() whose outputs make the data flow visible:
//   vision:      image_tokens = pixel_values[0] of that camera
//   prefix:      past_key_0 = image_tokens_0, past_value_0 = image_tokens_1, past_key_1 = state[0],
//                past_value_1 = lang_tokens[0], prefix_pad_mask = lang_mask[0]
//   expert_step: v_t = x_t * t + (past_key_0 - past_value_0), so the trajectory depends on both cameras
struct FakeModel {
  SplitConfig config = small_config();
  DType pixel_dtype = DType::kFloat32;
  DType activation_dtype = DType::kFloat32;  // image tokens and K/V
  MemoryKind kind = MemoryKind::kHost;
  std::chrono::milliseconds expert_delay{0};  // added to every expert step, to stand in for slow inference
  std::vector<float> seen_t;
  std::vector<float> seen_pixels;  // pixel_values[0] as each vision run read it

  std::vector<TensorSpec> vision_in() const {
    return {{"pixel_values", pixel_dtype, {1, 3, config.image_height, config.image_width}}};
  }
  std::vector<TensorSpec> vision_out() const {
    return {{"image_tokens", activation_dtype, {1, config.image_tokens, kHidden}}};
  }
  std::vector<TensorSpec> prefix_in() const {
    std::vector<TensorSpec> specs;
    for (int64_t c = 0; c < config.num_cameras(); ++c) {
      specs.push_back({"image_tokens_" + std::to_string(c), activation_dtype, {1, config.image_tokens, kHidden}});
    }
    specs.push_back({"lang_tokens", DType::kInt64, {1, config.lang_len}});
    specs.push_back({"lang_mask", DType::kBool, {1, config.lang_len}});
    specs.push_back({"state", DType::kFloat32, {1, config.max_state_dim}});
    return specs;
  }
  std::vector<TensorSpec> prefix_out() const {
    std::vector<TensorSpec> specs = {{"prefix_pad_mask", DType::kBool, {1, config.prefix_len}}};
    for (const auto& name : kv_names(config.num_layers)) {
      specs.push_back({name, activation_dtype, {1, config.kv_heads, config.prefix_len, config.head_dim}});
    }
    return specs;
  }
  std::vector<TensorSpec> expert_in() const {
    std::vector<TensorSpec> specs = {{"x_t", DType::kFloat32, {1, config.chunk_size, config.max_action_dim}},
                                     {"t", DType::kFloat32, {1}}};
    for (const auto& s : prefix_out()) specs.push_back(s);
    return specs;
  }
  std::vector<TensorSpec> expert_out() const {
    return {{"v_t", DType::kFloat32, {1, config.chunk_size, config.max_action_dim}}};
  }

  std::unique_ptr<FakeGraph> vision(std::vector<TensorSpec> in, std::vector<TensorSpec> out) {
    return std::make_unique<FakeGraph>(
        "vision", in, out,
        [this, in, out](FakeGraph& g) {
          seen_pixels.push_back(read(g, in[0], 0));
          fill(g, out[0], read(g, in[0], 0));
        },
        kind);
  }
  std::unique_ptr<FakeGraph> prefix(std::vector<TensorSpec> in, std::vector<TensorSpec> out) {
    return std::make_unique<FakeGraph>(
        "prefix", in, out,
        [this, in, out](FakeGraph& g) {
          const auto spec = [&](const std::string& name) {
            for (const auto* specs : {&in, &out}) {
              for (const auto& s : *specs) {
                if (s.name == name) return s;
              }
            }
            throw std::invalid_argument(name);
          };
          fill(g, spec("prefix_pad_mask"), read(g, spec("lang_mask"), 0));
          fill(g, spec("past_key_0"), read(g, spec("image_tokens_0"), 0));
          fill(g, spec("past_value_0"), read(g, spec("image_tokens_1"), 0));
          fill(g, spec("past_key_1"), read(g, spec("state"), 0));
          fill(g, spec("past_value_1"), read(g, spec("lang_tokens"), 0));
        },
        kind);
  }
  std::unique_ptr<FakeGraph> expert(std::vector<TensorSpec> in, std::vector<TensorSpec> out) {
    return std::make_unique<FakeGraph>(
        "expert_step", in, out,
        [this, in](FakeGraph& g) {
          if (expert_delay.count() > 0) std::this_thread::sleep_for(expert_delay);
          const float t = g.data<float>("t")[0];
          seen_t.push_back(t);
          const TensorSpec* key = nullptr;
          const TensorSpec* value = nullptr;
          for (const auto& s : in) {
            if (s.name == "past_key_0") key = &s;
            if (s.name == "past_value_0") value = &s;
          }
          const float offset = read(g, *key, 0) - read(g, *value, 0);
          const float* x = g.data<float>("x_t");
          float* v = g.data<float>("v_t");
          for (size_t i = 0; i < g.count("x_t"); ++i) v[i] = x[i] * t + offset;
        },
        kind);
  }

  // The runtime plus pointers to its graphs, which it owns.
  struct Built {
    std::unique_ptr<SmolVLARuntime> runtime;
    FakeGraph *vision, *prefix, *expert;
  };
  Built build(std::shared_ptr<Device> device) {
    auto v = vision(vision_in(), vision_out());
    auto p = prefix(prefix_in(), prefix_out());
    auto e = expert(expert_in(), expert_out());
    Built b{nullptr, v.get(), p.get(), e.get()};
    b.runtime = std::make_unique<SmolVLARuntime>(config, std::move(v), std::move(p), std::move(e), std::move(device));
    return b;
  }
};

// The observation and noise for small_config(): camera c's pixels are all pixel[c], noise[i] = i / 8.
struct Inputs {
  explicit Inputs(const SplitConfig& c, std::vector<float> pixel = {0.5f, 0.25f})
      : lang_tokens(static_cast<size_t>(c.lang_len), 7),
        lang_mask(static_cast<size_t>(c.lang_len), 1),
        state(static_cast<size_t>(c.max_state_dim), 0.0f),
        noise(static_cast<size_t>(c.chunk_size * c.max_action_dim)) {
    for (float p : pixel) pixels.emplace_back(static_cast<size_t>(3 * c.image_height * c.image_width), p);
    state[0] = 3.0f;
    for (size_t i = 0; i < noise.size(); ++i) noise[i] = static_cast<float>(i) / 8;
    for (const auto& p : pixels) obs.pixel_values.push_back(p.data());
    obs.lang_tokens = lang_tokens.data();
    obs.lang_mask = lang_mask.data();
    obs.state = state.data();
  }
  std::vector<std::vector<float>> pixels;
  std::vector<int64_t> lang_tokens;
  std::vector<uint8_t> lang_mask;
  std::vector<float> state, noise;
  Observation obs;
};

// x <- x + dt * (x * t + offset) over the schedule, each operation rounded to float32 as LeRobot does.
inline std::vector<float> expected_actions(const SplitConfig& c, const std::vector<float>& noise, float offset) {
  const EulerSchedule s = euler_schedule(c.num_steps);
  std::vector<float> x = noise;
  for (float t : s.t) {
    for (float& xi : x) {
      const float v = xi * t + offset;
      const float step = s.dt * v;
      xi = xi + step;
    }
  }
  std::vector<float> actions;
  for (int64_t i = 0; i < c.chunk_size; ++i) {
    for (int64_t a = 0; a < c.action_dim; ++a) actions.push_back(x[static_cast<size_t>(i * c.max_action_dim + a)]);
  }
  return actions;
}

}  // namespace smolvla_rt::testing
