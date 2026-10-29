#include "smolvla_rt/contract.h"

#include <stdexcept>

namespace smolvla_rt {

void SplitConfig::validate() const {
  // Far above any real size, and low enough that the arithmetic below cannot overflow.
  constexpr int64_t kMaxSize = (int64_t{1} << 31) - 1;
  const std::pair<const char*, int64_t> sizes[] = {{"image_hw[0]", image_height},  {"image_hw[1]", image_width},
                                                   {"image_tokens", image_tokens}, {"lang_len", lang_len},
                                                   {"state_dim", state_dim},       {"max_state_dim", max_state_dim},
                                                   {"action_dim", action_dim},     {"max_action_dim", max_action_dim},
                                                   {"chunk_size", chunk_size},     {"num_steps", num_steps},
                                                   {"num_layers", num_layers},     {"expert_layers", expert_layers},
                                                   {"kv_heads", kv_heads},         {"head_dim", head_dim},
                                                   {"prefix_len", prefix_len}};
  for (const auto& [name, value] : sizes) {
    if (value <= 0 || value > kMaxSize) {
      throw std::invalid_argument(std::string("split config: ") + name + " is " + std::to_string(value) +
                                  "; it must be positive and at most 2^31 - 1");
    }
  }
  if (camera_keys.empty() || num_cameras() > kMaxSize) {
    throw std::invalid_argument("split config: " + std::to_string(camera_keys.size()) + " cameras");
  }
  if (state_dim > max_state_dim || action_dim > max_action_dim) {
    throw std::invalid_argument("split config: state_dim/action_dim exceed their padded sizes");
  }
  if (expert_layers != num_layers) {
    throw std::invalid_argument("split config: the action expert has " + std::to_string(expert_layers) +
                                " layers for " + std::to_string(num_layers) +
                                " VLM layers (num_expert_layers); the split graphs need one per VLM layer");
  }
  const int64_t expected = num_cameras() * image_tokens + lang_len + 1;
  if (prefix_len != expected) {
    std::string why;
    if (prefix_len == num_cameras() * (image_tokens + 3) + lang_len + 1) {
      why = ": 3 more positions per camera, the image start and end tokens of add_image_special_tokens=True";
    } else if (prefix_len > expected) {
      why = ": padding to a fixed length, as prefix_length > 0 adds";
    }
    throw std::invalid_argument("split config: prefix_len " + std::to_string(prefix_len) +
                                " is not cameras * image_tokens + lang_len + 1 = " + std::to_string(expected) + why +
                                (why.empty() ? "" : ", which the split graphs do not support"));
  }
}

bool operator==(const SplitConfig& a, const SplitConfig& b) {
  return a.camera_keys == b.camera_keys && a.image_height == b.image_height && a.image_width == b.image_width &&
         a.image_tokens == b.image_tokens && a.lang_len == b.lang_len && a.state_dim == b.state_dim &&
         a.max_state_dim == b.max_state_dim && a.action_dim == b.action_dim && a.max_action_dim == b.max_action_dim &&
         a.chunk_size == b.chunk_size && a.num_steps == b.num_steps && a.num_layers == b.num_layers &&
         a.expert_layers == b.expert_layers && a.kv_heads == b.kv_heads && a.head_dim == b.head_dim &&
         a.prefix_len == b.prefix_len;
}

std::vector<std::string> kv_names(int64_t num_layers) {
  std::vector<std::string> names;
  for (int64_t i = 0; i < num_layers; ++i) {
    names.push_back("past_key_" + std::to_string(i));
    names.push_back("past_value_" + std::to_string(i));
  }
  return names;
}

std::vector<std::string> prefix_input_names(int64_t num_cameras) {
  std::vector<std::string> names;
  for (int64_t c = 0; c < num_cameras; ++c) names.push_back("image_tokens_" + std::to_string(c));
  names.insert(names.end(), {"lang_tokens", "lang_mask", "state"});
  return names;
}

std::vector<std::string> prefix_output_names(int64_t num_layers) {
  std::vector<std::string> names = {"prefix_pad_mask"};
  for (auto& name : kv_names(num_layers)) names.push_back(std::move(name));
  return names;
}

std::vector<std::string> expert_input_names(int64_t num_layers) {
  std::vector<std::string> names = {"x_t", "t"};
  for (auto& name : prefix_output_names(num_layers)) names.push_back(std::move(name));
  return names;
}

EulerSchedule euler_schedule(int64_t num_steps) {
  if (num_steps <= 0) throw std::invalid_argument("euler_schedule: num_steps must be positive");
  const double dt = -1.0 / static_cast<double>(num_steps);
  EulerSchedule schedule;
  schedule.dt = static_cast<float>(dt);
  for (int64_t k = 0; k < num_steps; ++k) schedule.t.push_back(static_cast<float>(1.0 + static_cast<double>(k) * dt));
  return schedule;
}

void euler_update(float* x, const float* v, float dt, size_t n) {
  for (size_t i = 0; i < n; ++i) x[i] = x[i] + dt * v[i];
}

}  // namespace smolvla_rt
