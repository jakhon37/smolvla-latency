#include "smolvla_rt/runtime.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <iterator>
#include <stdexcept>

namespace smolvla_rt {

namespace {

using Clock = std::chrono::steady_clock;

double ms_since(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

std::string join_names(const std::vector<std::string>& names) {
  std::string s;
  for (const auto& n : names) s += (s.empty() ? "" : ", ") + n;
  return s;
}

// The graphs bind by name, so only the set of names has to match; ONNX graphs from export_split.py also keep
// the contract's order, but a TensorRT engine need not.
void check_names(const std::vector<TensorSpec>& specs, const std::vector<std::string>& expected,
                 const std::string& what) {
  std::vector<std::string> names;
  for (const auto& s : specs) names.push_back(s.name);
  std::vector<std::string> want = expected;
  std::sort(names.begin(), names.end());
  std::sort(want.begin(), want.end());
  if (names == want) return;
  std::vector<std::string> missing, unexpected;
  std::set_difference(want.begin(), want.end(), names.begin(), names.end(), std::back_inserter(missing));
  std::set_difference(names.begin(), names.end(), want.begin(), want.end(), std::back_inserter(unexpected));
  throw std::invalid_argument(what + " do not match the contract: missing [" + join_names(missing) + "], unexpected [" +
                              join_names(unexpected) + "]");
}

void check_spec(const TensorSpec& spec, std::initializer_list<DType> dtypes, const Shape& shape,
                const std::string& graph) {
  const bool dtype_ok = std::find(dtypes.begin(), dtypes.end(), spec.dtype) != dtypes.end();
  if (dtype_ok && spec.shape == shape) return;
  std::string allowed;
  for (DType d : dtypes) allowed += std::string(allowed.empty() ? "" : " or ") + dtype_name(d);
  throw std::invalid_argument(graph + " " + to_string(spec) + ": the contract expects " + allowed + to_string(shape));
}

TensorSpec renamed(TensorSpec spec, std::string name) {
  spec.name = std::move(name);
  return spec;
}

}  // namespace

SmolVLARuntime::SmolVLARuntime(SplitConfig config, std::unique_ptr<Graph> vision, std::unique_ptr<Graph> prefix,
                               std::unique_ptr<Graph> expert_step, std::shared_ptr<Device> device)
    : config_(std::move(config)),
      vision_(std::move(vision)),
      prefix_(std::move(prefix)),
      expert_step_(std::move(expert_step)),
      device_(std::move(device)) {
  config_.validate();
  schedule_ = euler_schedule(config_.num_steps);
  if (!vision_ || !prefix_ || !expert_step_ || !device_) {
    throw std::invalid_argument("SmolVLARuntime: null graph or device");
  }
  for (const Graph* g : {vision_.get(), prefix_.get(), expert_step_.get()}) {
    if (g->memory_kind() != device_->kind()) {
      throw std::invalid_argument(std::string("SmolVLARuntime: a graph binds ") + to_string(g->memory_kind()) +
                                  " memory but the device allocates " + to_string(device_->kind()) + " memory");
    }
  }
  const SplitConfig& c = config_;
  const int64_t cams = c.num_cameras();

  // vision: pixel_values [1, 3, H, W] -> image_tokens [1, T, D]
  check_names(vision_->inputs(), {"pixel_values"}, "vision inputs");
  check_names(vision_->outputs(), {"image_tokens"}, "vision outputs");
  const TensorSpec& pixels = vision_->inputs()[0];
  check_spec(pixels, {DType::kFloat32, DType::kFloat16}, {1, 3, c.image_height, c.image_width}, "vision");
  const TensorSpec& tokens = vision_->outputs()[0];
  if (tokens.shape.size() != 3 || tokens.shape[2] <= 0) {
    throw std::invalid_argument("vision " + to_string(tokens) + ": the contract expects [1, T, D]");
  }
  check_spec(tokens, {DType::kFloat32, DType::kFloat16}, {1, c.image_tokens, tokens.shape[2]}, "vision");

  // prefix: image tokens per camera, language, state -> pad mask and per-layer K/V
  const auto& prefix_in = prefix_->inputs();
  const auto& prefix_out = prefix_->outputs();
  const std::vector<std::string> out_names = prefix_output_names(c.num_layers);
  check_names(prefix_in, prefix_input_names(cams), "prefix inputs");
  check_names(prefix_out, out_names, "prefix outputs");
  for (int64_t cam = 0; cam < cams; ++cam) {
    const TensorSpec in = find_spec(prefix_in, "image_tokens_" + std::to_string(cam), "prefix");
    if (in != renamed(tokens, in.name)) {
      throw std::invalid_argument("prefix " + to_string(in) + " does not match vision " + to_string(tokens));
    }
  }
  check_spec(find_spec(prefix_in, "lang_tokens", "prefix"), {DType::kInt64}, {1, c.lang_len}, "prefix");
  check_spec(find_spec(prefix_in, "lang_mask", "prefix"), {DType::kBool}, {1, c.lang_len}, "prefix");
  check_spec(find_spec(prefix_in, "state", "prefix"), {DType::kFloat32}, {1, c.max_state_dim}, "prefix");
  check_spec(find_spec(prefix_out, out_names[0], "prefix"), {DType::kBool}, {1, c.prefix_len}, "prefix");
  const DType kv_dtype = find_spec(prefix_out, out_names[1], "prefix").dtype;
  for (size_t i = 1; i < out_names.size(); ++i) {
    const TensorSpec kv = find_spec(prefix_out, out_names[i], "prefix");
    check_spec(kv, {DType::kFloat32, DType::kFloat16}, {1, c.kv_heads, c.prefix_len, c.head_dim}, "prefix");
    if (kv.dtype != kv_dtype) throw std::invalid_argument("prefix K/V outputs mix dtypes");
  }

  // expert_step: x_t, t and exactly the prefix outputs -> v_t
  const auto& expert_in = expert_step_->inputs();
  check_names(expert_in, expert_input_names(c.num_layers), "expert_step inputs");
  check_names(expert_step_->outputs(), {"v_t"}, "expert_step outputs");
  const Shape action_shape = {1, c.chunk_size, c.max_action_dim};
  check_spec(find_spec(expert_in, "x_t", "expert_step"), {DType::kFloat32}, action_shape, "expert_step");
  check_spec(find_spec(expert_in, "t", "expert_step"), {DType::kFloat32}, {1}, "expert_step");
  for (const auto& name : out_names) {
    const TensorSpec in = find_spec(expert_in, name, "expert_step");
    const TensorSpec out = find_spec(prefix_out, name, "prefix");
    if (in != out) {
      throw std::invalid_argument("expert_step " + to_string(in) + " does not match prefix " + to_string(out));
    }
  }
  check_spec(expert_step_->outputs()[0], {DType::kFloat32}, action_shape, "expert_step");

  // Every buffer is allocated and bound here, once. Only the vision graph is rebound, to each camera's pair.
  for (int64_t cam = 0; cam < cams; ++cam) {
    const std::string suffix = "_" + std::to_string(cam);
    pixels_.push_back(make_slot(renamed(pixels, "pixel_values" + suffix), pixels.dtype != DType::kFloat32));
    image_tokens_.push_back(make_slot(renamed(tokens, "image_tokens" + suffix), false));
    prefix_->bind(image_tokens_.back().spec.name, image_tokens_.back().device.data());
  }
  lang_tokens_ = make_slot(find_spec(prefix_in, "lang_tokens", "prefix"), false);
  lang_mask_ = make_slot(find_spec(prefix_in, "lang_mask", "prefix"), false);
  state_ = make_slot(find_spec(prefix_in, "state", "prefix"), false);
  for (const Slot* s : {&lang_tokens_, &lang_mask_, &state_}) prefix_->bind(s->spec.name, s->device.data());

  x_t_ = make_slot(find_spec(expert_in, "x_t", "expert_step"), true);
  t_ = make_slot(find_spec(expert_in, "t", "expert_step"), true);
  v_t_ = make_slot(expert_step_->outputs()[0], true);
  for (const Slot* s : {&x_t_, &t_, &v_t_}) expert_step_->bind(s->spec.name, s->device.data());
  for (const auto& name : out_names) {
    prefix_out_.push_back(make_slot(find_spec(prefix_out, name, "prefix"), false));
    prefix_->bind(name, prefix_out_.back().device.data());
    expert_step_->bind(name, prefix_out_.back().device.data());
  }
  vision_->bind("pixel_values", pixels_[0].device.data());
  vision_->bind("image_tokens", image_tokens_[0].device.data());

  timings_.vision.assign(static_cast<size_t>(cams), 0.0);
  timings_.expert_step.assign(static_cast<size_t>(c.num_steps), 0.0);
}

DType SmolVLARuntime::pixel_dtype() const { return pixels_.front().spec.dtype; }

SmolVLARuntime::Slot SmolVLARuntime::make_slot(const TensorSpec& spec, bool staged) const {
  Slot s;
  s.spec = spec;
  s.device = device_->allocate(spec.bytes());
  if (staged && device_->kind() != MemoryKind::kHost) s.host = allocate_host(spec.bytes());
  return s;
}

void SmolVLARuntime::upload(const Slot& slot) const {
  if (slot.host) device_->copy_from_host(slot.device.data(), slot.host.data(), slot.spec.bytes());
}

void SmolVLARuntime::download(const Slot& slot) const {
  if (slot.host) device_->copy_to_host(slot.host.data(), slot.device.data(), slot.spec.bytes());
}

void SmolVLARuntime::predict_action_chunk(const Observation& obs, const float* noise, float* actions) {
  const auto start = Clock::now();
  const int64_t cams = config_.num_cameras();
  if (static_cast<int64_t>(obs.pixel_values.size()) != cams) {
    throw std::invalid_argument("observation has " + std::to_string(obs.pixel_values.size()) + " cameras, the graphs " +
                                std::to_string(cams));
  }
  if (std::count(obs.pixel_values.begin(), obs.pixel_values.end(), nullptr) || !obs.lang_tokens || !obs.lang_mask ||
      !obs.state || !noise || !actions) {
    throw std::invalid_argument("predict_action_chunk: null input or output");
  }

  for (size_t cam = 0; cam < pixels_.size(); ++cam) {
    const Slot& slot = pixels_[cam];
    if (slot.spec.dtype == DType::kFloat16) {
      float_to_half(obs.pixel_values[cam], static_cast<uint16_t*>(slot.host_data()),
                    static_cast<size_t>(element_count(slot.spec.shape)));
      upload(slot);
    } else {
      device_->copy_from_host(slot.device.data(), obs.pixel_values[cam], slot.spec.bytes());
    }
  }
  device_->copy_from_host(lang_tokens_.device.data(), obs.lang_tokens, lang_tokens_.spec.bytes());
  device_->copy_from_host(lang_mask_.device.data(), obs.lang_mask, lang_mask_.spec.bytes());
  device_->copy_from_host(state_.device.data(), obs.state, state_.spec.bytes());
  timings_.inputs = ms_since(start);

  for (size_t cam = 0; cam < pixels_.size(); ++cam) {
    vision_->bind("pixel_values", pixels_[cam].device.data());
    vision_->bind("image_tokens", image_tokens_[cam].device.data());
    const auto t0 = Clock::now();
    vision_->run();
    timings_.vision[cam] = ms_since(t0);
  }
  const auto prefix_start = Clock::now();
  prefix_->run();
  timings_.prefix = ms_since(prefix_start);

  // On the host, x_t and v_t are the expert graph's own buffers and the update happens in place.
  const auto loop_start = Clock::now();
  auto* x = static_cast<float*>(x_t_.host_data());
  const auto* v = static_cast<const float*>(v_t_.host_data());
  auto* t = static_cast<float*>(t_.host_data());
  const auto n = static_cast<size_t>(element_count(x_t_.spec.shape));
  std::memcpy(x, noise, n * sizeof(float));
  double graph_ms = 0;
  for (size_t k = 0; k < schedule_.t.size(); ++k) {
    *t = schedule_.t[k];
    upload(t_);
    upload(x_t_);
    const auto t0 = Clock::now();
    expert_step_->run();
    timings_.expert_step[k] = ms_since(t0);
    graph_ms += timings_.expert_step[k];
    download(v_t_);
    if (step_observer_) step_observer_(static_cast<int64_t>(k), x, v);
    euler_update(x, v, schedule_.dt, n);
  }
  timings_.euler = ms_since(loop_start) - graph_ms;

  const auto a = static_cast<size_t>(config_.max_action_dim);
  const auto keep = static_cast<size_t>(config_.action_dim);
  for (size_t step = 0; step < static_cast<size_t>(config_.chunk_size); ++step) {
    std::memcpy(actions + step * keep, x + step * a, keep * sizeof(float));
  }
  timings_.total = ms_since(start);
}

const SmolVLARuntime::Slot& SmolVLARuntime::slot(const std::string& name) const {
  for (const auto* group : {&pixels_, &image_tokens_, &prefix_out_}) {
    for (const auto& s : *group) {
      if (s.spec.name == name) return s;
    }
  }
  for (const Slot* s : {&lang_tokens_, &lang_mask_, &state_, &x_t_, &t_, &v_t_}) {
    if (s->spec.name == name) return *s;
  }
  throw std::invalid_argument("the runtime has no tensor '" + name + "'");
}

const TensorSpec& SmolVLARuntime::tensor_spec(const std::string& name) const { return slot(name).spec; }

void SmolVLARuntime::copy_to_host(const std::string& name, void* dst) const {
  const Slot& s = slot(name);
  device_->copy_to_host(dst, s.device.data(), s.spec.bytes());
}

}  // namespace smolvla_rt
