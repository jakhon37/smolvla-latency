// SmolVLARuntime on fake graphs: the orchestration, the Euler loop, buffer reuse and the contract checks,
// without ONNX Runtime.
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <random>
#include <vector>

#include "fake_model.h"
#include "smolvla_rt/runtime.h"

namespace smolvla_rt {
namespace {

using testing::CountingDevice;
using testing::expected_actions;
using testing::FakeGraph;
using testing::FakeModel;
using testing::Inputs;

TEST(Runtime, RunsVisionPrefixAndTheEulerLoop) {
  FakeModel model;
  auto built = model.build(std::make_shared<HostDevice>());
  const SplitConfig& c = model.config;
  Inputs in(c);
  std::vector<float> actions(static_cast<size_t>(c.chunk_size * c.action_dim));
  built.runtime->predict_action_chunk(in.obs, in.noise.data(), actions.data());

  EXPECT_EQ(built.vision->bound_at_run.size(), 2u);
  EXPECT_EQ(built.prefix->bound_at_run.size(), 1u);
  EXPECT_EQ(built.expert->bound_at_run.size(), 4u);
  EXPECT_EQ(model.seen_pixels, (std::vector<float>{0.5f, 0.25f}));
  EXPECT_EQ(model.seen_t, euler_schedule(4).t);
  EXPECT_EQ(actions, expected_actions(c, in.noise, 0.25f));

  // x_0 = 1, worked by hand: v = x t + 1/4 and x += -1/4 v give 0.6875, 0.49609375, 0.37158203125 and
  // 0.285858154296875, all exact in float32.
  std::vector<float> noise(in.noise.size(), 0.0f);
  noise[0] = 1.0f;
  built.runtime->predict_action_chunk(in.obs, noise.data(), actions.data());
  EXPECT_EQ(actions[0], 0.285858154296875f);

  // The prefix saw the language tokens, mask and state.
  const auto tensor = [&](const std::string& name) {
    std::vector<float> values(static_cast<size_t>(element_count(built.runtime->tensor_spec(name).shape)));
    EXPECT_EQ(built.runtime->tensor_spec(name).bytes(), values.size() * sizeof(float));
    built.runtime->copy_to_host(name, values.data());
    return values;
  };
  EXPECT_EQ(tensor("past_key_1"), std::vector<float>(24, 3.0f));
  EXPECT_EQ(tensor("past_value_1"), std::vector<float>(24, 7.0f));
  std::vector<uint8_t> mask(static_cast<size_t>(c.prefix_len));
  built.runtime->copy_to_host("prefix_pad_mask", mask.data());
  EXPECT_EQ(mask, std::vector<uint8_t>(mask.size(), 1));
  EXPECT_THROW(built.runtime->tensor_spec("no_such_tensor"), std::invalid_argument);
}

TEST(Runtime, SwappedCamerasChangeTheResult) {
  FakeModel model;
  auto built = model.build(std::make_shared<HostDevice>());
  const SplitConfig& c = model.config;
  Inputs swapped(c, {0.25f, 0.5f});
  std::vector<float> actions(static_cast<size_t>(c.chunk_size * c.action_dim));
  built.runtime->predict_action_chunk(swapped.obs, swapped.noise.data(), actions.data());
  EXPECT_EQ(actions, expected_actions(c, swapped.noise, -0.25f));
}

TEST(Runtime, ReusesItsBuffersAcrossChunks) {
  FakeModel model;
  auto device = std::make_shared<CountingDevice>(MemoryKind::kHost);
  auto built = model.build(device);
  const SplitConfig& c = model.config;
  const int allocations = device->allocations;
  const int prefix_binds = built.prefix->binds, expert_binds = built.expert->binds;
  Inputs in(c);
  std::vector<float> actions(static_cast<size_t>(c.chunk_size * c.action_dim));
  for (int chunk = 0; chunk < 3; ++chunk) built.runtime->predict_action_chunk(in.obs, in.noise.data(), actions.data());

  EXPECT_EQ(device->allocations, allocations);
  EXPECT_EQ(built.prefix->binds, prefix_binds);  // bound once, at construction
  EXPECT_EQ(built.expert->binds, expert_binds);
  for (const auto& run : built.prefix->bound_at_run) EXPECT_EQ(run, built.prefix->bound_at_run.front());
  for (const auto& run : built.expert->bound_at_run) EXPECT_EQ(run, built.expert->bound_at_run.front());
  // expert_step reads the prefix outputs in place.
  for (const auto& name : prefix_output_names(c.num_layers)) {
    EXPECT_EQ(built.expert->bound_at_run.front().at(name), built.prefix->bound_at_run.front().at(name)) << name;
  }
  // Each camera keeps its own vision buffers, and they feed the prefix directly.
  const auto& vision_runs = built.vision->bound_at_run;
  ASSERT_EQ(vision_runs.size(), 6u);
  EXPECT_NE(vision_runs[0].at("image_tokens"), vision_runs[1].at("image_tokens"));
  for (size_t i = 2; i < vision_runs.size(); ++i) EXPECT_EQ(vision_runs[i], vision_runs[i % 2]);
  EXPECT_EQ(vision_runs[1].at("image_tokens"), built.prefix->bound_at_run.front().at("image_tokens_1"));
}

TEST(Runtime, StagesThroughHostCopiesWhenGraphsBindDeviceMemory) {
  FakeModel host_model;
  auto host = host_model.build(std::make_shared<HostDevice>());
  FakeModel device_model;
  device_model.kind = MemoryKind::kCuda;
  auto device = std::make_shared<CountingDevice>(MemoryKind::kCuda);
  auto built = device_model.build(device);
  const SplitConfig& c = host_model.config;
  Inputs in(c);
  std::vector<float> expected(static_cast<size_t>(c.chunk_size * c.action_dim)), actions(expected.size());
  host.runtime->predict_action_chunk(in.obs, in.noise.data(), expected.data());

  const int uploads = device->uploads, downloads = device->downloads;
  built.runtime->predict_action_chunk(in.obs, in.noise.data(), actions.data());
  EXPECT_EQ(actions, expected);
  // Per chunk: each camera's pixels, the language tokens, mask and state, then t and x_t every step; only v_t
  // comes back. The K/V never leave the device.
  EXPECT_EQ(device->uploads - uploads, 2 + 3 + 2 * 4);
  EXPECT_EQ(device->downloads - downloads, 4);
}

TEST(Runtime, CastsPixelsToHalfForFp16Graphs) {
  FakeModel model;
  model.pixel_dtype = DType::kFloat16;
  model.activation_dtype = DType::kFloat16;
  auto built = model.build(std::make_shared<HostDevice>());
  const SplitConfig& c = model.config;
  EXPECT_EQ(built.runtime->pixel_dtype(), DType::kFloat16);
  Inputs in(c, {0.1f, 1.0f / 3});
  std::vector<float> actions(static_cast<size_t>(c.chunk_size * c.action_dim));
  built.runtime->predict_action_chunk(in.obs, in.noise.data(), actions.data());
  EXPECT_EQ(model.seen_pixels,
            (std::vector<float>{half_to_float(float_to_half(0.1f)), half_to_float(float_to_half(1.0f / 3))}));
  const float offset = half_to_float(float_to_half(0.1f)) - half_to_float(float_to_half(1.0f / 3));
  EXPECT_EQ(actions, expected_actions(c, in.noise, offset));
}

TEST(Runtime, ObserverSeesEveryStep) {
  FakeModel model;
  auto built = model.build(std::make_shared<HostDevice>());
  const SplitConfig& c = model.config;
  Inputs in(c);
  std::vector<int64_t> steps;
  std::vector<float> first_x;
  built.runtime->set_step_observer([&](int64_t k, const float* x, const float* v) {
    steps.push_back(k);
    first_x.push_back(x[0]);
    EXPECT_EQ(v[0], x[0] * euler_schedule(4).t[static_cast<size_t>(k)] + 0.25f);
  });
  std::vector<float> actions(static_cast<size_t>(c.chunk_size * c.action_dim));
  built.runtime->predict_action_chunk(in.obs, in.noise.data(), actions.data());
  EXPECT_EQ(steps, (std::vector<int64_t>{0, 1, 2, 3}));
  EXPECT_EQ(first_x.front(), 0.0f);  // x_t at step 0 is the noise

  const ChunkTimings& t = built.runtime->last_timings();
  EXPECT_EQ(t.vision.size(), 2u);
  EXPECT_EQ(t.expert_step.size(), 4u);
  EXPECT_GE(t.total, t.prefix);
}

TEST(Runtime, AcceptsGraphsWhoseTensorsComeInAnotherOrder) {
  FakeModel model;
  auto in = model.expert_in();
  std::reverse(in.begin(), in.end());
  auto out = model.prefix_out();
  std::rotate(out.begin(), out.begin() + 1, out.end());
  SmolVLARuntime rt(model.config, model.vision(model.vision_in(), model.vision_out()),
                    model.prefix(model.prefix_in(), out), model.expert(in, model.expert_out()),
                    std::make_shared<HostDevice>());
  Inputs inputs(model.config);
  std::vector<float> actions(static_cast<size_t>(model.config.chunk_size * model.config.action_dim));
  rt.predict_action_chunk(inputs.obs, inputs.noise.data(), actions.data());
  EXPECT_EQ(actions, expected_actions(model.config, inputs.noise, 0.25f));
}

void expect_rejected(FakeModel& model, std::vector<TensorSpec> vision_in, std::vector<TensorSpec> prefix_out,
                     std::vector<TensorSpec> expert_in, const std::string& fragment,
                     std::shared_ptr<Device> device = std::make_shared<HostDevice>()) {
  try {
    SmolVLARuntime rt(model.config, model.vision(vision_in, model.vision_out()),
                      model.prefix(model.prefix_in(), prefix_out), model.expert(expert_in, model.expert_out()),
                      std::move(device));
    ADD_FAILURE() << "accepted graphs that should fail with '" << fragment << "'";
  } catch (const std::invalid_argument& e) {
    EXPECT_NE(std::string(e.what()).find(fragment), std::string::npos) << e.what();
  }
}

TEST(Runtime, RejectsGraphsThatBreakTheContract) {
  FakeModel m;
  auto expert_in = m.expert_in();
  expert_in[3].dtype = DType::kFloat16;  // past_key_0 no longer matches the prefix output
  expect_rejected(m, m.vision_in(), m.prefix_out(), expert_in, "past_key_0");

  auto vision_in = m.vision_in();
  vision_in[0].shape = {1, 3, 4, 5};
  expect_rejected(m, vision_in, m.prefix_out(), m.expert_in(), "pixel_values");

  auto prefix_out = m.prefix_out();
  prefix_out.pop_back();  // no past_value_1
  expect_rejected(m, m.vision_in(), prefix_out, m.expert_in(), "missing [past_value_1]");

  expert_in = m.expert_in();
  expert_in[1].shape = {1, 1};  // t
  expect_rejected(m, m.vision_in(), m.prefix_out(), expert_in, "t float32[1,1]");

  expect_rejected(m, m.vision_in(), m.prefix_out(), m.expert_in(), "memory",
                  std::make_shared<CountingDevice>(MemoryKind::kCuda));

  FakeModel wrong_config;
  wrong_config.config.lang_len = 6;  // the graphs were built for 5
  wrong_config.config.prefix_len = 2 * 3 + 6 + 1;
  FakeModel graphs;
  try {
    SmolVLARuntime rt(wrong_config.config, graphs.vision(graphs.vision_in(), graphs.vision_out()),
                      graphs.prefix(graphs.prefix_in(), graphs.prefix_out()),
                      graphs.expert(graphs.expert_in(), graphs.expert_out()), std::make_shared<HostDevice>());
    ADD_FAILURE() << "accepted graphs built for another config";
  } catch (const std::invalid_argument& e) {
    EXPECT_NE(std::string(e.what()).find("lang_tokens"), std::string::npos) << e.what();
  }
}

TEST(Runtime, RejectsBadObservations) {
  FakeModel model;
  auto built = model.build(std::make_shared<HostDevice>());
  const SplitConfig& c = model.config;
  Inputs in(c);
  std::vector<float> actions(static_cast<size_t>(c.chunk_size * c.action_dim));
  Observation one_camera = in.obs;
  one_camera.pixel_values.pop_back();
  EXPECT_THROW(built.runtime->predict_action_chunk(one_camera, in.noise.data(), actions.data()), std::invalid_argument);
  Observation no_state = in.obs;
  no_state.state = nullptr;
  EXPECT_THROW(built.runtime->predict_action_chunk(no_state, in.noise.data(), actions.data()), std::invalid_argument);
  EXPECT_THROW(built.runtime->predict_action_chunk(in.obs, nullptr, actions.data()), std::invalid_argument);
}

}  // namespace
}  // namespace smolvla_rt
