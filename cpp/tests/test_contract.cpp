#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "smolvla_rt/contract.h"

namespace smolvla_rt {
namespace {

TEST(Contract, TensorNamesMatchSplitContractPy) {
  EXPECT_EQ(kv_names(2), (std::vector<std::string>{"past_key_0", "past_value_0", "past_key_1", "past_value_1"}));
  EXPECT_EQ(prefix_input_names(2),
            (std::vector<std::string>{"image_tokens_0", "image_tokens_1", "lang_tokens", "lang_mask", "state"}));
  EXPECT_EQ(prefix_output_names(1), (std::vector<std::string>{"prefix_pad_mask", "past_key_0", "past_value_0"}));
  EXPECT_EQ(expert_input_names(1),
            (std::vector<std::string>{"x_t", "t", "prefix_pad_mask", "past_key_0", "past_value_0"}));
}

// split_contract.euler_schedule(10) evaluates 1 + k * -0.1 in double and rounds once, which gives the float32
// nearest to 1.0, 0.9, ..., 0.1. Evaluating in float32 instead would give 0.39999998, 0.19999999 and
// 0.099999964 at k = 6, 8 and 9.
TEST(Contract, EulerScheduleOfTenStepsIsLeRobots) {
  const EulerSchedule s = euler_schedule(10);
  EXPECT_EQ(s.dt, -0.1f);
  const std::vector<float> expected = {1.0f, 0.9f, 0.8f, 0.7f, 0.6f, 0.5f, 0.4f, 0.3f, 0.2f, 0.1f};
  EXPECT_EQ(s.t, expected);
  EXPECT_NE(1.0f + 6.0f * -0.1f, 0.4f) << "float32 arithmetic no longer differs here; the comment above is stale";
  EXPECT_THROW(euler_schedule(0), std::invalid_argument);
}

// A hand-computed case where every intermediate value is exact in float32: four steps of dt = -1/4 with
// v = x * t give x * (1 - t/4) per step, i.e. x * 0.75 * 0.8125 * 0.875 * 0.9375 = x * 0.4998779296875.
TEST(Contract, EulerUpdateHandComputed) {
  const EulerSchedule s = euler_schedule(4);
  EXPECT_EQ(s.dt, -0.25f);
  EXPECT_EQ(s.t, (std::vector<float>{1.0f, 0.75f, 0.5f, 0.25f}));
  std::vector<float> x = {1.0f, -2.0f, 8.0f};
  std::vector<float> v(x.size());
  for (float t : s.t) {
    for (size_t i = 0; i < x.size(); ++i) v[i] = x[i] * t;
    euler_update(x.data(), v.data(), s.dt, x.size());
  }
  EXPECT_EQ(x, (std::vector<float>{0.4998779296875f, -0.999755859375f, 3.9990234375f}));
}

// LeRobot's update rounds dt * v to float32 before adding; a fused multiply-add would round once. For these
// inputs the two differ (0.97f vs 0.96999997f), so the test fails if the build lets the compiler contract the
// update into an FMA.
TEST(Contract, EulerUpdateRoundsTheProductSeparately) {
  float x = 1.0f;
  const float v = 0.3f;
  euler_update(&x, &v, -0.1f, 1);
  EXPECT_EQ(x, 0.97f);
  EXPECT_EQ(static_cast<float>(1.0 + static_cast<double>(-0.1f) * static_cast<double>(0.3f)), 0.96999997f);
}

SplitConfig full_size_config() {
  SplitConfig c;
  c.camera_keys = {"a", "b"};
  c.image_height = c.image_width = 512;
  c.image_tokens = 64;
  c.lang_len = 48;
  c.state_dim = 8;
  c.max_state_dim = 32;
  c.action_dim = 7;
  c.max_action_dim = 32;
  c.chunk_size = 50;
  c.num_steps = 10;
  c.num_layers = 16;
  c.expert_layers = 16;
  c.kv_heads = 5;
  c.head_dim = 64;
  c.prefix_len = 177;
  return c;
}

// The validate() message for `config`, or "" if it is valid.
std::string validation_error(const SplitConfig& config) {
  try {
    config.validate();
    return "";
  } catch (const std::invalid_argument& e) {
    return e.what();
  }
}

TEST(Contract, ConfigValidation) {
  const SplitConfig c = full_size_config();
  EXPECT_NO_THROW(c.validate());
  SplitConfig bad = c;
  bad.prefix_len = 176;
  EXPECT_THROW(bad.validate(), std::invalid_argument);
  bad = c;
  bad.action_dim = 33;
  EXPECT_THROW(bad.validate(), std::invalid_argument);
  bad = c;
  bad.num_steps = 0;
  EXPECT_THROW(bad.validate(), std::invalid_argument);
  bad = c;
  bad.image_tokens = int64_t{1} << 62;  // cameras * image_tokens would overflow
  EXPECT_NE(validation_error(bad).find("image_tokens is 4611686018427387904"), std::string::npos);
}

// SmolVLA settings the split graphs do not reproduce, as an export made with them would describe itself.
TEST(Contract, ConfigValidationNamesUnsupportedSettings) {
  SplitConfig special_tokens = full_size_config();
  special_tokens.prefix_len = 2 * (64 + 3) + 48 + 1;  // [fake, global] image tokens [fake] per camera
  EXPECT_NE(validation_error(special_tokens).find("add_image_special_tokens=True"), std::string::npos)
      << validation_error(special_tokens);

  SplitConfig padded = full_size_config();
  padded.prefix_len = 200;
  EXPECT_NE(validation_error(padded).find("prefix_length > 0"), std::string::npos) << validation_error(padded);

  SplitConfig fewer_expert_layers = full_size_config();
  fewer_expert_layers.expert_layers = 8;
  EXPECT_NE(validation_error(fewer_expert_layers).find("the action expert has 8 layers for 16 VLM layers"),
            std::string::npos)
      << validation_error(fewer_expert_layers);
}

}  // namespace
}  // namespace smolvla_rt
