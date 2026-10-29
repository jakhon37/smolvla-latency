// The control-loop harness on the fake model. Timing on a shared test machine varies, so the tests check what
// the loop must do (tick count, chunk handoff, starvation when inference is too slow), not latency values.
#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

#include "fake_model.h"
#include "smolvla_rt/control_loop.h"

namespace smolvla_rt {
namespace {

using testing::FakeModel;
using testing::Inputs;

TEST(ControlLoop, KeepsTheChunkBufferFilledWhenInferenceIsFast) {
  FakeModel model;
  model.config.chunk_size = 20;
  auto built = model.build(std::make_shared<HostDevice>());
  Inputs in(model.config);
  ControlLoopOptions options;
  options.rate_hz = 100;
  options.ticks = 100;
  const ControlLoopReport r = run_control_loop(*built.runtime, in.obs, in.noise.data(), options);

  EXPECT_EQ(r.ticks, 100);
  EXPECT_DOUBLE_EQ(r.period_ms, 10.0);
  EXPECT_EQ(r.refill_threshold, 10);  // half a chunk by default
  EXPECT_EQ(r.wake_latency_ms.n, 100u);
  EXPECT_EQ(r.period_error_ms.n, 99u);
  EXPECT_GE(r.wake_latency_ms.min, 0.0);
  // A chunk lasts 20 ticks and is replaced once 10 are left: 100 ticks need about 9 more chunks.
  EXPECT_GE(r.chunks, 8);
  // The last chunk may have been computed but not switched to before the loop ended.
  EXPECT_GE(r.inference_ms.n, static_cast<size_t>(r.chunks));
  EXPECT_LE(r.inference_ms.n, static_cast<size_t>(r.chunks) + 1);
  // The fake graphs take microseconds, so with 10 ticks (100 ms) of slack nothing runs dry.
  EXPECT_EQ(r.starved_ticks, 0);
  EXPECT_GE(r.chunk_latency_ticks.min, 1.0);  // the switch happens at the first tick after the chunk is ready
  EXPECT_FALSE(r.rt_priority_applied);
  EXPECT_FALSE(r.control_pinned);
}

TEST(ControlLoop, StarvesWhenInferenceIsSlowerThanTheSlack) {
  FakeModel model;
  model.config.chunk_size = 20;
  model.expert_delay = std::chrono::milliseconds(20);  // 4 steps: at least 80 ms per chunk
  auto built = model.build(std::make_shared<HostDevice>());
  Inputs in(model.config);
  ControlLoopOptions options;
  options.rate_hz = 100;
  options.ticks = 120;
  options.refill_threshold = 2;  // asks for the next chunk with 20 ms of actions left
  const ControlLoopReport r = run_control_loop(*built.runtime, in.obs, in.noise.data(), options);

  EXPECT_GT(r.starved_ticks, 0);
  EXPECT_GE(r.inference_ms.min, 80.0);
  EXPECT_GE(r.chunks, 1);
  // A chunk arrives 8 ticks after its request at the earliest; allow for a control thread that woke late.
  EXPECT_GE(r.chunk_latency_ticks.min, 4.0);
}

TEST(ControlLoop, RejectsBadOptions) {
  FakeModel model;
  auto built = model.build(std::make_shared<HostDevice>());
  Inputs in(model.config);
  ControlLoopOptions options;
  options.rate_hz = 0;
  EXPECT_THROW(run_control_loop(*built.runtime, in.obs, in.noise.data(), options), std::invalid_argument);
  options.rate_hz = 50;
  options.refill_threshold = model.config.chunk_size;
  EXPECT_THROW(run_control_loop(*built.runtime, in.obs, in.noise.data(), options), std::invalid_argument);
}

// An allocation failure must reach the caller. Thrown while the inference thread is running, it would unwind
// past a joinable std::thread and end the process in std::terminate.
TEST(ControlLoop, AllocationFailureReachesTheCaller) {
  FakeModel model;
  auto built = model.build(std::make_shared<HostDevice>());
  Inputs in(model.config);
  ControlLoopOptions options;
  options.ticks = std::numeric_limits<int64_t>::max();  // more per-tick records than a vector can hold
  EXPECT_THROW(run_control_loop(*built.runtime, in.obs, in.noise.data(), options), std::length_error);
}

}  // namespace
}  // namespace smolvla_rt
