// A fixed-rate control loop fed by action chunks: how a robot would consume SmolVLARuntime's output.
#pragma once

#include <cstdint>

#include "smolvla_rt/runtime.h"
#include "smolvla_rt/stats.h"

namespace smolvla_rt {

struct ControlLoopOptions {
  double rate_hz = 50.0;
  int64_t ticks = 10000;
  // Request the next chunk when this many actions of the current one are left; negative means half a chunk.
  int64_t refill_threshold = -1;
  int control_cpu = -1;    // pin the control thread to this CPU; -1 leaves it to the scheduler
  int inference_cpu = -1;  // same for the inference thread
  int rt_priority = 0;     // SCHED_FIFO priority for the control thread; 0 keeps normal scheduling
};

struct ControlLoopReport {
  int64_t ticks = 0;
  double period_ms = 0;
  int64_t refill_threshold = 0;
  Summary wake_latency_ms;  // when the control thread woke, minus when the tick was scheduled
  Summary period_error_ms;  // |time between consecutive wake-ups - period|
  // Ticks that woke after the next tick was already due.
  int64_t missed_deadlines = 0;
  // Ticks with no action for their time step: the chunk ran out before the next one arrived.
  int64_t starved_ticks = 0;
  int64_t chunks = 0;           // delivered during the loop
  Summary inference_ms;         // predict_action_chunk wall time
  Summary chunk_latency_ticks;  // from requesting a chunk to the tick that switched to it
  bool rt_priority_applied = false;
  bool control_pinned = false;
  bool inference_pinned = false;

  double missed_per_10k() const {
    return ticks ? 1e4 * static_cast<double>(missed_deadlines) / static_cast<double>(ticks) : 0;
  }
  double starved_per_10k() const {
    return ticks ? 1e4 * static_cast<double>(starved_ticks) / static_cast<double>(ticks) : 0;
  }
};

// Runs a control thread at options.rate_hz for options.ticks ticks. Each tick it sleeps until its absolute
// deadline (CLOCK_MONOTONIC), takes the action for that time step from the current chunk, and, when few actions
// are left, asks an inference thread for the next chunk. The new chunk is computed from `obs` and `noise` at
// the requesting tick, and the control thread switches to it at the first tick after it is ready, skipping the
// actions whose time has passed. One chunk is computed before the clock starts.
//
// This is soft real-time: on a stock Linux kernel (not PREEMPT_RT), even with SCHED_FIFO and a pinned CPU,
// nothing bounds the worst-case wake-up latency, and the numbers describe only the machine they ran on.
ControlLoopReport run_control_loop(SmolVLARuntime& runtime, const Observation& obs, const float* noise,
                                   const ControlLoopOptions& options);

}  // namespace smolvla_rt
