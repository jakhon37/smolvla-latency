#include "smolvla_rt/control_loop.h"

#include <pthread.h>
#include <sched.h>
#include <time.h>

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

namespace smolvla_rt {

namespace {

constexpr int64_t kNsPerSec = 1000000000;

int64_t monotonic_ns() {
  timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return static_cast<int64_t>(ts.tv_sec) * kNsPerSec + ts.tv_nsec;
}

void sleep_until(int64_t deadline_ns) {
  timespec ts;
  ts.tv_sec = static_cast<time_t>(deadline_ns / kNsPerSec);
  ts.tv_nsec = static_cast<long>(deadline_ns % kNsPerSec);
  while (clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &ts, nullptr) == EINTR) {
  }
}

bool pin_to_cpu(int cpu) {
  if (cpu < 0) return false;
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(cpu, &set);
  return pthread_setaffinity_np(pthread_self(), sizeof set, &set) == 0;
}

// The chunk handoff between the two threads. The control thread only asks when the state is kIdle, the
// inference thread writes the buffer the control thread is not reading and publishes it with kReady, and the
// control thread switches buffers and returns to kIdle. Two buffers are enough because no new request is made
// before the previous chunk has been switched to.
enum class Handoff : int { kIdle, kRequested, kReady };

}  // namespace

ControlLoopReport run_control_loop(SmolVLARuntime& runtime, const Observation& obs, const float* noise,
                                   const ControlLoopOptions& options) {
  if (!(options.rate_hz > 0) || options.ticks <= 0) {
    throw std::invalid_argument("control loop: rate and ticks must be positive");
  }
  const SplitConfig& c = runtime.config();
  const auto chunk_size = c.chunk_size;
  const auto action_dim = static_cast<size_t>(c.action_dim);
  const int64_t threshold = options.refill_threshold < 0 ? chunk_size / 2 : options.refill_threshold;
  if (threshold >= chunk_size) {
    throw std::invalid_argument("control loop: refill threshold must be below the chunk size");
  }
  const auto period_ns = static_cast<int64_t>(std::llround(1e9 / options.rate_hz));

  ControlLoopReport report;
  report.ticks = options.ticks;
  report.period_ms = static_cast<double>(period_ns) / 1e6;
  report.refill_threshold = threshold;

  std::vector<float> chunks[2] = {std::vector<float>(static_cast<size_t>(chunk_size) * action_dim),
                                  std::vector<float>(static_cast<size_t>(chunk_size) * action_dim)};
  runtime.predict_action_chunk(obs, noise, chunks[0].data());

  // Everything the control thread records is allocated before the clock starts, and before the inference thread
  // exists: an allocation failure unwinding past a joinable std::thread would end in std::terminate.
  std::vector<double> wake_latency_ms, period_error_ms, chunk_latency_ticks;
  wake_latency_ms.reserve(static_cast<size_t>(options.ticks));
  period_error_ms.reserve(static_cast<size_t>(options.ticks));
  chunk_latency_ticks.reserve(static_cast<size_t>(options.ticks));

  std::atomic<Handoff> handoff{Handoff::kIdle};
  std::atomic<int> target{1};
  std::atomic<bool> stop{false};
  std::mutex mutex;
  std::condition_variable wake;
  std::vector<double> inference_ms;
  std::exception_ptr inference_error;

  std::thread inference([&] {
    report.inference_pinned = pin_to_cpu(options.inference_cpu);
    for (;;) {
      {
        std::unique_lock<std::mutex> lock(mutex);
        wake.wait(lock, [&] { return stop.load() || handoff.load(std::memory_order_acquire) == Handoff::kRequested; });
      }
      if (stop.load()) return;
      const auto start = std::chrono::steady_clock::now();
      try {
        runtime.predict_action_chunk(obs, noise, chunks[target.load(std::memory_order_relaxed)].data());
      } catch (...) {
        inference_error = std::current_exception();
        return;
      }
      inference_ms.push_back(
          std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
      handoff.store(Handoff::kReady, std::memory_order_release);
    }
  });

  std::thread control([&] {
    report.control_pinned = pin_to_cpu(options.control_cpu);
    if (options.rt_priority > 0) {
      sched_param param{};
      param.sched_priority = options.rt_priority;
      report.rt_priority_applied = pthread_setschedparam(pthread_self(), SCHED_FIFO, &param) == 0;
    }
    int active = 0;
    int64_t next_action = 0;                          // index into the active chunk of the action for the current tick
    int64_t request_tick = -1;                        // tick whose observation the pending chunk is computed from
    [[maybe_unused]] volatile float last_action = 0;  // stands in for sending the action to the robot
    const int64_t start = monotonic_ns() + 2 * period_ns;
    int64_t previous_wake = 0;
    for (int64_t tick = 0; tick < options.ticks; ++tick) {
      const int64_t due = start + tick * period_ns;
      sleep_until(due);
      const int64_t woke = monotonic_ns();
      wake_latency_ms.push_back(static_cast<double>(woke - due) / 1e6);
      if (tick > 0) period_error_ms.push_back(std::fabs(static_cast<double>(woke - previous_wake - period_ns)) / 1e6);
      if (woke - due > period_ns) ++report.missed_deadlines;
      previous_wake = woke;

      if (handoff.load(std::memory_order_acquire) == Handoff::kReady) {
        active = 1 - active;
        next_action = tick - request_tick;  // the chunk starts at the observation's tick
        chunk_latency_ticks.push_back(static_cast<double>(tick - request_tick));
        ++report.chunks;
        handoff.store(Handoff::kIdle, std::memory_order_release);
      }
      if (next_action < chunk_size) {
        last_action = chunks[active][static_cast<size_t>(next_action) * action_dim];
        ++next_action;
      } else {
        ++report.starved_ticks;  // a robot would hold its last command
      }
      if (chunk_size - next_action <= threshold && handoff.load(std::memory_order_acquire) == Handoff::kIdle) {
        request_tick = tick;
        target.store(1 - active, std::memory_order_relaxed);
        {
          std::lock_guard<std::mutex> lock(mutex);
          handoff.store(Handoff::kRequested, std::memory_order_release);
        }
        wake.notify_one();
      }
    }
  });

  control.join();
  {
    std::lock_guard<std::mutex> lock(mutex);
    stop.store(true);
  }
  wake.notify_one();
  inference.join();
  if (inference_error) std::rethrow_exception(inference_error);

  report.wake_latency_ms = summarize(std::move(wake_latency_ms));
  report.period_error_ms = summarize(std::move(period_error_ms));
  report.inference_ms = summarize(std::move(inference_ms));
  report.chunk_latency_ticks = summarize(std::move(chunk_latency_ticks));
  return report;
}

}  // namespace smolvla_rt
