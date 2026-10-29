// smolvla_rt_bench: run the split SmolVLA graphs from C++ on a bundle's observation, check the action chunk
// against the bundle's PyTorch reference, and time it.
//
//   smolvla_rt_bench --onnx-dir onnx_split --bundle onnx_split/bundle.bin [--precision fp16] [--iters 50]
//                    [--max-abs 1e-4] [--control-loop-ticks 10000] [--out report.json]
//
// The bundle comes from tools/dump_bundle.py and the graphs from export_split.py (docs/SPLIT_GRAPH.md).
// Exit status: 0 on success, 1 if a --max-abs / --min-cosine check fails, 2 on bad usage, 3 on any other error.

#include <sched.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>
#include <map>
#include <nlohmann/json.hpp>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "smolvla_rt/build_config.h"
#include "smolvla_rt/bundle.h"
#include "smolvla_rt/control_loop.h"
#include "smolvla_rt/loader.h"
#include "smolvla_rt/manifest.h"
#include "smolvla_rt/stats.h"

namespace {

using json = nlohmann::ordered_json;
using namespace smolvla_rt;

constexpr const char* kUsage = R"(usage: smolvla_rt_bench --onnx-dir DIR --bundle FILE [options]

Runs the split SmolVLA graphs from export_split.py on the observation in a tools/dump_bundle.py bundle,
compares the action chunk with the bundle's PyTorch reference, and times the pipeline.

  --onnx-dir DIR           export_split.py output (split_manifest.json and the graphs)
  --bundle FILE            tools/dump_bundle.py output for the same export
  --precision P            fp32 (default) or fp16
  --backend B              ort (default) or trt
  --provider P             ort: cpu (default), cuda, or auto for CUDA when the library has it
  --threads N              ort: intra-op threads per graph; 0 (default) lets ONNX Runtime choose
  --no-spin                ort: do not let idle worker threads spin
  --engine-dir DIR         trt: directory of <graph>_<precision>.engine files (default: --onnx-dir)
  --device N               CUDA device (default 0)
  --warmup N               untimed chunks before timing (default 5)
  --iters N                timed chunks (default 50; 0 checks parity only)
  --max-abs X              exit 1 if the action chunk differs from the bundle's by more than X; default 1e-4
                           for fp32, none for fp16, a negative X turns it off. The tests gate at 1e-5 on the
                           CPU; GPU kernels sum in another order than PyTorch's CPU reference, hence the margin
  --min-cosine X           exit 1 if its cosine similarity with the bundle's is below X
  --control-loop-ticks N   then run the control-loop harness for N ticks (0, the default, skips it)
  --rate HZ                control-loop rate (default 50)
  --refill N               request the next chunk when N actions are left (default: half a chunk)
  --control-cpu N          pin the control thread to CPU N
  --inference-cpu N        pin the inference thread to CPU N. Neither pins ONNX Runtime's worker threads: with
                           --threads 0 it pins them across the cores itself, ignoring taskset. With --threads 1
                           it has none and runs the graphs on the inference thread
  --rt-priority N          SCHED_FIFO priority for the control thread (needs CAP_SYS_NICE)
  --out FILE               write the JSON report here
)";

constexpr double kFp32MaxAbs = 1e-4;  // the default --max-abs for fp32; see kUsage

struct Args {
  std::string onnx_dir, bundle, out, engine_dir;
  LoadOptions load;
  int warmup = 5, iters = 50;
  std::optional<double> max_abs;
  double min_cosine = -2;
  ControlLoopOptions control;
  bool control_loop = false;
};

[[noreturn]] void usage_error(const std::string& message) {
  std::cerr << "smolvla_rt_bench: " << message << "\n\n" << kUsage;
  std::exit(2);
}

Args parse_args(int argc, char** argv) {
  Args a;
  auto value = [&](int& i) -> std::string {
    if (i + 1 >= argc) usage_error(std::string(argv[i]) + " needs a value");
    return argv[++i];
  };
  auto number = [&](int& i) -> double {
    const std::string flag = argv[i], v = value(i);
    char* end = nullptr;
    const double x = std::strtod(v.c_str(), &end);
    if (end == v.c_str() || *end != '\0') usage_error(flag + " expects a number, got '" + v + "'");
    return x;
  };
  auto integer = [&](int& i) -> int {
    const std::string flag = argv[i];
    const double x = number(i);
    if (x != std::floor(x)) usage_error(flag + " expects an integer");
    return static_cast<int>(x);
  };
  for (int i = 1; i < argc; ++i) {
    const std::string flag = argv[i];
    if (flag == "-h" || flag == "--help") {
      std::cout << kUsage;
      std::exit(0);
    } else if (flag == "--onnx-dir") {
      a.onnx_dir = value(i);
    } else if (flag == "--bundle") {
      a.bundle = value(i);
    } else if (flag == "--precision") {
      a.load.precision = value(i);
    } else if (flag == "--backend") {
      try {
        a.load.backend = parse_backend(value(i));
      } catch (const std::invalid_argument& e) {
        usage_error(e.what());
      }
    } else if (flag == "--provider") {
      a.load.ort.provider = value(i);
    } else if (flag == "--threads") {
      a.load.ort.intra_op_threads = integer(i);
    } else if (flag == "--no-spin") {
      a.load.ort.allow_spinning = false;
    } else if (flag == "--engine-dir") {
      a.load.engine_dir = value(i);
    } else if (flag == "--device") {
      a.load.cuda_device = a.load.ort.cuda_device = integer(i);
    } else if (flag == "--warmup") {
      a.warmup = integer(i);
    } else if (flag == "--iters") {
      a.iters = integer(i);
    } else if (flag == "--max-abs") {
      a.max_abs = number(i);
    } else if (flag == "--min-cosine") {
      a.min_cosine = number(i);
    } else if (flag == "--control-loop-ticks") {
      a.control.ticks = integer(i);
      a.control_loop = a.control.ticks > 0;
    } else if (flag == "--rate") {
      a.control.rate_hz = number(i);
    } else if (flag == "--refill") {
      a.control.refill_threshold = integer(i);
    } else if (flag == "--control-cpu") {
      a.control.control_cpu = integer(i);
    } else if (flag == "--inference-cpu") {
      a.control.inference_cpu = integer(i);
    } else if (flag == "--rt-priority") {
      a.control.rt_priority = integer(i);
    } else if (flag == "--out") {
      a.out = value(i);
    } else {
      usage_error("unknown argument '" + flag + "'");
    }
  }
  if (a.onnx_dir.empty() || a.bundle.empty()) usage_error("--onnx-dir and --bundle are required");
  if (a.warmup < 0 || a.iters < 0) usage_error("--warmup and --iters must not be negative");
  if (a.control_loop && (a.control.control_cpu >= 0 || a.control.inference_cpu >= 0) &&
      a.load.backend == Backend::kOnnxRuntime && a.load.ort.intra_op_threads != 1) {
    std::cerr << "smolvla_rt_bench: warning: --control-cpu and --inference-cpu do not pin ONNX Runtime's worker "
                 "threads, which may run on any CPU; pass --threads 1 to run the graphs on the inference thread\n";
  }
  return a;
}

json to_json(const Summary& s) {
  return {{"p50", s.p50}, {"p90", s.p90}, {"p99", s.p99}, {"mean", s.mean}, {"min", s.min}, {"max", s.max}, {"n", s.n}};
}

json to_json(const Diff& d) {
  return {{"max_abs", d.max_abs},
          {"mean_abs", d.mean_abs},
          {"rel_l2", d.rel_l2},
          {"cosine", d.cosine},
          {"nonfinite", d.nonfinite}};
}

json to_json(const SplitConfig& c) {
  return {{"camera_keys", c.camera_keys},   {"image_hw", {c.image_height, c.image_width}},
          {"image_tokens", c.image_tokens}, {"lang_len", c.lang_len},
          {"state_dim", c.state_dim},       {"max_state_dim", c.max_state_dim},
          {"action_dim", c.action_dim},     {"max_action_dim", c.max_action_dim},
          {"chunk_size", c.chunk_size},     {"num_steps", c.num_steps},
          {"num_layers", c.num_layers},     {"expert_layers", c.expert_layers},
          {"kv_heads", c.kv_heads},         {"head_dim", c.head_dim},
          {"prefix_len", c.prefix_len}};
}

std::vector<double> bundle_doubles(const Bundle& bundle, const std::string& name) {
  const BundleTensor& t = bundle.tensor(name);
  return to_doubles(t.data, t.spec.dtype, static_cast<size_t>(element_count(t.spec.shape)));
}

std::vector<double> runtime_doubles(const SmolVLARuntime& rt, const std::string& name) {
  const TensorSpec& spec = rt.tensor_spec(name);
  std::vector<unsigned char> host(spec.bytes());
  rt.copy_to_host(name, host.data());
  return to_doubles(host.data(), spec.dtype, static_cast<size_t>(element_count(spec.shape)));
}

void append(std::vector<double>& to, const std::vector<double>& from) { to.insert(to.end(), from.begin(), from.end()); }

std::string cpu_model() {
  std::ifstream in("/proc/cpuinfo");
  for (std::string line; std::getline(in, line);) {
    if (line.rfind("model name", 0) == 0) return line.substr(line.find(':') + 2);
  }
  return "unknown";
}

json system_info() {
  cpu_set_t set;
  const int usable = sched_getaffinity(0, sizeof set, &set) == 0 ? CPU_COUNT(&set) : -1;
  char now[32];
  const std::time_t t = std::time(nullptr);
  std::strftime(now, sizeof now, "%Y-%m-%dT%H:%M:%SZ", std::gmtime(&t));
  return {{"cpu", cpu_model()},
          {"hardware_threads", std::thread::hardware_concurrency()},
          {"usable_cpus", usable},
          {"compiler", __VERSION__},
#ifdef NDEBUG
          {"assertions", false},
#else
          {"assertions", true},
#endif
          {"smolvla_rt", SMOLVLA_RT_VERSION},
          {"utc", now}};
}

// A bundle's observation and noise; they point into the bundle.
struct Problem {
  Observation obs;
  const float* noise = nullptr;
};

// Compares the chunk, and where the bundle has LeRobot's intermediate tensors, every stage of the pipeline with
// them. Stage values come from this run end to end, so a stage's error includes what upstream stages passed on.
json check_parity(SmolVLARuntime& rt, const Bundle& bundle, const Problem& problem, const std::vector<float>& actions) {
  const SplitConfig& c = rt.config();
  json parity;
  if (bundle.contains("actions")) {
    std::vector<double> out(actions.begin(), actions.end());
    parity["actions"] = to_json(compare(out, bundle_doubles(bundle, "actions")));
  }
  if (!bundle.contains("ref.image_tokens_0")) return parity;

  // Rerun with an observer to see every step's x_t and v_t.
  std::vector<double> x_steps, v_steps;
  const auto n = static_cast<size_t>(c.chunk_size * c.max_action_dim);
  rt.set_step_observer([&](int64_t, const float* x, const float* v) {
    x_steps.insert(x_steps.end(), x, x + n);
    v_steps.insert(v_steps.end(), v, v + n);
  });
  std::vector<float> rerun(actions.size());
  rt.predict_action_chunk(problem.obs, problem.noise, rerun.data());
  rt.set_step_observer(nullptr);
  parity["rerun_identical"] = rerun == actions;

  json stages;
  for (int64_t cam = 0; cam < c.num_cameras(); ++cam) {
    const std::string name = "image_tokens_" + std::to_string(cam);
    stages[name] = to_json(compare(runtime_doubles(rt, name), bundle_doubles(bundle, "ref." + name)));
  }
  std::vector<double> kv, kv_ref;
  for (const auto& name : kv_names(c.num_layers)) {
    append(kv, runtime_doubles(rt, name));
    append(kv_ref, bundle_doubles(bundle, "ref." + name));
  }
  stages["past_key_values"] = to_json(compare(kv, kv_ref));
  stages["prefix_pad_mask_equal"] =
      runtime_doubles(rt, "prefix_pad_mask") == bundle_doubles(bundle, "ref.prefix_pad_mask");
  std::vector<double> x_ref, v_ref;
  for (int64_t k = 0; k < c.num_steps; ++k) {
    append(x_ref, bundle_doubles(bundle, "ref.x_t_" + std::to_string(k)));
    append(v_ref, bundle_doubles(bundle, "ref.v_t_" + std::to_string(k)));
  }
  stages["x_t_all_steps"] = to_json(compare(x_steps, x_ref));
  stages["v_t_all_steps"] = to_json(compare(v_steps, v_ref));
  parity["pipeline_vs_lerobot"] = stages;
  return parity;
}

// Whether the bundle's meta.euler is the runtime's schedule exactly; null if the bundle records none. Bundle::read
// has checked that meta.euler is {"dt": number, "t": [number, ...]}.
json euler_check(const SmolVLARuntime& rt, const Bundle& bundle) {
  const json meta = json::parse(bundle.meta_json());
  if (!meta.contains("euler")) return nullptr;
  const json& dt = meta.at("euler").at("dt");
  const json& t = meta.at("euler").at("t");
  const EulerSchedule& s = rt.schedule();
  bool same = dt.get<double>() == static_cast<double>(s.dt) && t.size() == s.t.size();
  for (size_t k = 0; same && k < s.t.size(); ++k) same = t.at(k).get<double>() == static_cast<double>(s.t[k]);
  return same;
}

void print_latency(const std::string& name, const Summary& s) {
  std::printf("  %-12s p50 %9.3f ms  p99 %9.3f ms  (n=%zu)\n", name.c_str(), s.p50, s.p99, s.n);
}

int run(const Args& args) {
  const Manifest manifest = Manifest::read(args.onnx_dir);
  const Bundle bundle = Bundle::read(args.bundle);
  const std::optional<SplitConfig>& bundle_config = bundle.config();
  if (!bundle_config) throw std::runtime_error(args.bundle + " has no meta.config");
  if (*bundle_config != manifest.config()) {
    throw std::runtime_error(args.bundle + " was written for another policy or export than " + args.onnx_dir);
  }
  auto rt = load_runtime(manifest, args.load);
  const SplitConfig& c = rt->config();
  const Problem problem{bundle_observation(bundle, c), bundle_noise(bundle, c)};

  std::vector<float> actions(static_cast<size_t>(c.chunk_size * c.action_dim));
  rt->predict_action_chunk(problem.obs, problem.noise, actions.data());

  json report;
  report["tool"] = "smolvla_rt_bench";
  report["onnx_dir"] = args.onnx_dir;
  report["bundle"] = args.bundle;
  report["precision"] = args.load.precision;
  report["backend"] = rt->backend();
  report["threads"] = args.load.ort.intra_op_threads;
  report["allow_spinning"] = args.load.ort.allow_spinning;
  report["config"] = to_json(c);
  report["system"] = system_info();
  report["euler_schedule_matches_bundle"] = euler_check(*rt, bundle);
  report["parity"] = check_parity(*rt, bundle, problem, actions);
  const double max_abs = args.max_abs.value_or(args.load.precision == "fp32" ? kFp32MaxAbs : -1);
  report["max_abs_gate"] = max_abs >= 0 ? json(max_abs) : json(nullptr);

  std::printf("%s action chunk [1, %lld, %lld] via %s\n", args.load.precision.c_str(),
              static_cast<long long>(c.chunk_size), static_cast<long long>(c.action_dim), rt->backend().c_str());
  bool ok = true;
  if (report["parity"].contains("actions")) {
    const json& a = report["parity"]["actions"];
    const double chunk_max_abs = a["max_abs"], cosine = a["cosine"];
    std::printf("  vs bundle actions  max_abs %.2e  cos %.8f  nonfinite %lld\n", chunk_max_abs, cosine,
                static_cast<long long>(a["nonfinite"].get<int64_t>()));
    if (max_abs >= 0 && !(chunk_max_abs <= max_abs)) {
      std::printf("  FAIL: max_abs above %.2e\n", max_abs);
      ok = false;
    }
    if (args.min_cosine > -2 && !(cosine >= args.min_cosine)) {
      std::printf("  FAIL: cosine below %.8f\n", args.min_cosine);
      ok = false;
    }
  } else if (args.max_abs.value_or(-1) >= 0 || args.min_cosine > -2) {
    throw std::runtime_error(args.bundle + " has no reference actions to check against");
  }

  if (args.iters > 0) {
    for (int i = 0; i < args.warmup; ++i) rt->predict_action_chunk(problem.obs, problem.noise, actions.data());
    std::map<std::string, std::vector<double>> samples;
    for (int i = 0; i < args.iters; ++i) {
      rt->predict_action_chunk(problem.obs, problem.noise, actions.data());
      const ChunkTimings& t = rt->last_timings();
      samples["chunk"].push_back(t.total);
      append(samples["vision"], t.vision);
      samples["prefix"].push_back(t.prefix);
      append(samples["expert_step"], t.expert_step);
      samples["inputs"].push_back(t.inputs);
      samples["euler"].push_back(t.euler);
    }
    json latency;
    for (const char* name : {"chunk", "vision", "prefix", "expert_step", "inputs", "euler"}) {
      const Summary s = summarize(samples[name]);
      latency[name] = to_json(s);
      print_latency(name, s);
    }
    report["warmup"] = args.warmup;
    report["iters"] = args.iters;
    report["latency_ms"] = latency;
  }

  if (args.control_loop) {
    const ControlLoopReport r = run_control_loop(*rt, problem.obs, problem.noise, args.control);
    report["control_loop"] = {{"ticks", r.ticks},
                              {"period_ms", r.period_ms},
                              {"refill_threshold", r.refill_threshold},
                              {"wake_latency_ms", to_json(r.wake_latency_ms)},
                              {"period_error_ms", to_json(r.period_error_ms)},
                              {"missed_deadlines", r.missed_deadlines},
                              {"missed_per_10k_ticks", r.missed_per_10k()},
                              {"starved_ticks", r.starved_ticks},
                              {"starved_per_10k_ticks", r.starved_per_10k()},
                              {"chunks", r.chunks},
                              {"inference_ms", to_json(r.inference_ms)},
                              {"chunk_latency_ticks", to_json(r.chunk_latency_ticks)},
                              {"rt_priority_requested", args.control.rt_priority},
                              {"rt_priority_applied", r.rt_priority_applied},
                              {"control_pinned", r.control_pinned},
                              {"inference_pinned", r.inference_pinned},
                              {"note", "soft real-time on a stock Linux kernel: no bound on the worst case"}};
    std::printf(
        "control loop %.0f Hz, %lld ticks: period error p99 %.3f ms, wake latency p99 %.3f ms (max %.3f), "
        "missed %.1f / 10k ticks, starved %.1f / 10k ticks, %lld chunks (inference p50 %.1f ms)\n",
        args.control.rate_hz, static_cast<long long>(r.ticks), r.period_error_ms.p99, r.wake_latency_ms.p99,
        r.wake_latency_ms.max, r.missed_per_10k(), r.starved_per_10k(), static_cast<long long>(r.chunks),
        r.inference_ms.p50);
  }

  report["passed"] = ok;
  if (!args.out.empty()) {
    std::ofstream out(args.out);
    out << report.dump(1) << "\n";
    if (!out) throw std::runtime_error("cannot write " + args.out);
    std::printf("wrote %s\n", args.out.c_str());
  }
  return ok ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
  const Args args = parse_args(argc, argv);
  try {
    return run(args);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "smolvla_rt_bench: %s\n", e.what());
    return 3;
  }
}
