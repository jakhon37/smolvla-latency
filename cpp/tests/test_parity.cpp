// End to end: the C++ runtime on ONNX Runtime against LeRobot's PyTorch outputs stored in a bundle.
//
// Needs SMOLVLA_RT_FIXTURE_DIR, an export_split.py output directory holding bundle.bin from tools/dump_bundle.py
// for the same checkpoint; tests/cpp_parity.sh builds one from the tiny random test policy. Without it the
// tests are skipped.
#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

#include "smolvla_rt/bundle.h"
#include "smolvla_rt/loader.h"
#include "smolvla_rt/manifest.h"
#include "smolvla_rt/stats.h"

namespace smolvla_rt {
namespace {

// As tests/test_split_graph.py asserts for ort_flow.py: the fp32 action chunk on the CPU, and each stage.
constexpr double kChunkTolerance = 1e-5;
constexpr double kStageTolerance = 1e-4;

struct Fixture {
  std::string dir;
  std::unique_ptr<Manifest> manifest;
  std::unique_ptr<Bundle> bundle;
};

// Loaded once for all tests; nullptr when SMOLVLA_RT_FIXTURE_DIR is unset.
const Fixture* fixture() {
  static const std::unique_ptr<Fixture> loaded = []() -> std::unique_ptr<Fixture> {
    const char* dir = std::getenv("SMOLVLA_RT_FIXTURE_DIR");
    if (!dir || !*dir) return nullptr;
    auto f = std::make_unique<Fixture>();
    f->dir = dir;
    f->manifest = std::make_unique<Manifest>(Manifest::read(dir));
    f->bundle = std::make_unique<Bundle>(Bundle::read(f->dir + "/bundle.bin"));
    return f;
  }();
  return loaded.get();
}

#define REQUIRE_FIXTURE()                                                                        \
  if (!fixture()) GTEST_SKIP() << "SMOLVLA_RT_FIXTURE_DIR is not set (see tests/cpp_parity.sh)"; \
  const Fixture& fx = *fixture()

std::unique_ptr<SmolVLARuntime> load(const Fixture& fx, const std::string& precision) {
  LoadOptions options;
  options.precision = precision;
  return load_runtime(*fx.manifest, options);
}

std::vector<float> run(SmolVLARuntime& rt, const Bundle& bundle) {
  const SplitConfig& c = rt.config();
  std::vector<float> actions(static_cast<size_t>(c.chunk_size * c.action_dim));
  rt.predict_action_chunk(bundle_observation(bundle, c), bundle_noise(bundle, c), actions.data());
  return actions;
}

std::vector<double> doubles(const Bundle& bundle, const std::string& name) {
  const BundleTensor& t = bundle.tensor(name);
  return to_doubles(t.data, t.spec.dtype, static_cast<size_t>(element_count(t.spec.shape)));
}

std::vector<double> doubles(const SmolVLARuntime& rt, const std::string& name) {
  const TensorSpec& spec = rt.tensor_spec(name);
  std::vector<unsigned char> host(spec.bytes());
  rt.copy_to_host(name, host.data());
  return to_doubles(host.data(), spec.dtype, static_cast<size_t>(element_count(spec.shape)));
}

TEST(Parity, BundleMatchesTheExport) {
  REQUIRE_FIXTURE();
  ASSERT_TRUE(fx.bundle->config().has_value());
  EXPECT_EQ(*fx.bundle->config(), fx.manifest->config());
  // With one K/V head, K/V read as [seq, heads] instead of [heads, seq] are the same numbers: the parity checks
  // below would not see a layout mistake.
  EXPECT_GT(fx.manifest->config().kv_heads, 1) << "the fixture should have several K/V heads";
  // The Euler schedule the bundle records is the one the C++ loop uses, bit for bit.
  const auto meta = nlohmann::json::parse(fx.bundle->meta_json());
  const EulerSchedule s = euler_schedule(fx.manifest->config().num_steps);
  EXPECT_EQ(meta["euler"]["dt"].get<float>(), s.dt);
  EXPECT_EQ(meta["euler"]["t"].get<std::vector<float>>(), s.t);
}

TEST(Parity, Fp32ActionChunkMatchesPyTorch) {
  REQUIRE_FIXTURE();
  auto rt = load(fx, "fp32");
  const std::vector<float> actions = run(*rt, *fx.bundle);
  const Diff d = compare(std::vector<double>(actions.begin(), actions.end()), doubles(*fx.bundle, "actions"));
  std::printf("fp32 action chunk vs PyTorch: max_abs %.2e  cos %.10f  (%s)\n", d.max_abs, d.cosine,
              rt->backend().c_str());
  EXPECT_LT(d.max_abs, kChunkTolerance);
  EXPECT_EQ(d.nonfinite, 0);
}

// Each stage of the C++ pipeline against LeRobot's value at that split point. The stages run end to end, so
// later ones carry the earlier ones' rounding; the step trajectory is compared through the observer.
TEST(Parity, Fp32StagesMatchLeRobot) {
  REQUIRE_FIXTURE();
  if (!fx.bundle->contains("ref.image_tokens_0")) GTEST_SKIP() << "bundle written with --no-intermediates";
  auto rt = load(fx, "fp32");
  const SplitConfig& c = rt->config();
  std::vector<double> x_steps, v_steps;
  const auto n = static_cast<size_t>(c.chunk_size * c.max_action_dim);
  rt->set_step_observer([&](int64_t, const float* x, const float* v) {
    x_steps.insert(x_steps.end(), x, x + n);
    v_steps.insert(v_steps.end(), v, v + n);
  });
  run(*rt, *fx.bundle);

  for (int64_t cam = 0; cam < c.num_cameras(); ++cam) {
    const std::string name = "image_tokens_" + std::to_string(cam);
    const Diff d = compare(doubles(*rt, name), doubles(*fx.bundle, "ref." + name));
    std::printf("fp32 %-16s max_abs %.2e\n", name.c_str(), d.max_abs);
    EXPECT_LT(d.max_abs, kStageTolerance) << name;
  }
  EXPECT_EQ(doubles(*rt, "prefix_pad_mask"), doubles(*fx.bundle, "ref.prefix_pad_mask"));
  double kv_max = 0;
  for (const auto& name : kv_names(c.num_layers)) {
    kv_max = std::max(kv_max, compare(doubles(*rt, name), doubles(*fx.bundle, "ref." + name)).max_abs);
  }
  std::printf("fp32 %-16s max_abs %.2e\n", "past_key_values", kv_max);
  EXPECT_LT(kv_max, kStageTolerance);
  for (int64_t k = 0; k < c.num_steps; ++k) {
    const std::vector<double> ref_x = doubles(*fx.bundle, "ref.x_t_" + std::to_string(k));
    const std::vector<double> ref_v = doubles(*fx.bundle, "ref.v_t_" + std::to_string(k));
    const auto at = static_cast<std::ptrdiff_t>(static_cast<size_t>(k) * n);
    const Diff dx = compare(
        std::vector<double>(x_steps.begin() + at, x_steps.begin() + at + static_cast<std::ptrdiff_t>(n)), ref_x);
    const Diff dv = compare(
        std::vector<double>(v_steps.begin() + at, v_steps.begin() + at + static_cast<std::ptrdiff_t>(n)), ref_v);
    EXPECT_LT(dx.max_abs, kStageTolerance) << "x_t step " << k;
    EXPECT_LT(dv.max_abs, kStageTolerance) << "v_t step " << k;
  }
}

// LeRobot's own trajectory, fed through the C++ update, gives LeRobot's next x_t bit for bit at every step: the
// update is the same float32 multiply and add. The last update gives the action chunk, the bundle's `actions`.
TEST(Parity, EulerUpdateReproducesLeRobotsTrajectoryExactly) {
  REQUIRE_FIXTURE();
  if (!fx.bundle->contains("ref.x_t_0")) GTEST_SKIP() << "bundle written with --no-intermediates";
  const SplitConfig& c = fx.manifest->config();
  const EulerSchedule s = euler_schedule(c.num_steps);
  const auto n = static_cast<size_t>(c.chunk_size * c.max_action_dim);
  for (int64_t k = 0; k < c.num_steps; ++k) {
    const BundleTensor& x = fx.bundle->tensor("ref.x_t_" + std::to_string(k));
    const BundleTensor& v = fx.bundle->tensor("ref.v_t_" + std::to_string(k));
    std::vector<float> updated(x.as<float>(), x.as<float>() + n);
    euler_update(updated.data(), v.as<float>(), s.dt, n);
    if (k + 1 < c.num_steps) {
      const BundleTensor& next = fx.bundle->tensor("ref.x_t_" + std::to_string(k + 1));
      EXPECT_EQ(updated, std::vector<float>(next.as<float>(), next.as<float>() + n)) << "step " << k;
      continue;
    }
    std::vector<float> chunk;  // x_t[:, :, :action_dim]
    for (int64_t i = 0; i < c.chunk_size; ++i) {
      const auto row = updated.begin() + static_cast<std::ptrdiff_t>(i * c.max_action_dim);
      chunk.insert(chunk.end(), row, row + static_cast<std::ptrdiff_t>(c.action_dim));
    }
    const BundleTensor& actions = fx.bundle->tensor("actions");
    ASSERT_EQ(actions.spec, (TensorSpec{"actions", DType::kFloat32, {1, c.chunk_size, c.action_dim}}));
    EXPECT_EQ(chunk, std::vector<float>(actions.as<float>(), actions.as<float>() + chunk.size())) << "last step";
  }
}

TEST(Parity, RepeatedChunksAreIdentical) {
  REQUIRE_FIXTURE();
  auto rt = load(fx, "fp32");
  const std::vector<float> first = run(*rt, *fx.bundle);
  for (int i = 0; i < 3; ++i) EXPECT_EQ(run(*rt, *fx.bundle), first);
}

TEST(Parity, Fp16ActionChunkIsClose) {
  REQUIRE_FIXTURE();
  if (!fx.manifest->has_precision("fp16")) GTEST_SKIP() << fx.dir << " has no fp16 graphs";
  auto rt = load(fx, "fp16");
  EXPECT_EQ(rt->pixel_dtype(), DType::kFloat16);
  const std::vector<float> actions = run(*rt, *fx.bundle);
  const Diff d = compare(std::vector<double>(actions.begin(), actions.end()), doubles(*fx.bundle, "actions"));
  std::printf("fp16 action chunk vs PyTorch fp32: max_abs %.2e  cos %.10f\n", d.max_abs, d.cosine);
  EXPECT_EQ(d.nonfinite, 0);
  EXPECT_GT(d.cosine, 0.999);  // as tests/test_split_graph.py checks the Python fp16 path
}

TEST(Parity, RejectsAnotherPrecisionOrBackend) {
  REQUIRE_FIXTURE();
  LoadOptions options;
  options.precision = "int8";
  EXPECT_THROW(load_runtime(*fx.manifest, options), std::out_of_range);
  options.precision = "fp32";
  options.ort.provider = "tpu";
  EXPECT_THROW(load_runtime(*fx.manifest, options), std::invalid_argument);
  if (!ort_cuda_available()) {
    options.ort.provider = "cuda";
    EXPECT_THROW(load_runtime(*fx.manifest, options), std::runtime_error);
    options.ort.provider = "auto";  // falls back to the CPU
    EXPECT_NE(load_runtime(*fx.manifest, options)->backend().find("CPUExecutionProvider"), std::string::npos);
  }
  if (!backend_compiled(Backend::kTensorRT)) {
    options.backend = Backend::kTensorRT;
    EXPECT_THROW(load_runtime(*fx.manifest, options), std::runtime_error);
  }
}

}  // namespace
}  // namespace smolvla_rt
