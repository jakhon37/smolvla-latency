// Manifest::read, the C++ loader's first step: what it accepts from split_manifest.json and what it reports.
#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "smolvla_rt/manifest.h"

namespace smolvla_rt {
namespace {

using json = nlohmann::ordered_json;

// The config export_split.py writes for the tiny test policy of tests/cpp_parity.sh.
json tiny_config() {
  return {{"camera_keys", {"observation.images.image", "observation.images.image2"}},
          {"image_hw", {512, 512}},
          {"image_tokens", 64},
          {"lang_len", 48},
          {"state_dim", 8},
          {"max_state_dim", 32},
          {"action_dim", 7},
          {"max_action_dim", 32},
          {"chunk_size", 50},
          {"num_steps", 10},
          {"num_layers", 4},
          {"expert_layers", 4},
          {"kv_heads", 2},
          {"head_dim", 64},
          {"prefix_len", 177}};
}

// Writes <TempDir>/<name>/split_manifest.json with `config` and an fp32 vision graph; returns the directory.
// `edit` may corrupt the manifest first.
std::string write_manifest(const std::string& name, const json& config,
                           const std::function<void(json&)>& edit = nullptr) {
  const std::string dir = ::testing::TempDir() + "/" + name;
  std::filesystem::create_directories(dir);
  const json vision = {{"file", "vision_fp32.onnx"},
                       {"inputs", {{{"name", "pixel_values"}, {"dtype", "float32"}, {"shape", {1, 3, 512, 512}}}}},
                       {"outputs", {{{"name", "image_tokens"}, {"dtype", "float32"}, {"shape", {1, 64, 192}}}}}};
  json manifest = {
      {"format", "smolvla-split"}, {"version", 1}, {"config", config}, {"variants", {{"fp32", {{"vision", vision}}}}}};
  if (edit) edit(manifest);
  std::ofstream(dir + "/split_manifest.json") << manifest.dump();
  return dir;
}

TEST(Manifest, ReadsTheExportIndex) {
  const Manifest m = Manifest::read(write_manifest("manifest_ok", tiny_config()));
  EXPECT_EQ(m.config().kv_heads, 2);
  EXPECT_EQ(m.config().expert_layers, 4);
  EXPECT_EQ(m.precisions(), std::vector<std::string>{"fp32"});
  EXPECT_EQ(m.graph("fp32", "vision").outputs.at(0), (TensorSpec{"image_tokens", DType::kFloat32, {1, 64, 192}}));
  EXPECT_EQ(m.path("fp32", "vision"), m.dir() + "/vision_fp32.onnx");
  EXPECT_THROW(m.graph("fp16", "vision"), std::out_of_range);
}

// export_split.py refuses checkpoints with these SmolVLA settings. An export made with them anyway would describe
// itself like this, and the loader has to say which setting it is rather than fail on a size later.
TEST(Manifest, NamesTheSmolVLASettingAnUnsupportedExportUses) {
  json special_tokens = tiny_config();
  special_tokens["prefix_len"] = 2 * (64 + 3) + 48 + 1;  // [fake, global] image [fake] for each camera
  json padded = tiny_config();
  padded["prefix_len"] = 200;
  json fewer_expert_layers = tiny_config();
  fewer_expert_layers["expert_layers"] = 2;
  const std::vector<std::tuple<std::string, json, std::string>> cases = {
      {"manifest_special_tokens", special_tokens, "add_image_special_tokens=True"},
      {"manifest_prefix_length", padded, "prefix_length > 0"},
      {"manifest_expert_layers", fewer_expert_layers, "the action expert has 2 layers for 4 VLM layers"}};
  for (const auto& [name, config, fragment] : cases) {
    const std::string dir = write_manifest(name, config);
    try {
      Manifest::read(dir);
      ADD_FAILURE() << name << " was accepted";
    } catch (const std::runtime_error& e) {
      const std::string error = e.what();
      EXPECT_NE(error.find(dir + "/split_manifest.json"), std::string::npos) << error;
      EXPECT_NE(error.find(fragment), std::string::npos) << error;
    }
  }
}

TEST(Manifest, RejectsAManifestWithoutExpertLayers) {
  json config = tiny_config();
  config.erase("expert_layers");
  EXPECT_THROW(Manifest::read(write_manifest("manifest_no_expert_layers", config)), std::runtime_error);
}

// Fields of the wrong JSON type get a std::runtime_error naming the file, not a JSON library exception.
TEST(Manifest, RejectsWronglyTypedFields) {
  const std::vector<std::tuple<std::string, std::function<void(json&)>, std::string>> cases = {
      {"manifest_array", [](json& m) { m = json::array({1, 2}); }, "not a version 1 smolvla-split manifest"},
      {"manifest_format_number", [](json& m) { m["format"] = 7; }, "not a version 1 smolvla-split manifest"},
      {"manifest_version_wraps", [](json& m) { m["version"] = (uint64_t{1} << 32) + 1; },
       "not a version 1 smolvla-split manifest"},
      {"manifest_variants_array", [](json& m) { m["variants"] = json::array(); }, "\"variants\" must be an object"},
      {"manifest_graphs_array", [](json& m) { m["variants"]["fp32"] = json::array({1}); },
       "variants.fp32 must be an object, not array"},
      {"manifest_file_number", [](json& m) { m["variants"]["fp32"]["vision"]["file"] = 5; },
       "fp32/vision: \"file\" must be a string, not number"},
      {"manifest_inputs_object", [](json& m) { m["variants"]["fp32"]["vision"]["inputs"] = json::object(); },
       "fp32/vision: \"inputs\" must be an array, not object"},
  };
  for (const auto& [name, edit, fragment] : cases) {
    const std::string dir = write_manifest(name, tiny_config(), edit);
    try {
      Manifest::read(dir);
      ADD_FAILURE() << name << " was accepted";
    } catch (const std::runtime_error& e) {
      const std::string error = e.what();
      EXPECT_NE(error.find(dir + "/split_manifest.json"), std::string::npos) << error;
      EXPECT_NE(error.find(fragment), std::string::npos) << error;
    }
  }
}

}  // namespace
}  // namespace smolvla_rt
