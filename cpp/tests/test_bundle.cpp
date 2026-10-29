#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <fstream>
#include <functional>
#include <limits>
#include <nlohmann/json.hpp>
#include <string>
#include <tuple>
#include <vector>

#include "smolvla_rt/bundle.h"

namespace smolvla_rt {
namespace {

using json = nlohmann::ordered_json;

struct RawTensor {
  std::string name, dtype;
  Shape shape;
  std::vector<uint8_t> bytes;
};

template <typename T>
RawTensor raw(const std::string& name, const std::string& dtype, Shape shape, const std::vector<T>& values) {
  RawTensor t{name, dtype, std::move(shape), std::vector<uint8_t>(values.size() * sizeof(T))};
  std::memcpy(t.bytes.data(), values.data(), t.bytes.size());
  return t;
}

// Writes a header and data section as smolvla_latency/bundle.py lays them out; `edit` may corrupt the header.
std::string write_bundle(const std::string& file, const std::vector<RawTensor>& tensors, const json& meta,
                         const std::function<void(json&)>& edit = nullptr) {
  const std::string path = ::testing::TempDir() + "/" + file;
  json header = {{"format", "smolvla-split-bundle"}, {"version", 1}, {"meta", meta}, {"tensors", json::array()}};
  std::vector<uint8_t> data;
  for (const auto& t : tensors) {
    const size_t offset = (data.size() + 63) / 64 * 64;
    data.resize(offset + t.bytes.size());
    std::memcpy(data.data() + offset, t.bytes.data(), t.bytes.size());
    header["tensors"].push_back(
        {{"name", t.name}, {"dtype", t.dtype}, {"shape", t.shape}, {"offset", offset}, {"nbytes", t.bytes.size()}});
  }
  if (edit) edit(header);
  std::string text = header.dump();
  text.append((64 - (16 + text.size()) % 64) % 64, ' ');
  const uint64_t size = text.size();
  std::ofstream out(path, std::ios::binary);
  out.write("SVLABNDL", 8);
  out.write(reinterpret_cast<const char*>(&size), 8);
  out << text;
  out.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
  return path;
}

std::vector<RawTensor> sample_tensors() {
  return {raw<float>("x", "float32", {1, 7}, {0, 1, 2, 3, 4, 5, 6}), raw<uint8_t>("mask", "bool", {1, 3}, {1, 0, 1}),
          raw<int64_t>("ids", "int64", {1, 3}, {1, -2, 3}),
          raw<uint16_t>("half", "float16", {2, 3}, std::vector<uint16_t>(6, 0x3800)),  // 0.5
          raw<float>("t", "float32", {1}, {0.9f})};
}

TEST(Bundle, ReadsWhatBundlePyWrites) {
  const json meta = {{"note", "test"}, {"dims", {1, 2}}, {"euler", {{"dt", -0.5f}, {"t", {1.0f, 0.5f}}}}};
  const Bundle b = Bundle::read(write_bundle("ok.bin", sample_tensors(), meta));
  EXPECT_EQ(json::parse(b.meta_json()), meta);
  EXPECT_FALSE(b.config().has_value());
  ASSERT_EQ(b.tensors().size(), 5u);
  EXPECT_EQ(b.tensors()[0].spec.name, "x");  // file order
  EXPECT_EQ(b.tensor("x").spec, (TensorSpec{"x", DType::kFloat32, {1, 7}}));
  EXPECT_EQ(b.tensor("x").as<float>()[6], 6.0f);
  EXPECT_EQ(b.tensor("mask").as<uint8_t>()[1], 0);
  EXPECT_EQ(b.tensor("ids").as<int64_t>()[1], -2);
  EXPECT_EQ(b.tensor("half").spec.dtype, DType::kFloat16);
  EXPECT_EQ(b.tensor("half").as<uint16_t>()[5], 0x3800);
  EXPECT_EQ(b.tensor("t").as<float>()[0], 0.9f);
  for (const auto& t : b.tensors()) EXPECT_EQ(reinterpret_cast<uintptr_t>(t.data) % 64, 0u) << t.spec.name;
  EXPECT_TRUE(b.contains("ids"));
  EXPECT_FALSE(b.contains("noise"));
  EXPECT_THROW(b.tensor("noise"), std::out_of_range);
}

TEST(Bundle, ParsesTheSplitConfig) {
  const json config = {{"camera_keys", {"observation.images.image", "observation.images.image2"}},
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
  const Bundle b = Bundle::read(write_bundle("config.bin", sample_tensors(), {{"config", config}}));
  ASSERT_TRUE(b.config().has_value());
  EXPECT_EQ(b.config()->num_cameras(), 2);
  EXPECT_EQ(b.config()->image_height, 512);
  EXPECT_EQ(b.config()->prefix_len, 177);
  EXPECT_EQ(b.config()->num_layers, 4);

  json broken = config;
  broken["prefix_len"] = 176;
  EXPECT_THROW(Bundle::read(write_bundle("bad_config.bin", sample_tensors(), {{"config", broken}})),
               std::runtime_error);
}

TEST(Bundle, EmptyBundle) {
  const Bundle b = Bundle::read(write_bundle("empty.bin", {}, json::object()));
  EXPECT_TRUE(b.tensors().empty());
}

TEST(Bundle, RejectsCorruptFiles) {
  const auto tensors = sample_tensors();
  auto expect_error = [](const std::string& path, const std::string& fragment) {
    try {
      Bundle::read(path);
      ADD_FAILURE() << path << " was accepted";
    } catch (const std::runtime_error& e) {
      EXPECT_NE(std::string(e.what()).find(fragment), std::string::npos) << e.what();
    }
  };
  expect_error(::testing::TempDir() + "/does_not_exist.bin", "cannot open");
  expect_error(write_bundle("misaligned.bin", tensors, json::object(), [](json& h) { h["tensors"][1]["offset"] = 65; }),
               "aligned");
  expect_error(write_bundle("nbytes.bin", tensors, json::object(), [](json& h) { h["tensors"][0]["nbytes"] = 24; }),
               "bytes");
  expect_error(
      write_bundle("past_end.bin", tensors, json::object(), [](json& h) { h["tensors"][4]["offset"] = 1 << 20; }),
      "past the end");
  expect_error(
      write_bundle("dtype.bin", tensors, json::object(), [](json& h) { h["tensors"][0]["dtype"] = "float64"; }),
      "float64");
  expect_error(write_bundle("version.bin", tensors, json::object(), [](json& h) { h["version"] = 2; }), "unsupported");
  expect_error(write_bundle("duplicate.bin", tensors, json::object(), [](json& h) { h["tensors"][1]["name"] = "x"; }),
               "duplicate");
  expect_error(write_bundle("no_meta.bin", tensors, json::object(), [](json& h) { h.erase("meta"); }), "meta");

  const std::string bad_magic = write_bundle("magic.bin", tensors, json::object());
  {
    std::fstream f(bad_magic, std::ios::in | std::ios::out | std::ios::binary);
    f.write("SVLAXXXX", 8);
  }
  expect_error(bad_magic, "magic");

  const std::string short_header = write_bundle("header.bin", tensors, json::object());
  {
    std::fstream f(short_header, std::ios::in | std::ios::out | std::ios::binary);
    f.seekp(8);
    const uint64_t huge = uint64_t{1} << 40;
    f.write(reinterpret_cast<const char*>(&huge), 8);
  }
  expect_error(short_header, "header length");

  const std::string unaligned = write_bundle("unaligned.bin", tensors, json::object());
  {
    std::fstream f(unaligned, std::ios::in | std::ios::out | std::ios::binary);
    uint64_t header_bytes = 0;
    f.seekg(8);
    f.read(reinterpret_cast<char*>(&header_bytes), 8);
    header_bytes += 1;  // the data section would start one byte past a 64-byte boundary
    f.seekp(8);
    f.write(reinterpret_cast<const char*>(&header_bytes), 8);
  }
  expect_error(unaligned, "data section is not 64-byte aligned");
}

// A header of the wrong JSON types gets a std::runtime_error naming the file and the field, not a JSON library
// exception or undefined behaviour in a later reader of meta.
TEST(Bundle, RejectsWronglyTypedHeaders) {
  const auto tensors = sample_tensors();
  const std::vector<std::tuple<std::string, std::function<void(json&)>, std::string>> cases = {
      {"header_array.bin", [](json& h) { h = json::array({1, 2}); }, "unsupported bundle format"},
      {"format_number.bin", [](json& h) { h["format"] = 7; }, "unsupported bundle format"},
      {"version_string.bin", [](json& h) { h["version"] = "1"; }, "unsupported bundle format"},
      {"version_float.bin", [](json& h) { h["version"] = 1.0; }, "unsupported bundle format"},
      {"version_wraps.bin", [](json& h) { h["version"] = (uint64_t{1} << 32) + 1; }, "unsupported bundle format"},
      {"meta_array.bin", [](json& h) { h["meta"] = json::array(); }, "\"meta\" must be an object, not array"},
      {"tensors_object.bin", [](json& h) { h["tensors"] = json::object(); }, "\"tensors\" must be an array"},
      {"name_number.bin", [](json& h) { h["tensors"][0]["name"] = 5; }, "\"name\" must be a string, not number"},
      {"euler_empty.bin", [](json& h) { h["meta"]["euler"] = json::object(); }, "meta.euler must be"},
      {"euler_no_dt.bin", [](json& h) { h["meta"]["euler"] = {{"t", {1.0, 0.5}}}; }, "meta.euler must be"},
      {"euler_t_string.bin", [](json& h) { h["meta"]["euler"] = {{"dt", -0.5}, {"t", {1.0, "0.5"}}}; },
       "meta.euler must be"},
      {"euler_array.bin", [](json& h) { h["meta"]["euler"] = {-0.5, 1.0}; }, "meta.euler must be"},
  };
  for (const auto& [file, edit, fragment] : cases) {
    const std::string path = write_bundle(file, tensors, json::object(), edit);
    try {
      Bundle::read(path);
      ADD_FAILURE() << file << " was accepted";
    } catch (const std::runtime_error& e) {
      EXPECT_NE(std::string(e.what()).find(path), std::string::npos) << e.what();
      EXPECT_NE(std::string(e.what()).find(fragment), std::string::npos) << e.what();
    }
  }
}

// Shapes and sizes whose arithmetic would overflow, or that a JSON parser only holds as floats, get an error that
// names the file and the tensor instead of undefined behaviour.
TEST(Bundle, RejectsAbsurdShapes) {
  const auto tensors = sample_tensors();
  auto expect_error = [&](const std::string& file, const std::function<void(json&)>& edit,
                          const std::string& fragment) {
    const std::string path = write_bundle(file, tensors, json::object(), edit);
    try {
      Bundle::read(path);
      ADD_FAILURE() << file << " was accepted";
    } catch (const std::runtime_error& e) {
      EXPECT_NE(std::string(e.what()).find(path), std::string::npos) << e.what();
      EXPECT_NE(std::string(e.what()).find(fragment), std::string::npos) << e.what();
    }
  };
  const uint64_t two32 = uint64_t{1} << 32;
  const uint64_t uint64_max = std::numeric_limits<uint64_t>::max();
  expect_error(
      "elements.bin", [&](json& h) { h["tensors"][0]["shape"] = {two32, two32}; },  // 2^64 elements
      "x: shape [4294967296,4294967296] has more elements than an int64 can count");
  expect_error(
      "wrap.bin", [&](json& h) { h["tensors"][0]["shape"] = {(uint64_t{1} << 62) + 1, 4}; },  // 4 in int64 arithmetic
      "x: shape [4611686018427387905,4] has more elements");
  expect_error(
      "bytes.bin", [&](json& h) { h["tensors"][2]["shape"] = {uint64_t{1} << 62}; },  // int64: 2^65 bytes
      "ids: int64[4611686018427387904] has more bytes than a size_t can count");
  expect_error(
      "dim_uint64.bin", [&](json& h) { h["tensors"][0]["shape"] = {uint64_max}; }, "dimensions must be int64 integers");
  expect_error(
      "dim_float.bin", [&](json& h) { h["tensors"][0]["shape"] = {1e20, 1}; }, "dimensions must be int64 integers");
  expect_error("nbytes_negative.bin", [&](json& h) { h["tensors"][0]["nbytes"] = -28; }, "\"nbytes\" is -28");
  expect_error("offset_float.bin", [&](json& h) { h["tensors"][0]["offset"] = 1e30; }, "\"offset\" is 1e+30");
  expect_error(
      "config_float.bin", [&](json& h) { h["meta"]["config"] = {{"camera_keys", {"a"}}, {"image_hw", {1e20, 1}}}; },
      "image_hw must be [height, width]");
}

}  // namespace
}  // namespace smolvla_rt
