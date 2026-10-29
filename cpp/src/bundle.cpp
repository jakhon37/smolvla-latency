#include "smolvla_rt/bundle.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <stdexcept>

#include "json_util.h"

#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
#error "the bundle format is little-endian and this reader does not byte-swap"
#endif

namespace smolvla_rt {

namespace {

constexpr char kMagic[8] = {'S', 'V', 'L', 'A', 'B', 'N', 'D', 'L'};
constexpr uint64_t kAlign = 64;
constexpr uint64_t kMaxHeaderBytes = uint64_t{1} << 30;

// meta.euler, as tools/dump_bundle.py writes it: {"dt": number, "t": [number, ...]}.
void check_euler(const nlohmann::ordered_json& euler, const std::string& where) {
  const auto is_number = [](const nlohmann::ordered_json& x) { return x.is_number(); };
  if (!euler.is_object() || !euler.contains("dt") || !euler.at("dt").is_number() || !euler.contains("t") ||
      !euler.at("t").is_array() || !std::all_of(euler.at("t").begin(), euler.at("t").end(), is_number)) {
    throw std::runtime_error(where + " must be {\"dt\": number, \"t\": [number, ...]}");
  }
}

}  // namespace

Bundle Bundle::read(const std::string& path) {
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  if (!in) throw std::runtime_error("cannot open bundle " + path);
  const auto file_size = static_cast<uint64_t>(in.tellg());
  in.seekg(0);
  auto fail = [&path](const std::string& why) { return std::runtime_error(path + ": " + why); };

  char prefix[16];
  if (file_size < sizeof prefix || !in.read(prefix, sizeof prefix)) throw fail("too short to be a bundle");
  if (std::memcmp(prefix, kMagic, sizeof kMagic) != 0) throw fail("not a bundle (bad magic)");
  uint64_t header_bytes;
  std::memcpy(&header_bytes, prefix + 8, sizeof header_bytes);
  if (header_bytes > kMaxHeaderBytes || 16 + header_bytes > file_size) throw fail("header length out of range");
  if ((16 + header_bytes) % kAlign != 0) throw fail("data section is not 64-byte aligned");

  std::string header(header_bytes, '\0');
  if (!in.read(header.data(), static_cast<std::streamsize>(header_bytes))) throw fail("truncated header");
  nlohmann::ordered_json j;
  try {
    j = nlohmann::ordered_json::parse(header);
  } catch (const nlohmann::ordered_json::exception& e) {
    throw fail(std::string("invalid header JSON: ") + e.what());
  }
  if (!detail::has_format(j, "smolvla-split-bundle", 1)) {
    throw fail("unsupported bundle format (expected smolvla-split-bundle version 1)");
  }

  Bundle b;
  b.path_ = path;
  try {
    const auto& meta = detail::require_object(j, "meta", path);
    b.meta_json_ = meta.dump();
    if (meta.contains("config")) b.config_ = detail::config_from_json(meta.at("config"), path + " meta.config");
    if (meta.contains("euler")) check_euler(meta.at("euler"), path + " meta.euler");
    const auto& entries = detail::require_array(j, "tensors", path);

    const uint64_t data_bytes = file_size - 16 - header_bytes;
    b.data_ = allocate_host(data_bytes);
    if (data_bytes && !in.read(b.data_.as<char>(), static_cast<std::streamsize>(data_bytes))) {
      throw fail("truncated data");
    }

    for (const auto& entry : entries) {
      BundleTensor t;
      t.spec = detail::spec_from_json(entry, path);
      const uint64_t offset = detail::require_uint64(entry, "offset", path);
      const uint64_t nbytes = detail::require_uint64(entry, "nbytes", path);
      const std::string what = "tensor " + to_string(t.spec);
      if (nbytes != t.spec.bytes()) throw fail(what + " has " + std::to_string(nbytes) + " bytes");
      if (offset % kAlign != 0) throw fail(what + " is not 64-byte aligned");
      if (offset > data_bytes || nbytes > data_bytes - offset) throw fail(what + " runs past the end of the file");
      if (b.index_.count(t.spec.name)) throw fail("duplicate " + what);
      t.data = b.data_.as<char>() + offset;
      b.index_[t.spec.name] = b.tensors_.size();
      b.tensors_.push_back(std::move(t));
    }
  } catch (const nlohmann::ordered_json::exception& e) {  // the checks above should leave none of these
    throw fail(std::string("malformed header: ") + e.what());
  }
  return b;
}

namespace {

const void* checked(const Bundle& bundle, const std::string& name, DType dtype, const Shape& shape) {
  if (!bundle.contains(name)) throw std::runtime_error("the bundle has no " + name + " tensor");
  const BundleTensor& t = bundle.tensor(name);
  if (t.spec.dtype != dtype || t.spec.shape != shape) {
    throw std::runtime_error("bundle tensor " + to_string(t.spec) + " should be " + dtype_name(dtype) +
                             to_string(shape));
  }
  return t.data;
}

}  // namespace

Observation bundle_observation(const Bundle& bundle, const SplitConfig& c) {
  Observation obs;
  for (int64_t cam = 0; cam < c.num_cameras(); ++cam) {
    obs.pixel_values.push_back(static_cast<const float*>(checked(
        bundle, "pixel_values_" + std::to_string(cam), DType::kFloat32, {1, 3, c.image_height, c.image_width})));
  }
  obs.lang_tokens = static_cast<const int64_t*>(checked(bundle, "lang_tokens", DType::kInt64, {1, c.lang_len}));
  obs.lang_mask = static_cast<const uint8_t*>(checked(bundle, "lang_mask", DType::kBool, {1, c.lang_len}));
  obs.state = static_cast<const float*>(checked(bundle, "state", DType::kFloat32, {1, c.max_state_dim}));
  return obs;
}

const float* bundle_noise(const Bundle& bundle, const SplitConfig& c) {
  return static_cast<const float*>(checked(bundle, "noise", DType::kFloat32, {1, c.chunk_size, c.max_action_dim}));
}

const BundleTensor& Bundle::tensor(const std::string& name) const {
  const auto it = index_.find(name);
  if (it == index_.end()) throw std::out_of_range(path_ + " has no tensor '" + name + "'");
  return tensors_[it->second];
}

}  // namespace smolvla_rt
