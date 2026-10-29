#include "json_util.h"

#include <limits>
#include <stdexcept>

namespace smolvla_rt::detail {

namespace {

constexpr auto kInt64Max = static_cast<uint64_t>(std::numeric_limits<int64_t>::max());

// `value` as an int64, if it is a JSON integer in int64's range.
bool as_int64(const nlohmann::ordered_json& value, int64_t& out) {
  if (!value.is_number_integer()) return false;
  if (value.is_number_unsigned() && value.get<uint64_t>() > kInt64Max) return false;
  out = value.get<int64_t>();
  return true;
}

[[noreturn]] void wrong_type(const nlohmann::ordered_json& value, const char* key, const char* expected,
                             const std::string& where) {
  throw std::runtime_error(where + ": \"" + key + "\" must be " + expected + ", not " + value.type_name());
}

}  // namespace

const nlohmann::ordered_json& require(const nlohmann::ordered_json& j, const char* key, const std::string& where) {
  if (!j.is_object() || !j.contains(key)) throw std::runtime_error(where + ": missing \"" + key + "\"");
  return j.at(key);
}

const nlohmann::ordered_json& require_array(const nlohmann::ordered_json& j, const char* key,
                                            const std::string& where) {
  const auto& value = require(j, key, where);
  if (!value.is_array()) wrong_type(value, key, "an array", where);
  return value;
}

const nlohmann::ordered_json& require_object(const nlohmann::ordered_json& j, const char* key,
                                             const std::string& where) {
  const auto& value = require(j, key, where);
  if (!value.is_object()) wrong_type(value, key, "an object", where);
  return value;
}

std::string require_string(const nlohmann::ordered_json& j, const char* key, const std::string& where) {
  const auto& value = require(j, key, where);
  if (!value.is_string()) wrong_type(value, key, "a string", where);
  return value.get<std::string>();
}

bool has_format(const nlohmann::ordered_json& j, const char* format, int64_t version) {
  if (!j.is_object()) return false;
  const auto f = j.find("format");
  const auto v = j.find("version");
  int64_t got = 0;
  return f != j.end() && f->is_string() && f->get<std::string>() == format && v != j.end() && as_int64(*v, got) &&
         got == version;
}

int64_t require_int64(const nlohmann::ordered_json& j, const char* key, const std::string& where) {
  const auto& value = require(j, key, where);
  int64_t out = 0;
  if (!as_int64(value, out)) {
    throw std::runtime_error(where + ": \"" + key + "\" is " + value.dump() + ", not an int64 integer");
  }
  return out;
}

uint64_t require_uint64(const nlohmann::ordered_json& j, const char* key, const std::string& where) {
  const auto& value = require(j, key, where);
  if (!value.is_number_unsigned()) {
    throw std::runtime_error(where + ": \"" + key + "\" is " + value.dump() + ", not a uint64 integer");
  }
  return value.get<uint64_t>();
}

SplitConfig config_from_json(const nlohmann::ordered_json& j, const std::string& where) {
  try {
    SplitConfig c;
    c.camera_keys = require(j, "camera_keys", where).get<std::vector<std::string>>();
    const auto& hw = require(j, "image_hw", where);
    if (!hw.is_array() || hw.size() != 2 || !as_int64(hw[0], c.image_height) || !as_int64(hw[1], c.image_width)) {
      throw std::runtime_error(where + ": image_hw must be [height, width], not " + hw.dump());
    }
    c.image_tokens = require_int64(j, "image_tokens", where);
    c.lang_len = require_int64(j, "lang_len", where);
    c.state_dim = require_int64(j, "state_dim", where);
    c.max_state_dim = require_int64(j, "max_state_dim", where);
    c.action_dim = require_int64(j, "action_dim", where);
    c.max_action_dim = require_int64(j, "max_action_dim", where);
    c.chunk_size = require_int64(j, "chunk_size", where);
    c.num_steps = require_int64(j, "num_steps", where);
    c.num_layers = require_int64(j, "num_layers", where);
    c.expert_layers = require_int64(j, "expert_layers", where);
    c.kv_heads = require_int64(j, "kv_heads", where);
    c.head_dim = require_int64(j, "head_dim", where);
    c.prefix_len = require_int64(j, "prefix_len", where);
    c.validate();
    return c;
  } catch (const nlohmann::ordered_json::exception& e) {
    throw std::runtime_error(where + ": malformed config: " + e.what());
  } catch (const std::invalid_argument& e) {
    throw std::runtime_error(where + ": " + e.what());
  }
}

TensorSpec spec_from_json(const nlohmann::ordered_json& j, const std::string& where) {
  TensorSpec spec;
  try {
    spec.name = require_string(j, "name", where);
    spec.dtype = parse_dtype(require_string(j, "dtype", where));
    const auto& shape = require(j, "shape", where);
    if (!shape.is_array()) throw std::runtime_error(where + ": " + spec.name + " has shape " + shape.dump());
    for (const auto& d : shape) {
      int64_t dim = 0;
      if (!as_int64(d, dim)) {
        throw std::runtime_error(where + ": " + spec.name + " has shape " + shape.dump() +
                                 "; dimensions must be int64 integers");
      }
      spec.shape.push_back(dim);
    }
    spec.bytes();  // rejects dynamic dimensions and sizes past int64 elements or size_t bytes
    return spec;
  } catch (const nlohmann::ordered_json::exception& e) {
    throw std::runtime_error(where + ": malformed tensor entry: " + e.what());
  } catch (const std::invalid_argument& e) {
    throw std::runtime_error(where + ": " + (spec.name.empty() ? "" : spec.name + ": ") + e.what());
  }
}

}  // namespace smolvla_rt::detail
