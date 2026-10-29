// Internal: reading the shared JSON pieces of split_manifest.json and bundle headers.
#pragma once

#include <nlohmann/json.hpp>
#include <string>

#include "smolvla_rt/contract.h"
#include "smolvla_rt/tensor.h"

namespace smolvla_rt::detail {

// `j[key]`, or std::runtime_error naming `where` and the key when it is missing.
const nlohmann::ordered_json& require(const nlohmann::ordered_json& j, const char* key, const std::string& where);

// `j[key]` when it is a JSON array, object or string; otherwise std::runtime_error naming `where` and the key.
const nlohmann::ordered_json& require_array(const nlohmann::ordered_json& j, const char* key, const std::string& where);
const nlohmann::ordered_json& require_object(const nlohmann::ordered_json& j, const char* key,
                                             const std::string& where);
std::string require_string(const nlohmann::ordered_json& j, const char* key, const std::string& where);

// Whether `j` is an object whose "format" is the string `format` and whose "version" is the integer `version`.
bool has_format(const nlohmann::ordered_json& j, const char* format, int64_t version);

// `j[key]` as an integer. Throws std::runtime_error naming `where` and the key if it is missing, not an integer
// (JSON parsers turn integers too large for 64 bits into floats), or outside the type's range.
int64_t require_int64(const nlohmann::ordered_json& j, const char* key, const std::string& where);
uint64_t require_uint64(const nlohmann::ordered_json& j, const char* key, const std::string& where);

// The "config" object written by smolvla_latency.split_graph.split_config.
SplitConfig config_from_json(const nlohmann::ordered_json& j, const std::string& where);

// {"name", "dtype", "shape"}, as in the manifest's inputs/outputs and the bundle's tensor entries. The shape must
// be static and small enough that its element count and byte size fit in int64_t and size_t.
TensorSpec spec_from_json(const nlohmann::ordered_json& j, const std::string& where);

}  // namespace smolvla_rt::detail
