// Reader for the test-vector bundles tools/dump_bundle.py writes (format: smolvla_latency/bundle.py).
#pragma once

#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "smolvla_rt/contract.h"
#include "smolvla_rt/memory.h"
#include "smolvla_rt/runtime.h"
#include "smolvla_rt/tensor.h"

namespace smolvla_rt {

struct BundleTensor {
  TensorSpec spec;
  const void* data = nullptr;  // C order, 64-byte aligned, owned by the Bundle

  template <typename T>
  const T* as() const {
    return static_cast<const T*>(data);
  }
};

// Layout, all integers little-endian:
//   offset 0       8 bytes   ASCII "SVLABNDL"
//   offset 8       8 bytes   uint64 H, the header length
//   offset 16      H bytes   JSON {"format": "smolvla-split-bundle", "version": 1, "meta": {...}, "tensors": [...]},
//                            space-padded so that 16 + H is a multiple of 64
//   offset 16 + H            data; each tensor at its entry's "offset" (a multiple of 64) from here
class Bundle {
 public:
  // Reads and validates the whole file. Throws std::runtime_error naming the file and the problem.
  static Bundle read(const std::string& path);

  // The header's "meta" object, serialized as JSON.
  const std::string& meta_json() const { return meta_json_; }
  // meta.config, which tools/dump_bundle.py copies from the export; nullopt if the bundle has none.
  const std::optional<SplitConfig>& config() const { return config_; }

  bool contains(const std::string& name) const { return index_.count(name) != 0; }
  // Throws std::out_of_range naming the tensor and the file.
  const BundleTensor& tensor(const std::string& name) const;
  // In file order.
  const std::vector<BundleTensor>& tensors() const { return tensors_; }

 private:
  std::string path_;
  Buffer data_;  // the data section, in 64-byte aligned host memory
  std::vector<BundleTensor> tensors_;
  std::unordered_map<std::string, size_t> index_;
  std::string meta_json_;
  std::optional<SplitConfig> config_;
};

// The observation a bundle holds for `config`: pixel_values_<c>, lang_tokens, lang_mask and state, pointing into
// the bundle's memory. Throws std::runtime_error if a tensor is missing or its dtype or shape is not the
// contract's.
Observation bundle_observation(const Bundle& bundle, const SplitConfig& config);
// The bundle's noise, [1, chunk_size, max_action_dim] float32: the first x_t.
const float* bundle_noise(const Bundle& bundle, const SplitConfig& config);

}  // namespace smolvla_rt
