// Building a SmolVLARuntime from an export_split.py directory with one of the compiled-in backends.
#pragma once

#include <memory>
#include <string>

#include "smolvla_rt/manifest.h"
#include "smolvla_rt/ort_graph.h"
#include "smolvla_rt/runtime.h"

namespace smolvla_rt {

enum class Backend { kOnnxRuntime, kTensorRT };

// "ort" or "trt"; throws std::invalid_argument otherwise.
Backend parse_backend(const std::string& name);
const char* backend_name(Backend backend);
// Whether this build of the library includes the backend.
bool backend_compiled(Backend backend);

struct LoadOptions {
  Backend backend = Backend::kOnnxRuntime;
  std::string precision = "fp32";
  OrtOptions ort;
  // TensorRT reads <engine_dir>/<graph>_<precision>.engine; empty means the manifest's directory.
  std::string engine_dir;
  int cuda_device = 0;
};

// The three graphs of `options.precision` on `options.backend`, checked against the manifest's config.
std::unique_ptr<SmolVLARuntime> load_runtime(const Manifest& manifest, const LoadOptions& options);

}  // namespace smolvla_rt
