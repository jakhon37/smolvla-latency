#include "smolvla_rt/loader.h"

#include <stdexcept>

#include "smolvla_rt/build_config.h"

#if SMOLVLA_RT_WITH_TENSORRT
#include "smolvla_rt/cuda_device.h"
#include "smolvla_rt/trt_graph.h"
#endif

namespace smolvla_rt {

namespace {

constexpr const char* kGraphs[3] = {kVision, kPrefix, kExpertStep};

}  // namespace

Backend parse_backend(const std::string& name) {
  if (name == "ort") return Backend::kOnnxRuntime;
  if (name == "trt") return Backend::kTensorRT;
  throw std::invalid_argument("unknown backend '" + name + "'; expected ort or trt");
}

const char* backend_name(Backend backend) { return backend == Backend::kOnnxRuntime ? "ort" : "trt"; }

bool backend_compiled(Backend backend) {
  return backend == Backend::kOnnxRuntime ? SMOLVLA_RT_WITH_ORT : SMOLVLA_RT_WITH_TENSORRT;
}

std::unique_ptr<SmolVLARuntime> load_runtime(const Manifest& manifest, const LoadOptions& options) {
  if (!backend_compiled(options.backend)) {
    throw std::runtime_error(std::string("this build of smolvla_rt has no ") + backend_name(options.backend) +
                             " backend; reconfigure CMake with it enabled");
  }
  // Fails early, listing the precisions the export has.
  for (const char* graph : kGraphs) manifest.graph(options.precision, graph);

  std::unique_ptr<Graph> graphs[3];
  std::shared_ptr<Device> device;
  if (options.backend == Backend::kOnnxRuntime) {
#if SMOLVLA_RT_WITH_ORT
    for (int i = 0; i < 3; ++i) {
      graphs[i] = std::make_unique<OrtGraph>(manifest.path(options.precision, kGraphs[i]), options.ort);
    }
    device = std::make_shared<HostDevice>();
#endif
  } else {
#if SMOLVLA_RT_WITH_TENSORRT
    const std::string dir = options.engine_dir.empty() ? manifest.dir() : options.engine_dir;
    device = std::make_shared<CudaDevice>(options.cuda_device);
    for (int i = 0; i < 3; ++i) {
      const std::string engine = dir + "/" + kGraphs[i] + "_" + options.precision + ".engine";
      graphs[i] = std::make_unique<TrtGraph>(engine, options.cuda_device);
    }
#endif
  }
  return std::make_unique<SmolVLARuntime>(manifest.config(), std::move(graphs[0]), std::move(graphs[1]),
                                          std::move(graphs[2]), std::move(device));
}

}  // namespace smolvla_rt
