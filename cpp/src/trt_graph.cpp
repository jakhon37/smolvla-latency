#include "smolvla_rt/trt_graph.h"

#include <NvInferRuntime.h>

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <stdexcept>

#include "cuda_check.h"

namespace smolvla_rt {

namespace {

class Logger final : public nvinfer1::ILogger {
 public:
  void log(Severity severity, nvinfer1::AsciiChar const* msg) noexcept override {
    if (severity <= Severity::kWARNING) std::fprintf(stderr, "[TensorRT] %s\n", msg);
  }
};

// TensorRT keeps a reference to the logger for as long as the runtime and its engines live.
Logger& logger() {
  static Logger instance;
  return instance;
}

DType to_dtype(nvinfer1::DataType type, const std::string& what) {
  switch (type) {
    case nvinfer1::DataType::kFLOAT:
      return DType::kFloat32;
    case nvinfer1::DataType::kHALF:
      return DType::kFloat16;
    case nvinfer1::DataType::kINT64:
      return DType::kInt64;
    case nvinfer1::DataType::kBOOL:
      return DType::kBool;
    default:
      throw std::runtime_error(what + " has TensorRT data type " + std::to_string(static_cast<int>(type)) +
                               "; the split graphs only use float, half, int64 and bool");
  }
}

}  // namespace

struct TrtGraph::Impl {
  std::string path;
  int device = 0;
  std::unique_ptr<nvinfer1::IRuntime> runtime;
  std::unique_ptr<nvinfer1::ICudaEngine> engine;
  std::unique_ptr<nvinfer1::IExecutionContext> context;  // declared last: destroyed before the engine
  cudaStream_t stream = nullptr;
  std::vector<TensorSpec> inputs, outputs;
  std::vector<std::string> unbound;

  // The CUDA device is per thread and the graph may be destroyed on another thread than the one that built it.
  // The members are released after this body, on this thread, so the engine's device is selected for them too.
  ~Impl() {
    cudaSetDevice(device);
    if (stream) cudaStreamDestroy(stream);
  }
};

TrtGraph::TrtGraph(const std::string& engine_path, int device) : impl_(std::make_unique<Impl>()) {
  Impl& m = *impl_;
  m.path = engine_path;
  m.device = device;

  std::ifstream in(engine_path, std::ios::binary | std::ios::ate);
  if (!in) throw std::runtime_error("cannot open TensorRT engine " + engine_path);
  std::vector<char> blob(static_cast<size_t>(in.tellg()));
  in.seekg(0);
  if (!in.read(blob.data(), static_cast<std::streamsize>(blob.size()))) {
    throw std::runtime_error("cannot read " + engine_path);
  }

  detail::check_cuda(cudaSetDevice(device), "cudaSetDevice");
  m.runtime.reset(nvinfer1::createInferRuntime(logger()));
  if (!m.runtime) throw std::runtime_error("TensorRT createInferRuntime failed");
  m.engine.reset(m.runtime->deserializeCudaEngine(blob.data(), blob.size()));
  if (!m.engine) {
    throw std::runtime_error(engine_path +
                             ": deserializeCudaEngine failed; engines only load in the TensorRT version "
                             "and on the GPU model they were built for");
  }
  m.context.reset(m.engine->createExecutionContext());
  if (!m.context) throw std::runtime_error(engine_path + ": createExecutionContext failed");
  // A blocking stream: it waits for earlier work on the legacy default stream, where CudaDevice copies the
  // inputs, so a run never starts before an asynchronous part of those copies has finished.
  detail::check_cuda(cudaStreamCreate(&m.stream), "cudaStreamCreate");

  for (int32_t i = 0; i < m.engine->getNbIOTensors(); ++i) {
    const char* name = m.engine->getIOTensorName(i);
    const bool is_input = m.engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT;
    const std::string what = engine_path + (is_input ? " input " : " output ") + name;
    TensorSpec spec;
    spec.name = name;
    spec.dtype = to_dtype(m.engine->getTensorDataType(name), what);
    const nvinfer1::Dims dims = m.engine->getTensorShape(name);
    spec.shape.assign(dims.d, dims.d + dims.nbDims);
    if (std::any_of(spec.shape.begin(), spec.shape.end(), [](int64_t d) { return d < 0; })) {
      throw std::runtime_error(what + " has a dynamic shape " + to_string(spec.shape) +
                               "; build the engine with static shapes");
    }
    if (m.engine->getTensorLocation(name) != nvinfer1::TensorLocation::kDEVICE) {
      throw std::runtime_error(what + " is a host tensor; the runtime binds device memory only");
    }
    m.unbound.push_back(spec.name);
    (is_input ? m.inputs : m.outputs).push_back(std::move(spec));
  }
}

TrtGraph::~TrtGraph() = default;

const std::vector<TensorSpec>& TrtGraph::inputs() const { return impl_->inputs; }
const std::vector<TensorSpec>& TrtGraph::outputs() const { return impl_->outputs; }

void TrtGraph::bind(const std::string& name, void* data) {
  Impl& m = *impl_;
  if (!data) throw std::invalid_argument(m.path + ": null buffer for " + name);
  const bool known =
      std::any_of(m.inputs.begin(), m.inputs.end(), [&](const TensorSpec& s) { return s.name == name; }) ||
      std::any_of(m.outputs.begin(), m.outputs.end(), [&](const TensorSpec& s) { return s.name == name; });
  if (!known) throw std::invalid_argument(m.path + " has no input or output named '" + name + "'");
  if (!m.context->setTensorAddress(name.c_str(), data)) {
    throw std::runtime_error(m.path + ": setTensorAddress(" + name + ") failed");
  }
  m.unbound.erase(std::remove(m.unbound.begin(), m.unbound.end(), name), m.unbound.end());
}

void TrtGraph::run() {
  Impl& m = *impl_;
  if (!m.unbound.empty()) throw std::runtime_error(m.path + ": " + m.unbound.front() + " is not bound");
  // The CUDA device is per thread, and run() may be called from another thread than the constructor.
  detail::check_cuda(cudaSetDevice(m.device), "cudaSetDevice");
  if (!m.context->enqueueV3(m.stream)) throw std::runtime_error(m.path + ": enqueueV3 failed");
  detail::check_cuda(cudaStreamSynchronize(m.stream), "cudaStreamSynchronize");
}

std::string TrtGraph::backend() const { return "TensorRT " + tensorrt_version(); }

std::string tensorrt_version() {
  const int32_t v = getInferLibVersion();  // major * 10000 + minor * 100 + patch since TensorRT 10
  return std::to_string(v / 10000) + "." + std::to_string(v / 100 % 100) + "." + std::to_string(v % 100);
}

}  // namespace smolvla_rt
