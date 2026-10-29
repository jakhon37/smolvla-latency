#include "smolvla_rt/ort_graph.h"

#include <onnxruntime_cxx_api.h>
#include <onnxruntime_session_options_config_keys.h>

#include <algorithm>
#include <stdexcept>

namespace smolvla_rt {

namespace {

DType to_dtype(ONNXTensorElementDataType type, const std::string& what) {
  switch (type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      return DType::kFloat32;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
      return DType::kFloat16;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      return DType::kInt64;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
      return DType::kBool;
    default:
      throw std::runtime_error(what + " has ONNX element type " + std::to_string(type) +
                               "; the split graphs only use float, float16, int64 and bool");
  }
}

ONNXTensorElementDataType to_onnx(DType dtype) {
  switch (dtype) {
    case DType::kFloat32:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    case DType::kFloat16:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
    case DType::kInt64:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
    case DType::kBool:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL;
  }
  throw std::invalid_argument("unknown DType");
}

bool use_cuda(const std::string& provider) {
  if (provider == "cpu") return false;
  if (provider == "cuda") return true;
  if (provider == "auto") return ort_cuda_available();
  throw std::invalid_argument("unknown execution provider '" + provider + "'; expected cpu, cuda or auto");
}

std::vector<TensorSpec> read_specs(const Ort::Session& session, bool inputs, const std::string& path) {
  Ort::AllocatorWithDefaultOptions allocator;
  const size_t count = inputs ? session.GetInputCount() : session.GetOutputCount();
  std::vector<TensorSpec> specs;
  for (size_t i = 0; i < count; ++i) {
    TensorSpec spec;
    spec.name =
        (inputs ? session.GetInputNameAllocated(i, allocator) : session.GetOutputNameAllocated(i, allocator)).get();
    const std::string what = path + " " + (inputs ? "input " : "output ") + spec.name;
    const Ort::TypeInfo info = inputs ? session.GetInputTypeInfo(i) : session.GetOutputTypeInfo(i);
    if (info.GetONNXType() != ONNX_TYPE_TENSOR) throw std::runtime_error(what + " is not a tensor");
    const auto tensor = info.GetTensorTypeAndShapeInfo();
    spec.dtype = to_dtype(tensor.GetElementType(), what);
    spec.shape = tensor.GetShape();
    if (std::any_of(spec.shape.begin(), spec.shape.end(), [](int64_t d) { return d < 0; })) {
      throw std::runtime_error(what + " has a dynamic shape " + to_string(spec.shape) +
                               "; the runtime needs static shapes");
    }
    specs.push_back(std::move(spec));
  }
  return specs;
}

}  // namespace

struct OrtGraph::Impl {
  // Declared first so it is destroyed last. ONNX Runtime keeps one environment per process and counts
  // references to it, so every graph holding its own handle is how the environment outlives every session.
  Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "smolvla_rt"};
  Ort::Session session{nullptr};
  Ort::MemoryInfo memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  Ort::RunOptions run_options;
  std::string path;
  std::string provider;
  std::vector<TensorSpec> inputs, outputs;
  std::vector<const char*> input_names, output_names;   // point into inputs / outputs
  std::vector<Ort::Value> input_values, output_values;  // null until bound

  bool bind(std::vector<TensorSpec>& specs, std::vector<Ort::Value>& values, const std::string& name, void* data) {
    for (size_t i = 0; i < specs.size(); ++i) {
      if (specs[i].name != name) continue;
      const TensorSpec& s = specs[i];
      values[i] = Ort::Value::CreateTensor(memory, data, s.bytes(), s.shape.data(), s.shape.size(), to_onnx(s.dtype));
      return true;
    }
    return false;
  }
};

OrtGraph::OrtGraph(const std::string& onnx_path, const OrtOptions& options) : impl_(std::make_unique<Impl>()) {
  Impl& m = *impl_;
  m.path = onnx_path;
  try {
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(options.intra_op_threads);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    if (!options.allow_spinning) session_options.AddConfigEntry(kOrtSessionOptionsConfigAllowIntraOpSpinning, "0");
    m.provider = "CPUExecutionProvider";
    if (use_cuda(options.provider)) {
      Ort::CUDAProviderOptions cuda;
      cuda.Update({{"device_id", std::to_string(options.cuda_device)}, {"use_tf32", "0"}});
      session_options.AppendExecutionProvider_CUDA_V2(*cuda);
      m.provider = "CUDAExecutionProvider";
    }
    m.session = Ort::Session(m.env, onnx_path.c_str(), session_options);
    m.inputs = read_specs(m.session, true, onnx_path);
    m.outputs = read_specs(m.session, false, onnx_path);
  } catch (const Ort::Exception& e) {
    throw std::runtime_error("onnxruntime could not load " + onnx_path + ": " + e.what());
  }
  for (const auto& s : m.inputs) {
    m.input_names.push_back(s.name.c_str());
    m.input_values.emplace_back(nullptr);
  }
  for (const auto& s : m.outputs) {
    m.output_names.push_back(s.name.c_str());
    m.output_values.emplace_back(nullptr);
  }
}

OrtGraph::~OrtGraph() = default;

const std::vector<TensorSpec>& OrtGraph::inputs() const { return impl_->inputs; }
const std::vector<TensorSpec>& OrtGraph::outputs() const { return impl_->outputs; }

void OrtGraph::bind(const std::string& name, void* data) {
  Impl& m = *impl_;
  if (!data) throw std::invalid_argument(m.path + ": null buffer for " + name);
  if (!m.bind(m.inputs, m.input_values, name, data) && !m.bind(m.outputs, m.output_values, name, data)) {
    throw std::invalid_argument(m.path + " has no input or output named '" + name + "'");
  }
}

void OrtGraph::run() {
  Impl& m = *impl_;
  for (size_t i = 0; i < m.input_values.size(); ++i) {
    if (!static_cast<const OrtValue*>(m.input_values[i])) {
      throw std::runtime_error(m.path + ": input " + m.inputs[i].name + " is not bound");
    }
  }
  for (size_t i = 0; i < m.output_values.size(); ++i) {
    if (!static_cast<const OrtValue*>(m.output_values[i])) {
      throw std::runtime_error(m.path + ": output " + m.outputs[i].name + " is not bound");
    }
  }
  try {
    m.session.Run(m.run_options, m.input_names.data(), m.input_values.data(), m.input_values.size(),
                  m.output_names.data(), m.output_values.data(), m.output_values.size());
  } catch (const Ort::Exception& e) {
    throw std::runtime_error("onnxruntime failed running " + m.path + ": " + e.what());
  }
}

std::string OrtGraph::backend() const { return "onnxruntime " + ort_version() + " " + impl_->provider; }

std::string ort_version() { return Ort::GetVersionString(); }

bool ort_cuda_available() {
  const auto providers = Ort::GetAvailableProviders();
  return std::find(providers.begin(), providers.end(), "CUDAExecutionProvider") != providers.end();
}

}  // namespace smolvla_rt
