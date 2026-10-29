// Python bindings: the C++ runtime with ort_flow.SplitRuntime's calling convention.
//
//   import smolvla_rt
//   rt = smolvla_rt.Runtime("onnx_split", precision="fp32")
//   actions = rt.predict_action_chunk(inputs, noise)  # inputs: pixel_values_<c>, lang_tokens, lang_mask, state
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include "smolvla_rt/build_config.h"
#include "smolvla_rt/loader.h"
#include "smolvla_rt/manifest.h"
#include "smolvla_rt/runtime.h"

namespace py = pybind11;
using namespace smolvla_rt;

namespace {

template <typename T>
using CArray = py::array_t<T, py::array::c_style | py::array::forcecast>;

// The array under `name`, cast to T, with the contract's shape.
template <typename T>
CArray<T> input(const py::dict& inputs, const std::string& name, const Shape& shape) {
  if (!inputs.contains(name)) throw std::invalid_argument("inputs has no '" + name + "'");
  auto array = py::cast<CArray<T>>(inputs[name.c_str()]);
  if (Shape(array.shape(), array.shape() + array.ndim()) != shape) {
    throw std::invalid_argument(name + " has shape " + to_string(Shape(array.shape(), array.shape() + array.ndim())) +
                                ", expected " + to_string(shape));
  }
  return array;
}

py::dict config_dict(const SplitConfig& c) {
  py::dict d;
  d["camera_keys"] = c.camera_keys;
  d["image_hw"] = std::vector<int64_t>{c.image_height, c.image_width};
  d["image_tokens"] = c.image_tokens;
  d["lang_len"] = c.lang_len;
  d["state_dim"] = c.state_dim;
  d["max_state_dim"] = c.max_state_dim;
  d["action_dim"] = c.action_dim;
  d["max_action_dim"] = c.max_action_dim;
  d["chunk_size"] = c.chunk_size;
  d["num_steps"] = c.num_steps;
  d["num_layers"] = c.num_layers;
  d["expert_layers"] = c.expert_layers;
  d["kv_heads"] = c.kv_heads;
  d["head_dim"] = c.head_dim;
  d["prefix_len"] = c.prefix_len;
  return d;
}

class PyRuntime {
 public:
  PyRuntime(const std::string& onnx_dir, const std::string& precision, const std::string& backend, int threads,
            bool allow_spinning, const std::string& provider, const std::string& engine_dir, int device) {
    LoadOptions options;
    options.backend = parse_backend(backend);
    options.precision = precision;
    options.ort.intra_op_threads = threads;
    options.ort.allow_spinning = allow_spinning;
    options.ort.provider = provider;
    options.ort.cuda_device = device;
    options.engine_dir = engine_dir;
    options.cuda_device = device;
    runtime_ = load_runtime(Manifest::read(onnx_dir), options);
  }

  py::array_t<float> predict_action_chunk(const py::dict& inputs, const py::object& noise_object) {
    const SplitConfig& c = runtime_->config();
    std::vector<CArray<float>> pixels;
    Observation obs;
    for (int64_t cam = 0; cam < c.num_cameras(); ++cam) {
      pixels.push_back(
          input<float>(inputs, "pixel_values_" + std::to_string(cam), {1, 3, c.image_height, c.image_width}));
      obs.pixel_values.push_back(pixels.back().data());
    }
    const auto lang_tokens = input<int64_t>(inputs, "lang_tokens", {1, c.lang_len});
    const auto lang_mask = input<bool>(inputs, "lang_mask", {1, c.lang_len});
    const auto state = input<float>(inputs, "state", {1, c.max_state_dim});
    const auto noise = py::cast<CArray<float>>(noise_object);
    if (Shape(noise.shape(), noise.shape() + noise.ndim()) != Shape{1, c.chunk_size, c.max_action_dim}) {
      throw std::invalid_argument("noise must be [1, chunk_size, max_action_dim]");
    }
    obs.lang_tokens = lang_tokens.data();
    static_assert(sizeof(bool) == 1, "numpy bools are one byte");
    obs.lang_mask = reinterpret_cast<const uint8_t*>(lang_mask.data());
    obs.state = state.data();

    py::array_t<float> actions({int64_t{1}, c.chunk_size, c.action_dim});
    float* out = actions.mutable_data();
    {
      // With the GIL released, other Python threads may call in; the runtime has one set of buffers and graph
      // bindings, so calls take turns. The lock is taken after the GIL is released and dropped before it is
      // taken back, so the two cannot deadlock.
      py::gil_scoped_release release;
      std::lock_guard<std::mutex> lock(mutex_);
      runtime_->predict_action_chunk(obs, noise.data(), out);
    }
    return actions;
  }

  py::dict last_timings() const {
    ChunkTimings t;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      t = runtime_->last_timings();
    }
    py::dict d;
    d["inputs"] = t.inputs;
    d["vision"] = t.vision;
    d["prefix"] = t.prefix;
    d["expert_step"] = t.expert_step;
    d["euler"] = t.euler;
    d["total"] = t.total;
    return d;
  }

  py::dict config() const { return config_dict(runtime_->config()); }
  std::string backend() const { return runtime_->backend(); }

 private:
  std::unique_ptr<SmolVLARuntime> runtime_;
  mutable std::mutex mutex_;  // serialises predict_action_chunk and reads of its timings
};

}  // namespace

PYBIND11_MODULE(smolvla_rt, m) {
  m.doc() = "C++ runtime for the split SmolVLA graphs (docs/SPLIT_GRAPH.md)";
  m.attr("__version__") = SMOLVLA_RT_VERSION;
  // The ONNX Runtime library the module runs on, e.g. "1.31.0"; None in a build without it.
#if SMOLVLA_RT_WITH_ORT
  m.attr("onnxruntime_version") = ort_version();
#else
  m.attr("onnxruntime_version") = py::none();
#endif
  py::class_<PyRuntime>(m, "Runtime")
      .def(py::init<const std::string&, const std::string&, const std::string&, int, bool, const std::string&,
                    const std::string&, int>(),
           py::arg("onnx_dir"), py::arg("precision") = "fp32", py::arg("backend") = "ort", py::arg("threads") = 0,
           py::arg("allow_spinning") = true, py::arg("provider") = "cpu", py::arg("engine_dir") = "",
           py::arg("device") = 0, "Loads the three graphs of one precision from an export_split.py directory.")
      .def("predict_action_chunk", &PyRuntime::predict_action_chunk, py::arg("inputs"), py::arg("noise"),
           "Action chunk [1, chunk_size, action_dim] float32 (normalised) from the graph inputs and the noise.")
      .def_property_readonly("last_timings", &PyRuntime::last_timings, "Milliseconds per part of the last chunk.")
      .def_property_readonly("config", &PyRuntime::config)
      .def_property_readonly("backend", &PyRuntime::backend);
}
