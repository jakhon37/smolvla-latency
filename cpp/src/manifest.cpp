#include "smolvla_rt/manifest.h"

#include <fstream>
#include <stdexcept>

#include "json_util.h"

namespace smolvla_rt {

Manifest Manifest::read(const std::string& onnx_dir) {
  const std::string path = onnx_dir + "/" + kManifestFile;
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open " + path + " (is " + onnx_dir + " an export_split.py output?)");
  nlohmann::ordered_json j;
  try {
    in >> j;
  } catch (const nlohmann::ordered_json::exception& e) {
    throw std::runtime_error(path + ": invalid JSON: " + e.what());
  }
  if (!detail::has_format(j, "smolvla-split", 1)) {
    throw std::runtime_error(path + ": not a version 1 smolvla-split manifest");
  }

  Manifest m;
  m.dir_ = onnx_dir;
  try {
    m.config_ = detail::config_from_json(detail::require(j, "config", path), path);
    for (const auto& [precision, graphs] : detail::require_object(j, "variants", path).items()) {
      if (!graphs.is_object()) {
        throw std::runtime_error(path + ": variants." + precision + " must be an object, not " + graphs.type_name());
      }
      m.precision_order_.push_back(precision);
      for (const auto& [name, g] : graphs.items()) {
        const std::string where = path + " " + precision + "/" + name;
        GraphInfo info;
        info.file = detail::require_string(g, "file", where);
        for (const auto& spec : detail::require_array(g, "inputs", where)) {
          info.inputs.push_back(detail::spec_from_json(spec, where));
        }
        for (const auto& spec : detail::require_array(g, "outputs", where)) {
          info.outputs.push_back(detail::spec_from_json(spec, where));
        }
        m.variants_[precision][name] = std::move(info);
      }
    }
  } catch (const nlohmann::ordered_json::exception& e) {  // the checks above should leave none of these
    throw std::runtime_error(path + ": malformed manifest: " + e.what());
  }
  return m;
}

std::vector<std::string> Manifest::precisions() const { return precision_order_; }

const GraphInfo& Manifest::graph(const std::string& precision, const std::string& graph) const {
  const auto variant = variants_.find(precision);
  if (variant == variants_.end()) {
    std::string have;
    for (const auto& p : precision_order_) have += (have.empty() ? "" : ", ") + p;
    throw std::out_of_range(dir_ + " has no " + precision + " graphs; it has " + have);
  }
  const auto it = variant->second.find(graph);
  if (it == variant->second.end()) throw std::out_of_range(dir_ + " has no " + precision + " " + graph + " graph");
  return it->second;
}

std::string Manifest::path(const std::string& precision, const std::string& graph) const {
  return dir_ + "/" + this->graph(precision, graph).file;
}

}  // namespace smolvla_rt
