// Reader for split_manifest.json, the index export_split.py writes next to the graphs.
#pragma once

#include <map>
#include <string>
#include <vector>

#include "smolvla_rt/contract.h"
#include "smolvla_rt/tensor.h"

namespace smolvla_rt {

struct GraphInfo {
  std::string file;  // relative to the manifest's directory
  std::vector<TensorSpec> inputs;
  std::vector<TensorSpec> outputs;
};

class Manifest {
 public:
  // Reads <onnx_dir>/split_manifest.json. Throws std::runtime_error naming the file on any problem.
  static Manifest read(const std::string& onnx_dir);

  const std::string& dir() const { return dir_; }
  const SplitConfig& config() const { return config_; }
  // "fp32", "fp16", ... in the manifest's order.
  std::vector<std::string> precisions() const;
  bool has_precision(const std::string& precision) const { return variants_.count(precision) != 0; }
  // Throws std::out_of_range listing what the manifest has.
  const GraphInfo& graph(const std::string& precision, const std::string& graph) const;
  // dir() joined with graph(precision, graph).file.
  std::string path(const std::string& precision, const std::string& graph) const;

 private:
  std::string dir_;
  SplitConfig config_;
  std::vector<std::string> precision_order_;
  std::map<std::string, std::map<std::string, GraphInfo>> variants_;
};

}  // namespace smolvla_rt
