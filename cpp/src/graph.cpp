#include "smolvla_rt/graph.h"

#include <stdexcept>

namespace smolvla_rt {

const TensorSpec& find_spec(const std::vector<TensorSpec>& specs, const std::string& name, const std::string& what) {
  for (const auto& s : specs) {
    if (s.name == name) return s;
  }
  throw std::invalid_argument(what + " has no tensor named '" + name + "'");
}

}  // namespace smolvla_rt
