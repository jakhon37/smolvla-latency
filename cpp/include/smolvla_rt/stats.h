// Latency percentiles and numerical parity, computed as the Python tools compute them.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "smolvla_rt/tensor.h"

namespace smolvla_rt {

struct Summary {
  double p50 = 0, p90 = 0, p99 = 0, mean = 0, min = 0, max = 0;
  size_t n = 0;
};

// Percentiles with numpy.percentile's default linear interpolation, so the numbers line up with ort_flow.py's.
// All zero with n = 0 for no values.
Summary summarize(std::vector<double> values);

// smolvla_latency.metrics.compare: statistics of out - ref over all elements, and the non-finite count of out.
struct Diff {
  double max_abs = 0, mean_abs = 0, rel_l2 = 0, cosine = 0;
  int64_t nonfinite = 0;
};
Diff compare(const std::vector<double>& out, const std::vector<double>& ref);

// The elements of a float32, float16, int64 or bool buffer in host memory, as doubles.
std::vector<double> to_doubles(const void* data, DType dtype, size_t count);

}  // namespace smolvla_rt
