#include "smolvla_rt/stats.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace smolvla_rt {

namespace {

double percentile(const std::vector<double>& sorted, double q) {
  const double rank = q / 100.0 * static_cast<double>(sorted.size() - 1);
  const auto lo = static_cast<size_t>(std::floor(rank));
  const auto hi = static_cast<size_t>(std::ceil(rank));
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (rank - static_cast<double>(lo));
}

}  // namespace

Summary summarize(std::vector<double> values) {
  Summary s;
  if (values.empty()) return s;
  std::sort(values.begin(), values.end());
  s.n = values.size();
  s.p50 = percentile(values, 50);
  s.p90 = percentile(values, 90);
  s.p99 = percentile(values, 99);
  s.mean = std::accumulate(values.begin(), values.end(), 0.0) / static_cast<double>(s.n);
  s.min = values.front();
  s.max = values.back();
  return s;
}

Diff compare(const std::vector<double>& out, const std::vector<double>& ref) {
  if (out.size() != ref.size() || out.empty()) {
    throw std::invalid_argument("compare: " + std::to_string(out.size()) + " vs " + std::to_string(ref.size()) +
                                " elements");
  }
  Diff d;
  double sum_abs = 0, sum_sq = 0, dot = 0, out_sq = 0, ref_sq = 0;
  for (size_t i = 0; i < out.size(); ++i) {
    const double diff = std::fabs(out[i] - ref[i]);
    // numpy's max propagates NaN; std::max would silently drop it.
    d.max_abs = (std::isnan(diff) || std::isnan(d.max_abs)) ? std::numeric_limits<double>::quiet_NaN()
                                                            : std::max(d.max_abs, diff);
    sum_abs += diff;
    sum_sq += diff * diff;
    dot += out[i] * ref[i];
    out_sq += out[i] * out[i];
    ref_sq += ref[i] * ref[i];
    if (!std::isfinite(out[i])) ++d.nonfinite;
  }
  d.mean_abs = sum_abs / static_cast<double>(out.size());
  d.rel_l2 = std::sqrt(sum_sq) / std::max(std::sqrt(ref_sq), 1e-12);
  d.cosine = dot / std::max(std::sqrt(out_sq) * std::sqrt(ref_sq), 1e-12);
  return d;
}

std::vector<double> to_doubles(const void* data, DType dtype, size_t count) {
  std::vector<double> out(count);
  switch (dtype) {
    case DType::kFloat32: {
      const auto* p = static_cast<const float*>(data);
      for (size_t i = 0; i < count; ++i) out[i] = p[i];
      break;
    }
    case DType::kFloat16: {
      const auto* p = static_cast<const uint16_t*>(data);
      for (size_t i = 0; i < count; ++i) out[i] = half_to_float(p[i]);
      break;
    }
    case DType::kInt64: {
      const auto* p = static_cast<const int64_t*>(data);
      for (size_t i = 0; i < count; ++i) out[i] = static_cast<double>(p[i]);
      break;
    }
    case DType::kBool: {
      const auto* p = static_cast<const uint8_t*>(data);
      for (size_t i = 0; i < count; ++i) out[i] = p[i] ? 1.0 : 0.0;
      break;
    }
  }
  return out;
}

}  // namespace smolvla_rt
