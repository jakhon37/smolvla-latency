#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <vector>

#include "smolvla_rt/stats.h"

namespace smolvla_rt {
namespace {

// Expected values from numpy.percentile(values, q) with the default linear interpolation.
TEST(Stats, PercentilesMatchNumpy) {
  const Summary s = summarize({5.0, 1.0, 4.0, 2.0, 3.0, 10.0});
  EXPECT_DOUBLE_EQ(s.p50, 3.5);
  EXPECT_DOUBLE_EQ(s.p90, 7.5);
  EXPECT_DOUBLE_EQ(s.p99, 9.75);
  EXPECT_DOUBLE_EQ(s.mean, 25.0 / 6);
  EXPECT_EQ(s.min, 1.0);
  EXPECT_EQ(s.max, 10.0);
  EXPECT_EQ(s.n, 6u);

  const Summary one = summarize({2.5});
  EXPECT_EQ(one.p50, 2.5);
  EXPECT_EQ(one.p99, 2.5);
  EXPECT_EQ(summarize({}).n, 0u);
}

// Expected values from smolvla_latency.metrics.compare on the same arrays.
TEST(Stats, CompareMatchesMetricsPy) {
  const Diff d = compare({1.0, 2.0, 3.5}, {1.0, 2.5, 3.0});
  EXPECT_DOUBLE_EQ(d.max_abs, 0.5);
  EXPECT_DOUBLE_EQ(d.mean_abs, 1.0 / 3);
  EXPECT_DOUBLE_EQ(d.rel_l2, std::sqrt(0.5) / std::sqrt(1 + 6.25 + 9));
  EXPECT_DOUBLE_EQ(d.cosine, (1 + 5 + 10.5) / (std::sqrt(1 + 4 + 12.25) * std::sqrt(1 + 6.25 + 9)));
  EXPECT_EQ(d.nonfinite, 0);

  const double nan = std::numeric_limits<double>::quiet_NaN();
  const Diff bad = compare({nan, 1.0}, {0.0, 1.0});
  EXPECT_TRUE(std::isnan(bad.max_abs));
  EXPECT_EQ(bad.nonfinite, 1);
  EXPECT_THROW(compare({1.0}, {1.0, 2.0}), std::invalid_argument);
}

TEST(Stats, ConvertsEveryDtype) {
  const float f[] = {1.5f, -2.0f};
  const uint16_t h[] = {0x3c00, 0xc000};
  const int64_t i[] = {3, -4};
  const uint8_t b[] = {1, 0};
  EXPECT_EQ(to_doubles(f, DType::kFloat32, 2), (std::vector<double>{1.5, -2.0}));
  EXPECT_EQ(to_doubles(h, DType::kFloat16, 2), (std::vector<double>{1.0, -2.0}));
  EXPECT_EQ(to_doubles(i, DType::kInt64, 2), (std::vector<double>{3.0, -4.0}));
  EXPECT_EQ(to_doubles(b, DType::kBool, 2), (std::vector<double>{1.0, 0.0}));
}

}  // namespace
}  // namespace smolvla_rt
