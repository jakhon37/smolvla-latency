#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <new>
#include <string>
#include <vector>

#include "smolvla_rt/memory.h"
#include "smolvla_rt/tensor.h"

namespace smolvla_rt {
namespace {

TEST(DType, NamesRoundTripAndSizes) {
  for (DType d : {DType::kFloat32, DType::kFloat16, DType::kInt64, DType::kBool}) {
    EXPECT_EQ(parse_dtype(dtype_name(d)), d);
  }
  EXPECT_EQ(dtype_size(DType::kFloat32), 4u);
  EXPECT_EQ(dtype_size(DType::kFloat16), 2u);
  EXPECT_EQ(dtype_size(DType::kInt64), 8u);
  EXPECT_EQ(dtype_size(DType::kBool), 1u);
  EXPECT_THROW(parse_dtype("float64"), std::invalid_argument);
}

TEST(TensorSpec, BytesAndDescription) {
  const TensorSpec spec{"past_key_0", DType::kFloat16, {1, 5, 177, 64}};
  EXPECT_EQ(spec.bytes(), 2u * 5 * 177 * 64);
  EXPECT_EQ(to_string(spec), "past_key_0 float16[1,5,177,64]");
  EXPECT_EQ(element_count({}), 1);  // a scalar
  EXPECT_THROW(element_count({1, -1}), std::invalid_argument);
  EXPECT_EQ(spec, (TensorSpec{"past_key_0", DType::kFloat16, {1, 5, 177, 64}}));
  EXPECT_NE(spec, (TensorSpec{"past_key_0", DType::kFloat32, {1, 5, 177, 64}}));
}

TEST(TensorSpec, SizesThatOverflowAreRejected) {
  constexpr int64_t kBig = int64_t{1} << 32;
  EXPECT_EQ(element_count({kBig, kBig / 8, 2}), int64_t{1} << 62);
  EXPECT_THROW(element_count({kBig, kBig}), std::invalid_argument);                 // 2^64 elements
  EXPECT_THROW(element_count({(int64_t{1} << 62) + 1, 4}), std::invalid_argument);  // wraps to 4 in int64
  EXPECT_EQ(element_count({kBig, kBig, 0}), 0);  // empty, however long the other sides
  const TensorSpec fits{"x", DType::kFloat32, {int64_t{1} << 61}};
  EXPECT_EQ(fits.bytes(), size_t{1} << 63);
  const TensorSpec too_many_bytes{"x", DType::kInt64, {int64_t{1} << 62}};  // 2^62 elements, 2^65 bytes
  try {
    too_many_bytes.bytes();
    ADD_FAILURE() << "2^65 bytes were accepted";
  } catch (const std::invalid_argument& e) {
    EXPECT_NE(std::string(e.what()).find("int64[4611686018427387904]"), std::string::npos) << e.what();
  }
  // The largest float32 size bytes() allows, 2^64 - 4, must not wrap to a small block when padded for alignment.
  const TensorSpec largest{"x", DType::kFloat32, {(int64_t{1} << 62) - 1}};
  EXPECT_THROW(allocate_host(largest.bytes()), std::bad_alloc);
}

TEST(Half, KnownValues) {
  EXPECT_EQ(float_to_half(1.0f), 0x3c00);
  EXPECT_EQ(float_to_half(-2.0f), 0xc000);
  EXPECT_EQ(float_to_half(65504.0f), 0x7bff);
  EXPECT_EQ(float_to_half(0.0f), 0x0000);
  EXPECT_EQ(float_to_half(-0.0f), 0x8000);
  EXPECT_EQ(float_to_half(std::ldexp(1.0f, -24)), 0x0001);  // smallest subnormal
  EXPECT_EQ(float_to_half(std::ldexp(1.0f, -14)), 0x0400);  // smallest normal
  EXPECT_EQ(float_to_half(std::numeric_limits<float>::infinity()), 0x7c00);
  EXPECT_EQ(float_to_half(-1e10f), 0xfc00);
  EXPECT_EQ(float_to_half(1e-10f), 0x0000);
  EXPECT_EQ(float_to_half(-std::numeric_limits<float>::denorm_min()), 0x8000);
  const uint16_t nan = float_to_half(std::numeric_limits<float>::quiet_NaN());
  EXPECT_EQ(nan & 0x7c00, 0x7c00);
  EXPECT_NE(nan & 0x03ff, 0);

  EXPECT_EQ(half_to_float(0x3c00), 1.0f);
  EXPECT_EQ(half_to_float(0x7bff), 65504.0f);
  EXPECT_EQ(half_to_float(0x0001), std::ldexp(1.0f, -24));
  EXPECT_EQ(half_to_float(0x03ff), std::ldexp(1023.0f, -24));  // largest subnormal
  EXPECT_TRUE(std::isinf(half_to_float(0xfc00)) && half_to_float(0xfc00) < 0);
  EXPECT_TRUE(std::isnan(half_to_float(0x7e00)));
  EXPECT_TRUE(std::signbit(half_to_float(0x8000)));
}

TEST(Half, EveryHalfRoundTrips) {
  for (uint32_t h = 0; h <= 0xffff; ++h) {
    const auto bits = static_cast<uint16_t>(h);
    const float f = half_to_float(bits);
    if (std::isnan(f)) {
      EXPECT_TRUE(std::isnan(half_to_float(float_to_half(f)))) << std::hex << h;
    } else {
      ASSERT_EQ(float_to_half(f), bits) << std::hex << h;
    }
  }
}

// Every float between two neighbouring halves rounds to the nearer one, and a float exactly halfway rounds to
// the one with an even significand. With round trips exact, this pins down round-to-nearest-even everywhere.
TEST(Half, RoundsToNearestEven) {
  for (uint16_t sign : {uint16_t{0}, uint16_t{0x8000}}) {
    for (uint32_t h = 0; h < 0x7c00; ++h) {
      const auto lo = static_cast<uint16_t>(sign | h), hi = static_cast<uint16_t>(sign | (h + 1));
      const float a = half_to_float(lo);
      // Above 65504 the next "half" is 65536, which only exists as the rounding boundary to infinity.
      const float b = h + 1 == 0x7c00 ? std::copysign(65536.0f, a) : half_to_float(hi);
      const float mid = (a + b) / 2;  // exact: at most 12 significant bits
      ASSERT_EQ(float_to_half(mid), (lo & 1) ? hi : lo) << std::hex << lo;
      ASSERT_EQ(float_to_half(std::nextafter(mid, a)), lo) << std::hex << lo;
      ASSERT_EQ(float_to_half(std::nextafter(mid, b)), hi) << std::hex << lo;
    }
  }
}

TEST(Half, BulkConversionMatchesScalar) {
  std::vector<float> values = {0.0f,
                               -0.0f,
                               1e-10f,
                               -1e-10f,
                               65519.0f,
                               65520.0f,
                               1e10f,
                               -1e10f,
                               std::numeric_limits<float>::infinity(),
                               std::numeric_limits<float>::quiet_NaN(),
                               std::numeric_limits<float>::denorm_min()};
  for (uint32_t h = 0; h < 0x7c00; ++h) {
    const float a = half_to_float(static_cast<uint16_t>(h)), b = half_to_float(static_cast<uint16_t>(h + 1));
    for (float v : {a, (a + b) / 2, std::nextafter((a + b) / 2, a), std::nextafter((a + b) / 2, b)}) {
      values.push_back(v);
      values.push_back(-v);
    }
  }
  std::vector<uint16_t> bulk(values.size());
  float_to_half(values.data(), bulk.data(), values.size());
  for (size_t i = 0; i < values.size(); ++i) ASSERT_EQ(bulk[i], float_to_half(values[i])) << values[i];
}

}  // namespace
}  // namespace smolvla_rt
