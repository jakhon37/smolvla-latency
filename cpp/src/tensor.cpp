#include "smolvla_rt/tensor.h"

#include <cstring>
#include <limits>
#include <stdexcept>

namespace smolvla_rt {

size_t dtype_size(DType dtype) {
  switch (dtype) {
    case DType::kFloat32:
      return 4;
    case DType::kFloat16:
      return 2;
    case DType::kInt64:
      return 8;
    case DType::kBool:
      return 1;
  }
  throw std::invalid_argument("unknown DType");
}

const char* dtype_name(DType dtype) {
  switch (dtype) {
    case DType::kFloat32:
      return "float32";
    case DType::kFloat16:
      return "float16";
    case DType::kInt64:
      return "int64";
    case DType::kBool:
      return "bool";
  }
  throw std::invalid_argument("unknown DType");
}

DType parse_dtype(std::string_view name) {
  for (DType d : {DType::kFloat32, DType::kFloat16, DType::kInt64, DType::kBool}) {
    if (name == dtype_name(d)) return d;
  }
  throw std::invalid_argument("unsupported dtype '" + std::string(name) +
                              "'; expected float32, float16, int64 or bool");
}

int64_t element_count(const Shape& shape) {
  bool empty = false;
  for (int64_t d : shape) {
    if (d < 0) throw std::invalid_argument("shape " + to_string(shape) + " has a dynamic dimension");
    empty = empty || d == 0;
  }
  if (empty) return 0;
  int64_t n = 1;
  for (int64_t d : shape) {
    if (n > std::numeric_limits<int64_t>::max() / d) {
      throw std::invalid_argument("shape " + to_string(shape) + " has more elements than an int64 can count");
    }
    n *= d;
  }
  return n;
}

std::string to_string(const Shape& shape) {
  std::string s = "[";
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i) s += ",";
    s += std::to_string(shape[i]);
  }
  return s + "]";
}

size_t TensorSpec::bytes() const {
  const auto count = static_cast<uint64_t>(element_count(shape));
  const size_t size = dtype_size(dtype);
  if (count > std::numeric_limits<size_t>::max() / size) {
    throw std::invalid_argument(std::string(dtype_name(dtype)) + to_string(shape) +
                                " has more bytes than a size_t can count");
  }
  return static_cast<size_t>(count) * size;
}

bool operator==(const TensorSpec& a, const TensorSpec& b) {
  return a.name == b.name && a.dtype == b.dtype && a.shape == b.shape;
}

std::string to_string(const TensorSpec& spec) {
  return spec.name + " " + dtype_name(spec.dtype) + to_string(spec.shape);
}

namespace {

// Branch-free so that the bulk conversion below vectorizes; every path is computed and the right one selected.
inline uint16_t half_bits(float value) {
  uint32_t f;
  std::memcpy(&f, &value, sizeof f);
  const uint32_t sign = (f >> 16) & 0x8000u;
  const uint32_t abs = f & 0x7fffffffu;

  // Normal range: rebias the exponent from 127 to 15 and round the 13 dropped mantissa bits to nearest even.
  // A carry out of the mantissa correctly bumps the exponent, up to infinity for values in [65520, 65536).
  const uint32_t normal = (abs - 0x38000000u + 0x0fffu + ((abs >> 13) & 1u)) >> 13;

  // Below the smallest normal half (2^-14) the result is a multiple of 2^-24. Adding 0.5, whose ulp is 2^-24,
  // makes the float addition itself round to that grid, ties to even, leaving the multiple in the low bits.
  float magnitude;
  std::memcpy(&magnitude, &abs, sizeof magnitude);
  const float shifted = magnitude + 0.5f;
  uint32_t shifted_bits;
  std::memcpy(&shifted_bits, &shifted, sizeof shifted_bits);
  const uint32_t subnormal = shifted_bits - 0x3f000000u;

  // Infinity stays infinity, NaN stays a quiet NaN with the top of its payload; >= 65536 overflows to infinity.
  const uint32_t nan = 0u - static_cast<uint32_t>(abs > 0x7f800000u);  // all ones for NaN
  const uint32_t special = 0x7c00u | (nan & (0x0200u | ((abs >> 13) & 0x03ffu)));

  // Selected with masks rather than ?: so the compiler keeps the loop free of branches.
  const uint32_t tiny = 0u - static_cast<uint32_t>(abs < 0x38800000u);
  const uint32_t huge = 0u - static_cast<uint32_t>(abs >= 0x47800000u);
  const uint32_t finite = (subnormal & tiny) | (normal & ~tiny);
  return static_cast<uint16_t>(sign | (special & huge) | (finite & ~huge));
}

}  // namespace

uint16_t float_to_half(float value) { return half_bits(value); }

void float_to_half(const float* src, uint16_t* dst, size_t count) {
  for (size_t i = 0; i < count; ++i) dst[i] = half_bits(src[i]);
}

float half_to_float(uint16_t bits) {
  const uint32_t sign = static_cast<uint32_t>(bits & 0x8000u) << 16;
  const uint32_t exponent = (bits >> 10) & 0x1fu;
  const uint32_t mantissa = bits & 0x03ffu;
  uint32_t f;
  if (exponent == 0x1fu) {
    f = sign | 0x7f800000u | (mantissa << 13);
  } else if (exponent == 0) {
    // Zero or subnormal: mantissa * 2^-24, exact in float32.
    const float magnitude = static_cast<float>(mantissa) * 5.9604644775390625e-8f;
    std::memcpy(&f, &magnitude, sizeof f);
    f |= sign;
  } else {
    f = sign | ((exponent + 112u) << 23) | (mantissa << 13);
  }
  float out;
  std::memcpy(&out, &f, sizeof out);
  return out;
}

}  // namespace smolvla_rt
