// Element types, tensor descriptions and fp16 conversion shared by every part of the runtime.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace smolvla_rt {

// The element types of the split graphs and of the bundle format (docs/SPLIT_GRAPH.md). kBool is one byte per
// element, 0 or 1, as in ONNX Runtime, TensorRT and numpy.
enum class DType { kFloat32, kFloat16, kInt64, kBool };

size_t dtype_size(DType dtype);
// The numpy names that split_manifest.json and bundles use: "float32", "float16", "int64", "bool".
const char* dtype_name(DType dtype);
// Inverse of dtype_name; throws std::invalid_argument for any other name.
DType parse_dtype(std::string_view name);

using Shape = std::vector<int64_t>;

// Product of the dimensions; throws std::invalid_argument if one is negative (a dynamic dimension) or if the
// product does not fit in int64_t.
int64_t element_count(const Shape& shape);
std::string to_string(const Shape& shape);

struct TensorSpec {
  std::string name;
  DType dtype = DType::kFloat32;
  Shape shape;

  // element_count(shape) * dtype_size(dtype); throws std::invalid_argument as element_count does, or if the size
  // does not fit in size_t.
  size_t bytes() const;
};

bool operator==(const TensorSpec& a, const TensorSpec& b);
inline bool operator!=(const TensorSpec& a, const TensorSpec& b) { return !(a == b); }
// "name float32[1,3,512,512]"
std::string to_string(const TensorSpec& spec);

// IEEE 754 binary32 to binary16 with round-to-nearest-even, overflow to infinity and NaN kept NaN: the same
// result as numpy's astype(np.float16), which is how the Python tools feed the fp16 graphs.
uint16_t float_to_half(float value);
void float_to_half(const float* src, uint16_t* dst, size_t count);
float half_to_float(uint16_t bits);

}  // namespace smolvla_rt
