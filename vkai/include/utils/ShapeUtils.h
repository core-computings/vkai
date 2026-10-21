#pragma once
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <vector>
namespace vkai {
inline int ElementCount(const std::vector<int>& shape) {
  int count = 1;
  for (int d : shape) {
    if (d <= 0 || count > std::numeric_limits<int>::max() / d)
      throw std::invalid_argument("Invalid or oversized tensor shape");
    count *= d;
  }
  return count;
}
inline std::vector<int> ContiguousStrides(const std::vector<int>& shape) {
  ElementCount(shape);
  std::vector<int> strides(shape.size());
  int stride = 1;
  for (size_t i = shape.size(); i-- > 0;) {
    strides[i] = stride;
    stride *= shape[i];
  }
  return strides;
}
inline std::vector<int> BroadcastShape(const std::vector<int>& a, const std::vector<int>& b) {
  ElementCount(a);
  ElementCount(b);
  std::vector<int> out(std::max(a.size(), b.size()), 1);
  for (size_t i = 0; i < out.size(); ++i) {
    int av = i < a.size() ? a[a.size() - 1 - i] : 1, bv = i < b.size() ? b[b.size() - 1 - i] : 1;
    if (av != bv && av != 1 && bv != 1)
      throw std::invalid_argument("Incompatible broadcast shapes");
    out[out.size() - 1 - i] = std::max(av, bv);
  }
  ElementCount(out);
  return out;
}
}  // namespace vkai
