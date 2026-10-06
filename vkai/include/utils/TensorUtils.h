#pragma once

#include <limits>
#include <stdexcept>

#include "graph/Tensor.h"

namespace vkai {

inline int ToLayerDimension(int64_t dimension, const std::string& tensor_name) {
  if (dimension <= 0 || dimension > std::numeric_limits<int>::max()) {
    throw std::runtime_error("Tensor has an unsupported dimension: " + tensor_name);
  }
  return static_cast<int>(dimension);
}

// A symbolic leading batch dimension defaults to 1. Other dimensions must be known.
inline int TensorElementCount(const Tensor& tensor) {
  if (tensor.GetShape().empty()) {
    throw std::runtime_error("Tensor shape is missing: " + tensor.Name());
  }
  int element_count = 1;
  for (size_t index = 0; index < tensor.GetShape().size(); ++index) {
    const int64_t dimension = tensor.GetShape()[index];
    const int extent =
        index == 0 && dimension == 0 ? 1 : ToLayerDimension(dimension, tensor.Name());
    if (element_count > std::numeric_limits<int>::max() / extent) {
      throw std::runtime_error("Tensor has too many elements: " + tensor.Name());
    }
    element_count *= extent;
  }
  return element_count;
}

}  // namespace vkai
