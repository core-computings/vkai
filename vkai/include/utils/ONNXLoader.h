#pragma once

#include <string>

#include "graph/Graph.h"

namespace vkai {

// Loads the float32 ONNX subset used by the exported MNIST model into the
// engine graph representation. Unsupported ONNX operators or tensor types
// produce a descriptive error instead of a partially executable graph.
class ONNXLoader {
 public:
  static bool Load(const std::string& filename, Graph& graph, std::string* error_message = nullptr);
};

}  // namespace vkai
