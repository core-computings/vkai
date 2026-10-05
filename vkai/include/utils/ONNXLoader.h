#pragma once

#include <string>

#include "graph/Graph.h"

namespace vkai {

// Builds an engine graph from the float32 ONNX subset used by the exported
// MNIST model. Unsupported ONNX operators or tensor types throw
// std::runtime_error.
Graph BuildGraphFromONNX(const std::string& filename);

}  // namespace vkai
