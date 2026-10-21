#pragma once

#include <string>

#include "graph/Graph.h"

namespace vkai {

// Builds a statically shaped float32 graph for the supported ONNX inference
// subset (including ResNet and U-Net). A symbolic leading batch defaults to 1.
// Unsupported ONNX operators or tensor types throw
// std::runtime_error.
Graph BuildGraphFromONNX(const std::string& filename);

}  // namespace vkai
