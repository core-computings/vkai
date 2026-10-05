#pragma once

#include <memory>
#include <string>
#include <vector>

#include "graph/Graph.h"

namespace vkai {

class Engine {
 public:
  // Loads the ONNX model and builds its graph. Loading errors propagate as
  // std::runtime_error from the ONNX loader.
  explicit Engine(const std::string& onnx_path);

  Engine(const Engine&) = delete;
  Engine& operator=(const Engine&) = delete;
  Engine(Engine&&) = delete;
  Engine& operator=(Engine&&) = delete;

  const Graph& GetGraph() const { return graph_; }

  const std::vector<std::shared_ptr<Operation>>& GetTopoOrder() const { return topo_order_; }

 private:
  // Tensors with unknown shapes are deferred until shape inference resolves them.
  void AllocateVulkanBuffers();

  // Declared before graph_ so tensor buffers are destroyed before their context.
  std::unique_ptr<core::vulkan::VulkanContext> context_;

  Graph graph_;

  std::vector<std::shared_ptr<Operation>> topo_order_;
};

}  // namespace vkai
