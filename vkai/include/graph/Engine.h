#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "graph/Graph.h"
#include "layers/Layer.h"

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

  // Convenience accessors for the first graph input and output.
  std::shared_ptr<Tensor> GetInputTensor() const { return graph_.Inputs().at(0); }

  std::shared_ptr<Tensor> GetOutputTensor() const { return graph_.Outputs().at(0); }

  const std::vector<std::shared_ptr<Operation>>& GetTopoOrder() const { return topo_order_; }

  const std::unordered_map<std::string, std::unique_ptr<Layer>>& GetLayers() const {
    return layers_;
  }

  // Call SetData() on graph input tensors first. Executes synchronously and
  // stores results in graph output tensor Data(). Dynamic batch defaults to 1.
  void ExecuteGraph();

 private:
  void AllocateVulkanBuffers();

  void UploadWeights();

  void CreatePipeline();

  void UploadInputs();

  void DownloadOutputs();

  // Declared before graph_ so tensor buffers are destroyed before their context.
  std::unique_ptr<core::vulkan::VulkanContext> context_;

  Graph graph_;

  std::vector<std::shared_ptr<Operation>> topo_order_;

  // Only operations with runtime layers, in dependency order.
  std::vector<std::shared_ptr<Operation>> execution_order_;

  std::unordered_map<std::string, std::unique_ptr<Layer>> layers_;
};

}  // namespace vkai
