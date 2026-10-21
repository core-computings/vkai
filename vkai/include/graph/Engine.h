#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "graph/Graph.h"
#include "layers/Layer.h"

namespace vkai {

struct InferenceBenchmark {
  std::string device_name;
  uint32_t timestamp_valid_bits;
  float timestamp_period_ns;
  std::vector<double> gpu_ms;
};

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

  // Call PopulateTensor() on graph input tensors first. Executes synchronously and
  // stores results in graph output tensor Data(). Dynamic batch defaults to 1.
  void ExecuteGraph();

  // Uploads inputs once and reuses a recorded command buffer. GPU timestamps
  // bracket the complete graph, including its internal barriers/copies, while
  // excluding model setup, host recording/submission and input/output copies.
  // Runs synchronously and copies the final result to output tensor Data().
  InferenceBenchmark BenchmarkGraph(int warmup_runs = 5, int measured_runs = 30);

 private:
  void RecordOperations(VkCommandBuffer command);

  void AllocateVulkanBuffers();

  void UploadWeights();

  void CreatePipeline();

  void UploadInputs();

  void CopyOutputs();

  // Declared before graph_ so tensor buffers are destroyed before their context.
  std::unique_ptr<core::vulkan::VulkanContext> context_;

  Graph graph_;

  std::vector<std::shared_ptr<Operation>> topo_order_;

  // Only operations with runtime layers, in dependency order.
  std::vector<std::shared_ptr<Operation>> execution_order_;

  std::unordered_map<std::string, std::unique_ptr<Layer>> layers_;
};

}  // namespace vkai
