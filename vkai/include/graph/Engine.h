#pragma once

#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "graph/Graph.h"
#include "graph/MemoryPlanner.h"
#include "layers/Layer.h"

namespace vkai {

struct InferenceBenchmark {
  std::string device_name;
  uint32_t timestamp_valid_bits;
  float timestamp_period_ns;
  std::vector<double> gpu_ms;
  uint32_t submissions_per_run;
};

class Engine {
 public:
  // Loads the ONNX model and builds its graph. Loading errors propagate as
  // std::runtime_error from the ONNX loader.
  // Bound each submission for devices with a GPU job watchdog. The default
  // records the whole graph in one submission.
  explicit Engine(const std::string& onnx_path,
                  size_t operations_per_submission = std::numeric_limits<size_t>::max());

  Engine(const Engine&) = delete;
  Engine& operator=(const Engine&) = delete;
  Engine(Engine&&) = delete;
  Engine& operator=(Engine&&) = delete;

  // Intermediate buffers may be overwritten after their last consumer. CPU-only
  // parameters have no graph buffer; graph outputs remain available in Data().
  const Graph& GetGraph() const { return graph_; }

  // Convenience accessors for the first graph input and output. Retained tensors
  // keep their GPU storage and its Vulkan context alive after Engine destruction.
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
  // cover every submission, including its internal barriers/copies, while
  // excluding model setup, host recording/submission and input/output copies.
  // Durations are summed across submissions, excluding host waits between them.
  // Runs synchronously and copies the final result to output tensor Data().
  InferenceBenchmark BenchmarkGraph(int warmup_runs = 5, int measured_runs = 30);

 private:
  void FindConvReluFusions();

  void RecordOperations(VkCommandBuffer command, size_t first, size_t count);

  void AllocateVulkanBuffers(const MemoryPlan& plan);

  void UploadConstants(const MemoryPlan& plan);

  void CreatePipeline();

  void UploadInputs();

  void CopyOutputs();

  // Graph storage shares this context; layers are destroyed before this owner.
  std::shared_ptr<core::vulkan::VulkanContext> context_;

  Graph graph_;

  std::vector<std::shared_ptr<Operation>> topo_order_;

  // Runtime operations in dependency order. Fused Relu slots are retained to
  // preserve submission boundaries (and watchdog budgets) across optimizations.
  std::vector<std::shared_ptr<Operation>> execution_order_;
  std::unordered_set<const Operation*> fused_relus_;
  size_t operations_per_submission_;

  std::unordered_map<std::string, std::unique_ptr<Layer>> layers_;
};

}  // namespace vkai
