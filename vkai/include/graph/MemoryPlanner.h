#pragma once

#include <cstddef>
#include <unordered_map>
#include <vector>

#include "graph/Graph.h"

namespace vkai {

struct BufferSlot {
  VkDeviceSize capacity = 0;
  size_t last_use = 0;
  bool persistent = false;
};

struct MemoryPlan {
  std::vector<BufferSlot> slots;
  std::unordered_map<const Tensor*, size_t> tensor_slots;
  std::vector<std::shared_ptr<Tensor>> constants;
};

// Plans storage without creating Vulkan objects. Reshape aliases share one
// lifetime; CPU-only parameters need no graph buffer. Inputs, outputs and GPU
// constants have dedicated storage so repeated execution preserves their data.
MemoryPlan BuildMemoryPlan(const Graph& graph,
                           const std::vector<std::shared_ptr<Operation>>& order);

}  // namespace vkai
