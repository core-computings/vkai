#include "engine/Engine.h"

#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

#include "utils/BufferUtils.h"
#include "utils/ONNXLoader.h"

namespace vkai {

Engine::Engine(const std::string& onnx_path) : graph_(BuildGraphFromONNX(onnx_path)) {
  topo_order_ = graph_.TopologicalSort();
  context_ = std::make_unique<core::vulkan::VulkanContext>();
  context_->Init();
  AllocateVulkanBuffers();
  UploadWeights();
}

void Engine::AllocateVulkanBuffers() {
  for (const auto& [name, tensor] : graph_.Tensors()) {
    if (tensor->HasBuffer()) {
      continue;
    }

    const auto& shape = tensor->GetShape();
    // An empty shape currently means missing shape information in the graph IR.
    if (shape.empty()) {
      continue;
    }
    bool has_unknown_dimension = false;
    for (const auto dimension : shape) {
      if (dimension == 0) {
        has_unknown_dimension = true;
        break;
      }
    }
    if (has_unknown_dimension) {
      continue;
    }

    VkDeviceSize byte_size = sizeof(float);
    for (const auto dimension : shape) {
      const auto extent = static_cast<VkDeviceSize>(dimension);
      if (byte_size > std::numeric_limits<VkDeviceSize>::max() / extent) {
        throw std::overflow_error("Tensor buffer size overflow: " + name);
      }
      byte_size *= extent;
    }

    core::vulkan::VulkanBuffer buffer(context_.get(), byte_size,
                                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                          VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                          VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                      kHostVisibleMemory);
    tensor->SetBuffer(std::move(buffer));
  }
}

void Engine::UploadWeights() {
  for (const auto& [name, tensor] : graph_.Tensors()) {
    if (!tensor->HasData() || tensor->Data().empty()) {
      continue;
    }
    if (!tensor->HasBuffer()) {
      throw std::runtime_error("Cannot upload tensor data without a Vulkan buffer: " + name);
    }

    const auto& data = tensor->Data();
    auto& buffer = tensor->Buffer();
    if (buffer.Size() != data.size() * sizeof(float)) {
      throw std::runtime_error("Tensor data size does not match Vulkan buffer size: " + name);
    }
    buffer.MapData([&data](void* mapped_data) {
      std::memcpy(mapped_data, data.data(), data.size() * sizeof(float));
    });
  }
}

}  // namespace vkai
