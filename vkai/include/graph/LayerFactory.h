#pragma once

#include "graph/Operation.h"
#include "layers/Layer.h"

namespace vkai {

// Validate the supported ONNX subset and construct an initialized Vulkan layer.
std::unique_ptr<Layer> CreateLayer(core::vulkan::VulkanContext* context, const Operation& op);
void RecordLayer(Layer& layer, const Operation& op, VkCommandBuffer command);

}  // namespace vkai
