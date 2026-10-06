#pragma once

#include "layers/Layer.h"

namespace vkai {

class Reshape : public Layer {
 public:
  explicit Reshape(core::vulkan::VulkanContext* context);

  // Reshape uses a buffer copy, so no compute pipeline is needed.
  void Init() override;

  void Execute(const VkCommandBuffer& command_buffer,
               const core::vulkan::VulkanBuffer& input_buffer,
               core::vulkan::VulkanBuffer& output_buffer) override;

 protected:
  std::vector<core::vulkan::BindingInfo> GetBindingInfo() const override;

  const std::vector<uint32_t>& LoadShaderCode() const override;
};

}  // namespace vkai
