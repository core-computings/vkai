#pragma once

#include "layers/Layer.h"

namespace vkai {

class Reshape : public Layer {
 public:
  Reshape(core::vulkan::VulkanContext* context, VkDeviceSize byte_size);

  // Engine reshapes alias storage. Standalone execution can copy the payload.
  void Init() override;

  void Execute(const VkCommandBuffer& command_buffer,
               const core::vulkan::VulkanBuffer& input_buffer,
               core::vulkan::VulkanBuffer& output_buffer) override;

 protected:
  std::vector<core::vulkan::BindingInfo> GetBindingInfo() const override;

  const std::vector<uint32_t>& LoadShaderCode() const override;

 private:
  VkDeviceSize byte_size_;
};

}  // namespace vkai
