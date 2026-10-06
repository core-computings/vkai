#include "layers/Reshape.h"

namespace vkai {

Reshape::Reshape(core::vulkan::VulkanContext* context) : Layer(context) {}

void Reshape::Init() {}

void Reshape::Execute(const VkCommandBuffer& command_buffer,
                      const core::vulkan::VulkanBuffer& input_buffer,
                      core::vulkan::VulkanBuffer& output_buffer) {
  // Reshape preserves element order. Copy into its output tensor's buffer.
  const VkMemoryBarrier barrier{
      .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
      .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
      .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT |
                       VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
  };
  constexpr VkPipelineStageFlags stages =
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT;
  vkCmdPipelineBarrier(command_buffer, stages, stages, 0, 1, &barrier, 0, nullptr, 0, nullptr);
  const VkBufferCopy copy{.srcOffset = 0, .dstOffset = 0, .size = input_buffer.Size()};
  vkCmdCopyBuffer(command_buffer, input_buffer.buffer, output_buffer.buffer, 1, &copy);
  vkCmdPipelineBarrier(command_buffer, stages, stages, 0, 1, &barrier, 0, nullptr, 0, nullptr);
}

std::vector<core::vulkan::BindingInfo> Reshape::GetBindingInfo() const { return {}; }

const std::vector<uint32_t>& Reshape::LoadShaderCode() const {
  static const std::vector<uint32_t> shader_code;
  return shader_code;
}

}  // namespace vkai
