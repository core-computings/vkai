#include "layers/Reshape.h"

#include <stdexcept>

namespace vkai {

Reshape::Reshape(core::vulkan::VulkanContext* context, VkDeviceSize byte_size)
    : Layer(context), byte_size_(byte_size) {
  if (byte_size == 0) throw std::invalid_argument("Reshape requires a nonempty payload");
}

void Reshape::Init() {}

void Reshape::Execute(const VkCommandBuffer& command_buffer,
                      const core::vulkan::VulkanBuffer& input_buffer,
                      core::vulkan::VulkanBuffer& output_buffer) {
  if (input_buffer.Size() < byte_size_ || output_buffer.Size() < byte_size_)
    throw std::invalid_argument("Reshape buffer is too small");
  if (input_buffer.buffer == output_buffer.buffer) return;
  const VkMemoryBarrier barrier{
      .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
      .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
      .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT |
                       VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
  };
  constexpr VkPipelineStageFlags stages =
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT;
  vkCmdPipelineBarrier(command_buffer, stages, stages, 0, 1, &barrier, 0, nullptr, 0, nullptr);
  const VkBufferCopy copy{.srcOffset = 0, .dstOffset = 0, .size = byte_size_};
  vkCmdCopyBuffer(command_buffer, input_buffer.buffer, output_buffer.buffer, 1, &copy);
  vkCmdPipelineBarrier(command_buffer, stages, stages, 0, 1, &barrier, 0, nullptr, 0, nullptr);
}

std::vector<core::vulkan::BindingInfo> Reshape::GetBindingInfo() const { return {}; }

const std::vector<uint32_t>& Reshape::LoadShaderCode() const {
  static const std::vector<uint32_t> shader_code;
  return shader_code;
}

}  // namespace vkai
