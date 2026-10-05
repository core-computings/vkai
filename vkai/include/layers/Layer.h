#pragma once

#include "VulkanBuffer.h"
#include "VulkanCommandBuffer.h"
#include "VulkanCompute.h"

namespace vkai {

class Layer : public core::vulkan::VulkanCompute {
 public:
  explicit Layer(core::vulkan::VulkanContext* context) : core::vulkan::VulkanCompute(context) {
    descriptor_set_layout_ = VK_NULL_HANDLE;
    descriptor_pool_ = VK_NULL_HANDLE;
    descriptor_set_ = VK_NULL_HANDLE;
  }
  virtual ~Layer() = default;

  virtual void Execute(const VkCommandBuffer& command_buffer,
                       const core::vulkan::VulkanBuffer& input_buffer,
                       core::vulkan::VulkanBuffer& output_buffer) = 0;

 protected:
  void UpdateStorageBufferDescriptors(uint32_t input_binding, uint32_t output_binding,
                                      const core::vulkan::VulkanBuffer& input_buffer,
                                      const core::vulkan::VulkanBuffer& output_buffer) const {
    const VkDescriptorBufferInfo buffer_infos[] = {
        {
            .buffer = input_buffer.buffer,
            .offset = 0,
            .range = input_buffer.Size(),
        },
        {
            .buffer = output_buffer.buffer,
            .offset = 0,
            .range = output_buffer.Size(),
        },
    };

    const VkWriteDescriptorSet descriptor_writes[] = {
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = descriptor_set_,
            .dstBinding = input_binding,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &buffer_infos[0],
        },
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = descriptor_set_,
            .dstBinding = output_binding,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &buffer_infos[1],
        },
    };

    vkUpdateDescriptorSets(context_->logical_device, 2, descriptor_writes, 0, nullptr);
  }
};

}  // namespace vkai
