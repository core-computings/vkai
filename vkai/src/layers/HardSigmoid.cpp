#include "layers/HardSigmoid.h"

#include <cmath>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <stdexcept>

#include "utils/BufferUtils.h"

namespace vkai {

HardSigmoid::HardSigmoid(core::vulkan::VulkanContext* context, int element_count, float alpha,
                         float beta)
    : Layer(context),
      uniform_buffer_(context, sizeof(UniformData), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                      kHostVisibleMemory),
      uniform_data_{.element_count = element_count, .alpha = alpha, .beta = beta, .reserved = 0} {
  if (element_count <= 0 || !std::isfinite(alpha) || !std::isfinite(beta)) {
    throw std::invalid_argument("HardSigmoid requires positive size and finite alpha/beta");
  }
  uniform_buffer_.MapData(
      [this](void* data) { std::memcpy(data, &uniform_data_, sizeof(UniformData)); });
  valid_ = true;
}

void HardSigmoid::Init() {
  if (!valid_) {
    std::cerr << "Cannot initialize an invalid HardSigmoid\n";
    return;
  }
  VulkanCompute::Init();

  CreateUniformBufferDescriptorSet(0, uniform_buffer_);
  vkUpdateDescriptorSets(context_->logical_device, 1, &writes_[0], 0, nullptr);

  const std::string cache = GetPipelineCache();
  if (!cache.empty() && !std::filesystem::exists(cache)) {
    SavePipelineCache(cache);
  }
}

void HardSigmoid::Execute(const VkCommandBuffer& command_buffer,
                          const core::vulkan::VulkanBuffer& input_buffer,
                          core::vulkan::VulkanBuffer& output_buffer) {
  if (!valid_ || pipeline == VK_NULL_HANDLE) {
    std::cerr << "Cannot run an invalid or uninitialized HardSigmoid\n";
    return;
  }
  const VkDeviceSize required_size =
      static_cast<VkDeviceSize>(uniform_data_.element_count) * sizeof(float);
  if (input_buffer.Size() < required_size || output_buffer.Size() < required_size) {
    std::cerr << "HardSigmoid input or output buffer is too small\n";
    return;
  }
  UpdateStorageBufferDescriptors(1, 2, input_buffer, output_buffer);

  vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
  vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout, 0, 1,
                          &descriptor_set_, 0, nullptr);

  constexpr uint32_t kLocalSize = 256;
  const uint32_t group_x =
      (static_cast<uint32_t>(uniform_data_.element_count) + kLocalSize - 1) / kLocalSize;
  vkCmdDispatch(command_buffer, group_x, 1, 1);
}

std::vector<core::vulkan::BindingInfo> HardSigmoid::GetBindingInfo() const {
  return {{0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT},
          {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT},
          {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT}};
}

const std::vector<uint32_t>& HardSigmoid::LoadShaderCode() const {
  static const std::vector<uint32_t> shader_code =
#include "HardSigmoid.comp.spv"
      ;
  return shader_code;
}

const std::string HardSigmoid::GetPipelineCache() const {
  const std::string cache_dir = PIPELINE_CACHE_DIR;
  if (cache_dir.empty()) {
    return "";
  }
  return cache_dir + "/hard_sigmoid.cache";
}

}  // namespace vkai
