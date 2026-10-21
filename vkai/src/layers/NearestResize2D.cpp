#include "layers/NearestResize2D.h"

#include <cstring>
#include <filesystem>
#include <iostream>
#include <stdexcept>

#include "utils/BufferUtils.h"

namespace vkai {

NearestResize2D::NearestResize2D(core::vulkan::VulkanContext* context, int input_channels,
                                 int input_height, int input_width, int output_height,
                                 int output_width, int batch_size)
    : Layer(context),
      uniform_data_{.input_width = input_width,
                    .input_height = input_height,
                    .input_channels = input_channels,
                    .batch_size = batch_size,
                    .output_width = output_width,
                    .output_height = output_height,
                    .reserved_0 = 0,
                    .reserved_1 = 0},
      uniform_buffer_(context, sizeof(UniformData), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                      kHostVisibleMemory) {
  if (input_channels <= 0 || input_height <= 0 || input_width <= 0 || output_height <= 0 ||
      output_width <= 0 || batch_size <= 0) {
    throw std::invalid_argument("NearestResize2D dimensions must be positive");
  }
  uniform_buffer_.MapData(
      [this](void* data) { std::memcpy(data, &uniform_data_, sizeof(UniformData)); });
  valid_ = true;
}

void NearestResize2D::Init() {
  if (!valid_) {
    std::cerr << "Cannot initialize an invalid NearestResize2D\n";
    return;
  }
  VulkanCompute::Init();
  CreateUniformBufferDescriptorSet(0, uniform_buffer_);
  vkUpdateDescriptorSets(context_->logical_device, 1, &writes_[0], 0, nullptr);
  const std::string cache = GetPipelineCache();
  if (!cache.empty() && !std::filesystem::exists(cache)) SavePipelineCache(cache);
}

void NearestResize2D::Execute(const VkCommandBuffer& command_buffer,
                              const core::vulkan::VulkanBuffer& input_buffer,
                              core::vulkan::VulkanBuffer& output_buffer) {
  if (!valid_ || pipeline == VK_NULL_HANDLE) {
    std::cerr << "Cannot run an invalid or uninitialized NearestResize2D\n";
    return;
  }
  const VkDeviceSize input_size = static_cast<VkDeviceSize>(uniform_data_.batch_size) *
                                  uniform_data_.input_channels * uniform_data_.input_height *
                                  uniform_data_.input_width * sizeof(float);
  const VkDeviceSize output_size = static_cast<VkDeviceSize>(uniform_data_.batch_size) *
                                   uniform_data_.input_channels * uniform_data_.output_height *
                                   uniform_data_.output_width * sizeof(float);
  if (input_buffer.Size() < input_size || output_buffer.Size() < output_size) {
    std::cerr << "NearestResize2D input or output buffer is too small\n";
    return;
  }
  UpdateStorageBufferDescriptors(1, 2, input_buffer, output_buffer);
  vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
  vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout, 0, 1,
                          &descriptor_set_, 0, nullptr);
  const uint32_t output_elements =
      static_cast<uint32_t>(uniform_data_.batch_size * uniform_data_.input_channels *
                            uniform_data_.output_height * uniform_data_.output_width);
  constexpr uint32_t kLocalSize = 256;
  vkCmdDispatch(command_buffer, (output_elements + kLocalSize - 1) / kLocalSize, 1, 1);
}

std::vector<core::vulkan::BindingInfo> NearestResize2D::GetBindingInfo() const {
  return {{0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT},
          {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT},
          {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT}};
}

const std::vector<uint32_t>& NearestResize2D::LoadShaderCode() const {
  static const std::vector<uint32_t> shader_code =
#include "NearestResize2D.comp.spv"
      ;
  return shader_code;
}

const std::string NearestResize2D::GetPipelineCache() const {
  const std::string cache_dir = PIPELINE_CACHE_DIR;
  return cache_dir.empty() ? "" : cache_dir + "/nearest_resize_2d.cache";
}

}  // namespace vkai
