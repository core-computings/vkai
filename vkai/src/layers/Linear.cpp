#include "layers/Linear.h"

#include <cstring>
#include <filesystem>
#include <iostream>

#include "utils/BufferUtils.h"

namespace vkai {

Linear::Linear(core::vulkan::VulkanContext* context, int input_size, int output_size,
               int batch_size)
    : Layer(context),
      uniform_buffer_(context, sizeof(UniformData), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                      kHostVisibleMemory),
      weights_buffer_(context, static_cast<VkDeviceSize>(input_size) * output_size * sizeof(float),
                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, kHostVisibleMemory),
      bias_buffer_(context, static_cast<VkDeviceSize>(output_size) * sizeof(float),
                   VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, kHostVisibleMemory),
      uniform_data_{
          .input_size = input_size, .output_size = output_size, .batch_size = batch_size} {
  if (input_size <= 0 || output_size <= 0 || batch_size <= 0) {
    std::cerr << "Linear dimensions must be positive\n";
    return;
  }
  uniform_buffer_.MapData(
      [this](void* data) { memcpy(data, &uniform_data_, sizeof(UniformData)); });
  valid_ = true;
}

void Linear::MapWeights(const std::vector<float>& weights, const std::vector<float>& bias) {
  if (!valid_) {
    throw std::logic_error("Cannot map weights for an invalid Linear");
  }
  if (weights.size() != static_cast<size_t>(uniform_data_.input_size) * uniform_data_.output_size ||
      bias.size() != static_cast<size_t>(uniform_data_.output_size)) {
    throw std::invalid_argument("Linear weights or bias size does not match its dimensions");
  }
  weights_buffer_.MapData([&weights](void* data) {
    std::memcpy(data, weights.data(), weights.size() * sizeof(float));
  });
  bias_buffer_.MapData(
      [&bias](void* data) { std::memcpy(data, bias.data(), bias.size() * sizeof(float)); });
  weights_mapped_ = true;
}

void Linear::Init() {
  if (!valid_ || !weights_mapped_) {
    throw std::logic_error("Linear must have valid dimensions and mapped weights before Init");
  }
  VulkanCompute::Init();

  CreateUniformBufferDescriptorSet(0, uniform_buffer_);
  CreateStorageBufferDescriptorSet(2, weights_buffer_);
  CreateStorageBufferDescriptorSet(3, bias_buffer_);

  const VkWriteDescriptorSet fixed_writes[] = {writes_[0], writes_[2], writes_[3]};
  vkUpdateDescriptorSets(context_->logical_device, 3, fixed_writes, 0, nullptr);

  // Save cache if file doesn't exist
  if (!std::filesystem::exists(GetPipelineCache())) {
    SavePipelineCache(GetPipelineCache());
  }
}

void Linear::Execute(const VkCommandBuffer& command_buffer,
                     const core::vulkan::VulkanBuffer& input_buffer,
                     core::vulkan::VulkanBuffer& output_buffer) {
  if (!valid_ || pipeline == VK_NULL_HANDLE) {
    std::cerr << "Cannot run an invalid or uninitialized Linear\n";
    return;
  }

  const VkDeviceSize required_input_size = static_cast<VkDeviceSize>(uniform_data_.batch_size) *
                                           uniform_data_.input_size * sizeof(float);
  const VkDeviceSize required_output_size = static_cast<VkDeviceSize>(uniform_data_.batch_size) *
                                            uniform_data_.output_size * sizeof(float);
  if (input_buffer.Size() < required_input_size) {
    std::cerr << "Linear input buffer is too small\n";
    return;
  }
  if (output_buffer.Size() < required_output_size) {
    std::cerr << "Linear output buffer is too small\n";
    return;
  }
  UpdateStorageBufferDescriptors(1, 4, input_buffer, output_buffer);

  vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
  vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout, 0, 1,
                          &descriptor_set_, 0, nullptr);

  constexpr uint32_t kTileSize = 16;
  const uint32_t group_x =
      (static_cast<uint32_t>(uniform_data_.output_size) + kTileSize - 1) / kTileSize;
  const uint32_t group_y =
      (static_cast<uint32_t>(uniform_data_.batch_size) + kTileSize - 1) / kTileSize;
  vkCmdDispatch(command_buffer, group_x, group_y, 1);
}

std::vector<core::vulkan::BindingInfo> Linear::GetBindingInfo() const {
  return {{0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT},
          {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT},
          {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT},
          {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT},
          {4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT}};
}

const std::vector<uint32_t>& Linear::LoadShaderCode() const {
  static const std::vector<uint32_t> shader_code =
#include "Linear.comp.spv"
      ;
  return shader_code;
}

const std::string Linear::GetPipelineCache() const {
  const std::string cache_dir = PIPELINE_CACHE_DIR;
  if (cache_dir.empty()) {
    return "";
  }
  std::string pipeline_cache = cache_dir + "/linear.cache";
  printf("Using pipeline cache file: %s\n", pipeline_cache.c_str());
  return pipeline_cache;
}

}  // namespace vkai
