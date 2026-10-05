#include "layers/BatchNorm2D.h"

#include <cmath>
#include <cstring>
#include <filesystem>
#include <limits>
#include <stdexcept>

#include "utils/BufferUtils.h"

namespace vkai {
namespace {

core::vulkan::VulkanContext* ValidateConfiguration(core::vulkan::VulkanContext* context,
                                                   int channels, int spatial, int batch,
                                                   float eps) {
  if (channels <= 0 || spatial <= 0 || batch <= 0 || !std::isfinite(eps) || eps < 0 ||
      channels > std::numeric_limits<int>::max() / spatial ||
      channels * spatial > std::numeric_limits<int>::max() / batch) {
    throw std::invalid_argument("Invalid BatchNorm2D dimensions or epsilon");
  }
  return context;
}

}  // namespace

BatchNorm2D::BatchNorm2D(core::vulkan::VulkanContext* context, int channels,
                         int elements_per_channel, int batch_size, float eps)
    : Layer(ValidateConfiguration(context, channels, elements_per_channel, batch_size, eps)),
      eps_(eps),
      uniform_buffer_(context, sizeof(UniformData), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                      kHostVisibleMemory),
      parameters_buffer_(context, static_cast<VkDeviceSize>(channels) * 2 * sizeof(float),
                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, kHostVisibleMemory) {
  uniform_data_ = {channels, elements_per_channel, channels * elements_per_channel * batch_size, 0};
  uniform_buffer_.MapData(
      [this](void* data) { std::memcpy(data, &uniform_data_, sizeof(UniformData)); });
}

void BatchNorm2D::MapParameters(const std::vector<float>& running_mean,
                                const std::vector<float>& running_var,
                                const std::vector<float>& weight, const std::vector<float>& bias) {
  const size_t count = static_cast<size_t>(uniform_data_.channels);
  if (running_mean.size() != count || running_var.size() != count ||
      (!weight.empty() && weight.size() != count) || (!bias.empty() && bias.size() != count)) {
    throw std::invalid_argument("BatchNorm2D parameters must match channel count");
  }
  std::vector<float> parameters(count * 2);
  for (size_t c = 0; c < count; ++c) {
    if (!std::isfinite(running_mean[c]) || !std::isfinite(running_var[c]) || running_var[c] < 0 ||
        !std::isfinite(running_var[c] + eps_) || running_var[c] + eps_ <= 0 ||
        !std::isfinite(weight.empty() ? 1.0F : weight[c]) ||
        !std::isfinite(bias.empty() ? 0.0F : bias[c])) {
      throw std::invalid_argument("Invalid BatchNorm2D running statistics or affine parameters");
    }
    const float scale = (weight.empty() ? 1.0F : weight[c]) / std::sqrt(running_var[c] + eps_);
    parameters[2 * c] = scale;
    parameters[2 * c + 1] = (bias.empty() ? 0.0F : bias[c]) - running_mean[c] * scale;
  }
  parameters_buffer_.MapData([&parameters](void* data) {
    std::memcpy(data, parameters.data(), parameters.size() * sizeof(float));
  });
  parameters_mapped_ = true;
}

void BatchNorm2D::Init() {
  if (!parameters_mapped_) {
    throw std::logic_error("BatchNorm2D parameters must be mapped before Init");
  }
  VulkanCompute::Init();
  CreateUniformBufferDescriptorSet(0, uniform_buffer_);
  vkUpdateDescriptorSets(context_->logical_device, 1, &writes_[0], 0, nullptr);
  const VkDescriptorBufferInfo info = {parameters_buffer_.buffer, 0, parameters_buffer_.Size()};
  const VkWriteDescriptorSet write = {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                                      .dstSet = descriptor_set_,
                                      .dstBinding = 3,
                                      .descriptorCount = 1,
                                      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                      .pBufferInfo = &info};
  vkUpdateDescriptorSets(context_->logical_device, 1, &write, 0, nullptr);
  const std::string cache = GetPipelineCache();
  if (!cache.empty() && !std::filesystem::exists(cache)) SavePipelineCache(cache);
}

void BatchNorm2D::Execute(const VkCommandBuffer& command_buffer,
                          const core::vulkan::VulkanBuffer& input_buffer,
                          core::vulkan::VulkanBuffer& output_buffer) {
  if (pipeline == VK_NULL_HANDLE) throw std::logic_error("BatchNorm2D is not initialized");
  const VkDeviceSize bytes = static_cast<VkDeviceSize>(uniform_data_.element_count) * sizeof(float);
  if (input_buffer.Size() < bytes || output_buffer.Size() < bytes) {
    throw std::invalid_argument("BatchNorm2D input or output buffer is too small");
  }
  UpdateStorageBufferDescriptors(1, 2, input_buffer, output_buffer);
  vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
  vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout, 0, 1,
                          &descriptor_set_, 0, nullptr);
  vkCmdDispatch(command_buffer, (static_cast<uint32_t>(uniform_data_.element_count) + 255) / 256, 1,
                1);
}

std::vector<core::vulkan::BindingInfo> BatchNorm2D::GetBindingInfo() const {
  return {{0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT},
          {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT},
          {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT},
          {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT}};
}

const std::vector<uint32_t>& BatchNorm2D::LoadShaderCode() const {
  static const std::vector<uint32_t> shader_code =
#include "BatchNorm2D.comp.spv"
      ;
  return shader_code;
}

const std::string BatchNorm2D::GetPipelineCache() const {
  const std::string directory = PIPELINE_CACHE_DIR;
  return directory.empty() ? "" : directory + "/batch_norm_2d.cache";
}

}  // namespace vkai
