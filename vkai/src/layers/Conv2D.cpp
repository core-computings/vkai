#include "layers/Conv2D.h"

#include <cstring>
#include <filesystem>
#include <iostream>

#include "utils/BufferUtils.h"

namespace vkai {

Conv2D::Conv2D(core::vulkan::VulkanContext* context, int input_channels, int output_channels,
               int input_height, int input_width, int kernel_height, int kernel_width,
               int stride_height, int stride_width, int padding_height, int padding_width,
               PaddingType padding_type, int batch_size, int dilation_height, int dilation_width)
    : Layer(context),
      uniform_buffer_(context, sizeof(UniformData), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                      kHostVisibleMemory),
      weights_buffer_(context,
                      static_cast<VkDeviceSize>(input_channels) * output_channels * kernel_height *
                          kernel_width * sizeof(float),
                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, kHostVisibleMemory),
      bias_buffer_(context, static_cast<VkDeviceSize>(output_channels) * sizeof(float),
                   VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, kHostVisibleMemory),
      uniform_data_{
          .input_width = input_width,
          .input_height = input_height,
          .input_channels = input_channels,
          .batch_size = batch_size,
          .output_width = 0,
          .output_height = 0,
          .output_channels = output_channels,
          .padding_type = static_cast<int>(padding_type),
          .kernel_width = kernel_width,
          .kernel_height = kernel_height,
          .stride_width = stride_width,
          .stride_height = stride_height,
          .padding_width = padding_width,
          .padding_height = padding_height,
          .dilation_width = dilation_width,
          .dilation_height = dilation_height,
      } {
  if (input_channels <= 0 || output_channels <= 0 || input_height <= 0 || input_width <= 0 ||
      kernel_height <= 0 || kernel_width <= 0 || stride_height <= 0 || stride_width <= 0 ||
      padding_height < 0 || padding_width < 0 || dilation_height <= 0 || dilation_width <= 0 ||
      batch_size <= 0) {
    std::cerr << "Conv2D dimensions must be valid and positive\n";
    return;
  }
  if (padding_type != PaddingType::Zero && padding_type != PaddingType::Reflect) {
    std::cerr << "Conv2D padding type must be Zero or Reflect\n";
    return;
  }

  const int effective_kernel_width = dilation_width * (kernel_width - 1) + 1;
  const int effective_kernel_height = dilation_height * (kernel_height - 1) + 1;
  uniform_data_.output_width =
      (input_width + 2 * padding_width - effective_kernel_width) / stride_width + 1;
  uniform_data_.output_height =
      (input_height + 2 * padding_height - effective_kernel_height) / stride_height + 1;
  if (uniform_data_.output_height <= 0 || uniform_data_.output_width <= 0) {
    std::cerr << "Conv2D output dimensions must be positive\n";
    return;
  }

  uniform_buffer_.MapData(
      [this](void* data) { std::memcpy(data, &uniform_data_, sizeof(UniformData)); });
  valid_ = true;
}

void Conv2D::MapWeights(const std::vector<float>& weights, const std::vector<float>& bias) {
  if (!valid_) {
    throw std::logic_error("Cannot map weights for an invalid Conv2D");
  }
  const size_t expected_weight_count = static_cast<size_t>(uniform_data_.output_channels) *
                                       uniform_data_.input_channels * uniform_data_.kernel_height *
                                       uniform_data_.kernel_width;
  if (weights.size() != expected_weight_count ||
      (!bias.empty() && bias.size() != static_cast<size_t>(uniform_data_.output_channels))) {
    throw std::invalid_argument("Conv2D weights or bias size does not match its dimensions");
  }
  weights_buffer_.MapData([&weights](void* data) {
    std::memcpy(data, weights.data(), weights.size() * sizeof(float));
  });
  bias_buffer_.MapData([&bias, this](void* data) {
    std::memset(data, 0, static_cast<size_t>(uniform_data_.output_channels) * sizeof(float));
    if (!bias.empty()) {
      std::memcpy(data, bias.data(), bias.size() * sizeof(float));
    }
  });
  weights_mapped_ = true;
}

void Conv2D::Init() {
  if (!valid_ || !weights_mapped_) {
    throw std::logic_error("Conv2D must have valid dimensions and mapped weights before Init");
  }
  VulkanCompute::Init();

  CreateUniformBufferDescriptorSet(0, uniform_buffer_);
  CreateStorageBufferDescriptorSet(2, weights_buffer_);
  CreateStorageBufferDescriptorSet(4, bias_buffer_);

  const VkWriteDescriptorSet fixed_writes[] = {writes_[0], writes_[2], writes_[4]};
  vkUpdateDescriptorSets(context_->logical_device, 3, fixed_writes, 0, nullptr);

  const std::string cache = GetPipelineCache();
  if (!cache.empty() && !std::filesystem::exists(cache)) {
    SavePipelineCache(cache);
  }
}

void Conv2D::Execute(const VkCommandBuffer& command_buffer,
                     const core::vulkan::VulkanBuffer& input_buffer,
                     core::vulkan::VulkanBuffer& output_buffer) {
  if (!valid_ || pipeline == VK_NULL_HANDLE) {
    std::cerr << "Cannot run an invalid or uninitialized Conv2D\n";
    return;
  }

  const VkDeviceSize required_input_size =
      static_cast<VkDeviceSize>(uniform_data_.batch_size) * uniform_data_.input_channels *
      uniform_data_.input_height * uniform_data_.input_width * sizeof(float);
  const VkDeviceSize required_output_size =
      static_cast<VkDeviceSize>(uniform_data_.batch_size) * uniform_data_.output_channels *
      uniform_data_.output_height * uniform_data_.output_width * sizeof(float);
  if (input_buffer.Size() < required_input_size) {
    std::cerr << "Conv2D input buffer is too small\n";
    return;
  }
  if (output_buffer.Size() < required_output_size) {
    std::cerr << "Conv2D output buffer is too small\n";
    return;
  }
  UpdateStorageBufferDescriptors(1, 3, input_buffer, output_buffer);

  vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
  vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout, 0, 1,
                          &descriptor_set_, 0, nullptr);

  const uint32_t output_elements =
      static_cast<uint32_t>(uniform_data_.batch_size * uniform_data_.output_channels *
                            uniform_data_.output_height * uniform_data_.output_width);
  constexpr uint32_t kLocalSize = 256;
  const uint32_t group_x = (output_elements + kLocalSize - 1) / kLocalSize;
  vkCmdDispatch(command_buffer, group_x, 1, 1);
}

std::vector<core::vulkan::BindingInfo> Conv2D::GetBindingInfo() const {
  return {{0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT},
          {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT},
          {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT},
          {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT},
          {4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT}};
}

const std::vector<uint32_t>& Conv2D::LoadShaderCode() const {
  static const std::vector<uint32_t> shader_code =
#include "Conv2D.comp.spv"
      ;
  return shader_code;
}

const std::string Conv2D::GetPipelineCache() const {
  const std::string cache_dir = PIPELINE_CACHE_DIR;
  if (cache_dir.empty()) {
    return "";
  }
  return cache_dir + "/conv2d.cache";
}

}  // namespace vkai
