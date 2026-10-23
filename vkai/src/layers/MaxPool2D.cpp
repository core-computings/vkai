#include "layers/MaxPool2D.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <limits>

#include "utils/BufferUtils.h"

namespace vkai {
namespace {

int PoolOutputSize(int input, int kernel, int stride, int padding, bool ceil_mode) {
  const int64_t numerator = static_cast<int64_t>(input) + 2LL * padding - kernel;
  // C++ division truncates toward zero. Round explicitly for both modes, including
  // a ceil window whose kernel is slightly larger than the padded input.
  int64_t output = numerator / stride + 1;
  if (ceil_mode && numerator % stride > 0) ++output;
  if (!ceil_mode && numerator % stride < 0) --output;
  if (ceil_mode) {
    const int64_t valid_starts = (static_cast<int64_t>(input) + padding + stride - 1) / stride;
    output = std::min(output, valid_starts);
  }
  return output > 0 && output <= std::numeric_limits<int>::max() ? static_cast<int>(output) : 0;
}

}  // namespace

MaxPool2D::MaxPool2D(core::vulkan::VulkanContext* context, int input_channels, int input_height,
                     int input_width, int kernel_height, int kernel_width, int stride_height,
                     int stride_width, int batch_size, int padding_height, int padding_width,
                     bool ceil_mode)
    : Layer(context),
      uniform_buffer_(context, sizeof(UniformData), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                      kHostVisibleMemory),
      uniform_data_{
          .input_width = input_width,
          .input_height = input_height,
          .input_channels = input_channels,
          .batch_size = batch_size,
          .output_width = 0,
          .output_height = 0,
          .kernel_width = kernel_width,
          .kernel_height = kernel_height,
          .stride_width = stride_width,
          .stride_height = stride_height,
          .padding_width = padding_width,
          .padding_height = padding_height,
      } {
  if (input_channels <= 0 || input_height <= 0 || input_width <= 0 || kernel_height <= 0 ||
      kernel_width <= 0 || stride_height <= 0 || stride_width <= 0 || batch_size <= 0) {
    std::cerr << "MaxPool2D dimensions must be positive\n";
    return;
  }

  if (padding_height < 0 || padding_width < 0) {
    std::cerr << "MaxPool2D padding must not be negative\n";
    return;
  }
  uniform_data_.output_width =
      PoolOutputSize(input_width, kernel_width, stride_width, padding_width, ceil_mode);
  uniform_data_.output_height =
      PoolOutputSize(input_height, kernel_height, stride_height, padding_height, ceil_mode);
  if (uniform_data_.output_height <= 0 || uniform_data_.output_width <= 0) {
    std::cerr << "MaxPool2D kernel must fit within the input\n";
    return;
  }

  uniform_buffer_.MapData(
      [this](void* data) { std::memcpy(data, &uniform_data_, sizeof(UniformData)); });
  valid_ = true;
}

void MaxPool2D::Init() {
  if (!valid_) {
    std::cerr << "Cannot initialize an invalid MaxPool2D\n";
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

void MaxPool2D::Execute(const VkCommandBuffer& command_buffer,
                        const core::vulkan::VulkanBuffer& input_buffer,
                        core::vulkan::VulkanBuffer& output_buffer) {
  if (!valid_ || pipeline == VK_NULL_HANDLE) {
    std::cerr << "Cannot run an invalid or uninitialized MaxPool2D\n";
    return;
  }

  const VkDeviceSize required_input_size =
      static_cast<VkDeviceSize>(uniform_data_.batch_size) * uniform_data_.input_channels *
      uniform_data_.input_height * uniform_data_.input_width * sizeof(float);
  const VkDeviceSize required_output_size =
      static_cast<VkDeviceSize>(uniform_data_.batch_size) * uniform_data_.input_channels *
      uniform_data_.output_height * uniform_data_.output_width * sizeof(float);
  if (input_buffer.Size() < required_input_size || output_buffer.Size() < required_output_size) {
    std::cerr << "MaxPool2D input or output buffer is too small\n";
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

std::vector<core::vulkan::BindingInfo> MaxPool2D::GetBindingInfo() const {
  return {{0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT},
          {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT},
          {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT}};
}

const std::vector<uint32_t>& MaxPool2D::LoadShaderCode() const {
  static const std::vector<uint32_t> shader_code =
#include "MaxPool2D.comp.spv"
      ;
  return shader_code;
}

const std::string MaxPool2D::GetPipelineCache() const {
  const std::string cache_dir = PIPELINE_CACHE_DIR;
  if (cache_dir.empty()) {
    return "";
  }
  return cache_dir + "/maxpool2d.cache";
}

}  // namespace vkai
