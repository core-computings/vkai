#include "layers/ConvTranspose2D.h"

#include <cstring>
#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>

#include "utils/BufferUtils.h"

namespace vkai {
namespace {
VkDeviceSize CheckedWeightBytes(int ic, int oc, int h, int w, int kh, int kw, int sh, int sw,
                                int ph, int pw, int n, int dh, int dw, int oph, int opw) {
  if (ic <= 0 || oc <= 0 || h <= 0 || w <= 0 || kh <= 0 || kw <= 0 || sh <= 0 || sw <= 0 ||
      ph < 0 || pw < 0 || n <= 0 || dh <= 0 || dw <= 0 || oph < 0 || opw < 0 || oph >= sh ||
      opw >= sw) {
    throw std::invalid_argument("Invalid ConvTranspose2D dimensions or output padding");
  }
  const int64_t oh = (int64_t(h) - 1) * sh - 2LL * ph + int64_t(dh) * (kh - 1) + oph + 1;
  const int64_t ow = (int64_t(w) - 1) * sw - 2LL * pw + int64_t(dw) * (kw - 1) + opw + 1;
  constexpr int64_t limit = std::numeric_limits<int>::max();
  if (oh <= 0 || ow <= 0 || oh > limit || ow > limit) {
    throw std::invalid_argument("Invalid ConvTranspose2D output dimensions");
  }
  auto count = [](std::initializer_list<int64_t> extents) {
    int64_t result = 1;
    for (auto d : extents) {
      if (d > limit / result) throw std::invalid_argument("ConvTranspose2D tensor is too large");
      result *= d;
    }
    return result;
  };
  count({n, ic, h, w});
  count({n, oc, oh, ow});
  return static_cast<VkDeviceSize>(count({ic, oc, kh, kw})) * sizeof(float);
}
}  // namespace

ConvTranspose2D::ConvTranspose2D(core::vulkan::VulkanContext* context, int input_channels,
                                 int output_channels, int input_height, int input_width,
                                 int kernel_height, int kernel_width, int stride_height,
                                 int stride_width, int padding_height, int padding_width,
                                 int batch_size, int dilation_height, int dilation_width,
                                 int output_padding_height, int output_padding_width)
    : Layer(context),
      uniform_buffer_(context, sizeof(UniformData), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                      kHostVisibleMemory),
      weights_buffer_(
          context,
          CheckedWeightBytes(input_channels, output_channels, input_height, input_width,
                             kernel_height, kernel_width, stride_height, stride_width,
                             padding_height, padding_width, batch_size, dilation_height,
                             dilation_width, output_padding_height, output_padding_width),
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
          .reserved = 0,
          .kernel_width = kernel_width,
          .kernel_height = kernel_height,
          .stride_width = stride_width,
          .stride_height = stride_height,
          .padding_width = padding_width,
          .padding_height = padding_height,
          .dilation_width = dilation_width,
          .dilation_height = dilation_height,
      } {
  const int64_t effective_kernel_width = int64_t(dilation_width) * (kernel_width - 1) + 1;
  const int64_t effective_kernel_height = int64_t(dilation_height) * (kernel_height - 1) + 1;
  uniform_data_.output_width =
      static_cast<int>((int64_t(input_width) - 1) * stride_width - 2LL * padding_width +
                       effective_kernel_width + output_padding_width);
  uniform_data_.output_height =
      static_cast<int>((int64_t(input_height) - 1) * stride_height - 2LL * padding_height +
                       effective_kernel_height + output_padding_height);
  uniform_buffer_.MapData(
      [this](void* data) { std::memcpy(data, &uniform_data_, sizeof(UniformData)); });
  valid_ = true;
}

void ConvTranspose2D::MapWeights(const std::vector<float>& weights,
                                 const std::vector<float>& bias) {
  if (!valid_) {
    throw std::logic_error("Cannot map weights for an invalid ConvTranspose2D");
  }
  const size_t expected_weight_count = static_cast<size_t>(uniform_data_.output_channels) *
                                       uniform_data_.input_channels * uniform_data_.kernel_height *
                                       uniform_data_.kernel_width;
  if (weights.size() != expected_weight_count ||
      (!bias.empty() && bias.size() != static_cast<size_t>(uniform_data_.output_channels))) {
    throw std::invalid_argument(
        "ConvTranspose2D weights or bias size does not match its dimensions");
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

void ConvTranspose2D::Init() {
  if (!valid_ || !weights_mapped_) {
    throw std::logic_error(
        "ConvTranspose2D must have valid dimensions and mapped weights before Init");
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

void ConvTranspose2D::Execute(const VkCommandBuffer& command_buffer,
                              const core::vulkan::VulkanBuffer& input_buffer,
                              core::vulkan::VulkanBuffer& output_buffer) {
  if (!valid_ || pipeline == VK_NULL_HANDLE) {
    std::cerr << "Cannot run an invalid or uninitialized ConvTranspose2D\n";
    return;
  }

  const VkDeviceSize required_input_size =
      static_cast<VkDeviceSize>(uniform_data_.batch_size) * uniform_data_.input_channels *
      uniform_data_.input_height * uniform_data_.input_width * sizeof(float);
  const VkDeviceSize required_output_size =
      static_cast<VkDeviceSize>(uniform_data_.batch_size) * uniform_data_.output_channels *
      uniform_data_.output_height * uniform_data_.output_width * sizeof(float);
  if (input_buffer.Size() < required_input_size) {
    std::cerr << "ConvTranspose2D input buffer is too small\n";
    return;
  }
  if (output_buffer.Size() < required_output_size) {
    std::cerr << "ConvTranspose2D output buffer is too small\n";
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

std::vector<core::vulkan::BindingInfo> ConvTranspose2D::GetBindingInfo() const {
  return {{0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT},
          {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT},
          {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT},
          {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT},
          {4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT}};
}

const std::vector<uint32_t>& ConvTranspose2D::LoadShaderCode() const {
  static const std::vector<uint32_t> shader_code =
#include "ConvTranspose2D.comp.spv"
      ;
  return shader_code;
}

const std::string ConvTranspose2D::GetPipelineCache() const {
  const std::string cache_dir = PIPELINE_CACHE_DIR;
  if (cache_dir.empty()) {
    return "";
  }
  return cache_dir + "/conv_transpose2d.cache";
}

}  // namespace vkai
