#pragma once

#include "layers/Layer.h"

namespace vkai {

// Float32 NCHW transposed convolution, group=1, IOHW weights and optional bias.
// Supports symmetric padding, dilation, and output_padding < stride.
class ConvTranspose2D : public Layer {
 public:
  ConvTranspose2D(core::vulkan::VulkanContext* context, int input_channels, int output_channels,
                  int input_height, int input_width, int kernel_height, int kernel_width,
                  int stride_height, int stride_width, int padding_height, int padding_width,
                  int batch_size = 1, int dilation_height = 1, int dilation_width = 1,
                  int output_padding_height = 0, int output_padding_width = 0);

  void MapWeights(const std::vector<float>& weights, const std::vector<float>& bias = {});

  void Init() override;

  void Execute(const VkCommandBuffer& command_buffer,
               const core::vulkan::VulkanBuffer& input_buffer,
               core::vulkan::VulkanBuffer& output_buffer) override;

  int OutputHeight() const { return uniform_data_.output_height; }
  int OutputWidth() const { return uniform_data_.output_width; }

 protected:
  std::vector<core::vulkan::BindingInfo> GetBindingInfo() const override;

  const std::vector<uint32_t>& LoadShaderCode() const override;

  const std::string GetPipelineCache() const override;

 private:
  struct alignas(16) UniformData {
    int input_width;
    int input_height;
    int input_channels;
    int batch_size;

    int output_width;
    int output_height;
    int output_channels;
    int reserved;

    int kernel_width;
    int kernel_height;
    int stride_width;
    int stride_height;

    int padding_width;
    int padding_height;
    int dilation_width;
    int dilation_height;
  };

  static_assert(sizeof(UniformData) == 64);

  core::vulkan::VulkanBuffer uniform_buffer_;
  core::vulkan::VulkanBuffer weights_buffer_;
  core::vulkan::VulkanBuffer bias_buffer_;
  UniformData uniform_data_;
  bool valid_ = false;
  bool weights_mapped_ = false;
};

}  // namespace vkai
