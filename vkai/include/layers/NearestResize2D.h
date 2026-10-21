#pragma once

#include "layers/Layer.h"

namespace vkai {

// Nearest resize for contiguous float32 NCHW tensors using ONNX asymmetric coordinates and floor
// rounding.
class NearestResize2D : public Layer {
 public:
  NearestResize2D(core::vulkan::VulkanContext* context, int input_channels, int input_height,
                  int input_width, int output_height, int output_width, int batch_size = 1);

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
    int reserved_0;
    int reserved_1;
  } uniform_data_;

  static_assert(sizeof(UniformData) == 32);

  core::vulkan::VulkanBuffer uniform_buffer_;
  bool valid_ = false;
};

}  // namespace vkai
