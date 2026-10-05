#pragma once

#include "layers/Layer.h"

namespace vkai {

// Float32 contiguous NCHW inference using fixed running statistics.
// Empty weight/bias vectors mean gamma=1/beta=0 respectively.
class BatchNorm2D : public Layer {
 public:
  BatchNorm2D(core::vulkan::VulkanContext* context, int channels, int elements_per_channel,
              int batch_size, float eps);

  void MapParameters(const std::vector<float>& running_mean, const std::vector<float>& running_var,
                     const std::vector<float>& weight = {}, const std::vector<float>& bias = {});
  void Init() override;
  void Execute(const VkCommandBuffer& command_buffer,
               const core::vulkan::VulkanBuffer& input_buffer,
               core::vulkan::VulkanBuffer& output_buffer) override;

 protected:
  std::vector<core::vulkan::BindingInfo> GetBindingInfo() const override;
  const std::vector<uint32_t>& LoadShaderCode() const override;
  const std::string GetPipelineCache() const override;

 private:
  struct alignas(16) UniformData {
    int channels;
    int elements_per_channel;
    int element_count;
    int reserved;
  } uniform_data_{};
  float eps_ = 0.0F;
  core::vulkan::VulkanBuffer uniform_buffer_;
  core::vulkan::VulkanBuffer parameters_buffer_;
  bool parameters_mapped_ = false;
};

}  // namespace vkai
