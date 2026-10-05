#pragma once

#include "layers/Layer.h"

namespace vkai {

class Linear : public Layer {
 public:
  Linear(core::vulkan::VulkanContext* context, int input_size, int output_size, int batch_size = 1);

  void MapWeights(const std::vector<float>& weights, const std::vector<float>& bias);

  void Init() override;

  void Execute(const VkCommandBuffer& command_buffer,
               const core::vulkan::VulkanBuffer& input_buffer,
               core::vulkan::VulkanBuffer& output_buffer) override;

 protected:
  std::vector<core::vulkan::BindingInfo> GetBindingInfo() const override;

  const std::vector<uint32_t>& LoadShaderCode() const override;

  const std::string GetPipelineCache() const override;

 private:
  core::vulkan::VulkanBuffer uniform_buffer_;
  core::vulkan::VulkanBuffer weights_buffer_;
  core::vulkan::VulkanBuffer bias_buffer_;

  struct UniformData {
    int input_size;
    int output_size;
    int batch_size;
  } uniform_data_;

  bool valid_ = false;
  bool weights_mapped_ = false;
};

}  // namespace vkai
