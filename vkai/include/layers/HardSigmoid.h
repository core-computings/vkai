#pragma once

#include "layers/Layer.h"

namespace vkai {

class HardSigmoid : public Layer {
 public:
  HardSigmoid(core::vulkan::VulkanContext* context, int element_count, float alpha, float beta);

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

  struct UniformData {
    int element_count;
    float alpha;
    float beta;
    int reserved;
  } uniform_data_;

  bool valid_ = false;
};

}  // namespace vkai
