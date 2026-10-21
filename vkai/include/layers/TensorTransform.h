#pragma once
#include "layers/Layer.h"
namespace vkai {
// Affine gather for Slice and Transpose. Strides are indexed by output axes.
class TensorTransform : public Layer {
 public:
  TensorTransform(core::vulkan::VulkanContext*, int input_count,
                  const std::vector<int>& output_shape, const std::vector<int>& input_strides,
                  int offset = 0);
  void Init() override;
  void Execute(const VkCommandBuffer&, const core::vulkan::VulkanBuffer&,
               core::vulkan::VulkanBuffer&) override;

 protected:
  std::vector<core::vulkan::BindingInfo> GetBindingInfo() const override;
  const std::vector<uint32_t>& LoadShaderCode() const override;

 private:
  struct alignas(16) UniformData {
    int count, rank, offset, reserved;
    int dims[8], strides[8];
  } data_{};
  core::vulkan::VulkanBuffer uniform_;
  int input_count_;
};
}  // namespace vkai
