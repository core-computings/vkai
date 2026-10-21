#pragma once
#include "layers/Layer.h"
namespace vkai {
enum class BinaryKind { Add = 0, Sub = 1, Mul = 2, Div = 3 };
// ONNX multidirectional broadcasting of contiguous float32 tensors, rank <= 8.
class BinaryElementwise : public Layer {
 public:
  BinaryElementwise(core::vulkan::VulkanContext* context, const std::vector<int>& a,
                    const std::vector<int>& b, BinaryKind kind);
  void Init() override;
  void Execute(const VkCommandBuffer&, const core::vulkan::VulkanBuffer&,
               core::vulkan::VulkanBuffer&) override;
  void Execute(const VkCommandBuffer&, const core::vulkan::VulkanBuffer&,
               const core::vulkan::VulkanBuffer&, core::vulkan::VulkanBuffer&);

 protected:
  std::vector<core::vulkan::BindingInfo> GetBindingInfo() const override;
  const std::vector<uint32_t>& LoadShaderCode() const override;

 private:
  struct alignas(16) UniformData {
    int count, rank, kind, reserved;
    int dims[8], a_strides[8], b_strides[8];
  } data_{};
  core::vulkan::VulkanBuffer uniform_;
  int a_count_, b_count_;
};
}  // namespace vkai
