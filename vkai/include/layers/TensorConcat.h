#pragma once
#include "layers/Layer.h"
namespace vkai {
// Concatenates contiguous float32 tensors on any axis using GPU buffer copies.
class TensorConcat : public Layer {
 public:
  TensorConcat(core::vulkan::VulkanContext*, const std::vector<std::vector<int>>& shapes, int axis);
  void Init() override {}
  void Execute(const VkCommandBuffer&, const core::vulkan::VulkanBuffer&,
               core::vulkan::VulkanBuffer&) override;
  void Execute(const VkCommandBuffer&, const std::vector<const core::vulkan::VulkanBuffer*>&,
               core::vulkan::VulkanBuffer&);

 protected:
  std::vector<core::vulkan::BindingInfo> GetBindingInfo() const override { return {}; }
  const std::vector<uint32_t>& LoadShaderCode() const override {
    static const std::vector<uint32_t> empty;
    return empty;
  }

 private:
  int outer_ = 1, inner_ = 1, total_axis_ = 0;
  std::vector<int> extents_;
};
}  // namespace vkai
