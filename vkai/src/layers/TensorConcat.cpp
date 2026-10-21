#include "layers/TensorConcat.h"

#include <limits>
#include <stdexcept>

#include "utils/ShapeUtils.h"
namespace vkai {
TensorConcat::TensorConcat(core::vulkan::VulkanContext* context,
                           const std::vector<std::vector<int>>& shapes, int axis)
    : Layer(context) {
  if (shapes.empty() || shapes[0].empty() || axis < 0 || axis >= int(shapes[0].size()))
    throw std::invalid_argument("Invalid concat axis or inputs");
  const auto& first = shapes[0];
  for (const auto& shape : shapes) {
    ElementCount(shape);
    if (shape.size() != first.size()) throw std::invalid_argument("Concat rank mismatch");
    for (size_t i = 0; i < shape.size(); ++i)
      if (int(i) != axis && shape[i] != first[i])
        throw std::invalid_argument("Concat shape mismatch");
    extents_.push_back(shape[axis]);
    if (total_axis_ > std::numeric_limits<int>::max() - shape[axis])
      throw std::invalid_argument("Concat axis is too large");
    total_axis_ += shape[axis];
  }
  auto output = first;
  output[axis] = total_axis_;
  ElementCount(output);
  for (int i = 0; i < axis; ++i) outer_ *= first[i];
  for (size_t i = axis + 1; i < first.size(); ++i) inner_ *= first[i];
}
void TensorConcat::Execute(const VkCommandBuffer&, const core::vulkan::VulkanBuffer&,
                           core::vulkan::VulkanBuffer&) {
  throw std::logic_error("Concat requires all inputs");
}
void TensorConcat::Execute(const VkCommandBuffer& cmd,
                           const std::vector<const core::vulkan::VulkanBuffer*>& inputs,
                           core::vulkan::VulkanBuffer& out) {
  if (inputs.size() != extents_.size() ||
      out.Size() < VkDeviceSize(outer_) * total_axis_ * inner_ * 4)
    throw std::invalid_argument("Concat buffer mismatch");
  const VkMemoryBarrier barrier{
      .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
      .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
      .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT |
                       VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT};
  constexpr VkPipelineStageFlags stages =
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT;
  vkCmdPipelineBarrier(cmd, stages, stages, 0, 1, &barrier, 0, nullptr, 0, nullptr);
  int axis_offset = 0;
  for (size_t i = 0; i < inputs.size(); ++i) {
    const auto* input = inputs[i];
    VkDeviceSize chunk = VkDeviceSize(extents_[i]) * inner_ * 4;
    if (!input || input->Size() < VkDeviceSize(outer_) * chunk)
      throw std::invalid_argument("Concat input buffer too small");
    std::vector<VkBufferCopy> regions(outer_);
    for (int row = 0; row < outer_; ++row)
      regions[row] = {VkDeviceSize(row) * chunk,
                      VkDeviceSize(int64_t(row) * total_axis_ + axis_offset) * inner_ * 4, chunk};
    vkCmdCopyBuffer(cmd, input->buffer, out.buffer, regions.size(), regions.data());
    axis_offset += extents_[i];
  }
  vkCmdPipelineBarrier(cmd, stages, stages, 0, 1, &barrier, 0, nullptr, 0, nullptr);
}
}  // namespace vkai
