#include "layers/BinaryElementwise.h"

#include <cstring>
#include <stdexcept>

#include "utils/BufferUtils.h"
#include "utils/ShapeUtils.h"
namespace vkai {
BinaryElementwise::BinaryElementwise(core::vulkan::VulkanContext* context,
                                     const std::vector<int>& a, const std::vector<int>& b,
                                     BinaryKind kind)
    : Layer(context),
      uniform_(context, sizeof(UniformData), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
               kHostVisibleMemory),
      a_count_(ElementCount(a)),
      b_count_(ElementCount(b)) {
  auto shape = BroadcastShape(a, b);
  if (shape.size() > 8 || kind < BinaryKind::Add || kind > BinaryKind::Div)
    throw std::invalid_argument("Unsupported BinaryElementwise configuration");
  data_.rank = shape.size();
  data_.count = ElementCount(shape);
  data_.kind = static_cast<int>(kind);
  auto as = ContiguousStrides(a), bs = ContiguousStrides(b);
  for (size_t i = 0; i < shape.size(); ++i) {
    data_.dims[i] = shape[i];
    if (i >= shape.size() - a.size()) {
      size_t j = i - (shape.size() - a.size());
      data_.a_strides[i] = a[j] == 1 ? 0 : as[j];
    }
    if (i >= shape.size() - b.size()) {
      size_t j = i - (shape.size() - b.size());
      data_.b_strides[i] = b[j] == 1 ? 0 : bs[j];
    }
  }
  uniform_.MapData([&](void* p) { std::memcpy(p, &data_, sizeof(data_)); });
}
void BinaryElementwise::Init() {
  VulkanCompute::Init();
  CreateUniformBufferDescriptorSet(0, uniform_);
  vkUpdateDescriptorSets(context_->logical_device, 1, &writes_[0], 0, nullptr);
}
void BinaryElementwise::Execute(const VkCommandBuffer&, const core::vulkan::VulkanBuffer&,
                                core::vulkan::VulkanBuffer&) {
  throw std::logic_error("BinaryElementwise requires two inputs");
}
void BinaryElementwise::Execute(const VkCommandBuffer& cmd, const core::vulkan::VulkanBuffer& a,
                                const core::vulkan::VulkanBuffer& b,
                                core::vulkan::VulkanBuffer& out) {
  if (pipeline == VK_NULL_HANDLE || a.Size() < VkDeviceSize(a_count_) * 4 ||
      b.Size() < VkDeviceSize(b_count_) * 4 || out.Size() < VkDeviceSize(data_.count) * 4)
    throw std::invalid_argument("Invalid BinaryElementwise buffers or uninitialized pipeline");
  const VkDescriptorBufferInfo info[] = {
      {a.buffer, 0, a.Size()}, {b.buffer, 0, b.Size()}, {out.buffer, 0, out.Size()}};
  VkWriteDescriptorSet writes[3]{};
  for (uint32_t i = 0; i < 3; ++i) {
    writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[i].dstSet = descriptor_set_;
    writes[i].dstBinding = i + 1;
    writes[i].descriptorCount = 1;
    writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[i].pBufferInfo = &info[i];
  }
  vkUpdateDescriptorSets(context_->logical_device, 3, writes, 0, nullptr);
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout, 0, 1,
                          &descriptor_set_, 0, nullptr);
  vkCmdDispatch(cmd, (uint32_t(data_.count) + 255) / 256, 1, 1);
}
std::vector<core::vulkan::BindingInfo> BinaryElementwise::GetBindingInfo() const {
  return {{0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT},
          {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT},
          {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT},
          {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT}};
}
const std::vector<uint32_t>& BinaryElementwise::LoadShaderCode() const {
  static const std::vector<uint32_t> code =
#include "BinaryElementwise.comp.spv"
      ;
  return code;
}
}  // namespace vkai
