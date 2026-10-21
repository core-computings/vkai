#include "layers/TensorTransform.h"

#include <cstring>
#include <stdexcept>

#include "utils/BufferUtils.h"
#include "utils/ShapeUtils.h"
namespace vkai {
TensorTransform::TensorTransform(core::vulkan::VulkanContext* context, int input_count,
                                 const std::vector<int>& shape, const std::vector<int>& strides,
                                 int offset)
    : Layer(context),
      uniform_(context, sizeof(UniformData), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
               kHostVisibleMemory),
      input_count_(input_count) {
  if (shape.size() > 8 || shape.size() != strides.size() || input_count <= 0)
    throw std::invalid_argument("Invalid TensorTransform configuration");
  data_.count = ElementCount(shape);
  data_.rank = shape.size();
  data_.offset = offset;
  int64_t lo = offset, hi = offset;
  for (size_t i = 0; i < shape.size(); ++i) {
    data_.dims[i] = shape[i];
    data_.strides[i] = strides[i];
    int64_t delta = int64_t(shape[i] - 1) * strides[i];
    lo += std::min<int64_t>(delta, 0);
    hi += std::max<int64_t>(delta, 0);
  }
  if (lo < 0 || hi >= input_count)
    throw std::invalid_argument("TensorTransform reads outside input");
  uniform_.MapData([&](void* p) { std::memcpy(p, &data_, sizeof(data_)); });
}
void TensorTransform::Init() {
  VulkanCompute::Init();
  CreateUniformBufferDescriptorSet(0, uniform_);
  vkUpdateDescriptorSets(context_->logical_device, 1, &writes_[0], 0, nullptr);
}
void TensorTransform::Execute(const VkCommandBuffer& cmd, const core::vulkan::VulkanBuffer& in,
                              core::vulkan::VulkanBuffer& out) {
  if (pipeline == VK_NULL_HANDLE || in.Size() < VkDeviceSize(input_count_) * 4 ||
      out.Size() < VkDeviceSize(data_.count) * 4)
    throw std::invalid_argument("Invalid TensorTransform buffers");
  UpdateStorageBufferDescriptors(1, 2, in, out);
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout, 0, 1,
                          &descriptor_set_, 0, nullptr);
  vkCmdDispatch(cmd, (uint32_t(data_.count) + 255) / 256, 1, 1);
}
std::vector<core::vulkan::BindingInfo> TensorTransform::GetBindingInfo() const {
  return {{0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT},
          {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT},
          {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT}};
}
const std::vector<uint32_t>& TensorTransform::LoadShaderCode() const {
  static const std::vector<uint32_t> code =
#include "TensorTransform.comp.spv"
      ;
  return code;
}
}  // namespace vkai
