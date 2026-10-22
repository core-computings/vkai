#include "graph/Tensor.h"

#include <stdexcept>
#include <utility>

#include "utils/TensorUtils.h"

namespace vkai {

Tensor::Tensor(std::string name, std::vector<int64_t> shape)
    : name_(std::move(name)), shape_(std::move(shape)) {
  for (const int64_t dimension : shape_) {
    if (dimension < 0) {
      throw std::invalid_argument("Tensor dimensions must not be negative");
    }
  }
}

void Tensor::PopulateTensor(std::vector<float> data) {
  integer_data_.reset();
  data_ = std::move(data);
}

void Tensor::PopulateInt64Tensor(std::vector<int64_t> data) {
  integer_data_ = std::move(data);
  data_ = std::vector<float>{};
}

const std::vector<float>& Tensor::Data() const {
  if (!data_.has_value()) {
    throw std::logic_error("Tensor does not have CPU data");
  }
  return *data_;
}

void Tensor::SetBuffer(core::vulkan::VulkanBuffer&& buffer) {
  auto storage = std::make_shared<core::vulkan::VulkanBuffer>();
  *storage = std::move(buffer);
  SetBuffer(std::move(storage));
}

void Tensor::SetBuffer(std::shared_ptr<core::vulkan::VulkanBuffer> buffer) {
  if (!buffer || buffer->Size() < ByteSize()) {
    throw std::invalid_argument("Tensor storage is missing or too small: " + name_);
  }
  buffer_ = std::move(buffer);
}

VkDeviceSize Tensor::ByteSize() const {
  return static_cast<VkDeviceSize>(TensorElementCount(*this)) * sizeof(float);
}

core::vulkan::VulkanBuffer& Tensor::Buffer() {
  if (!buffer_) {
    throw std::logic_error("Tensor does not have a Vulkan buffer");
  }
  return *buffer_;
}

const core::vulkan::VulkanBuffer& Tensor::Buffer() const {
  if (!buffer_) {
    throw std::logic_error("Tensor does not have a Vulkan buffer");
  }
  return *buffer_;
}

}  // namespace vkai
