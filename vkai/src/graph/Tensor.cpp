#include "graph/Tensor.h"

#include <stdexcept>
#include <utility>

namespace vkai {

Tensor::Tensor(std::string name, std::vector<int64_t> shape)
    : name_(std::move(name)), shape_(std::move(shape)) {
  for (const int64_t dimension : shape_) {
    if (dimension < 0) {
      throw std::invalid_argument("Tensor dimensions must not be negative");
    }
  }
}

void Tensor::SetData(std::vector<float> data) { data_ = std::move(data); }

const std::vector<float>& Tensor::Data() const {
  if (!data_.has_value()) {
    throw std::logic_error("Tensor does not have CPU data");
  }
  return *data_;
}

void Tensor::SetBuffer(core::vulkan::VulkanBuffer&& buffer) {
  if (!buffer_.has_value()) {
    buffer_.emplace();
  }
  *buffer_ = std::move(buffer);
}

core::vulkan::VulkanBuffer& Tensor::Buffer() {
  if (!buffer_.has_value()) {
    throw std::logic_error("Tensor does not have a Vulkan buffer");
  }
  return *buffer_;
}

const core::vulkan::VulkanBuffer& Tensor::Buffer() const {
  if (!buffer_.has_value()) {
    throw std::logic_error("Tensor does not have a Vulkan buffer");
  }
  return *buffer_;
}

}  // namespace vkai
