#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "VulkanBuffer.h"

namespace vkai {

class Tensor {
 public:
  Tensor(std::string name, std::vector<int64_t> shape);

  const std::string& Name() const { return name_; }

  const std::vector<int64_t>& GetShape() const { return shape_; }

  void PopulateTensor(std::vector<float> data);

  void PopulateInt64Tensor(std::vector<int64_t> data);

  bool HasInt64Data() const { return integer_data_.has_value(); }
  const std::vector<int64_t>& Int64Data() const { return integer_data_.value(); }

  bool HasData() const { return data_.has_value(); }

  const std::vector<float>& Data() const;

  // Owns or shares storage assigned by the inference executor. Size is capacity;
  // ByteSize() is the tensor's logical payload, which may be smaller.
  void SetBuffer(core::vulkan::VulkanBuffer&& buffer);
  void SetBuffer(std::shared_ptr<core::vulkan::VulkanBuffer> buffer);

  bool HasBuffer() const { return buffer_ != nullptr; }

  VkDeviceSize ByteSize() const;

  core::vulkan::VulkanBuffer& Buffer();

  const core::vulkan::VulkanBuffer& Buffer() const;

 private:
  std::string name_;
  std::vector<int64_t> shape_;
  std::optional<std::vector<float>> data_;
  std::optional<std::vector<int64_t>> integer_data_;
  std::shared_ptr<core::vulkan::VulkanBuffer> buffer_;
};

}  // namespace vkai
