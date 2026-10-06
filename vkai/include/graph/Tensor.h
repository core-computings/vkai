#pragma once

#include <cstdint>
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

  bool HasData() const { return data_.has_value(); }

  const std::vector<float>& Data() const;

  // Takes ownership of a buffer allocated by the inference executor.
  void SetBuffer(core::vulkan::VulkanBuffer&& buffer);

  bool HasBuffer() const { return buffer_.has_value(); }

  core::vulkan::VulkanBuffer& Buffer();

  const core::vulkan::VulkanBuffer& Buffer() const;

 private:
  std::string name_;
  std::vector<int64_t> shape_;
  std::optional<std::vector<float>> data_;
  std::optional<core::vulkan::VulkanBuffer> buffer_;
};

}  // namespace vkai
