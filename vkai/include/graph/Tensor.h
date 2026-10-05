#pragma once

#include <vulkan/vulkan.h>

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace vkai {

class Tensor {
 public:
  Tensor(std::string name, std::vector<int64_t> shape);

  const std::string& Name() const { return name_; }

  const std::vector<int64_t>& GetShape() const { return shape_; }

  void SetData(std::vector<float> data);

  bool HasData() const { return data_.has_value(); }

  const std::vector<float>& Data() const;

  // The graph does not own the Vulkan resource. Allocation is handled by the
  // inference executor after graph construction.
  void SetBuffer(VkBuffer buffer);

  bool HasBuffer() const { return buffer_.has_value(); }

  VkBuffer Buffer() const;

 private:
  std::string name_;
  std::vector<int64_t> shape_;
  std::optional<std::vector<float>> data_;
  std::optional<VkBuffer> buffer_;
};

}  // namespace vkai
