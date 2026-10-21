#pragma once

#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <memory>
#include <vector>

#include "VulkanBuffer.h"
#include "VulkanCommandBuffer.h"
#include "VulkanContext.h"
#include "utils/BufferUtils.h"
#include "utils/Synchronization.h"

namespace vkai::test {

class GpuTest : public ::testing::Test {
 protected:
  core::vulkan::VulkanContext context{false, core::vulkan::QueueFamilyType::Compute,
                                      VK_NULL_HANDLE};
  void SetUp() override { context.Init(); }
};

inline std::vector<float> TestValues(size_t count, float scale = .125F, int offset = 0) {
  std::vector<float> values(count);
  for (size_t i = 0; i < count; ++i) values[i] = (static_cast<int>((i + offset) % 23) - 11) * scale;
  return values;
}

inline std::unique_ptr<core::vulkan::VulkanBuffer> Upload(core::vulkan::VulkanContext& context,
                                                          const std::vector<float>& values) {
  auto buffer = std::make_unique<core::vulkan::VulkanBuffer>(
      &context, values.size() * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
      kHostVisibleMemory);
  buffer->MapData(
      [&](void* data) { std::memcpy(data, values.data(), values.size() * sizeof(float)); });
  return buffer;
}

template <typename Record>
std::vector<float> RunLayer(core::vulkan::VulkanContext& context, const std::vector<float>& values,
                            size_t output_count, Record record) {
  auto input = Upload(context, values);
  core::vulkan::VulkanBuffer output(&context, output_count * sizeof(float),
                                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, kHostVisibleMemory);
  auto command = core::vulkan::VulkanCommandBuffer::BeginOneTimeCommands(&context);
  record(command.buffer(), *input, output);
  Synchronization::InsertHostReadBarrier(command.buffer(), output);
  command.EndOneTimeCommands();
  std::vector<float> actual(output_count);
  output.MapData(
      [&](void* data) { std::memcpy(actual.data(), data, actual.size() * sizeof(float)); });
  return actual;
}

inline void ExpectNear(const std::vector<float>& actual, const std::vector<float>& expected,
                       float atol = 1e-5F, float rtol = 1e-5F) {
  ASSERT_EQ(actual.size(), expected.size());
  for (size_t i = 0; i < actual.size(); ++i) {
    EXPECT_TRUE(std::isfinite(actual[i])) << "index " << i;
    EXPECT_NEAR(actual[i], expected[i], atol + rtol * std::abs(expected[i])) << "index " << i;
  }
}

}  // namespace vkai::test
