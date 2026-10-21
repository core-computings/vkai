#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <vector>

#include "VulkanCommandBuffer.h"
#include "layers/Sigmoid.h"
#include "utils/BufferUtils.h"
#include "utils/Synchronization.h"

TEST(SigmoidTest, HandlesExtremesAndPartialWorkgroups) {
  core::vulkan::VulkanContext context(false, core::vulkan::QueueFamilyType::Compute,
                                      VK_NULL_HANDLE);
  context.Init();
  std::vector<float> x(517);
  for (size_t i = 0; i < x.size(); ++i) x[i] = (static_cast<int>(i) - 258) * .25F;
  x[0] = -1000;
  x[1] = 1000;
  x[2] = 0;
  core::vulkan::VulkanBuffer input(&context, x.size() * sizeof(float),
                                   VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, vkai::kHostVisibleMemory);
  core::vulkan::VulkanBuffer output(&context, x.size() * sizeof(float),
                                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, vkai::kHostVisibleMemory);
  input.MapData([&](void* data) { std::memcpy(data, x.data(), x.size() * sizeof(float)); });
  vkai::Sigmoid layer(&context, x.size());
  layer.Init();
  auto command = core::vulkan::VulkanCommandBuffer::BeginOneTimeCommands(&context);
  layer.Execute(command.buffer(), input, output);
  vkai::Synchronization::InsertHostReadBarrier(command.buffer(), output);
  command.EndOneTimeCommands();
  std::vector<float> actual(x.size());
  output.MapData(
      [&](void* data) { std::memcpy(actual.data(), data, actual.size() * sizeof(float)); });
  for (size_t i = 0; i < x.size(); ++i) {
    EXPECT_TRUE(std::isfinite(actual[i]));
    EXPECT_NEAR(actual[i], 1.0 / (1.0 + std::exp(-static_cast<double>(x[i]))), 1e-6);
  }
  EXPECT_THROW(vkai::Sigmoid(&context, 0), std::invalid_argument);
}
