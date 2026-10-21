#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

#include "VulkanCommandBuffer.h"
#include "layers/Clip.h"
#include "layers/Conv2D.h"
#include "utils/BufferUtils.h"
#include "utils/Synchronization.h"

TEST(ClipTest, NonzeroBoundsExtremesAndPartialWorkgroup) {
  core::vulkan::VulkanContext context(false, core::vulkan::QueueFamilyType::Compute,
                                      VK_NULL_HANDLE);
  context.Init();
  std::vector<float> x(517);
  for (size_t i = 0; i < x.size(); ++i) x[i] = (static_cast<int>(i) - 258) * .125F;
  x[0] = -std::numeric_limits<float>::infinity();
  x[1] = std::numeric_limits<float>::infinity();
  core::vulkan::VulkanBuffer input(&context, x.size() * sizeof(float),
                                   VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, vkai::kHostVisibleMemory);
  core::vulkan::VulkanBuffer output(&context, x.size() * sizeof(float),
                                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, vkai::kHostVisibleMemory);
  input.MapData([&](void* p) { std::memcpy(p, x.data(), x.size() * sizeof(float)); });
  vkai::Clip clip(&context, x.size(), -1.5F, 3.5F);
  clip.Init();
  auto command = core::vulkan::VulkanCommandBuffer::BeginOneTimeCommands(&context);
  clip.Execute(command.buffer(), input, output);
  vkai::Synchronization::InsertHostReadBarrier(command.buffer(), output);
  command.EndOneTimeCommands();
  output.MapData([&](void* p) {
    const auto* values = static_cast<float*>(p);
    for (size_t i = 0; i < x.size(); ++i) EXPECT_FLOAT_EQ(values[i], std::clamp(x[i], -1.5F, 3.5F));
  });
  EXPECT_THROW(vkai::Clip(&context, 0, 0, 6), std::invalid_argument);
  EXPECT_THROW(vkai::Clip(&context, 1, 6, 0), std::invalid_argument);
  EXPECT_THROW(vkai::Clip(&context, 1, std::numeric_limits<float>::quiet_NaN(), 6),
               std::invalid_argument);
  EXPECT_THROW(
      (vkai::Conv2D(&context, 3, 4, 3, 3, 1, 1, 1, 1, 0, 0, vkai::PaddingType::Zero, 1, 1, 1, 2)),
      std::invalid_argument);
  EXPECT_THROW(
      (vkai::Conv2D(&context, 4, 4, 3, 3, 1, 1, 1, 1, 0, 0, vkai::PaddingType::Zero, 1, 1, 1, 0)),
      std::invalid_argument);
}
