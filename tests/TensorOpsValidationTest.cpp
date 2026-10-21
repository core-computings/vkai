#include <gtest/gtest.h>

#include "layers/BinaryElementwise.h"
#include "layers/NearestResize2D.h"
#include "layers/TensorConcat.h"
#include "layers/TensorTransform.h"
TEST(TensorOpsValidationTest, RejectsInvalidBroadcastsGatherBoundsConcatAxesAndResizeSizes) {
  core::vulkan::VulkanContext context(false, core::vulkan::QueueFamilyType::Compute,
                                      VK_NULL_HANDLE);
  context.Init();
  EXPECT_THROW((vkai::BinaryElementwise(&context, {2, 3}, {4}, vkai::BinaryKind::Mul)),
               std::invalid_argument);
  EXPECT_THROW(
      (vkai::BinaryElementwise(&context, std::vector<int>(9, 1), {}, vkai::BinaryKind::Add)),
      std::invalid_argument);
  EXPECT_THROW((vkai::TensorTransform(&context, 4, {2}, {4}, 0)), std::invalid_argument);
  EXPECT_THROW((vkai::TensorTransform(&context, 4, {2, 2}, {1})), std::invalid_argument);
  EXPECT_THROW((vkai::TensorConcat(&context, {{2, 3}, {3, 3}}, 1)), std::invalid_argument);
  EXPECT_THROW((vkai::TensorConcat(&context, {{2, 3}, {2, 3}}, 2)), std::invalid_argument);
  EXPECT_THROW((vkai::NearestResize2D(&context, 1, 0, 3, 2, 6)), std::invalid_argument);
}
