#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <vector>

#include "VulkanContext.h"
#include "layers/BatchNorm2D.h"
#include "utils/BufferUtils.h"

namespace vkai {
namespace test {
namespace {

void CheckReference(const std::string& name) {
  const auto directory =
      std::filesystem::path(VKAI_SOURCE_DIR) / "python/pspnet/test_data/batchnorm";
  std::ifstream shape_file(directory / (name + ".txt"));
  int n, c, h, w;
  float eps;
  ASSERT_TRUE(static_cast<bool>(shape_file >> n >> c >> h >> w >> eps))
      << "Generate BN fixtures using export_batchnorm_reference.py";
  ASSERT_GT(n, 0);
  ASSERT_GT(c, 0);
  ASSERT_GT(h, 0);
  ASSERT_GT(w, 0);
  std::vector<float> input, expected, mean, variance, weight, bias;
  auto read = [&](const char* key, std::vector<float>& values) {
    std::ifstream file(directory / (name + "_" + key + ".bin"), std::ios::binary | std::ios::ate);
    if (!file) return false;
    const auto bytes = file.tellg();
    if (bytes < 0 || bytes % sizeof(float) != 0) return false;
    values.resize(static_cast<size_t>(bytes) / sizeof(float));
    file.seekg(0);
    if (bytes == 0) return true;
    return static_cast<bool>(file.read(reinterpret_cast<char*>(values.data()), bytes));
  };
  ASSERT_TRUE(read("input", input));
  ASSERT_TRUE(read("output", expected));
  ASSERT_TRUE(read("mean", mean));
  ASSERT_TRUE(read("variance", variance));
  ASSERT_TRUE(read("weight", weight));
  ASSERT_TRUE(read("bias", bias));
  ASSERT_EQ(input.size(), static_cast<size_t>(n) * c * h * w);
  ASSERT_EQ(expected.size(), input.size());
  core::vulkan::VulkanContext context(false, core::vulkan::QueueFamilyType::Compute,
                                      VK_NULL_HANDLE);
  context.Init();
  core::vulkan::VulkanBuffer input_buffer(&context, input.size() * sizeof(float),
                                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                          vkai::kHostVisibleMemory);
  core::vulkan::VulkanBuffer output_buffer(&context, input.size() * sizeof(float),
                                           VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                           vkai::kHostVisibleMemory);
  input_buffer.MapData(
      [&](void* data) { std::memcpy(data, input.data(), input.size() * sizeof(float)); });
  BatchNorm2D bn(&context, c, h * w, n, eps);
  bn.MapParameters(mean, variance, weight, bias);
  bn.Init();
  auto command = core::vulkan::VulkanCommandBuffer::BeginOneTimeCommands(&context);
  bn.Execute(command.buffer(), input_buffer, output_buffer);
  command.EndOneTimeCommands();
  std::vector<float> actual(expected.size());
  output_buffer.MapData(
      [&](void* data) { std::memcpy(actual.data(), data, actual.size() * sizeof(float)); });
  for (size_t i = 0; i < actual.size(); ++i) {
    ASSERT_NEAR(actual[i], expected[i], 1e-5F + 1e-5F * std::abs(expected[i])) << "index " << i;
  }
}

TEST(BatchNorm2DTest, MatchesReferenceBeforeRelu) { CheckReference("pspnet_bn2"); }
TEST(BatchNorm2DTest, MatchesReferenceBeforeResidualAdd) { CheckReference("pspnet_bn3"); }
TEST(BatchNorm2DTest, BatchedNonDefaultEpsilonAndZeroVariance) { CheckReference("batched"); }
TEST(BatchNorm2DTest, SupportsNonAffineNormalization) { CheckReference("no_affine"); }

TEST(BatchNorm2DTest, RejectsInvalidParameters) {
  core::vulkan::VulkanContext context(false, core::vulkan::QueueFamilyType::Compute,
                                      VK_NULL_HANDLE);
  context.Init();
  BatchNorm2D bn(&context, 1, 1, 1, 1e-5F);
  EXPECT_THROW(bn.MapParameters({0}, {-1}), std::invalid_argument);
  EXPECT_THROW(bn.MapParameters({0}, {1}, {1, 2}), std::invalid_argument);
  EXPECT_THROW(BatchNorm2D(&context, 1, 1, 1, -1.0F), std::invalid_argument);
  EXPECT_THROW(BatchNorm2D(&context, std::numeric_limits<int>::max(), 2, 1, 1e-5F),
               std::invalid_argument);
}

}  // namespace
}  // namespace test
}  // namespace vkai
