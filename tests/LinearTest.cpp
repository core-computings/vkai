#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "VulkanBuffer.h"
#include "VulkanCommandBuffer.h"
#include "VulkanContext.h"
#include "layers/Linear.h"
#include "utils/BufferUtils.h"
#include "utils/WeightsLoader.h"

namespace {

bool ReadFloatBinary(const std::filesystem::path& path, std::vector<float>& values) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file.is_open()) {
    std::cerr << "Failed to open reference data: " << path << '\n';
    return false;
  }

  const std::streamsize byte_size = file.tellg();
  if (byte_size < 0 || byte_size % sizeof(float) != 0) {
    std::cerr << "Invalid float32 reference data: " << path << '\n';
    return false;
  }

  values.resize(static_cast<size_t>(byte_size) / sizeof(float));
  file.seekg(0, std::ios::beg);
  if (!file.read(reinterpret_cast<char*>(values.data()), byte_size)) {
    std::cerr << "Failed to read reference data: " << path << '\n';
    values.clear();
    return false;
  }
  return true;
}

std::vector<float> RunLinear(const std::vector<float>& input, const std::vector<float>& weights,
                             const std::vector<float>& bias, int input_size, int output_size,
                             int batch_size) {
  core::vulkan::VulkanContext context(false, core::vulkan::QueueFamilyType::Compute,
                                      VK_NULL_HANDLE);
  context.Init();

  core::vulkan::VulkanBuffer input_buffer(&context, input.size() * sizeof(float),
                                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                          vkai::kHostVisibleMemory);
  core::vulkan::VulkanBuffer output_buffer(
      &context, static_cast<VkDeviceSize>(output_size * batch_size) * sizeof(float),
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, vkai::kHostVisibleMemory);

  input_buffer.MapData(
      [&input](void* data) { std::memcpy(data, input.data(), input.size() * sizeof(float)); });
  vkai::Linear layer(&context, input_size, output_size, batch_size);
  layer.MapWeights(weights, bias);
  layer.Init();

  auto command_buffer = core::vulkan::VulkanCommandBuffer::BeginOneTimeCommands(&context);
  layer.Execute(command_buffer.buffer(), input_buffer, output_buffer);
  command_buffer.EndOneTimeCommands();

  std::vector<float> output(output_size * batch_size);
  output_buffer.MapData(
      [&output](void* data) { std::memcpy(output.data(), data, output.size() * sizeof(float)); });
  return output;
}

}  // namespace

namespace vkai {
namespace test {

TEST(LinearTest, fc1_test) {
  constexpr int kInputSize = 28 * 28;
  constexpr int kOutputSize = 128;
  constexpr int kBatchSize = 1;

  const std::filesystem::path source_dir = VKAI_SOURCE_DIR;
  const std::filesystem::path mnist_dir = source_dir / "python/mnist";
  std::vector<float> input;
  std::vector<float> reference;
  Weights weights;
  ASSERT_TRUE(ReadFloatBinary(mnist_dir / "test_data/input.bin", input));
  ASSERT_TRUE(ReadFloatBinary(mnist_dir / "test_data/fc1_output.bin", reference));
  ASSERT_TRUE(WeightsLoader::Load((mnist_dir / "mnist_weights.bin").string(), weights));

  const WeightTensor* fc1_weights = weights.Find("fc1.weight");
  const WeightTensor* fc1_bias = weights.Find("fc1.bias");
  ASSERT_NE(fc1_weights, nullptr);
  ASSERT_NE(fc1_bias, nullptr);

  ASSERT_EQ(input.size(), static_cast<size_t>(kInputSize * kBatchSize));
  ASSERT_EQ(fc1_weights->data.size(), static_cast<size_t>(kInputSize * kOutputSize));
  ASSERT_EQ(fc1_bias->data.size(), static_cast<size_t>(kOutputSize));
  ASSERT_EQ(reference.size(), static_cast<size_t>(kOutputSize * kBatchSize));

  const auto actual =
      RunLinear(input, fc1_weights->data, fc1_bias->data, kInputSize, kOutputSize, kBatchSize);
  ASSERT_EQ(actual.size(), reference.size());

  for (size_t i = 0; i < reference.size(); ++i) {
    const float tolerance = 1e-4F + 1e-5F * std::abs(reference[i]);
    EXPECT_NEAR(actual[i], reference[i], tolerance) << "FC1 output mismatch at element " << i;
  }
}

TEST(LinearTest, fc2_test) {
  constexpr int kInputSize = 128;
  constexpr int kOutputSize = 10;
  constexpr int kBatchSize = 1;

  const std::filesystem::path source_dir = VKAI_SOURCE_DIR;
  const std::filesystem::path mnist_dir = source_dir / "python/mnist";
  std::vector<float> input;
  std::vector<float> reference;
  Weights weights;
  ASSERT_TRUE(ReadFloatBinary(mnist_dir / "test_data/relu1_output.bin", input));
  ASSERT_TRUE(ReadFloatBinary(mnist_dir / "test_data/fc2_output.bin", reference));
  ASSERT_TRUE(WeightsLoader::Load((mnist_dir / "mnist_weights.bin").string(), weights));

  const WeightTensor* fc2_weights = weights.Find("fc2.weight");
  const WeightTensor* fc2_bias = weights.Find("fc2.bias");
  ASSERT_NE(fc2_weights, nullptr);
  ASSERT_NE(fc2_bias, nullptr);

  ASSERT_EQ(input.size(), static_cast<size_t>(kInputSize * kBatchSize));
  ASSERT_EQ(fc2_weights->data.size(), static_cast<size_t>(kInputSize * kOutputSize));
  ASSERT_EQ(fc2_bias->data.size(), static_cast<size_t>(kOutputSize));
  ASSERT_EQ(reference.size(), static_cast<size_t>(kOutputSize * kBatchSize));

  const auto actual =
      RunLinear(input, fc2_weights->data, fc2_bias->data, kInputSize, kOutputSize, kBatchSize);
  ASSERT_EQ(actual.size(), reference.size());

  for (size_t i = 0; i < reference.size(); ++i) {
    const float tolerance = 1e-4F + 1e-5F * std::abs(reference[i]);
    EXPECT_NEAR(actual[i], reference[i], tolerance) << "FC2 output mismatch at element " << i;
  }
}

}  // namespace test
}  // namespace vkai
