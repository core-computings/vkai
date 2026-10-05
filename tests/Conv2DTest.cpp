#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

#include "VulkanBuffer.h"
#include "VulkanCommandBuffer.h"
#include "VulkanContext.h"
#include "layers/Conv2D.h"
#include "utils/BufferUtils.h"

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

void ExpectConvMatchesReference(const std::filesystem::path& input_path,
                                const std::filesystem::path& weight_path,
                                const std::filesystem::path& output_path, int input_channels,
                                int output_channels, int input_height, int input_width,
                                int kernel_height, int kernel_width, int padding_height,
                                int padding_width, int dilation_height, int dilation_width,
                                const std::filesystem::path* bias_path = nullptr) {
  std::vector<float> input;
  std::vector<float> weights;
  std::vector<float> bias;
  std::vector<float> reference;
  ASSERT_TRUE(ReadFloatBinary(input_path, input));
  ASSERT_TRUE(ReadFloatBinary(weight_path, weights));
  ASSERT_TRUE(ReadFloatBinary(output_path, reference));
  if (bias_path != nullptr) {
    ASSERT_TRUE(ReadFloatBinary(*bias_path, bias));
    ASSERT_EQ(bias.size(), static_cast<size_t>(output_channels));
  }
  const int output_height =
      (input_height + 2 * padding_height - dilation_height * (kernel_height - 1) - 1) + 1;
  const int output_width =
      (input_width + 2 * padding_width - dilation_width * (kernel_width - 1) - 1) + 1;
  ASSERT_EQ(input.size(), static_cast<size_t>(input_channels * input_height * input_width));
  ASSERT_EQ(weights.size(),
            static_cast<size_t>(output_channels * input_channels * kernel_height * kernel_width));
  ASSERT_EQ(reference.size(), static_cast<size_t>(output_channels * output_height * output_width));

  core::vulkan::VulkanContext context(false, core::vulkan::QueueFamilyType::Compute,
                                      VK_NULL_HANDLE);
  context.Init();
  core::vulkan::VulkanBuffer input_buffer(&context, input.size() * sizeof(float),
                                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                          vkai::kHostVisibleMemory);
  core::vulkan::VulkanBuffer output_buffer(&context, reference.size() * sizeof(float),
                                           VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                           vkai::kHostVisibleMemory);
  input_buffer.MapData(
      [&input](void* data) { std::memcpy(data, input.data(), input.size() * sizeof(float)); });

  vkai::Conv2D conv(&context, input_channels, output_channels, input_height, input_width,
                    kernel_height, kernel_width, 1, 1, padding_height, padding_width,
                    vkai::PaddingType::Zero, 1, dilation_height, dilation_width);
  conv.MapWeights(weights, bias);
  ASSERT_EQ(conv.OutputHeight(), output_height);
  ASSERT_EQ(conv.OutputWidth(), output_width);
  conv.Init();
  auto command_buffer = core::vulkan::VulkanCommandBuffer::BeginOneTimeCommands(&context);
  conv.Execute(command_buffer.buffer(), input_buffer, output_buffer);
  command_buffer.EndOneTimeCommands();

  std::vector<float> actual(reference.size());
  output_buffer.MapData(
      [&actual](void* data) { std::memcpy(actual.data(), data, actual.size() * sizeof(float)); });
  for (size_t index = 0; index < reference.size(); ++index) {
    const float tolerance = 1e-4F + 1e-4F * std::abs(reference[index]);
    EXPECT_NEAR(actual[index], reference[index], tolerance)
        << "Conv2D output mismatch at element " << index;
  }
}

}  // namespace

namespace vkai {
namespace test {

TEST(Conv2DTest, MatchesMNIST12FirstConvolution) {
  constexpr int kInputChannels = 1;
  constexpr int kOutputChannels = 8;
  constexpr int kInputHeight = 28;
  constexpr int kInputWidth = 28;
  constexpr int kKernelHeight = 5;
  constexpr int kKernelWidth = 5;
  constexpr int kStrideHeight = 1;
  constexpr int kStrideWidth = 1;
  constexpr int kPaddingHeight = 2;
  constexpr int kPaddingWidth = 2;
  constexpr int kBatchSize = 1;
  const std::filesystem::path data_dir =
      std::filesystem::path(VKAI_SOURCE_DIR) / "python/mnist12_cnn/test_data";
  std::vector<float> input;
  std::vector<float> weights;
  std::vector<float> reference;
  ASSERT_TRUE(ReadFloatBinary(data_dir / "000_input.bin", input));
  ASSERT_TRUE(ReadFloatBinary(data_dir / "001_Convolution28_Conv_input_1.bin", weights));
  ASSERT_TRUE(ReadFloatBinary(data_dir / "001_Convolution28_Conv_output.bin", reference));

  ASSERT_EQ(input.size(),
            static_cast<size_t>(kBatchSize * kInputChannels * kInputHeight * kInputWidth));
  ASSERT_EQ(weights.size(),
            static_cast<size_t>(kOutputChannels * kInputChannels * kKernelHeight * kKernelWidth));
  ASSERT_EQ(reference.size(),
            static_cast<size_t>(kBatchSize * kOutputChannels * kInputHeight * kInputWidth));

  core::vulkan::VulkanContext context(false, core::vulkan::QueueFamilyType::Compute,
                                      VK_NULL_HANDLE);
  context.Init();

  core::vulkan::VulkanBuffer input_buffer(&context, input.size() * sizeof(float),
                                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                          vkai::kHostVisibleMemory);
  core::vulkan::VulkanBuffer output_buffer(&context, reference.size() * sizeof(float),
                                           VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                           vkai::kHostVisibleMemory);
  input_buffer.MapData(
      [&input](void* data) { std::memcpy(data, input.data(), input.size() * sizeof(float)); });

  Conv2D conv2d(&context, kInputChannels, kOutputChannels, kInputHeight, kInputWidth, kKernelHeight,
                kKernelWidth, kStrideHeight, kStrideWidth, kPaddingHeight, kPaddingWidth,
                PaddingType::Zero, kBatchSize);
  conv2d.MapWeights(weights);
  ASSERT_EQ(conv2d.OutputHeight(), kInputHeight);
  ASSERT_EQ(conv2d.OutputWidth(), kInputWidth);
  conv2d.Init();

  auto command_buffer = core::vulkan::VulkanCommandBuffer::BeginOneTimeCommands(&context);
  conv2d.Execute(command_buffer.buffer(), input_buffer, output_buffer);
  command_buffer.EndOneTimeCommands();

  std::vector<float> actual(reference.size());
  output_buffer.MapData(
      [&actual](void* data) { std::memcpy(actual.data(), data, actual.size() * sizeof(float)); });

  for (size_t i = 0; i < reference.size(); ++i) {
    const float tolerance = 1e-5F + 1e-5F * std::abs(reference[i]);
    EXPECT_NEAR(actual[i], reference[i], tolerance) << "Conv2D output mismatch at element " << i;
  }
}

TEST(Conv2DTest, SupportsReflectPadding) {
  const std::vector<float> input = {
      1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F, 9.0F,
  };
  const std::vector<float> weights(9, 1.0F);
  const std::vector<float> reference = {
      33.0F, 36.0F, 39.0F, 42.0F, 45.0F, 48.0F, 51.0F, 54.0F, 57.0F,
  };

  core::vulkan::VulkanContext context(false, core::vulkan::QueueFamilyType::Compute,
                                      VK_NULL_HANDLE);
  context.Init();

  core::vulkan::VulkanBuffer input_buffer(&context, input.size() * sizeof(float),
                                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                          vkai::kHostVisibleMemory);
  core::vulkan::VulkanBuffer output_buffer(&context, reference.size() * sizeof(float),
                                           VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                           vkai::kHostVisibleMemory);
  input_buffer.MapData(
      [&input](void* data) { std::memcpy(data, input.data(), input.size() * sizeof(float)); });

  Conv2D conv2d(&context, 1, 1, 3, 3, 3, 3, 1, 1, 1, 1, PaddingType::Reflect);
  conv2d.MapWeights(weights);
  conv2d.Init();

  auto command_buffer = core::vulkan::VulkanCommandBuffer::BeginOneTimeCommands(&context);
  conv2d.Execute(command_buffer.buffer(), input_buffer, output_buffer);
  command_buffer.EndOneTimeCommands();

  std::vector<float> actual(reference.size());
  output_buffer.MapData(
      [&actual](void* data) { std::memcpy(actual.data(), data, actual.size() * sizeof(float)); });
  EXPECT_EQ(actual, reference);
}

TEST(Conv2DTest, MatchesDilatedConvolutionReference) {
  const std::filesystem::path data_dir =
      std::filesystem::path(VKAI_SOURCE_DIR) / "python/pspnet/test_data";
  ExpectConvMatchesReference(
      data_dir / "dilated_conv_input.bin", data_dir / "dilated_conv_weight.bin",
      data_dir / "dilated_conv_output.bin", 256, 256, 6, 8, 3, 3, 2, 2, 2, 2);
}

TEST(Conv2DTest, MatchesBiasBearingClassifierConvolution) {
  const std::filesystem::path data_dir =
      std::filesystem::path(VKAI_SOURCE_DIR) / "python/pspnet/test_data";
  const std::filesystem::path bias_path = data_dir / "classifier_conv_bias.bin";
  ExpectConvMatchesReference(
      data_dir / "classifier_conv_input.bin", data_dir / "classifier_conv_weight.bin",
      data_dir / "classifier_conv_output.bin", 512, 150, 6, 8, 1, 1, 0, 0, 1, 1, &bias_path);
}

}  // namespace test
}  // namespace vkai
