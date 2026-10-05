#include "MNISTVulkan.h"

#include <iostream>
#include <utility>

#include "utils/Synchronization.h"

namespace vkai {
namespace {

constexpr int kInputSize = 28 * 28;
constexpr int kFC1OutputSize = 128;
constexpr int kFC2OutputSize = 10;
constexpr int kBatchSize = 1;

}  // namespace

MNISTVulkan::MNISTVulkan(core::vulkan::VulkanContext* context, const std::string& weights_file,
                         core::vulkan::VulkanBuffer& input, core::vulkan::VulkanBuffer& output)
    : context_(context), input_buffer_(input), output_buffer_(output) {
  if (!LoadWeights(weights_file)) {
    return;
  }

  const WeightTensor* fc1_weights = weights_.Find("fc1.weight");
  const WeightTensor* fc1_bias = weights_.Find("fc1.bias");
  const WeightTensor* fc2_weights = weights_.Find("fc2.weight");
  const WeightTensor* fc2_bias = weights_.Find("fc2.bias");
  if (fc1_weights == nullptr || fc1_bias == nullptr || fc2_weights == nullptr ||
      fc2_bias == nullptr ||
      fc1_weights->data.size() != static_cast<size_t>(kInputSize * kFC1OutputSize) ||
      fc1_bias->data.size() != static_cast<size_t>(kFC1OutputSize) ||
      fc2_weights->data.size() != static_cast<size_t>(kFC1OutputSize * kFC2OutputSize) ||
      fc2_bias->data.size() != static_cast<size_t>(kFC2OutputSize)) {
    std::cerr << "Unexpected MNIST weight dimensions\n";
    return;
  }

  if (input_buffer_.Size() < kInputSize * sizeof(float)) {
    std::cerr << "MNIST input buffer is too small\n";
    return;
  }
  if (output_buffer_.Size() < kFC1OutputSize * sizeof(float)) {
    std::cerr << "MNIST output buffer must fit the largest intermediate tensor\n";
    return;
  }

  auto fc1 = std::make_unique<Linear>(context_, kInputSize, kFC1OutputSize, kBatchSize);
  fc1->MapWeights(fc1_weights->data, fc1_bias->data);
  layers_.emplace_back(std::move(fc1));
  layers_.emplace_back(std::make_unique<Relu>(context_, kFC1OutputSize));
  auto fc2 = std::make_unique<Linear>(context_, kFC1OutputSize, kFC2OutputSize, kBatchSize);
  fc2->MapWeights(fc2_weights->data, fc2_bias->data);
  layers_.emplace_back(std::move(fc2));
  layers_.emplace_back(std::make_unique<Softmax>(context_, kFC2OutputSize, kBatchSize));
}

bool MNISTVulkan::LoadWeights(const std::string& weights_file) {
  std::string error_message;
  if (!WeightsLoader::Load(weights_file, weights_, &error_message)) {
    std::cerr << error_message << '\n';
    return false;
  }
  return true;
}

void MNISTVulkan::Init() {
  for (auto& layer : layers_) {
    layer->Init();
  }
}

core::vulkan::VulkanBuffer& MNISTVulkan::Run(const VkCommandBuffer& command_buffer) {
  core::vulkan::VulkanBuffer* input_buffer = &input_buffer_;
  core::vulkan::VulkanBuffer* output_buffer = &output_buffer_;

  for (auto& layer : layers_) {
    layer->Execute(command_buffer, *input_buffer, *output_buffer);
    Synchronization::InsertComputeBarrier(command_buffer);
    std::swap(input_buffer, output_buffer);
  }
  Synchronization::InsertHostReadBarrier(command_buffer, *input_buffer);
  return *input_buffer;
}

}  // namespace vkai
