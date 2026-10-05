#include "PSPNetVulkan.h"

#include <cstring>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "layers/AdaptiveAvgPool2D.h"
#include "layers/Add.h"
#include "layers/BatchNorm2D.h"
#include "layers/BilinearResize2D.h"
#include "layers/ChannelConcat.h"
#include "layers/Conv2D.h"
#include "layers/MaxPool2D.h"
#include "layers/Relu.h"
#include "layers/Softmax.h"
#include "utils/BufferUtils.h"
#include "utils/Synchronization.h"
#include "utils/WeightsLoader.h"

namespace vkai {
namespace {

int ConvOutputSize(int input, int kernel, int stride, int padding, int dilation) {
  return (input + 2 * padding - dilation * (kernel - 1) - 1) / stride + 1;
}

}  // namespace

struct PSPNetVulkan::Impl {
  struct Tensor {
    int channels;
    int height;
    int width;
    core::vulkan::VulkanBuffer* buffer;
  };
  struct Operation {
    std::function<void(const VkCommandBuffer&)> execute;
  };

  Impl(core::vulkan::VulkanContext* context, const std::string& weights_file, int input_height,
       int input_width, core::vulkan::VulkanBuffer& input, core::vulkan::VulkanBuffer& output)
      : context(context), input_buffer(input), output_buffer(output) {
    std::string error;
    if (!WeightsLoader::Load(weights_file, weights, &error)) {
      std::cerr << error << '\n';
      return;
    }
    if (input_height <= 0 || input_width <= 0 ||
        input_buffer.Size() <
            static_cast<VkDeviceSize>(3 * input_height * input_width) * sizeof(float) ||
        output_buffer.Size() <
            static_cast<VkDeviceSize>(150 * input_height * input_width) * sizeof(float)) {
      std::cerr << "PSPNet input or output buffer is too small\n";
      return;
    }
    input_tensor = {3, input_height, input_width, &input_buffer};
    Tensor* current = BuildStem(&input_tensor);
    if (current == nullptr) return;
    current = BuildStage("encoder.layer1", current, 64, 3, 1, 1);
    current = BuildStage("encoder.layer2", current, 128, 4, 2, 1);
    current = BuildStage("encoder.layer3", current, 256, 6, 1, 2);
    current = BuildStage("encoder.layer4", current, 512, 3, 1, 4);
    if (current == nullptr || !BuildDecoder(current, input_height, input_width)) return;
    valid = true;
  }

  const WeightTensor* Weight(const std::string& name) {
    const WeightTensor* tensor = weights.Find(name);
    if (tensor == nullptr) std::cerr << "Missing PSPNet tensor: " << name << '\n';
    return tensor;
  }

  Tensor* CreateTensor(int channels, int height, int width) {
    if (channels <= 0 || height <= 0 || width <= 0) return nullptr;
    const VkDeviceSize bytes = static_cast<VkDeviceSize>(channels) * height * width * sizeof(float);
    owned_buffers.push_back(std::make_unique<core::vulkan::VulkanBuffer>(
        context, bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, kHostVisibleMemory));
    tensors.push_back(
        std::make_unique<Tensor>(Tensor{channels, height, width, owned_buffers.back().get()}));
    return tensors.back().get();
  }

  template <typename LayerType>
  LayerType* KeepLayer(std::unique_ptr<LayerType> layer) {
    LayerType* raw = layer.get();
    layers.push_back(std::move(layer));
    return raw;
  }

  void AddUnary(Layer* layer, Tensor* input, Tensor* output) {
    operations.push_back({[layer, input, output](const VkCommandBuffer& command_buffer) {
      layer->Execute(command_buffer, *input->buffer, *output->buffer);
    }});
  }

  Tensor* AddConv(const std::string& name, Tensor* input, int output_channels, int kernel,
                  int stride, int padding, int dilation = 1) {
    const WeightTensor* weight = Weight(name + ".weight");
    if (weight == nullptr ||
        weight->data.size() !=
            static_cast<size_t>(output_channels * input->channels * kernel * kernel)) {
      std::cerr << "Unexpected Conv2D weight shape: " << name << '\n';
      return nullptr;
    }
    std::vector<float> bias;
    if (const WeightTensor* bias_tensor = weights.Find(name + ".bias"); bias_tensor != nullptr) {
      if (bias_tensor->data.size() != static_cast<size_t>(output_channels)) return nullptr;
      bias = bias_tensor->data;
    }
    const int output_height = ConvOutputSize(input->height, kernel, stride, padding, dilation);
    const int output_width = ConvOutputSize(input->width, kernel, stride, padding, dilation);
    Tensor* output = CreateTensor(output_channels, output_height, output_width);
    if (output == nullptr) return nullptr;
    auto layer = std::make_unique<Conv2D>(context, input->channels, output_channels, input->height,
                                          input->width, kernel, kernel, stride, stride, padding,
                                          padding, PaddingType::Zero, 1, dilation, dilation);
    layer->MapWeights(weight->data, bias);
    AddUnary(KeepLayer(std::move(layer)), input, output);
    return output;
  }

  Tensor* AddBatchNorm(const std::string& name, Tensor* input) {
    const WeightTensor* mean = Weight(name + ".running_mean");
    const WeightTensor* variance = Weight(name + ".running_var");
    const WeightTensor* weight = Weight(name + ".weight");
    const WeightTensor* bias = Weight(name + ".bias");
    if (mean == nullptr || variance == nullptr || weight == nullptr || bias == nullptr ||
        mean->data.size() != static_cast<size_t>(input->channels) ||
        variance->data.size() != static_cast<size_t>(input->channels) ||
        weight->data.size() != static_cast<size_t>(input->channels) ||
        bias->data.size() != static_cast<size_t>(input->channels))
      return nullptr;
    Tensor* output = CreateTensor(input->channels, input->height, input->width);
    if (output == nullptr) return nullptr;
    auto layer = std::make_unique<BatchNorm2D>(context, input->channels,
                                               input->height * input->width, 1, 1e-5F);
    layer->MapParameters(mean->data, variance->data, weight->data, bias->data);
    AddUnary(KeepLayer(std::move(layer)), input, output);
    return output;
  }

  Tensor* AddRelu(Tensor* input) {
    Tensor* output = CreateTensor(input->channels, input->height, input->width);
    if (output == nullptr) return nullptr;
    auto layer = std::make_unique<Relu>(context, input->channels * input->height * input->width);
    AddUnary(KeepLayer(std::move(layer)), input, output);
    return output;
  }

  Tensor* AddConvBnRelu(const std::string& conv_name, const std::string& bn_name, Tensor* input,
                        int output_channels, int kernel, int stride, int padding, int dilation,
                        bool relu) {
    Tensor* output = AddConv(conv_name, input, output_channels, kernel, stride, padding, dilation);
    if (output == nullptr || (output = AddBatchNorm(bn_name, output)) == nullptr) return nullptr;
    return relu ? AddRelu(output) : output;
  }

  Tensor* BuildStem(Tensor* input) {
    Tensor* current = AddConvBnRelu("encoder.conv1", "encoder.bn1", input, 64, 3, 2, 1, 1, true);
    if (current == nullptr) return nullptr;
    current = AddConvBnRelu("encoder.conv2", "encoder.bn2", current, 64, 3, 1, 1, 1, true);
    if (current == nullptr) return nullptr;
    current = AddConvBnRelu("encoder.conv3", "encoder.bn3", current, 128, 3, 1, 1, 1, true);
    if (current == nullptr) return nullptr;
    Tensor* output = CreateTensor(128, (current->height + 1) / 2, (current->width + 1) / 2);
    if (output == nullptr) return nullptr;
    auto layer = std::make_unique<MaxPool2D>(context, 128, current->height, current->width, 3, 3, 2,
                                             2, 1, 1, 1);
    AddUnary(KeepLayer(std::move(layer)), current, output);
    return output;
  }

  Tensor* BuildBlock(const std::string& name, Tensor* input, int planes, int stride, int dilation,
                     bool downsample) {
    Tensor* main = AddConvBnRelu(name + ".conv1", name + ".bn1", input, planes, 1, 1, 0, 1, true);
    if (main == nullptr) return nullptr;
    main = AddConvBnRelu(name + ".conv2", name + ".bn2", main, planes, 3, stride, dilation,
                         dilation, true);
    if (main == nullptr) return nullptr;
    main = AddConvBnRelu(name + ".conv3", name + ".bn3", main, planes * 4, 1, 1, 0, 1, false);
    if (main == nullptr) return nullptr;
    Tensor* shortcut = input;
    if (downsample) {
      shortcut = AddConvBnRelu(name + ".downsample.0", name + ".downsample.1", input, planes * 4, 1,
                               stride, 0, 1, false);
      if (shortcut == nullptr) return nullptr;
    }
    if (shortcut->channels != main->channels || shortcut->height != main->height ||
        shortcut->width != main->width)
      return nullptr;
    Tensor* added = CreateTensor(main->channels, main->height, main->width);
    if (added == nullptr) return nullptr;
    auto add = std::make_unique<Add>(context, main->channels, main->height * main->width, 1,
                                     AddMode::kElementwise);
    Add* add_raw = KeepLayer(std::move(add));
    operations.push_back({[add_raw, main, shortcut, added](const VkCommandBuffer& command_buffer) {
      add_raw->Execute(command_buffer, *main->buffer, *shortcut->buffer, *added->buffer);
    }});
    return AddRelu(added);
  }

  Tensor* BuildStage(const std::string& name, Tensor* input, int planes, int blocks, int stride,
                     int dilation) {
    Tensor* current = input;
    for (int index = 0; index < blocks; ++index) {
      const bool first = index == 0;
      const int block_stride = first ? stride : 1;
      const int block_dilation = first && dilation > 1 ? dilation / 2 : dilation;
      current = BuildBlock(name + "." + std::to_string(index), current, planes, block_stride,
                           block_dilation, first);
      if (current == nullptr) return nullptr;
    }
    return current;
  }

  bool BuildDecoder(Tensor* conv5, int output_height, int output_width) {
    std::vector<Tensor*> ppm_inputs{conv5};
    for (int index = 0; index < 4; ++index) {
      const int scale = index == 0 ? 1 : index == 1 ? 2 : index == 2 ? 3 : 6;
      Tensor* pooled = CreateTensor(conv5->channels, scale, scale);
      auto pool = std::make_unique<AdaptiveAvgPool2D>(context, conv5->channels, conv5->height,
                                                      conv5->width, scale, scale);
      AddUnary(KeepLayer(std::move(pool)), conv5, pooled);
      Tensor* branch = AddConvBnRelu("decoder.ppm." + std::to_string(index) + ".1",
                                     "decoder.ppm." + std::to_string(index) + ".2", pooled, 512, 1,
                                     1, 0, 1, true);
      if (branch == nullptr) return false;
      Tensor* resized = CreateTensor(512, conv5->height, conv5->width);
      auto resize = std::make_unique<BilinearResize2D>(context, 512, branch->height, branch->width,
                                                       conv5->height, conv5->width);
      AddUnary(KeepLayer(std::move(resize)), branch, resized);
      ppm_inputs.push_back(resized);
    }
    Tensor* concatenated = CreateTensor(4096, conv5->height, conv5->width);
    auto concat = std::make_unique<ChannelConcat>(
        context, std::vector<int>{2048, 512, 512, 512, 512}, conv5->height, conv5->width);
    ChannelConcat* concat_raw = KeepLayer(std::move(concat));
    operations.push_back(
        {[concat_raw, ppm_inputs, concatenated](const VkCommandBuffer& command_buffer) {
          std::vector<const core::vulkan::VulkanBuffer*> buffers;
          for (const Tensor* tensor : ppm_inputs) buffers.push_back(tensor->buffer);
          concat_raw->Execute(command_buffer, buffers, *concatenated->buffer);
        }});
    Tensor* decoder = AddConvBnRelu("decoder.conv_last.0", "decoder.conv_last.1", concatenated, 512,
                                    3, 1, 1, 1, true);
    if (decoder == nullptr) return false;
    Tensor* logits = AddConv("decoder.conv_last.4", decoder, 150, 1, 1, 0, 1);
    if (logits == nullptr) return false;
    Tensor* resized_logits = CreateTensor(150, output_height, output_width);
    auto resize = std::make_unique<BilinearResize2D>(context, 150, logits->height, logits->width,
                                                     output_height, output_width);
    AddUnary(KeepLayer(std::move(resize)), logits, resized_logits);
    auto softmax = std::make_unique<Softmax>(context, 150, output_height, output_width, 1,
                                             SoftmaxMode::kChannelNCHW);
    Softmax* softmax_raw = KeepLayer(std::move(softmax));
    operations.push_back(
        {[softmax_raw, resized_logits, this](const VkCommandBuffer& command_buffer) {
          softmax_raw->Execute(command_buffer, *resized_logits->buffer, output_buffer);
        }});
    return true;
  }

  void Init() {
    if (!valid) return;
    for (auto& layer : layers) layer->Init();
  }

  core::vulkan::VulkanBuffer& Run(const VkCommandBuffer& command_buffer) {
    if (!valid) return output_buffer;
    for (const Operation& operation : operations) {
      operation.execute(command_buffer);
      Synchronization::InsertComputeBarrier(command_buffer);
    }
    Synchronization::InsertHostReadBarrier(command_buffer, output_buffer);
    return output_buffer;
  }

  core::vulkan::VulkanContext* context;
  core::vulkan::VulkanBuffer& input_buffer;
  core::vulkan::VulkanBuffer& output_buffer;
  Weights weights;
  Tensor input_tensor{};
  std::vector<std::unique_ptr<Layer>> layers;
  std::vector<std::unique_ptr<core::vulkan::VulkanBuffer>> owned_buffers;
  std::vector<std::unique_ptr<Tensor>> tensors;
  std::vector<Operation> operations;
  bool valid = false;
};

PSPNetVulkan::PSPNetVulkan(core::vulkan::VulkanContext* context, const std::string& weights_file,
                           int input_height, int input_width, core::vulkan::VulkanBuffer& input,
                           core::vulkan::VulkanBuffer& output)
    : impl_(std::make_unique<Impl>(context, weights_file, input_height, input_width, input,
                                   output)) {}

PSPNetVulkan::~PSPNetVulkan() = default;
void PSPNetVulkan::Init() { impl_->Init(); }
core::vulkan::VulkanBuffer& PSPNetVulkan::Run(const VkCommandBuffer& command_buffer) {
  return impl_->Run(command_buffer);
}

}  // namespace vkai
