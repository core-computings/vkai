#include "graph/Engine.h"

#include <cstring>
#include <stdexcept>
#include <utility>

#include "layers/Conv2D.h"
#include "layers/Linear.h"
#include "layers/Relu.h"
#include "layers/Reshape.h"
#include "utils/BufferUtils.h"
#include "utils/ONNXLoader.h"
#include "utils/Synchronization.h"
#include "utils/TensorUtils.h"

namespace vkai {

Engine::Engine(const std::string& onnx_path) : graph_(BuildGraphFromONNX(onnx_path)) {
  topo_order_ = graph_.TopologicalSort();
  context_ = std::make_unique<core::vulkan::VulkanContext>();
  context_->Init();
  AllocateVulkanBuffers();
  UploadWeights();
  CreatePipeline();

  for (const auto& operation : topo_order_) {
    if (layers_.contains(operation->Name())) {
      execution_order_.push_back(operation);
    }
  }
}

void Engine::AllocateVulkanBuffers() {
  for (const auto& entry : graph_.Tensors()) {
    const auto& tensor = entry.second;
    const VkDeviceSize byte_size =
        static_cast<VkDeviceSize>(TensorElementCount(*tensor)) * sizeof(float);

    core::vulkan::VulkanBuffer buffer(context_.get(), byte_size,
                                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                          VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                          VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                      kHostVisibleMemory);
    tensor->SetBuffer(std::move(buffer));
  }
}

void Engine::UploadWeights() {
  for (const auto& [name, tensor] : graph_.Tensors()) {
    if (!tensor->HasData() || tensor->Data().empty()) {
      continue;
    }
    if (!tensor->HasBuffer()) {
      throw std::runtime_error("Cannot upload tensor data without a Vulkan buffer: " + name);
    }

    const auto& data = tensor->Data();
    auto& buffer = tensor->Buffer();
    if (buffer.Size() != data.size() * sizeof(float)) {
      throw std::runtime_error("Tensor data size does not match Vulkan buffer size: " + name);
    }
    buffer.MapData([&data](void* mapped_data) {
      std::memcpy(mapped_data, data.data(), data.size() * sizeof(float));
    });
  }
}

void Engine::CreatePipeline() {
  for (const auto& operation : graph_.Operations()) {
    std::unique_ptr<Layer> layer;
    switch (operation->Type()) {
      case OpType::Constant:
        continue;
      case OpType::Reshape:
        layer = std::make_unique<Reshape>(context_.get());
        break;
        // clang-format off
      case OpType::Dense: {
        auto linear = std::make_unique<Linear>(
            context_.get(), 
            operation->GetAttribute<int>("input_size"),
            operation->GetAttribute<int>("output_size"), 
            operation->GetAttribute<int>("batch_size"));
        const int output_size = operation->GetAttribute<int>("output_size");
        std::vector<float> bias(static_cast<size_t>(output_size), 0.0F);
        if (operation->HasInput("bias")) {
          bias = operation->GetInput("bias")->Data();
        }
        linear->MapWeights(operation->GetInput("weights")->Data(), bias);
        layer = std::move(linear);
        break;
      }
      // clang-format on
      case OpType::Conv2D: {
        // Conv2D attribute-to-layer mapping will be added with the first Conv2D model.
        auto conv = std::make_unique<Conv2D>(context_.get(), 1, 1, 28, 28, 3, 3, 1, 1, 0, 0,
                                             PaddingType::Zero);
        const auto& weights = operation->GetInput("weights")->Data();
        const std::vector<float> bias = operation->HasInput("bias")
                                            ? operation->GetInput("bias")->Data()
                                            : std::vector<float>{};
        conv->MapWeights(weights, bias);
        layer = std::move(conv);
        break;
      }
      case OpType::Relu:
        layer =
            std::make_unique<Relu>(context_.get(), operation->GetAttribute<int>("element_count"));
        break;
      default:
        throw std::runtime_error("Layer type is not configured yet: " + operation->Name());
    }
    layer->Init();
    layers_.emplace(operation->Name(), std::move(layer));
  }
}

void Engine::UploadInputs() {
  for (const auto& input : graph_.Inputs()) {
    input->Buffer().MapData([&input](void* mapped) {
      std::memcpy(mapped, input->Data().data(), input->Data().size() * sizeof(float));
    });
  }
}

void Engine::ExecuteGraph() {
  UploadInputs();
  auto command = core::vulkan::VulkanCommandBuffer::BeginOneTimeCommands(context_.get());
  for (const auto& operation : execution_order_) {
    auto& layer = *layers_.at(operation->Name());
    layer.Execute(command.buffer(), operation->GetInput("input")->Buffer(),
                  operation->Outputs().front()->Buffer());
    Synchronization::InsertComputeBarrier(command.buffer());
  }
  for (const auto& output : graph_.Outputs()) {
    Synchronization::InsertHostReadBarrier(command.buffer(), output->Buffer());
  }
  command.EndOneTimeCommands();
  CopyOutputs();
}

void Engine::CopyOutputs() {
  for (const auto& output : graph_.Outputs()) {
    std::vector<float> data(output->Buffer().Size() / sizeof(float));
    output->Buffer().MapData(
        [&data](void* mapped) { std::memcpy(data.data(), mapped, data.size() * sizeof(float)); });
    output->PopulateTensor(std::move(data));
  }
}

}  // namespace vkai
