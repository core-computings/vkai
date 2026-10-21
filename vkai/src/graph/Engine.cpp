#include "graph/Engine.h"

#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

#include "graph/LayerFactory.h"
#include "utils/BufferUtils.h"
#include "utils/ONNXLoader.h"
#include "utils/Synchronization.h"
#include "utils/TensorUtils.h"

namespace vkai {

Engine::Engine(const std::string& onnx_path) : graph_(BuildGraphFromONNX(onnx_path)) {
  topo_order_ = graph_.TopologicalSort();
  context_ = std::make_unique<core::vulkan::VulkanContext>(
      false, core::vulkan::QueueFamilyType::Compute, VK_NULL_HANDLE);
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
  for (const auto& operation : topo_order_) {
    auto layer = CreateLayer(context_.get(), *operation);
    if (layer) layers_.emplace(operation->Name(), std::move(layer));
  }
}

void Engine::UploadInputs() {
  for (const auto& input : graph_.Inputs()) {
    if (!input->HasData() ||
        input->Data().size() != static_cast<size_t>(TensorElementCount(*input))) {
      throw std::runtime_error("Input data size does not match tensor shape: " + input->Name());
    }
    input->Buffer().MapData([&input](void* mapped) {
      std::memcpy(mapped, input->Data().data(), input->Data().size() * sizeof(float));
    });
  }
}

void Engine::RecordOperations(VkCommandBuffer command) {
  for (const auto& operation : execution_order_) {
    auto& layer = *layers_.at(operation->Name());
    RecordLayer(layer, *operation, command);
    Synchronization::InsertComputeBarrier(command);
  }
}

void Engine::ExecuteGraph() {
  UploadInputs();
  auto command = core::vulkan::VulkanCommandBuffer::BeginOneTimeCommands(context_.get());
  RecordOperations(command.buffer());
  for (const auto& output : graph_.Outputs()) {
    Synchronization::InsertHostReadBarrier(command.buffer(), output->Buffer());
  }
  command.EndOneTimeCommands();
  CopyOutputs();
}

InferenceBenchmark Engine::BenchmarkGraph(int warmup_runs, int measured_runs) {
  if (warmup_runs < 0 || measured_runs <= 0 ||
      warmup_runs > std::numeric_limits<int>::max() - measured_runs) {
    throw std::invalid_argument("Invalid benchmark iteration counts");
  }
  uint32_t family_count = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(context_->physical_device, &family_count, nullptr);
  std::vector<VkQueueFamilyProperties> families(family_count);
  vkGetPhysicalDeviceQueueFamilyProperties(context_->physical_device, &family_count,
                                           families.data());
  const auto family = context_->GetQueueFamilyIndices().compute_family.value();
  const uint32_t valid_bits = families.at(family).timestampValidBits;
  VkPhysicalDeviceProperties properties{};
  vkGetPhysicalDeviceProperties(context_->physical_device, &properties);
  if (valid_bits == 0 || properties.limits.timestampPeriod <= 0) {
    throw std::runtime_error("Compute queue does not support GPU timestamps");
  }
  InferenceBenchmark result{
      properties.deviceName, valid_bits, properties.limits.timestampPeriod, {}};
  result.gpu_ms.reserve(measured_runs);
  UploadInputs();

  struct QueryPool {
    VkDevice device;
    VkQueryPool handle = VK_NULL_HANDLE;
    ~QueryPool() {
      if (handle != VK_NULL_HANDLE) vkDestroyQueryPool(device, handle, nullptr);
    }
  } query{context_->logical_device};
  const VkQueryPoolCreateInfo query_info{.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
                                         .queryType = VK_QUERY_TYPE_TIMESTAMP,
                                         .queryCount = 2};
  if (vkCreateQueryPool(query.device, &query_info, nullptr, &query.handle) != VK_SUCCESS) {
    throw std::runtime_error("Cannot create GPU timestamp query pool");
  }
  core::vulkan::VulkanCommandBuffer command(context_.get());
  command.BeginCommandBuffer();
  vkCmdResetQueryPool(command.buffer(), query.handle, 0, 2);
  vkCmdWriteTimestamp(command.buffer(), VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, query.handle, 0);
  RecordOperations(command.buffer());
  vkCmdWriteTimestamp(command.buffer(), VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, query.handle, 1);
  for (const auto& output : graph_.Outputs()) {
    Synchronization::InsertHostReadBarrier(command.buffer(), output->Buffer());
  }
  if (vkEndCommandBuffer(command.buffer()) != VK_SUCCESS) {
    throw std::runtime_error("Cannot record benchmark command buffer");
  }
  const VkCommandBuffer raw_command = command.buffer();
  const VkSubmitInfo submit{.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                            .commandBufferCount = 1,
                            .pCommandBuffers = &raw_command};
  core::vulkan::VulkanFence fence(context_.get());
  const uint64_t mask = valid_bits == 64 ? UINT64_MAX : (uint64_t{1} << valid_bits) - 1;
  for (int run = 0; run < warmup_runs + measured_runs; ++run) {
    fence.Reset();
    if (vkQueueSubmit(context_->compute_queue(), 1, &submit, fence.fence) != VK_SUCCESS ||
        vkWaitForFences(query.device, 1, &fence.fence, VK_TRUE, UINT64_MAX) != VK_SUCCESS) {
      throw std::runtime_error("GPU benchmark submission failed");
    }
    uint64_t timestamps[2]{};
    if (vkGetQueryPoolResults(query.device, query.handle, 0, 2, sizeof(timestamps), timestamps,
                              sizeof(uint64_t),
                              VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT) != VK_SUCCESS) {
      throw std::runtime_error("Cannot read GPU timestamp results");
    }
    if (run >= warmup_runs) {
      const uint64_t ticks = (timestamps[1] - timestamps[0]) & mask;
      result.gpu_ms.push_back(static_cast<double>(ticks) * result.timestamp_period_ns / 1e6);
    }
  }
  CopyOutputs();
  return result;
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
