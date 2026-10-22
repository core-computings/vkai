#include "graph/Engine.h"

#include <algorithm>
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

Engine::Engine(const std::string& onnx_path, size_t operations_per_submission)
    : graph_(BuildGraphFromONNX(onnx_path)), operations_per_submission_(operations_per_submission) {
  if (operations_per_submission_ == 0)
    throw std::invalid_argument("Operations per submission must be positive");
  topo_order_ = graph_.TopologicalSort();
  context_ = std::make_unique<core::vulkan::VulkanContext>(
      false, core::vulkan::QueueFamilyType::Compute, VK_NULL_HANDLE);
  context_->Init();
  const auto memory_plan = BuildMemoryPlan(graph_, topo_order_);
  AllocateVulkanBuffers(memory_plan);
  UploadConstants(memory_plan);
  CreatePipeline();

  for (const auto& operation : topo_order_) {
    if (layers_.contains(operation->Name()) && operation->Type() != OpType::Reshape) {
      execution_order_.push_back(operation);
    }
  }
}

void Engine::AllocateVulkanBuffers(const MemoryPlan& plan) {
  std::vector<std::shared_ptr<core::vulkan::VulkanBuffer>> buffers;
  buffers.reserve(plan.slots.size());
  for (const auto& slot : plan.slots) {
    buffers.push_back(std::make_shared<core::vulkan::VulkanBuffer>(
        context_.get(), slot.capacity,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
            VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        kHostVisibleMemory));
  }
  for (const auto& [name, tensor] : graph_.Tensors()) {
    const auto assignment = plan.tensor_slots.find(tensor.get());
    if (assignment != plan.tensor_slots.end()) tensor->SetBuffer(buffers.at(assignment->second));
  }
}

void Engine::UploadConstants(const MemoryPlan& plan) {
  for (const auto& tensor : plan.constants) {
    const auto& name = tensor->Name();
    if (!tensor->HasBuffer()) {
      throw std::runtime_error("Cannot upload tensor data without a Vulkan buffer: " + name);
    }

    const auto& data = tensor->Data();
    auto& buffer = tensor->Buffer();
    if (tensor->ByteSize() != data.size() * sizeof(float)) {
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

void Engine::RecordOperations(VkCommandBuffer command, size_t first, size_t count) {
  for (size_t i = first; i < first + count; ++i) {
    const auto& operation = execution_order_[i];
    auto& layer = *layers_.at(operation->Name());
    RecordLayer(layer, *operation, command);
    Synchronization::InsertComputeBarrier(command);
  }
}

void Engine::ExecuteGraph() {
  UploadInputs();
  core::vulkan::VulkanFence fence(context_.get());
  size_t first = 0;
  do {
    const size_t count = std::min(operations_per_submission_, execution_order_.size() - first);
    auto command = core::vulkan::VulkanCommandBuffer::BeginOneTimeCommands(context_.get());
    RecordOperations(command.buffer(), first, count);
    first += count;
    if (first == execution_order_.size()) {
      for (const auto& output : graph_.Outputs())
        Synchronization::InsertHostReadBarrier(command.buffer(), output->Buffer());
    }
    if (vkEndCommandBuffer(command.buffer()) != VK_SUCCESS)
      throw std::runtime_error("Cannot record graph command buffer");
    const VkCommandBuffer raw_command = command.buffer();
    const VkSubmitInfo submit{.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                              .commandBufferCount = 1,
                              .pCommandBuffers = &raw_command};
    fence.Reset();
    if (vkQueueSubmit(context_->compute_queue(), 1, &submit, fence.fence) != VK_SUCCESS ||
        vkWaitForFences(context_->logical_device, 1, &fence.fence, VK_TRUE, UINT64_MAX) !=
            VK_SUCCESS)
      throw std::runtime_error("GPU graph submission failed");
  } while (first < execution_order_.size());
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
  const size_t submissions =
      execution_order_.empty() ? 1 : 1 + (execution_order_.size() - 1) / operations_per_submission_;
  if (submissions > std::numeric_limits<uint32_t>::max() / 2)
    throw std::runtime_error("Too many GPU timestamp queries");
  const uint32_t query_count = static_cast<uint32_t>(submissions) * 2;
  InferenceBenchmark result{properties.deviceName,
                            valid_bits,
                            properties.limits.timestampPeriod,
                            {},
                            static_cast<uint32_t>(submissions)};
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
                                         .queryCount = query_count};
  if (vkCreateQueryPool(query.device, &query_info, nullptr, &query.handle) != VK_SUCCESS) {
    throw std::runtime_error("Cannot create GPU timestamp query pool");
  }
  std::vector<core::vulkan::VulkanCommandBuffer> commands;
  commands.reserve(submissions);
  size_t first = 0;
  for (uint32_t chunk = 0; chunk < submissions; ++chunk) {
    auto& command = commands.emplace_back(context_.get());
    command.BeginCommandBuffer();
    vkCmdResetQueryPool(command.buffer(), query.handle, chunk * 2, 2);
    vkCmdWriteTimestamp(command.buffer(), VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, query.handle,
                        chunk * 2);
    const size_t count = std::min(operations_per_submission_, execution_order_.size() - first);
    RecordOperations(command.buffer(), first, count);
    first += count;
    vkCmdWriteTimestamp(command.buffer(), VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, query.handle,
                        chunk * 2 + 1);
    if (first == execution_order_.size()) {
      for (const auto& output : graph_.Outputs())
        Synchronization::InsertHostReadBarrier(command.buffer(), output->Buffer());
    }
    if (vkEndCommandBuffer(command.buffer()) != VK_SUCCESS)
      throw std::runtime_error("Cannot record benchmark command buffer");
  }
  core::vulkan::VulkanFence fence(context_.get());
  const uint64_t mask = valid_bits == 64 ? UINT64_MAX : (uint64_t{1} << valid_bits) - 1;
  std::vector<uint64_t> timestamps(query_count);
  for (int run = 0; run < warmup_runs + measured_runs; ++run) {
    for (const auto& command : commands) {
      const VkCommandBuffer raw_command = command.buffer();
      const VkSubmitInfo submit{.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                                .commandBufferCount = 1,
                                .pCommandBuffers = &raw_command};
      fence.Reset();
      if (vkQueueSubmit(context_->compute_queue(), 1, &submit, fence.fence) != VK_SUCCESS ||
          vkWaitForFences(query.device, 1, &fence.fence, VK_TRUE, UINT64_MAX) != VK_SUCCESS)
        throw std::runtime_error("GPU benchmark submission failed");
    }
    if (vkGetQueryPoolResults(query.device, query.handle, 0, query_count,
                              timestamps.size() * sizeof(uint64_t), timestamps.data(),
                              sizeof(uint64_t),
                              VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT) != VK_SUCCESS) {
      throw std::runtime_error("Cannot read GPU timestamp results");
    }
    if (run >= warmup_runs) {
      double elapsed_ms = 0;
      for (size_t chunk = 0; chunk < submissions; ++chunk) {
        const uint64_t ticks = (timestamps[chunk * 2 + 1] - timestamps[chunk * 2]) & mask;
        elapsed_ms += static_cast<double>(ticks) * result.timestamp_period_ns / 1e6;
      }
      result.gpu_ms.push_back(elapsed_ms);
    }
  }
  CopyOutputs();
  return result;
}

void Engine::CopyOutputs() {
  for (const auto& output : graph_.Outputs()) {
    std::vector<float> data(static_cast<size_t>(TensorElementCount(*output)));
    output->Buffer().MapData(
        [&data](void* mapped) { std::memcpy(data.data(), mapped, data.size() * sizeof(float)); });
    output->PopulateTensor(std::move(data));
  }
}

}  // namespace vkai
