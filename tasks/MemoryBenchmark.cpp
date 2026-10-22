#include <dlfcn.h>

#include <mutex>
#include <unordered_map>
#include <unordered_set>

#include "ModelTestUtils.h"
#include "utils/TensorUtils.h"

namespace {

struct TrackedBuffer {
  VkDeviceSize capacity = 0;
  VkBufferUsageFlags usage = 0;
  VkDeviceMemory memory = VK_NULL_HANDLE;
};

struct MemoryTracker {
  std::mutex mutex;
  std::unordered_map<VkDeviceMemory, VkDeviceSize> allocations;
  std::unordered_map<VkBuffer, TrackedBuffer> buffers;
  VkDeviceSize live_bytes = 0;
  VkDeviceSize peak_bytes = 0;
  size_t allocation_calls = 0;
  size_t buffer_calls = 0;
};

MemoryTracker& Tracker() {
  static MemoryTracker tracker;
  return tracker;
}

template <typename Function>
Function VulkanFunction(const char* name) {
  auto function = reinterpret_cast<Function>(dlsym(RTLD_NEXT, name));
  if (!function) throw std::runtime_error(std::string("Cannot resolve Vulkan function: ") + name);
  return function;
}

struct Snapshot {
  VkDeviceSize allocated_bytes = 0;
  VkDeviceSize peak_bytes = 0;
  VkDeviceSize capacity_bytes = 0;
  VkDeviceSize graph_bytes = 0;
  VkDeviceSize layer_bytes = 0;
  VkDeviceSize uniform_bytes = 0;
  VkDeviceSize logical_tensor_bytes = 0;
  VkDeviceSize bound_tensor_bytes = 0;
  VkDeviceSize graph_capacity_bytes = 0;
  VkDeviceSize aliased_view_bytes = 0;
  size_t allocation_count = 0;
  size_t graph_buffers = 0;
  size_t layer_buffers = 0;
  size_t allocation_calls = 0;
  size_t buffer_calls = 0;
};

Snapshot ReadSnapshot(const vkai::Engine& engine) {
  Snapshot result;
  std::unordered_set<VkBuffer> graph_buffers;
  for (const auto& [name, tensor] : engine.GetGraph().Tensors()) {
    const auto bytes = static_cast<VkDeviceSize>(vkai::TensorElementCount(*tensor)) * sizeof(float);
    result.logical_tensor_bytes += bytes;
    if (tensor->HasBuffer()) {
      graph_buffers.insert(tensor->Buffer().buffer);
      result.bound_tensor_bytes += bytes;
    }
  }
  for (const auto& op : engine.GetTopoOrder()) {
    if (op->Type() != vkai::OpType::Reshape) continue;
    const auto& input = op->GetInput("input");
    const auto& output = op->Outputs().front();
    if (input->HasBuffer() && output->HasBuffer() &&
        input->Buffer().buffer == output->Buffer().buffer)
      result.aliased_view_bytes +=
          static_cast<VkDeviceSize>(vkai::TensorElementCount(*output)) * sizeof(float);
  }
  auto& tracker = Tracker();
  const std::lock_guard lock(tracker.mutex);
  result.allocated_bytes = tracker.live_bytes;
  result.peak_bytes = tracker.peak_bytes;
  result.allocation_count = tracker.allocations.size();
  result.allocation_calls = tracker.allocation_calls;
  result.buffer_calls = tracker.buffer_calls;
  for (const auto& [buffer, info] : tracker.buffers) {
    const auto bytes = tracker.allocations.at(info.memory);
    result.capacity_bytes += info.capacity;
    if (graph_buffers.contains(buffer)) {
      result.graph_bytes += bytes;
      result.graph_capacity_bytes += info.capacity;
      ++result.graph_buffers;
    } else {
      result.layer_bytes += bytes;
      ++result.layer_buffers;
    }
    if ((info.usage & VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT) != 0) result.uniform_bytes += bytes;
  }
  return result;
}

void WriteSamples(const char* name, const std::vector<double>& values) {
  std::cout << ',' << std::quoted(name) << ":[";
  for (size_t i = 0; i < values.size(); ++i) {
    if (i) std::cout << ',';
    std::cout << values[i];
  }
  std::cout << ']';
}

}  // namespace

// Only this measurement executable interposes allocation calls. The byte counts
// include every application-owned VkDeviceMemory allocation, with driver-required
// alignment, but exclude private driver/pipeline allocations and host tensors.
extern "C" VKAPI_ATTR VkResult VKAPI_CALL vkAllocateMemory(VkDevice device,
                                                           const VkMemoryAllocateInfo* info,
                                                           const VkAllocationCallbacks* allocator,
                                                           VkDeviceMemory* memory) {
  static const auto allocate = VulkanFunction<PFN_vkAllocateMemory>("vkAllocateMemory");
  const auto result = allocate(device, info, allocator, memory);
  if (result == VK_SUCCESS) {
    auto& tracker = Tracker();
    const std::lock_guard lock(tracker.mutex);
    tracker.allocations.emplace(*memory, info->allocationSize);
    tracker.live_bytes += info->allocationSize;
    tracker.peak_bytes = std::max(tracker.peak_bytes, tracker.live_bytes);
    ++tracker.allocation_calls;
  }
  return result;
}

extern "C" VKAPI_ATTR void VKAPI_CALL vkFreeMemory(VkDevice device, VkDeviceMemory memory,
                                                   const VkAllocationCallbacks* allocator) {
  static const auto free_memory = VulkanFunction<PFN_vkFreeMemory>("vkFreeMemory");
  if (memory != VK_NULL_HANDLE) {
    auto& tracker = Tracker();
    const std::lock_guard lock(tracker.mutex);
    tracker.live_bytes -= tracker.allocations.at(memory);
    tracker.allocations.erase(memory);
  }
  free_memory(device, memory, allocator);
}

extern "C" VKAPI_ATTR VkResult VKAPI_CALL vkCreateBuffer(VkDevice device,
                                                         const VkBufferCreateInfo* info,
                                                         const VkAllocationCallbacks* allocator,
                                                         VkBuffer* buffer) {
  static const auto create = VulkanFunction<PFN_vkCreateBuffer>("vkCreateBuffer");
  const auto result = create(device, info, allocator, buffer);
  if (result == VK_SUCCESS) {
    auto& tracker = Tracker();
    const std::lock_guard lock(tracker.mutex);
    tracker.buffers.emplace(*buffer, TrackedBuffer{.capacity = info->size, .usage = info->usage});
    ++tracker.buffer_calls;
  }
  return result;
}

extern "C" VKAPI_ATTR void VKAPI_CALL vkDestroyBuffer(VkDevice device, VkBuffer buffer,
                                                      const VkAllocationCallbacks* allocator) {
  static const auto destroy = VulkanFunction<PFN_vkDestroyBuffer>("vkDestroyBuffer");
  {
    auto& tracker = Tracker();
    const std::lock_guard lock(tracker.mutex);
    tracker.buffers.erase(buffer);
  }
  destroy(device, buffer, allocator);
}

extern "C" VKAPI_ATTR VkResult VKAPI_CALL vkBindBufferMemory(VkDevice device, VkBuffer buffer,
                                                             VkDeviceMemory memory,
                                                             VkDeviceSize offset) {
  static const auto bind = VulkanFunction<PFN_vkBindBufferMemory>("vkBindBufferMemory");
  const auto result = bind(device, buffer, memory, offset);
  if (result == VK_SUCCESS) {
    auto& tracker = Tracker();
    const std::lock_guard lock(tracker.mutex);
    tracker.buffers.at(buffer).memory = memory;
  }
  return result;
}

int main(int argc, char** argv) {
  if (argc != 8) {
    std::cerr << "Usage: vkai_memory_benchmark MODEL INPUT REFERENCE OPS_PER_SUBMISSION "
                 "WARMUP GPU_RUNS E2E_RUNS\n";
    return 2;
  }
  const auto input = model_test::ReadFloats(argv[2]);
  const auto reference = model_test::ReadFloats(argv[3]);
  const size_t operations = std::stoull(argv[4]);
  const int warmup = std::stoi(argv[5]);
  const int runs = std::stoi(argv[6]);
  const int e2e_runs = std::stoi(argv[7]);
  if (e2e_runs <= 0) throw std::invalid_argument("E2E run count must be positive");
  {
    const auto start = std::chrono::steady_clock::now();
    vkai::Engine engine(argv[1], operations);
    const double init_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    const auto initialized = ReadSnapshot(engine);
    engine.GetInputTensor()->PopulateTensor(input);
    const auto gpu = engine.BenchmarkGraph(warmup, runs);
    model_test::CompareOutput(engine.GetOutputTensor()->Data(), reference);
    std::vector<double> e2e_ms;
    for (int i = 0; i < e2e_runs; ++i) {
      e2e_ms.push_back(model_test::RunInference(engine));
      model_test::CompareOutput(engine.GetOutputTensor()->Data(), reference);
    }
    const auto final = ReadSnapshot(engine);
    if (initialized.allocation_calls != final.allocation_calls)
      throw std::runtime_error("Inference allocated additional Vulkan memory");
    if (final.allocated_bytes != final.graph_bytes + final.layer_bytes ||
        final.allocation_count != final.graph_buffers + final.layer_buffers)
      throw std::runtime_error("Memory attribution does not cover every allocation");
    std::cout << std::setprecision(12) << "{\"device\":" << std::quoted(gpu.device_name)
              << ",\"init_ms\":" << init_ms << ",\"allocated_bytes\":" << final.allocated_bytes
              << ",\"peak_allocated_bytes\":" << final.peak_bytes
              << ",\"buffer_capacity_bytes\":" << final.capacity_bytes
              << ",\"graph_allocated_bytes\":" << final.graph_bytes
              << ",\"layer_allocated_bytes\":" << final.layer_bytes
              << ",\"uniform_allocated_bytes\":" << final.uniform_bytes
              << ",\"logical_tensor_bytes\":" << final.logical_tensor_bytes
              << ",\"cpu_only_tensor_bytes\":"
              << final.logical_tensor_bytes - final.bound_tensor_bytes
              << ",\"aliased_view_bytes\":" << final.aliased_view_bytes
              << ",\"slot_reuse_saved_bytes\":"
              << final.bound_tensor_bytes - final.aliased_view_bytes - final.graph_capacity_bytes
              << ",\"allocation_count\":" << final.allocation_count
              << ",\"graph_buffer_count\":" << final.graph_buffers
              << ",\"layer_buffer_count\":" << final.layer_buffers
              << ",\"allocation_calls\":" << final.allocation_calls
              << ",\"buffer_creation_calls\":" << final.buffer_calls
              << ",\"submissions_per_run\":" << gpu.submissions_per_run
              << ",\"warmup_runs\":" << warmup << ",\"mismatches\":0";
    WriteSamples("gpu_ms", gpu.gpu_ms);
    WriteSamples("e2e_ms", e2e_ms);
  }
  const auto& tracker = Tracker();
  if (tracker.live_bytes != 0 || !tracker.buffers.empty() || !tracker.allocations.empty())
    throw std::runtime_error("Engine destruction leaked Vulkan memory");
  std::cout << ",\"allocated_bytes_after_destruction\":0}\n";
}
