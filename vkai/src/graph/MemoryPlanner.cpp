#include "graph/MemoryPlanner.h"

#include <algorithm>
#include <stdexcept>

namespace vkai {
namespace {

std::vector<std::shared_ptr<Tensor>> GpuInputs(const Operation& operation) {
  if (operation.Type() == OpType::Constant || operation.Type() == OpType::Reshape) return {};
  std::vector<std::shared_ptr<Tensor>> inputs{operation.GetInput("input")};
  switch (operation.Type()) {
    case OpType::Add:
      inputs.push_back(operation.GetInput("addend"));
      break;
    case OpType::Sub:
    case OpType::Mul:
    case OpType::Div:
      inputs.push_back(operation.GetInput("rhs"));
      break;
    case OpType::ChannelConcat:
      for (int i = 1; i < operation.GetAttribute<int>("input_count"); ++i)
        inputs.push_back(operation.GetInput("input_" + std::to_string(i)));
      break;
    default:
      break;
  }
  return inputs;
}

struct Lifetime {
  const Tensor* root = nullptr;
  VkDeviceSize bytes = 0;
  size_t first = 0;
  size_t last = 0;
  bool required = false;
  bool persistent = false;
};

}  // namespace

MemoryPlan BuildMemoryPlan(const Graph& graph,
                           const std::vector<std::shared_ptr<Operation>>& order) {
  std::unordered_map<const Tensor*, const Tensor*> roots;
  std::unordered_map<const Tensor*, OpType> producers;
  for (const auto& [name, tensor] : graph.Tensors()) roots.emplace(tensor.get(), tensor.get());
  const auto root_of = [&roots](const Tensor* tensor) {
    while (roots.at(tensor) != tensor) tensor = roots.at(tensor);
    return tensor;
  };
  for (const auto& operation : order) {
    for (const auto& output : operation->Outputs())
      producers.emplace(output.get(), operation->Type());
    if (operation->Type() == OpType::Reshape) {
      if (operation->Outputs().size() != 1)
        throw std::runtime_error("Reshape requires one output: " + operation->Name());
      const auto& input = operation->GetInput("input");
      const auto& output = operation->Outputs().front();
      if (input->ByteSize() != output->ByteSize())
        throw std::runtime_error("Reshape must preserve the element count: " + operation->Name());
      roots.at(root_of(output.get())) = root_of(input.get());
    }
  }

  std::unordered_map<const Tensor*, Lifetime> lifetimes;
  for (const auto& [name, tensor] : graph.Tensors()) {
    const auto* root = root_of(tensor.get());
    roots.at(tensor.get()) = root;
    lifetimes.try_emplace(root, Lifetime{.root = root});
  }
  const auto lifetime_of = [&lifetimes,
                            &roots](const std::shared_ptr<Tensor>& tensor) -> Lifetime& {
    return lifetimes.at(roots.at(tensor.get()));
  };
  MemoryPlan plan;
  for (const auto& [name, tensor] : graph.Tensors()) {
    const auto producer = producers.find(tensor.get());
    if (tensor->HasData() &&
        (producer == producers.end() || producer->second == OpType::Constant)) {
      lifetime_of(tensor).persistent = true;
      if (!tensor->HasInt64Data()) plan.constants.push_back(tensor);
    }
  }
  for (const auto& input : graph.Inputs()) {
    auto& lifetime = lifetime_of(input);
    lifetime.required = true;
    lifetime.persistent = true;
  }
  for (const auto& output : graph.Outputs()) {
    auto& lifetime = lifetime_of(output);
    lifetime.required = true;
    lifetime.persistent = true;
    lifetime.last = order.size();
  }
  for (size_t i = 0; i < order.size(); ++i) {
    const auto& operation = *order[i];
    if (operation.Type() == OpType::Constant || operation.Type() == OpType::Reshape) continue;
    for (const auto& output : operation.Outputs()) {
      auto& lifetime = lifetime_of(output);
      lifetime.required = true;
      lifetime.first = i;
      lifetime.last = std::max(lifetime.last, i);
    }
    for (const auto& input : GpuInputs(operation)) {
      auto& lifetime = lifetime_of(input);
      lifetime.required = true;
      lifetime.last = std::max(lifetime.last, i);
    }
  }
  std::vector<Lifetime> required;
  for (auto& [root, lifetime] : lifetimes) {
    if (!lifetime.required) continue;
    if (root->HasInt64Data())
      throw std::runtime_error("GPU execution requires float32 data: " + root->Name());
    lifetime.bytes = root->ByteSize();
    required.push_back(lifetime);
  }
  std::sort(required.begin(), required.end(), [](const Lifetime& a, const Lifetime& b) {
    if (a.first != b.first) return a.first < b.first;
    if (a.bytes != b.bytes) return a.bytes > b.bytes;
    return a.root->Name() < b.root->Name();
  });
  std::unordered_map<const Tensor*, size_t> root_slots;
  for (const auto& lifetime : required) {
    size_t chosen = plan.slots.size();
    if (!lifetime.persistent) {
      for (size_t i = 0; i < plan.slots.size(); ++i) {
        const auto& slot = plan.slots[i];
        if (!slot.persistent && slot.last_use < lifetime.first && slot.capacity >= lifetime.bytes &&
            (chosen == plan.slots.size() || slot.capacity < plan.slots[chosen].capacity)) {
          chosen = i;
        }
      }
    }
    if (chosen == plan.slots.size()) {
      plan.slots.push_back(BufferSlot{.capacity = lifetime.bytes,
                                      .last_use = lifetime.last,
                                      .persistent = lifetime.persistent});
    } else {
      plan.slots[chosen].last_use = lifetime.last;
    }
    root_slots.emplace(lifetime.root, chosen);
  }
  for (const auto& [name, tensor] : graph.Tensors()) {
    const auto slot = root_slots.find(roots.at(tensor.get()));
    if (slot != root_slots.end()) plan.tensor_slots.emplace(tensor.get(), slot->second);
  }
  std::erase_if(plan.constants,
                [&plan](const auto& tensor) { return !plan.tensor_slots.contains(tensor.get()); });
  return plan;
}

}  // namespace vkai
