#include "graph/Graph.h"

#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace vkai {

std::shared_ptr<Tensor> Graph::AddTensor(std::string name, std::vector<int64_t> shape) {
  if (tensors_.contains(name)) {
    throw std::invalid_argument("A tensor with this name already exists: " + name);
  }
  auto tensor = std::make_shared<Tensor>(std::move(name), std::move(shape));
  tensors_.emplace(tensor->Name(), tensor);
  return tensor;
}

std::shared_ptr<Operation> Graph::AddOperation(std::string name, OpType type) {
  if (operations_by_name_.contains(name)) {
    throw std::invalid_argument("An operation with this name already exists: " + name);
  }
  auto operation = std::make_shared<Operation>(std::move(name), type);
  operations_by_name_.emplace(operation->Name(), operation);
  operations_.push_back(operation);
  return operation;
}

void Graph::AddInput(const std::shared_ptr<Tensor>& tensor) {
  ValidateTensor(tensor);
  inputs_.push_back(tensor);
}

void Graph::AddOutput(const std::shared_ptr<Tensor>& tensor) {
  ValidateTensor(tensor);
  outputs_.push_back(tensor);
}

std::shared_ptr<Tensor> Graph::FindTensor(const std::string& name) const {
  const auto it = tensors_.find(name);
  return it == tensors_.end() ? nullptr : it->second;
}

std::shared_ptr<Operation> Graph::FindOperation(const std::string& name) const {
  const auto it = operations_by_name_.find(name);
  return it == operations_by_name_.end() ? nullptr : it->second;
}

std::vector<std::shared_ptr<Operation>> Graph::TopologicalSort() const {
  std::unordered_map<const Tensor*, const Operation*> producers;
  for (const auto& operation : operations_) {
    for (const auto& output : operation->Outputs()) {
      ValidateTensor(output);
      const auto [producer, inserted] = producers.emplace(output.get(), operation.get());
      static_cast<void>(producer);
      if (!inserted) {
        throw std::logic_error("A tensor has more than one producing operation: " + output->Name());
      }
    }
    for (const auto& input : operation->Inputs()) {
      ValidateTensor(input);
    }
  }

  std::unordered_set<const Tensor*> available;
  for (const auto& input : inputs_) {
    available.insert(input.get());
  }
  for (const auto& entry : tensors_) {
    if (entry.second->HasData()) {
      available.insert(entry.second.get());
    }
  }

  std::unordered_set<const Operation*> executed;
  std::vector<std::shared_ptr<Operation>> ordered;
  ordered.reserve(operations_.size());
  while (ordered.size() != operations_.size()) {
    bool made_progress = false;
    for (const auto& operation : operations_) {
      if (executed.contains(operation.get())) {
        continue;
      }

      bool ready = true;
      for (const auto& input : operation->Inputs()) {
        if (!available.contains(input.get())) {
          ready = false;
          break;
        }
      }
      if (!ready) {
        continue;
      }

      executed.insert(operation.get());
      ordered.push_back(operation);
      for (const auto& output : operation->Outputs()) {
        available.insert(output.get());
      }
      made_progress = true;
    }
    if (!made_progress) {
      throw std::logic_error("Graph has a cycle or an input without a producer");
    }
  }
  return ordered;
}

void Graph::ValidateTensor(const std::shared_ptr<Tensor>& tensor) const {
  if (tensor == nullptr) {
    throw std::invalid_argument("Graph tensor must not be null");
  }
  const auto it = tensors_.find(tensor->Name());
  if (it == tensors_.end() || it->second != tensor) {
    throw std::invalid_argument("Tensor does not belong to this graph: " + tensor->Name());
  }
}

}  // namespace vkai
