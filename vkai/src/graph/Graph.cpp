#include "graph/Graph.h"

#include <queue>
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
  std::unordered_map<const Tensor*, size_t> producers;
  for (size_t index = 0; index < operations_.size(); ++index) {
    const auto& operation = operations_[index];
    for (const auto& output : operation->Outputs()) {
      ValidateTensor(output);
      if (!producers.emplace(output.get(), index).second) {
        throw std::logic_error("A tensor has more than one producing operation: " + output->Name());
      }
    }
    for (const auto& [role, input] : operation->Inputs()) {
      ValidateTensor(input);
    }
  }

  std::unordered_set<const Tensor*> graph_inputs;
  for (const auto& input : inputs_) {
    graph_inputs.insert(input.get());
  }

  // Each input edge adds one dependency, including repeated inputs.
  std::vector<size_t> indegree(operations_.size(), 0);
  std::vector<std::vector<size_t>> consumers(operations_.size());
  for (size_t index = 0; index < operations_.size(); ++index) {
    for (const auto& [role, input] : operations_[index]->Inputs()) {
      const auto producer = producers.find(input.get());
      if (producer != producers.end()) {
        ++indegree[index];
        consumers[producer->second].push_back(index);
      } else if (!graph_inputs.contains(input.get()) && !input->HasData()) {
        throw std::logic_error("Graph input has no producer or data: " + input->Name());
      }
    }
  }

  std::queue<size_t> ready;
  for (size_t index = 0; index < operations_.size(); ++index) {
    if (indegree[index] == 0) {
      ready.push(index);
    }
  }

  std::vector<std::shared_ptr<Operation>> ordered;
  ordered.reserve(operations_.size());
  while (!ready.empty()) {
    const size_t index = ready.front();
    ready.pop();
    ordered.push_back(operations_[index]);
    for (const size_t consumer : consumers[index]) {
      if (--indegree[consumer] == 0) {
        ready.push(consumer);
      }
    }
  }
  if (ordered.size() != operations_.size()) {
    throw std::logic_error("Graph has a cycle");
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
