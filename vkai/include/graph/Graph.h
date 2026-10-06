#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "graph/Operation.h"

namespace vkai {

class Graph {
 public:
  std::shared_ptr<Tensor> AddTensor(std::string name, std::vector<int64_t> shape);

  std::shared_ptr<Operation> AddOperation(std::string name, OpType type);

  void AddInput(const std::shared_ptr<Tensor>& tensor);

  void AddOutput(const std::shared_ptr<Tensor>& tensor);

  std::shared_ptr<Tensor> FindTensor(const std::string& name) const;

  std::shared_ptr<Operation> FindOperation(const std::string& name) const;

  const std::vector<std::shared_ptr<Tensor>>& Inputs() const { return inputs_; }

  const std::vector<std::shared_ptr<Tensor>>& Outputs() const { return outputs_; }

  const std::vector<std::shared_ptr<Operation>>& Operations() const { return operations_; }

  const std::unordered_map<std::string, std::shared_ptr<Tensor>>& Tensors() const {
    return tensors_;
  }

  // Kahn's algorithm with a FIFO ready queue. Inputs without a producer must
  // be graph inputs or have constant data. CPU data never bypasses a producer.
  std::vector<std::shared_ptr<Operation>> TopologicalSort() const;

 private:
  void ValidateTensor(const std::shared_ptr<Tensor>& tensor) const;

  std::unordered_map<std::string, std::shared_ptr<Tensor>> tensors_;

  std::unordered_map<std::string, std::shared_ptr<Operation>> operations_by_name_;

  std::vector<std::shared_ptr<Operation>> operations_;

  std::vector<std::shared_ptr<Tensor>> inputs_;

  std::vector<std::shared_ptr<Tensor>> outputs_;
};

}  // namespace vkai
