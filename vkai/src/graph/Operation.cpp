#include "graph/Operation.h"

#include <stdexcept>
#include <utility>

namespace vkai {

Operation::Operation(std::string name, OpType type) : name_(std::move(name)), type_(type) {}

void Operation::AddInput(const std::string& role, const std::shared_ptr<Tensor>& tensor) {
  if (tensor == nullptr) {
    throw std::invalid_argument("Operation input tensor must not be null");
  }
  if (!inputs_.emplace(role, tensor).second) {
    throw std::invalid_argument("Operation input role already exists: " + role);
  }
}

void Operation::AddOutput(const std::shared_ptr<Tensor>& tensor) {
  if (tensor == nullptr) {
    throw std::invalid_argument("Operation output tensor must not be null");
  }
  outputs_.push_back(tensor);
}

}  // namespace vkai
