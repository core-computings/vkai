#pragma once

#include <any>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "graph/Tensor.h"

namespace vkai {

enum class OpType {
  Dense,
  Conv2D,
  Relu,
  MaxPool2D,
  Add,
  BatchNorm2D,
  BilinearResize2D,
  AdaptiveAvgPool2D,
  ChannelConcat,
  Softmax,
};

class Operation {
 public:
  Operation(std::string name, OpType type);

  const std::string& Name() const { return name_; }

  OpType Type() const { return type_; }

  void AddInput(const std::shared_ptr<Tensor>& tensor);

  void AddOutput(const std::shared_ptr<Tensor>& tensor);

  const std::vector<std::shared_ptr<Tensor>>& Inputs() const { return inputs_; }

  const std::vector<std::shared_ptr<Tensor>>& Outputs() const { return outputs_; }

  template <typename T>
  void SetAttribute(const std::string& name, T value) {
    attributes_[name] = std::move(value);
  }

  template <typename T>
  const T* FindAttribute(const std::string& name) const {
    const auto it = attributes_.find(name);
    return it == attributes_.end() ? nullptr : std::any_cast<T>(&it->second);
  }

  bool HasAttribute(const std::string& name) const { return attributes_.contains(name); }

 private:
  std::string name_;
  OpType type_;
  std::vector<std::shared_ptr<Tensor>> inputs_;
  std::vector<std::shared_ptr<Tensor>> outputs_;
  std::unordered_map<std::string, std::any> attributes_;
};

}  // namespace vkai
