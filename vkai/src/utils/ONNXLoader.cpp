#include "utils/ONNXLoader.h"

#include <onnx/onnx_pb.h>

#include <cstring>
#include <fstream>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace vkai {
namespace {

void SetError(std::string* error_message, const std::string& message) {
  if (error_message != nullptr) {
    *error_message = message;
  }
}

std::vector<int64_t> ShapeFromValueInfo(const onnx::ValueInfoProto& value_info) {
  std::vector<int64_t> shape;
  if (!value_info.has_type() || !value_info.type().has_tensor_type()) {
    return shape;
  }
  const auto& tensor_shape = value_info.type().tensor_type().shape();
  shape.reserve(tensor_shape.dim_size());
  for (const auto& dimension : tensor_shape.dim()) {
    // A symbolic ONNX dimension has no numeric value. Zero represents an
    // unknown dimension in this lightweight graph IR.
    shape.push_back(dimension.has_dim_value() ? dimension.dim_value() : 0);
  }
  return shape;
}

std::vector<int64_t> ShapeFromTensor(const onnx::TensorProto& tensor) {
  return {tensor.dims().begin(), tensor.dims().end()};
}

std::vector<float> FloatData(const onnx::TensorProto& tensor) {
  if (tensor.data_type() != onnx::TensorProto::FLOAT) {
    throw std::runtime_error("Expected a float32 ONNX tensor: " + tensor.name());
  }
  if (!tensor.raw_data().empty()) {
    if (tensor.raw_data().size() % sizeof(float) != 0) {
      throw std::runtime_error("Invalid float32 raw_data size: " + tensor.name());
    }
    std::vector<float> data(tensor.raw_data().size() / sizeof(float));
    std::memcpy(data.data(), tensor.raw_data().data(), tensor.raw_data().size());
    return data;
  }
  return {tensor.float_data().begin(), tensor.float_data().end()};
}

std::vector<int64_t> Int64Data(const onnx::TensorProto& tensor) {
  if (tensor.data_type() != onnx::TensorProto::INT64) {
    throw std::runtime_error("Expected an int64 ONNX tensor: " + tensor.name());
  }
  if (!tensor.raw_data().empty()) {
    if (tensor.raw_data().size() % sizeof(int64_t) != 0) {
      throw std::runtime_error("Invalid int64 raw_data size: " + tensor.name());
    }
    std::vector<int64_t> data(tensor.raw_data().size() / sizeof(int64_t));
    std::memcpy(data.data(), tensor.raw_data().data(), tensor.raw_data().size());
    return data;
  }
  return {tensor.int64_data().begin(), tensor.int64_data().end()};
}

std::optional<OpType> ToOpType(const std::string& onnx_type) {
  static const std::unordered_map<std::string, OpType> kTypes = {
      {"Constant", OpType::Constant}, {"Reshape", OpType::Reshape}, {"Gemm", OpType::Dense},
      {"Relu", OpType::Relu},         {"Conv", OpType::Conv2D},     {"MaxPool", OpType::MaxPool2D},
      {"Add", OpType::Add},           {"Softmax", OpType::Softmax},
  };
  const auto it = kTypes.find(onnx_type);
  return it == kTypes.end() ? std::nullopt : std::optional<OpType>(it->second);
}

std::shared_ptr<Tensor> GetOrAddTensor(Graph& graph, const std::string& name,
                                       std::vector<int64_t> shape = {}) {
  const auto tensor = graph.FindTensor(name);
  return tensor == nullptr ? graph.AddTensor(name, std::move(shape)) : tensor;
}

void ImportAttribute(const onnx::AttributeProto& attribute, Operation& operation,
                     const std::vector<std::shared_ptr<Tensor>>& outputs) {
  if (attribute.has_f()) {
    operation.SetAttribute(attribute.name(), attribute.f());
  } else if (attribute.has_i()) {
    operation.SetAttribute(attribute.name(), attribute.i());
  } else if (attribute.has_s()) {
    operation.SetAttribute(attribute.name(), attribute.s());
  } else if (attribute.has_t()) {
    const auto& tensor = attribute.t();
    if (tensor.data_type() == onnx::TensorProto::FLOAT) {
      operation.SetAttribute(attribute.name(), FloatData(tensor));
      for (const auto& output : outputs) {
        output->SetData({});
      }
    } else if (tensor.data_type() == onnx::TensorProto::INT64) {
      operation.SetAttribute(attribute.name(), Int64Data(tensor));
      for (const auto& output : outputs) {
        output->SetData({});
      }
    } else {
      throw std::runtime_error("Unsupported ONNX constant tensor type");
    }
  }
}

}  // namespace

bool ONNXLoader::Load(const std::string& filename, Graph& graph, std::string* error_message) {
  std::ifstream input(filename, std::ios::binary);
  if (!input.is_open()) {
    SetError(error_message, "Failed to open ONNX file: " + filename);
    return false;
  }

  onnx::ModelProto model;
  if (!model.ParseFromIstream(&input) || !model.has_graph()) {
    SetError(error_message, "Failed to parse ONNX ModelProto: " + filename);
    return false;
  }

  try {
    const onnx::GraphProto& model_graph = model.graph();
    std::unordered_set<std::string> initializer_names;
    for (const auto& initializer : model_graph.initializer()) {
      initializer_names.insert(initializer.name());
      const auto tensor = GetOrAddTensor(graph, initializer.name(), ShapeFromTensor(initializer));
      tensor->SetData(FloatData(initializer));
    }

    for (const auto& input_value : model_graph.input()) {
      if (!initializer_names.contains(input_value.name())) {
        graph.AddInput(GetOrAddTensor(graph, input_value.name(), ShapeFromValueInfo(input_value)));
      }
    }
    for (const auto& output_value : model_graph.output()) {
      graph.AddOutput(GetOrAddTensor(graph, output_value.name(), ShapeFromValueInfo(output_value)));
    }

    for (int node_index = 0; node_index < model_graph.node_size(); ++node_index) {
      const onnx::NodeProto& node = model_graph.node(node_index);
      const auto type = ToOpType(node.op_type());
      if (!type.has_value()) {
        throw std::runtime_error("Unsupported ONNX operator: " + node.op_type());
      }

      const std::string node_name =
          node.name().empty() ? node.op_type() + "_" + std::to_string(node_index) : node.name();
      const auto operation = graph.AddOperation(node_name, *type);
      for (const std::string& input_name : node.input()) {
        if (!input_name.empty()) {
          operation->AddInput(GetOrAddTensor(graph, input_name));
        }
      }

      std::vector<std::shared_ptr<Tensor>> outputs;
      outputs.reserve(node.output_size());
      for (const std::string& output_name : node.output()) {
        const auto output = GetOrAddTensor(graph, output_name);
        operation->AddOutput(output);
        outputs.push_back(output);
      }
      for (const auto& attribute : node.attribute()) {
        ImportAttribute(attribute, *operation, outputs);
      }
    }
  } catch (const std::exception& error) {
    SetError(error_message, error.what());
    return false;
  }
  return true;
}

}  // namespace vkai
