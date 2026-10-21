#include "utils/ONNXLoader.h"

#include <onnx/onnx_pb.h>
#include <onnx/shape_inference/implementation.h>

#include <cstring>
#include <fstream>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "utils/TensorUtils.h"

namespace vkai {
namespace {

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

std::optional<OpType> GetOpType(const std::string& onnx_type) {
  static const std::unordered_map<std::string, OpType> kTypes = {
      {"Constant", OpType::Constant},
      {"Reshape", OpType::Reshape},
      {"Gemm", OpType::Dense},
      {"Relu", OpType::Relu},
      {"Conv", OpType::Conv2D},
      {"MaxPool", OpType::MaxPool2D},
      {"Add", OpType::Add},
      {"Softmax", OpType::Softmax},
      {"ConvTranspose", OpType::ConvTranspose2D},
      {"Sigmoid", OpType::Sigmoid},
      {"Clip", OpType::Clip},
      {"Mul", OpType::Mul},
      {"Sub", OpType::Sub},
      {"Div", OpType::Div},
      {"Slice", OpType::Slice},
      {"Transpose", OpType::Transpose},
      {"Resize", OpType::NearestResize2D},
      {"ReduceMean", OpType::AdaptiveAvgPool2D},
      {"BatchNormalization", OpType::BatchNorm2D},
      {"GlobalAveragePool", OpType::AdaptiveAvgPool2D},
      {"Concat", OpType::ChannelConcat},
      {"Flatten", OpType::Reshape},
      {"Identity", OpType::Reshape},
  };
  const auto it = kTypes.find(onnx_type);
  return it == kTypes.end() ? std::nullopt : std::optional<OpType>(it->second);
}

// ONNX input positions map to semantic roles, independently of tensor names.
std::string InputRole(OpType type, size_t index) {
  static const std::unordered_map<OpType, std::vector<std::string>> kRoles = {
      {OpType::Constant, {}},
      {OpType::Reshape, {"input", "shape"}},
      {OpType::Dense, {"input", "weights", "bias"}},
      {OpType::Conv2D, {"input", "weights", "bias"}},
      {OpType::Relu, {"input"}},
      {OpType::MaxPool2D, {"input"}},
      {OpType::Add, {"input", "addend"}},
      {OpType::Softmax, {"input"}},
      {OpType::ConvTranspose2D, {"input", "weights", "bias"}},
      {OpType::Sigmoid, {"input"}},
      {OpType::Clip, {"input", "min", "max"}},
      {OpType::Mul, {"input", "rhs"}},
      {OpType::Sub, {"input", "rhs"}},
      {OpType::Div, {"input", "rhs"}},
      {OpType::Slice, {"input", "starts", "ends", "axes", "steps"}},
      {OpType::Transpose, {"input"}},
      {OpType::NearestResize2D, {"input", "roi", "scales", "sizes"}},
      {OpType::BatchNorm2D, {"input", "scale", "bias", "mean", "variance"}},
      {OpType::AdaptiveAvgPool2D, {"input"}},
  };
  if (type == OpType::ChannelConcat) return index == 0 ? "input" : "input_" + std::to_string(index);
  const auto& roles = kRoles.at(type);
  if (index >= roles.size()) throw std::runtime_error("Too many ONNX node inputs");
  return roles[index];
}

std::shared_ptr<Tensor> AddTensor(Graph& graph, const std::string& name,
                                  std::vector<int64_t> shape = {}) {
  const auto tensor = graph.FindTensor(name);
  return tensor == nullptr ? graph.AddTensor(name, std::move(shape)) : tensor;
}

void ImportAttribute(const onnx::AttributeProto& attribute, Operation& operation,
                     const std::vector<std::shared_ptr<Tensor>>& outputs) {
  if (attribute.type() == onnx::AttributeProto::INTS) {
    operation.SetAttribute(attribute.name(),
                           std::vector<int64_t>(attribute.ints().begin(), attribute.ints().end()));
  } else if (attribute.has_f()) {
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
        output->PopulateTensor(FloatData(tensor));
      }
    } else if (tensor.data_type() == onnx::TensorProto::INT64) {
      operation.SetAttribute(attribute.name(), Int64Data(tensor));
      for (const auto& output : outputs) {
        output->PopulateInt64Tensor(Int64Data(tensor));
      }
    } else {
      throw std::runtime_error("Unsupported ONNX constant tensor type");
    }
  }
}

// Normalize shape-derived parameters once so the engine can directly create layers.
void ImportLayerParameters(Operation& operation) {
  switch (operation.Type()) {
    case OpType::Dense: {
      if (!operation.HasInput("input") || !operation.HasInput("weights")) {
        throw std::runtime_error("Linear operation requires an activation and weights: " +
                                 operation.Name());
      }
      const auto input = operation.GetInput("input");
      const auto weights = operation.GetInput("weights");
      const auto& input_shape = input->GetShape();
      const auto& weight_shape = weights->GetShape();
      if (input_shape.size() != 2 || weight_shape.size() != 2) {
        throw std::runtime_error("Linear operation requires rank-2 input and weights: " +
                                 operation.Name());
      }
      operation.SetAttribute("input_size", ToLayerDimension(input_shape[1], input->Name()));
      operation.SetAttribute("output_size", ToLayerDimension(weight_shape[0], weights->Name()));
      const int batch_size =
          input_shape[0] == 0 ? 1 : ToLayerDimension(input_shape[0], input->Name());
      operation.SetAttribute("batch_size", batch_size);
      break;
    }
    case OpType::Relu:
      if (!operation.HasInput("input")) {
        throw std::runtime_error("Relu operation requires one input: " + operation.Name());
      }
      operation.SetAttribute("element_count", TensorElementCount(*operation.GetInput("input")));
      break;
    default:
      break;
  }
}

}  // namespace

Graph BuildGraphFromONNX(const std::string& filename) {
  std::ifstream input(filename, std::ios::binary);
  if (!input.is_open()) {
    throw std::runtime_error("Failed to open ONNX file: " + filename);
  }

  onnx::ModelProto model;
  if (!model.ParseFromIstream(&input) || !model.has_graph()) {
    throw std::runtime_error("Failed to parse ONNX ModelProto: " + filename);
  }
  try {
    onnx::shape_inference::InferShapes(model);
  } catch (const std::exception& exception) {
    throw std::runtime_error("Failed to infer ONNX shapes: " + std::string(exception.what()));
  }

  Graph graph;
  const onnx::GraphProto& model_graph = model.graph();
  std::unordered_set<std::string> initializer_names;
  // weights tensors
  for (const auto& initializer : model_graph.initializer()) {
    initializer_names.insert(initializer.name());
    const auto tensor = AddTensor(graph, initializer.name(), ShapeFromTensor(initializer));
    if (initializer.data_type() == onnx::TensorProto::INT64) {
      // Shape-only initializers were consumed by shape inference above. They
      // have no float payload to upload to Vulkan.
      tensor->PopulateInt64Tensor(Int64Data(initializer));
    } else {
      tensor->PopulateTensor(FloatData(initializer));
    }
  }

  // input tensors
  for (const auto& input_value : model_graph.input()) {
    if (!initializer_names.contains(input_value.name())) {
      graph.AddInput(AddTensor(graph, input_value.name(), ShapeFromValueInfo(input_value)));
    }
  }

  // intermediate tensors
  for (const auto& value_info : model_graph.value_info()) {
    AddTensor(graph, value_info.name(), ShapeFromValueInfo(value_info));
  }

  // output tensors
  for (const auto& output_value : model_graph.output()) {
    graph.AddOutput(AddTensor(graph, output_value.name(), ShapeFromValueInfo(output_value)));
  }

  // operations
  for (int idx = 0; idx < model_graph.node_size(); ++idx) {
    const onnx::NodeProto& node = model_graph.node(idx);
    if (!node.domain().empty() && node.domain() != "ai.onnx") {
      throw std::runtime_error("Unsupported ONNX domain: " + node.domain());
    }
    // Lower a multi-output Split to independent GPU slices. This preserves
    // the graph's single-output layer interface and all original tensor names.
    if (node.op_type() == "Split") {
      if (node.input_size() < 1 || node.output_size() < 1)
        throw std::runtime_error("Split requires input and outputs");
      const auto source = graph.FindTensor(node.input(0));
      if (!source) throw std::runtime_error("Split input is missing");
      int64_t axis = 0;
      for (const auto& attr : node.attribute())
        if (attr.name() == "axis") axis = attr.i();
      const auto& shape = source->GetShape();
      if (axis < 0) axis += shape.size();
      if (axis < 0 || axis >= static_cast<int64_t>(shape.size()))
        throw std::runtime_error("Invalid Split axis");
      int64_t start = 0;
      for (int i = 0; i < node.output_size(); ++i) {
        auto output = graph.FindTensor(node.output(i));
        if (!output || output->GetShape().size() != shape.size())
          throw std::runtime_error("Split needs inferred output shapes");
        const auto extent = output->GetShape()[axis];
        if (extent <= 0) throw std::runtime_error("Split requires positive output extents");
        auto op = graph.AddOperation(
            (node.name().empty() ? "Split_" + std::to_string(idx) : node.name()) + "/part_" +
                std::to_string(i),
            OpType::Slice);
        op->AddInput("input", source);
        op->AddOutput(output);
        op->SetAttribute("slice_axis", axis);
        op->SetAttribute("slice_start", start);
        op->SetAttribute("slice_extent", extent);
        start += extent;
      }
      if (start != shape[axis]) throw std::runtime_error("Split extents do not cover input");
      continue;
    }
    const auto type = GetOpType(node.op_type());
    if (!type.has_value()) {
      throw std::runtime_error("Unsupported ONNX operator: " + node.op_type());
    }

    const std::string node_name =
        node.name().empty() ? node.op_type() + "_" + std::to_string(idx) : node.name();
    const auto operation = graph.AddOperation(node_name, *type);
    operation->SetAttribute("onnx_type", node.op_type());
    if (node.output_size() != 1)
      throw std::runtime_error("Only single-output inference nodes are supported: " + node_name);
    if (*type == OpType::ChannelConcat) operation->SetAttribute("input_count", node.input_size());
    for (int input_index = 0; input_index < node.input_size(); ++input_index) {
      const std::string& input_name = node.input(input_index);
      if (!input_name.empty()) {
        operation->AddInput(InputRole(*type, static_cast<size_t>(input_index)),
                            AddTensor(graph, input_name));
      }
    }

    std::vector<std::shared_ptr<Tensor>> outputs;
    outputs.reserve(node.output_size());
    for (const std::string& output_name : node.output()) {
      const auto output = AddTensor(graph, output_name);
      operation->AddOutput(output);
      outputs.push_back(output);
    }
    for (const auto& attribute : node.attribute()) {
      ImportAttribute(attribute, *operation, outputs);
    }
    if (node.op_type() == "Identity" && operation->GetInput("input")->HasData()) {
      const auto source = operation->GetInput("input");
      if (source->HasInt64Data())
        operation->Outputs().front()->PopulateInt64Tensor(source->Int64Data());
      else
        operation->Outputs().front()->PopulateTensor(source->Data());
    }
    ImportLayerParameters(*operation);
  }
  return graph;
}

}  // namespace vkai
