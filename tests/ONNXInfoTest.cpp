#include <gtest/gtest.h>
#include <onnx/onnx_pb.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>

namespace vkai {
namespace test {
namespace {

std::string ShapeString(const onnx::ValueInfoProto& value_info) {
  if (!value_info.has_type() || !value_info.type().has_tensor_type()) {
    return "<unknown>";
  }

  std::ostringstream stream;
  stream << '[';
  const auto& dimensions = value_info.type().tensor_type().shape().dim();
  for (int index = 0; index < dimensions.size(); ++index) {
    if (index != 0) {
      stream << ", ";
    }
    const auto& dimension = dimensions.Get(index);
    if (dimension.has_dim_value()) {
      stream << dimension.dim_value();
    } else if (dimension.has_dim_param()) {
      stream << dimension.dim_param();
    } else {
      stream << '?';
    }
  }
  stream << ']';
  return stream.str();
}

std::string ShapeString(const onnx::TensorProto& tensor) {
  std::ostringstream stream;
  stream << '[';
  for (int index = 0; index < tensor.dims_size(); ++index) {
    if (index != 0) {
      stream << ", ";
    }
    stream << tensor.dims(index);
  }
  stream << ']';
  return stream.str();
}

int64_t ElementCount(const onnx::TensorProto& tensor) {
  int64_t count = 1;
  for (int index = 0; index < tensor.dims_size(); ++index) {
    count *= tensor.dims(index);
  }
  return count;
}

const char* AttributeTypeName(onnx::AttributeProto::AttributeType type) {
  switch (type) {
    case onnx::AttributeProto::FLOAT:
      return "FLOAT";
    case onnx::AttributeProto::INT:
      return "INT";
    case onnx::AttributeProto::STRING:
      return "STRING";
    case onnx::AttributeProto::TENSOR:
      return "TENSOR";
    case onnx::AttributeProto::FLOATS:
      return "FLOATS";
    case onnx::AttributeProto::INTS:
      return "INTS";
    case onnx::AttributeProto::STRINGS:
      return "STRINGS";
    case onnx::AttributeProto::TENSORS:
      return "TENSORS";
    default:
      return "OTHER";
  }
}

void PrintValueInfo(const char* label, const onnx::ValueInfoProto& value_info) {
  std::cout << "  " << label << ": name=\"" << value_info.name()
            << "\", shape=" << ShapeString(value_info) << '\n';
}

}  // namespace

TEST(ONNXInfoTest, PrintsExportedMnistModel) {
  const std::filesystem::path onnx_path =
      std::filesystem::path(VKAI_SOURCE_DIR) / "python/mnist/mnist_model.onnx";
  ASSERT_TRUE(std::filesystem::exists(onnx_path))
      << "Generate the test model with: python python/mnist/train.py --export-onnx";

  std::ifstream input(onnx_path, std::ios::binary);
  ASSERT_TRUE(input.is_open());

  onnx::ModelProto model;
  ASSERT_TRUE(model.ParseFromIstream(&input));
  ASSERT_TRUE(model.has_graph());

  const auto& graph = model.graph();
  std::unordered_map<std::string, std::string> shapes;
  for (const auto& value_info : graph.input()) {
    shapes.emplace(value_info.name(), ShapeString(value_info));
  }
  for (const auto& value_info : graph.output()) {
    shapes.emplace(value_info.name(), ShapeString(value_info));
  }
  for (const auto& value_info : graph.value_info()) {
    shapes.emplace(value_info.name(), ShapeString(value_info));
  }
  for (const auto& initializer : graph.initializer()) {
    shapes.emplace(initializer.name(), ShapeString(initializer));
  }

  std::cout << "ONNX model: ir_version=" << model.ir_version() << ", producer=\""
            << model.producer_name() << "\", opset=";
  if (model.opset_import_size() != 0) {
    std::cout << model.opset_import(0).version();
  } else {
    std::cout << "<unspecified>";
  }
  std::cout << '\n';

  std::cout << "Graph: name=\"" << graph.name() << "\"\nInputs:\n";
  for (const auto& value_info : graph.input()) {
    PrintValueInfo("input", value_info);
  }

  std::cout << "Initializers:\n";
  for (const auto& initializer : graph.initializer()) {
    std::cout << "  name=\"" << initializer.name() << "\", shape=" << ShapeString(initializer)
              << ", data_type="
              << onnx::TensorProto_DataType_Name(
                     static_cast<onnx::TensorProto_DataType>(initializer.data_type()))
              << ", elements=" << ElementCount(initializer);
    if (!initializer.raw_data().empty()) {
      std::cout << ", raw_bytes=" << initializer.raw_data().size();
    }
    std::cout << '\n';
  }

  std::cout << "Nodes:\n";
  for (int node_index = 0; node_index < graph.node_size(); ++node_index) {
    const auto& node = graph.node(node_index);
    std::cout << "  [" << node_index << "] name=\"" << node.name() << "\", op_type=\""
              << node.op_type() << "\"\n";
    for (const auto& input_name : node.input()) {
      std::cout << "    input:  name=\"" << input_name << "\", shape="
                << (shapes.contains(input_name) ? shapes.at(input_name) : "<not recorded>") << '\n';
    }
    for (const auto& output_name : node.output()) {
      std::cout << "    output: name=\"" << output_name << "\", shape="
                << (shapes.contains(output_name) ? shapes.at(output_name) : "<not recorded>")
                << '\n';
    }
    for (const auto& attribute : node.attribute()) {
      std::cout << "    attribute: name=\"" << attribute.name()
                << "\", type=" << AttributeTypeName(attribute.type());
      if (attribute.has_t()) {
        std::cout << ", tensor_shape=" << ShapeString(attribute.t());
      }
      std::cout << '\n';
    }
  }

  std::cout << "Outputs:\n";
  for (const auto& value_info : graph.output()) {
    PrintValueInfo("output", value_info);
  }
}

}  // namespace test
}  // namespace vkai
