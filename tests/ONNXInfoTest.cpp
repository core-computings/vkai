#include <gtest/gtest.h>
#include <onnx/onnx_pb.h>
#include <onnx/shape_inference/implementation.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

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

void PrintShape(const std::string& name, const std::string& shape) {
  std::cout << name << ": " << shape << '\n';
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
  ASSERT_NO_THROW(onnx::shape_inference::InferShapes(model));

  const auto& graph = model.graph();
  for (const auto& value_info : graph.input()) {
    PrintShape(value_info.name(), ShapeString(value_info));
  }
  for (const auto& initializer : graph.initializer()) {
    PrintShape(initializer.name(), ShapeString(initializer));
  }
  for (const auto& value_info : graph.value_info()) {
    PrintShape(value_info.name(), ShapeString(value_info));
  }
  for (const auto& value_info : graph.output()) {
    PrintShape(value_info.name(), ShapeString(value_info));
  }
}

}  // namespace test
}  // namespace vkai
