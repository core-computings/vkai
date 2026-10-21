#pragma once

#include <gtest/gtest.h>
#include <onnx/onnx_pb.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "graph/Engine.h"

namespace vkai::test {

class TestModel {
 public:
  TestModel(const std::vector<int64_t>& input, const std::vector<int64_t>& output) {
    model_.set_ir_version(7);
    auto* version = model_.add_opset_import();
    version->set_version(13);
    auto* graph = model_.mutable_graph();
    graph->set_name("unit");
    Value(graph->add_input(), "input", input);
    Value(graph->add_output(), "output", output);
    static std::atomic<unsigned> counter{0};
    path_ = std::filesystem::temp_directory_path() /
            ("vkai-unit-" +
             std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + "-" +
             std::to_string(counter++) + ".onnx");
  }
  ~TestModel() { std::filesystem::remove(path_); }
  TestModel(const TestModel&) = delete;
  TestModel& operator=(const TestModel&) = delete;
  static void Value(onnx::ValueInfoProto* value, const std::string& name,
                    const std::vector<int64_t>& shape) {
    value->set_name(name);
    auto* type = value->mutable_type()->mutable_tensor_type();
    type->set_elem_type(onnx::TensorProto::FLOAT);
    for (auto d : shape) type->mutable_shape()->add_dim()->set_dim_value(d);
  }
  void Weight(const std::string& name, const std::vector<int64_t>& shape,
              const std::vector<float>& data) {
    auto* t = model_.mutable_graph()->add_initializer();
    t->set_name(name);
    t->set_data_type(onnx::TensorProto::FLOAT);
    for (auto d : shape) t->add_dims(d);
    for (auto f : data) t->add_float_data(f);
  }
  void IntWeight(const std::string& name, const std::vector<int64_t>& shape,
                 const std::vector<int64_t>& data) {
    auto* t = model_.mutable_graph()->add_initializer();
    t->set_name(name);
    t->set_data_type(onnx::TensorProto::INT64);
    for (auto d : shape) t->add_dims(d);
    for (auto value : data) t->add_int64_data(value);
  }
  static void String(onnx::NodeProto* node, const std::string& name, const std::string& value) {
    auto* a = node->add_attribute();
    a->set_name(name);
    a->set_type(onnx::AttributeProto::STRING);
    a->set_s(value);
  }
  onnx::NodeProto* Node(const std::string& type, const std::vector<std::string>& inputs,
                        const std::string& output = "output") {
    auto* node = model_.mutable_graph()->add_node();
    node->set_op_type(type);
    node->set_name(type + std::to_string(model_.graph().node_size()));
    for (const auto& input : inputs) node->add_input(input);
    node->add_output(output);
    return node;
  }
  static void Int(onnx::NodeProto* node, const std::string& name, int64_t value) {
    auto* a = node->add_attribute();
    a->set_name(name);
    a->set_type(onnx::AttributeProto::INT);
    a->set_i(value);
  }
  static void Float(onnx::NodeProto* node, const std::string& name, float value) {
    auto* a = node->add_attribute();
    a->set_name(name);
    a->set_type(onnx::AttributeProto::FLOAT);
    a->set_f(value);
  }
  static void Ints(onnx::NodeProto* node, const std::string& name,
                   const std::vector<int64_t>& values) {
    auto* a = node->add_attribute();
    a->set_name(name);
    a->set_type(onnx::AttributeProto::INTS);
    for (auto v : values) a->add_ints(v);
  }
  std::string Save() {
    std::ofstream stream(path_, std::ios::binary);
    if (!model_.SerializeToOstream(&stream))
      throw std::runtime_error("Cannot save unit ONNX model");
    return path_.string();
  }
  void Check(const std::vector<float>& input, const std::vector<float>& reference) {
    vkai::Engine engine(Save());
    engine.GetInputTensor()->PopulateTensor(input);
    // Check repeat execution and input upload after a previous command completes.
    for (int iteration = 0; iteration < 2; ++iteration) {
      engine.ExecuteGraph();
      const auto& actual = engine.GetOutputTensor()->Data();
      ASSERT_EQ(actual.size(), reference.size());
      for (size_t i = 0; i < actual.size(); ++i) EXPECT_NEAR(actual[i], reference[i], 1e-5F) << i;
    }
  }

  onnx::ModelProto& Proto() { return model_; }

 private:
  onnx::ModelProto model_;
  std::filesystem::path path_;
};

// A tiny generated graph exercises constants, reshape, parameters, and activation.
inline void MakeTestGraph(TestModel& model) {
  auto* constant = model.Node("Constant", {}, "shape");
  auto* attribute = constant->add_attribute();
  attribute->set_name("value");
  attribute->set_type(onnx::AttributeProto::TENSOR);
  auto* shape = attribute->mutable_t();
  shape->set_data_type(onnx::TensorProto::INT64);
  shape->add_dims(2);
  shape->add_int64_data(2);
  shape->add_int64_data(6);
  model.Node("Reshape", {"input", "shape"}, "flat");
  model.Weight("weights", {3, 6}, {1, 0, -1, 0, 1, 0, 0, 1, 0, -1, 0, 1, -1, 0, 1, 0, -1, 0});
  model.Weight("bias", {3}, {.5F, -.5F, 1});
  auto* dense = model.Node("Gemm", {"flat", "weights", "bias"}, "dense");
  TestModel::Int(dense, "transB", 1);
  TestModel::Float(dense, "alpha", 1);
  TestModel::Float(dense, "beta", 1);
  model.Node("Relu", {"dense"});
}

}  // namespace vkai::test
