#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <unordered_map>
#include <vector>

#include "graph/Engine.h"

namespace vkai {
namespace test {

TEST(EngineTest, BuildsGraphFromExportedMnistModel) {
  const std::filesystem::path onnx_path =
      std::filesystem::path(VKAI_SOURCE_DIR) / "python/mnist/mnist_model.onnx";
  ASSERT_TRUE(std::filesystem::exists(onnx_path))
      << "Generate the test model with: python python/mnist/train.py --export-onnx";

  const Engine engine(onnx_path.string());
  const Graph& graph = engine.GetGraph();

  ASSERT_EQ(graph.Inputs().size(), 1U);
  ASSERT_TRUE(graph.Inputs().front()->HasBuffer());
  EXPECT_EQ(graph.Inputs().front()->Buffer().Size(), 784U * sizeof(float));

  for (const auto& operation : graph.Operations()) {
    for (const auto& output : operation->Outputs()) {
      EXPECT_TRUE(output->HasBuffer()) << output->Name();
    }
  }
  const auto& layers = engine.GetLayers();
  ASSERT_EQ(layers.size(), 4U);
  EXPECT_FALSE(layers.contains("/Constant"));
  EXPECT_TRUE(layers.contains("/Reshape"));
  EXPECT_NE(layers.at("/fc1/Gemm")->pipeline, VK_NULL_HANDLE);
  EXPECT_NE(layers.at("/relu/Relu")->pipeline, VK_NULL_HANDLE);
  EXPECT_NE(layers.at("/fc2/Gemm")->pipeline, VK_NULL_HANDLE);

  const auto weights = graph.FindTensor("fc1.weight");
  ASSERT_NE(weights, nullptr);
  ASSERT_TRUE(weights->HasBuffer());
  EXPECT_NE(weights->Buffer().buffer, VK_NULL_HANDLE);
  EXPECT_EQ(weights->Buffer().Size(), 128U * 784U * sizeof(float));

  const auto bias = graph.FindTensor("fc1.bias");
  ASSERT_NE(bias, nullptr);
  ASSERT_TRUE(bias->HasBuffer());
  EXPECT_EQ(bias->Buffer().Size(), 128U * sizeof(float));

  const auto& order = engine.GetTopoOrder();
  ASSERT_EQ(order.size(), 5U);
  EXPECT_EQ(order[0]->Name(), "/Constant");
  EXPECT_EQ(order[1]->Name(), "/Reshape");
  EXPECT_EQ(order[2]->Name(), "/fc1/Gemm");
  EXPECT_EQ(order[3]->Name(), "/relu/Relu");
  EXPECT_EQ(order[4]->Name(), "/fc2/Gemm");
  EXPECT_EQ(order, graph.TopologicalSort());
  EXPECT_EQ(&order, &engine.GetTopoOrder());
}

TEST(EngineTest, ExecutesMnistGraph) {
  const auto model_dir = std::filesystem::path(VKAI_SOURCE_DIR) / "python/mnist";
  Engine engine((model_dir / "mnist_model.onnx").string());
  const auto input = engine.GetInput();

  // read input from file
  std::vector<float> input_data(784);
  std::ifstream input_file(model_dir / "test_data/input.bin", std::ios::binary);
  ASSERT_TRUE(input_file.read(reinterpret_cast<char*>(input_data.data()),
                              static_cast<std::streamsize>(input_data.size() * sizeof(float))));

  // read reference output from file
  std::vector<float> reference(10);
  std::ifstream reference_file(model_dir / "test_data/fc2_output.bin", std::ios::binary);
  ASSERT_TRUE(reference_file.read(reinterpret_cast<char*>(reference.data()),
                                  static_cast<std::streamsize>(reference.size() * sizeof(float))));

  // inference
  input->SetData(input_data);
  engine.ExecuteGraph();
  const auto output = engine.GetOutput();

  // validate output
  for (size_t index = 0; index < reference.size(); ++index) {
    EXPECT_NEAR(output->Data()[index], reference[index],
                1e-4F + 1e-4F * std::abs(reference[index]));
  }
}

}  // namespace test
}  // namespace vkai
