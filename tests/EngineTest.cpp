#include <gtest/gtest.h>

#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <vector>

#include "engine/Engine.h"

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
  EXPECT_FALSE(graph.Inputs().front()->HasBuffer());

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

TEST(EngineTest, UploadsTensorDataToVulkanBuffers) {
  const std::filesystem::path onnx_path =
      std::filesystem::path(VKAI_SOURCE_DIR) / "python/mnist/mnist_model.onnx";
  ASSERT_TRUE(std::filesystem::exists(onnx_path));

  const Engine engine(onnx_path.string());
  size_t uploaded_tensors = 0;
  for (const auto& [name, tensor] : engine.GetGraph().Tensors()) {
    if (!tensor->HasData() || tensor->Data().empty()) {
      continue;
    }
    SCOPED_TRACE(name);
    ASSERT_TRUE(tensor->HasBuffer());
    const auto& data = tensor->Data();
    ASSERT_EQ(tensor->Buffer().Size(), data.size() * sizeof(float));
    tensor->Buffer().MapData([&data](void* mapped_data) {
      EXPECT_EQ(std::memcmp(mapped_data, data.data(), data.size() * sizeof(float)), 0);
    });
    ++uploaded_tensors;
  }
  EXPECT_EQ(uploaded_tensors, 4U);
}

}  // namespace test
}  // namespace vkai
