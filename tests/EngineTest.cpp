#include "OnnxTestModel.h"

namespace vkai::test {
TEST(EngineTest, AllocatesBuffersAndBuildsTopologicalLayers) {
  TestModel model({2, 1, 2, 3}, {2, 3});
  MakeTestGraph(model);
  Engine engine(model.Save());
  const auto& graph = engine.GetGraph();
  ASSERT_EQ(graph.Inputs().size(), 1U);
  ASSERT_TRUE(graph.Inputs().front()->HasBuffer());
  EXPECT_EQ(graph.Inputs().front()->Buffer().Size(), 12U * sizeof(float));
  for (const auto& operation : graph.Operations())
    for (const auto& output : operation->Outputs()) EXPECT_TRUE(output->HasBuffer());
  const auto& layers = engine.GetLayers();
  ASSERT_EQ(layers.size(), 3U);
  EXPECT_FALSE(layers.contains("Constant1"));
  for (const auto* name : {"Reshape2", "Gemm3", "Relu4"}) {
    ASSERT_TRUE(layers.contains(name));
  }
  // Reshape records a buffer copy; only compute layers have a pipeline.
  for (const auto* name : {"Gemm3", "Relu4"}) {
    EXPECT_NE(layers.at(name)->pipeline, VK_NULL_HANDLE);
  }
  for (const auto* name : {"weights", "bias"}) {
    auto tensor = graph.FindTensor(name);
    ASSERT_NE(tensor, nullptr);
    ASSERT_TRUE(tensor->HasBuffer());
    EXPECT_NE(tensor->Buffer().buffer, VK_NULL_HANDLE);
  }
  EXPECT_EQ(graph.FindTensor("weights")->Buffer().Size(), 18U * sizeof(float));
  EXPECT_EQ(graph.FindTensor("bias")->Buffer().Size(), 3U * sizeof(float));
  const auto& order = engine.GetTopoOrder();
  ASSERT_EQ(order.size(), 4U);
  EXPECT_EQ(order[0]->Name(), "Constant1");
  EXPECT_EQ(order[1]->Name(), "Reshape2");
  EXPECT_EQ(order[2]->Name(), "Gemm3");
  EXPECT_EQ(order[3]->Name(), "Relu4");
  EXPECT_EQ(order, graph.TopologicalSort());
  EXPECT_EQ(&order, &engine.GetTopoOrder());
}

TEST(EngineTest, ExecutesGeneratedGraphAcrossBatches) {
  TestModel model({2, 1, 2, 3}, {2, 3});
  MakeTestGraph(model);
  model.Check({1, 2, 3, 4, 5, 6, -1, -2, -3, -4, -5, -6}, {3.5F, 3.5F, 0, 0, 0, 4});
}
}  // namespace vkai::test
