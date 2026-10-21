#include "OnnxTestModel.h"
#include "utils/ONNXLoader.h"

namespace vkai::test {
TEST(ONNXLoaderTest, LoadsGeneratedGraphAttributesParametersAndConnections) {
  TestModel model({2, 1, 2, 3}, {2, 3});
  MakeTestGraph(model);
  Graph graph = BuildGraphFromONNX(model.Save());
  auto input = graph.FindTensor("input");
  ASSERT_NE(input, nullptr);
  EXPECT_EQ(input->GetShape(), (std::vector<int64_t>{2, 1, 2, 3}));
  auto weights = graph.FindTensor("weights");
  ASSERT_NE(weights, nullptr);
  EXPECT_TRUE(weights->HasData());
  EXPECT_EQ(weights->GetShape(), (std::vector<int64_t>{3, 6}));
  EXPECT_EQ(weights->Data().size(), 18U);
  auto dense = graph.FindOperation("Gemm3");
  ASSERT_NE(dense, nullptr);
  EXPECT_EQ(dense->GetAttribute<int>("input_size"), 6);
  EXPECT_EQ(dense->GetAttribute<int>("output_size"), 3);
  EXPECT_EQ(dense->GetAttribute<int>("batch_size"), 2);
  EXPECT_FLOAT_EQ(dense->GetAttribute<float>("alpha"), 1.F);
  EXPECT_FLOAT_EQ(dense->GetAttribute<float>("beta"), 1.F);
  EXPECT_EQ(dense->GetAttribute<int64_t>("transB"), 1);
  EXPECT_EQ(dense->GetInput("input"), graph.FindTensor("flat"));
  EXPECT_EQ(dense->GetInput("weights"), weights);
  EXPECT_EQ(dense->GetInput("bias"), graph.FindTensor("bias"));
  auto relu = graph.FindOperation("Relu4");
  ASSERT_NE(relu, nullptr);
  EXPECT_EQ(relu->GetAttribute<int>("element_count"), 6);
  EXPECT_EQ(relu->GetInput("input"), graph.FindTensor("dense"));
  EXPECT_FALSE(relu->HasInput("weights"));
  auto reshape = graph.FindOperation("Reshape2");
  ASSERT_NE(reshape, nullptr);
  EXPECT_EQ(reshape->GetInput("input"), input);
  EXPECT_EQ(reshape->GetInput("shape"), graph.FindTensor("shape"));
  ASSERT_EQ(graph.Operations().size(), 4U);
  const auto order = graph.TopologicalSort();
  ASSERT_EQ(order.size(), 4U);
  EXPECT_EQ(order[0]->Name(), "Constant1");
  EXPECT_EQ(order[1]->Name(), "Reshape2");
  EXPECT_EQ(order[2]->Name(), "Gemm3");
  EXPECT_EQ(order[3]->Name(), "Relu4");
}
}  // namespace vkai::test
