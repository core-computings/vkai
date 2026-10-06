#include <gtest/gtest.h>

#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "graph/Graph.h"
#include "utils/ONNXLoader.h"

namespace vkai {
namespace test {

TEST(ONNXLoaderTest, LoadsExportedMnistModel) {
  const std::filesystem::path onnx_path =
      std::filesystem::path(VKAI_SOURCE_DIR) / "python/mnist/mnist_model.onnx";
  ASSERT_TRUE(std::filesystem::exists(onnx_path))
      << "Generate the test model with: python python/mnist/train.py --export-onnx";

  Graph graph = BuildGraphFromONNX(onnx_path.string());

  const auto input = graph.FindTensor("input");
  ASSERT_NE(input, nullptr);
  EXPECT_EQ(input->GetShape(), (std::vector<int64_t>{0, 1, 28, 28}));

  const auto fc1_weights = graph.FindTensor("fc1.weight");
  ASSERT_NE(fc1_weights, nullptr);
  EXPECT_TRUE(fc1_weights->HasData());
  EXPECT_EQ(fc1_weights->GetShape(), (std::vector<int64_t>{128, 784}));
  EXPECT_EQ(fc1_weights->Data().size(), 128U * 784U);

  const auto fc1_output = graph.FindTensor("/fc1/Gemm_output_0");
  ASSERT_NE(fc1_output, nullptr);
  EXPECT_EQ(fc1_output->GetShape(), (std::vector<int64_t>{0, 128}));

  const auto fc1 = graph.FindOperation("/fc1/Gemm");
  ASSERT_NE(fc1, nullptr);
  EXPECT_EQ(fc1->GetAttribute<int>("input_size"), 784);
  EXPECT_EQ(fc1->GetAttribute<int>("output_size"), 128);
  EXPECT_EQ(fc1->GetAttribute<int>("batch_size"), 1);
  // Original ONNX attributes retain their original types and values.
  EXPECT_FLOAT_EQ(fc1->GetAttribute<float>("alpha"), 1.0F);
  EXPECT_FLOAT_EQ(fc1->GetAttribute<float>("beta"), 1.0F);
  EXPECT_EQ(fc1->GetAttribute<int64_t>("transB"), 1);

  const auto fc2 = graph.FindOperation("/fc2/Gemm");
  ASSERT_NE(fc2, nullptr);
  EXPECT_EQ(fc2->GetAttribute<int>("input_size"), 128);
  EXPECT_EQ(fc2->GetAttribute<int>("output_size"), 10);
  EXPECT_EQ(fc2->GetAttribute<int>("batch_size"), 1);

  const auto relu = graph.FindOperation("/relu/Relu");
  ASSERT_NE(relu, nullptr);
  EXPECT_EQ(relu->GetAttribute<int>("element_count"), 128);

  ASSERT_EQ(graph.Operations().size(), 5U);
  const auto order = graph.TopologicalSort();
  ASSERT_EQ(order.size(), 5U);
  EXPECT_EQ(order[0]->Name(), "/Constant");
  EXPECT_EQ(order[1]->Name(), "/Reshape");
  EXPECT_EQ(order[2]->Name(), "/fc1/Gemm");
  EXPECT_EQ(order[3]->Name(), "/relu/Relu");
  EXPECT_EQ(order[4]->Name(), "/fc2/Gemm");

  std::cout << "Loaded ONNX topological order: ";
  for (size_t index = 0; index < order.size(); ++index) {
    std::cout << (index == 0 ? "" : " -> ") << order[index]->Name();
  }
  std::cout << '\n';
}

}  // namespace test
}  // namespace vkai
