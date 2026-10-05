#include <gtest/gtest.h>

#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "graph/Graph.h"

namespace vkai {
namespace test {

TEST(GraphTest, TopologicalSortPrintsExecutionOrder) {
  Graph graph;

  const auto input = graph.AddTensor("input", {1, 784});
  const auto weights = graph.AddTensor("fc1_weights", {128, 784});
  const auto bias = graph.AddTensor("fc1_bias", {128});
  const auto fc1_output = graph.AddTensor("fc1_output", {1, 128});
  const auto relu_output = graph.AddTensor("relu_output", {1, 128});
  const auto merged_output = graph.AddTensor("merged_output", {1, 128});
  const auto probabilities = graph.AddTensor("probabilities", {1, 128});
  graph.AddInput(input);
  graph.AddOutput(probabilities);
  weights->SetData({0.25F});
  bias->SetData({0.0F});

  // Add operations in reverse dependency order to verify the sort is based on
  // tensor dependencies rather than graph construction order.
  const auto softmax = graph.AddOperation("softmax", OpType::Softmax);
  softmax->AddInput(merged_output);
  softmax->AddOutput(probabilities);

  const auto add = graph.AddOperation("residual_add", OpType::Add);
  add->AddInput(relu_output);
  add->AddInput(fc1_output);
  add->AddOutput(merged_output);

  const auto relu = graph.AddOperation("relu", OpType::Relu);
  relu->AddInput(fc1_output);
  relu->AddOutput(relu_output);

  const auto dense = graph.AddOperation("fc1", OpType::Dense);
  dense->AddInput(input);
  dense->AddInput(weights);
  dense->AddInput(bias);
  dense->AddOutput(fc1_output);
  dense->SetAttribute("activation", std::string("none"));

  const auto order = graph.TopologicalSort();
  ASSERT_EQ(order.size(), 4U);
  EXPECT_EQ(order[0]->Name(), "fc1");
  EXPECT_EQ(order[1]->Name(), "relu");
  EXPECT_EQ(order[2]->Name(), "residual_add");
  EXPECT_EQ(order[3]->Name(), "softmax");
  ASSERT_NE(dense->FindAttribute<std::string>("activation"), nullptr);
  EXPECT_EQ(*dense->FindAttribute<std::string>("activation"), "none");

  std::cout << "Topological order: ";
  for (size_t index = 0; index < order.size(); ++index) {
    std::cout << (index == 0 ? "" : " -> ") << order[index]->Name();
  }
  std::cout << '\n';
}

}  // namespace test
}  // namespace vkai
