#include <gtest/gtest.h>

#include <iostream>
#include <memory>
#include <stdexcept>
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
  weights->PopulateTensor({0.25F});
  bias->PopulateTensor({0.0F});

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

TEST(GraphTest, OrdersBranchesBeforeTheirMerge) {
  Graph graph;
  const auto input = graph.AddTensor("input", {1});
  const auto left_output = graph.AddTensor("left_output", {1});
  const auto right_output = graph.AddTensor("right_output", {1});
  const auto output = graph.AddTensor("output", {1});
  graph.AddInput(input);
  graph.AddOutput(output);

  const auto merge = graph.AddOperation("merge", OpType::Add);
  merge->AddInput(left_output);
  merge->AddInput(right_output);
  merge->AddOutput(output);
  const auto left = graph.AddOperation("left", OpType::Relu);
  left->AddInput(input);
  left->AddOutput(left_output);
  const auto right = graph.AddOperation("right", OpType::Relu);
  right->AddInput(input);
  right->AddOutput(right_output);

  const std::vector<std::shared_ptr<Operation>> expected{left, right, merge};
  EXPECT_EQ(graph.TopologicalSort(), expected);
  EXPECT_EQ(graph.TopologicalSort(), expected);
}

TEST(GraphTest, RepeatedInputsAndCpuDataDoNotBypassProducer) {
  Graph graph;
  const auto input = graph.AddTensor("input", {1});
  const auto intermediate = graph.AddTensor("intermediate", {1});
  const auto output = graph.AddTensor("output", {1});
  graph.AddInput(input);
  // CPU data left by an earlier inference must not remove the dependency.
  intermediate->PopulateTensor({1.0F});
  output->PopulateTensor({2.0F});

  const auto add = graph.AddOperation("add", OpType::Add);
  add->AddInput(intermediate);
  add->AddInput(intermediate);
  add->AddOutput(output);
  const auto relu = graph.AddOperation("relu", OpType::Relu);
  relu->AddInput(input);
  relu->AddOutput(intermediate);

  const std::vector<std::shared_ptr<Operation>> expected{relu, add};
  EXPECT_EQ(graph.TopologicalSort(), expected);
}

TEST(GraphTest, RejectsCycleEvenWhenTensorsHaveCpuData) {
  Graph graph;
  const auto a = graph.AddTensor("a", {1});
  const auto b = graph.AddTensor("b", {1});
  a->PopulateTensor({1.0F});
  b->PopulateTensor({2.0F});
  const auto first = graph.AddOperation("first", OpType::Relu);
  first->AddInput(b);
  first->AddOutput(a);
  const auto second = graph.AddOperation("second", OpType::Relu);
  second->AddInput(a);
  second->AddOutput(b);

  EXPECT_THROW(graph.TopologicalSort(), std::logic_error);
}

TEST(GraphTest, RejectsInputWithoutProducerOrData) {
  Graph graph;
  const auto missing = graph.AddTensor("missing", {1});
  const auto output = graph.AddTensor("output", {1});
  const auto relu = graph.AddOperation("relu", OpType::Relu);
  relu->AddInput(missing);
  relu->AddOutput(output);

  EXPECT_THROW(graph.TopologicalSort(), std::logic_error);
}

TEST(GraphTest, RejectsDuplicateProducers) {
  Graph graph;
  const auto output = graph.AddTensor("output", {1});
  graph.AddOperation("first", OpType::Constant)->AddOutput(output);
  graph.AddOperation("second", OpType::Constant)->AddOutput(output);

  EXPECT_THROW(graph.TopologicalSort(), std::logic_error);
}

TEST(GraphTest, RejectsTensorFromAnotherGraph) {
  Graph graph;
  Graph other;
  const auto foreign = other.AddTensor("foreign", {1});
  graph.AddOperation("relu", OpType::Relu)->AddInput(foreign);
  EXPECT_THROW(graph.TopologicalSort(), std::invalid_argument);
}

}  // namespace test
}  // namespace vkai
