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
  softmax->AddInput("input", merged_output);
  softmax->AddOutput(probabilities);

  const auto add = graph.AddOperation("residual_add", OpType::Add);
  add->AddInput("input", relu_output);
  add->AddInput("addend", fc1_output);
  add->AddOutput(merged_output);

  const auto relu = graph.AddOperation("relu", OpType::Relu);
  relu->AddInput("input", fc1_output);
  relu->AddOutput(relu_output);

  const auto dense = graph.AddOperation("fc1", OpType::Dense);
  dense->AddInput("input", input);
  dense->AddInput("weights", weights);
  dense->AddInput("bias", bias);
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
  EXPECT_EQ(dense->GetInput("input"), input);
  EXPECT_EQ(dense->GetInput("weights"), weights);
  EXPECT_EQ(dense->GetInput("bias"), bias);
  EXPECT_FALSE(relu->HasInput("bias"));
  EXPECT_THROW(relu->GetInput("bias"), std::out_of_range);

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
  merge->AddInput("input", left_output);
  merge->AddInput("addend", right_output);
  merge->AddOutput(output);
  const auto left = graph.AddOperation("left", OpType::Relu);
  left->AddInput("input", input);
  left->AddOutput(left_output);
  const auto right = graph.AddOperation("right", OpType::Relu);
  right->AddInput("input", input);
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
  add->AddInput("input", intermediate);
  add->AddInput("addend", intermediate);
  add->AddOutput(output);
  const auto relu = graph.AddOperation("relu", OpType::Relu);
  relu->AddInput("input", input);
  relu->AddOutput(intermediate);

  const std::vector<std::shared_ptr<Operation>> expected{relu, add};
  EXPECT_EQ(graph.TopologicalSort(), expected);
  EXPECT_EQ(add->GetInput("input"), add->GetInput("addend"));
}

TEST(GraphTest, RejectsCycleEvenWhenTensorsHaveCpuData) {
  Graph graph;
  const auto a = graph.AddTensor("a", {1});
  const auto b = graph.AddTensor("b", {1});
  a->PopulateTensor({1.0F});
  b->PopulateTensor({2.0F});
  const auto first = graph.AddOperation("first", OpType::Relu);
  first->AddInput("input", b);
  first->AddOutput(a);
  const auto second = graph.AddOperation("second", OpType::Relu);
  second->AddInput("input", a);
  second->AddOutput(b);

  EXPECT_THROW(graph.TopologicalSort(), std::logic_error);
}

TEST(GraphTest, RejectsInputWithoutProducerOrData) {
  Graph graph;
  const auto missing = graph.AddTensor("missing", {1});
  const auto output = graph.AddTensor("output", {1});
  const auto relu = graph.AddOperation("relu", OpType::Relu);
  relu->AddInput("input", missing);
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
  graph.AddOperation("relu", OpType::Relu)->AddInput("input", foreign);
  EXPECT_THROW(graph.TopologicalSort(), std::invalid_argument);
}

TEST(GraphTest, RejectsDuplicateInputRoles) {
  Graph graph;
  const auto first = graph.AddTensor("first", {1});
  const auto second = graph.AddTensor("second", {1});
  const auto operation = graph.AddOperation("add", OpType::Add);
  operation->AddInput("input", first);

  EXPECT_THROW(operation->AddInput("input", second), std::invalid_argument);
  EXPECT_EQ(operation->GetInput("input"), first);
}

}  // namespace test
}  // namespace vkai
