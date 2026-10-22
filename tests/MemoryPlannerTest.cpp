#include <gtest/gtest.h>

#include "graph/MemoryPlanner.h"

namespace vkai::test {
namespace {

void Unary(Graph& graph, const std::shared_ptr<Tensor>& input,
           const std::shared_ptr<Tensor>& output, OpType type = OpType::Relu) {
  const auto op = graph.AddOperation(output->Name() + "_op", type);
  op->AddInput("input", input);
  op->AddOutput(output);
}

TEST(MemoryPlannerTest, ReusesOnlyAfterTheLastConsumerAndKeepsIoDedicated) {
  Graph graph;
  const auto input = graph.AddTensor("input", {8});
  const auto a = graph.AddTensor("a", {8});
  const auto b = graph.AddTensor("b", {8});
  const auto c = graph.AddTensor("c", {8});
  const auto output = graph.AddTensor("output", {8});
  graph.AddInput(input);
  graph.AddOutput(output);
  Unary(graph, input, a);
  Unary(graph, a, b);
  Unary(graph, b, c);
  Unary(graph, c, output);
  const auto plan = BuildMemoryPlan(graph, graph.TopologicalSort());
  EXPECT_EQ(plan.slots.size(), 4U);
  EXPECT_EQ(plan.tensor_slots.at(a.get()), plan.tensor_slots.at(c.get()));
  EXPECT_NE(plan.tensor_slots.at(a.get()), plan.tensor_slots.at(b.get()));
  EXPECT_NE(plan.tensor_slots.at(input.get()), plan.tensor_slots.at(c.get()));
  EXPECT_NE(plan.tensor_slots.at(output.get()), plan.tensor_slots.at(b.get()));
  EXPECT_EQ(plan.tensor_slots, BuildMemoryPlan(graph, graph.TopologicalSort()).tensor_slots);
}

TEST(MemoryPlannerTest, PreservesBranchesAndProducedCpuData) {
  Graph graph;
  const auto input = graph.AddTensor("input", {8});
  const auto a = graph.AddTensor("a", {8});
  const auto b = graph.AddTensor("b", {8});
  const auto c = graph.AddTensor("c", {8});
  const auto output = graph.AddTensor("output", {8});
  graph.AddInput(input);
  graph.AddOutput(output);
  a->PopulateTensor(std::vector<float>(8, 99));
  Unary(graph, input, a);
  Unary(graph, a, b);
  Unary(graph, b, c);
  const auto merge = graph.AddOperation("merge", OpType::Add);
  merge->AddInput("input", c);
  merge->AddInput("addend", a);
  merge->AddOutput(output);
  const auto plan = BuildMemoryPlan(graph, graph.TopologicalSort());
  EXPECT_NE(plan.tensor_slots.at(a.get()), plan.tensor_slots.at(c.get()));
  EXPECT_TRUE(plan.constants.empty());
  EXPECT_FALSE(plan.slots.at(plan.tensor_slots.at(a.get())).persistent);
}

TEST(MemoryPlannerTest, AliasesExtendStorageLifetimeThroughEveryView) {
  Graph graph;
  const auto input = graph.AddTensor("input", {8});
  const auto a = graph.AddTensor("a", {8});
  const auto view = graph.AddTensor("view", {2, 4});
  const auto view2 = graph.AddTensor("view2", {4, 2});
  const auto b = graph.AddTensor("b", {2, 4});
  const auto c = graph.AddTensor("c", {2, 4});
  const auto output = graph.AddTensor("output", {4, 2});
  graph.AddInput(input);
  graph.AddOutput(output);
  Unary(graph, input, a);
  Unary(graph, a, view, OpType::Reshape);
  Unary(graph, view, view2, OpType::Reshape);
  Unary(graph, view, b);
  Unary(graph, b, c);
  const auto merge = graph.AddOperation("merge", OpType::Add);
  merge->AddInput("input", view2);
  merge->AddInput("addend", c);
  merge->AddOutput(output);
  const auto plan = BuildMemoryPlan(graph, graph.TopologicalSort());
  EXPECT_EQ(plan.tensor_slots.at(a.get()), plan.tensor_slots.at(view.get()));
  EXPECT_EQ(plan.tensor_slots.at(a.get()), plan.tensor_slots.at(view2.get()));
  EXPECT_NE(plan.tensor_slots.at(a.get()), plan.tensor_slots.at(c.get()));
}

TEST(MemoryPlannerTest, OutputViewsPinTheUnderlyingStorage) {
  Graph graph;
  const auto input = graph.AddTensor("input", {8});
  const auto a = graph.AddTensor("a", {8});
  const auto output = graph.AddTensor("output", {2, 4});
  const auto b = graph.AddTensor("b", {8});
  const auto c = graph.AddTensor("c", {8});
  graph.AddInput(input);
  graph.AddOutput(output);
  Unary(graph, input, a);
  Unary(graph, a, output, OpType::Reshape);
  Unary(graph, a, b);
  Unary(graph, b, c);
  const auto plan = BuildMemoryPlan(graph, graph.TopologicalSort());
  EXPECT_EQ(plan.tensor_slots.at(a.get()), plan.tensor_slots.at(output.get()));
  EXPECT_NE(plan.tensor_slots.at(a.get()), plan.tensor_slots.at(c.get()));
  EXPECT_TRUE(plan.slots.at(plan.tensor_slots.at(a.get())).persistent);
}

TEST(MemoryPlannerTest, OmitsCpuOnlyParametersButRetainsGpuConstants) {
  Graph graph;
  const auto input = graph.AddTensor("input", {2});
  const auto weights = graph.AddTensor("weights", {2, 2});
  const auto bias = graph.AddTensor("bias", {2});
  const auto shape = graph.AddTensor("shape", {1});
  const auto dense = graph.AddTensor("dense", {2});
  const auto output = graph.AddTensor("output", {2});
  graph.AddInput(input);
  graph.AddOutput(output);
  weights->PopulateTensor({1, 0, 0, 1});
  bias->PopulateTensor({1, 2});
  shape->PopulateInt64Tensor({2});
  const auto fc = graph.AddOperation("fc", OpType::Dense);
  fc->AddInput("input", input);
  fc->AddInput("weights", weights);
  fc->AddInput("bias", bias);
  fc->AddOutput(dense);
  const auto add = graph.AddOperation("add", OpType::Add);
  add->AddInput("input", dense);
  add->AddInput("addend", bias);
  add->AddOutput(output);
  const auto plan = BuildMemoryPlan(graph, graph.TopologicalSort());
  EXPECT_FALSE(plan.tensor_slots.contains(weights.get()));
  EXPECT_FALSE(plan.tensor_slots.contains(shape.get()));
  EXPECT_TRUE(plan.tensor_slots.contains(bias.get()));
  EXPECT_TRUE(plan.slots.at(plan.tensor_slots.at(bias.get())).persistent);
  ASSERT_EQ(plan.constants.size(), 1U);
  EXPECT_EQ(plan.constants.front(), bias);
}

TEST(MemoryPlannerTest, ReusesLargerCapacityAndRejectsInvalidViews) {
  Graph graph;
  const auto input = graph.AddTensor("input", {16});
  const auto a = graph.AddTensor("a", {16});
  const auto b = graph.AddTensor("b", {8});
  const auto c = graph.AddTensor("c", {4});
  const auto output = graph.AddTensor("output", {4});
  graph.AddInput(input);
  graph.AddOutput(output);
  Unary(graph, input, a);
  Unary(graph, a, b, OpType::Slice);
  Unary(graph, b, c, OpType::Slice);
  Unary(graph, c, output);
  const auto plan = BuildMemoryPlan(graph, graph.TopologicalSort());
  EXPECT_EQ(plan.tensor_slots.at(a.get()), plan.tensor_slots.at(c.get()));
  EXPECT_GT(plan.slots.at(plan.tensor_slots.at(c.get())).capacity, c->ByteSize());
  const auto invalid_view = graph.AddTensor("invalid_view", {3});
  Unary(graph, output, invalid_view, OpType::Reshape);
  EXPECT_THROW(BuildMemoryPlan(graph, graph.TopologicalSort()), std::runtime_error);
}

}  // namespace
}  // namespace vkai::test
