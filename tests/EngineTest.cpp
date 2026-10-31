#include <cstring>

#include "OnnxTestModel.h"
#include "TestUtils.h"
#include "layers/Reshape.h"
#include "utils/BufferUtils.h"
#include "utils/Synchronization.h"

namespace vkai::test {
namespace {
void CheckConvRelu(int channels, int kernel) {
  constexpr int batch = 2, height = 7, width = 11;
  const std::vector<int64_t> shape{batch, channels, height, width};
  TestModel model(shape, shape);
  std::vector<float> weights(channels * channels * kernel * kernel, 0.F);
  for (int c = 0; c < channels; ++c)
    weights[(c * channels + c) * kernel * kernel + kernel * kernel / 2] = c % 2 ? -1.F : 1.F;
  model.Weight("w", {channels, channels, kernel, kernel}, weights);
  model.Weight("b", {channels}, std::vector<float>(channels, -.25F));
  model.Node("Relu", {"input"}, "features");
  auto* conv = model.Node("Conv", {"features", "w", "b"}, "raw");
  TestModel::Ints(conv, "pads", {kernel / 2, kernel / 2, kernel / 2, kernel / 2});
  model.Node("Relu", {"raw"}, "activated");
  model.Node("Sigmoid", {"activated"});
  Engine engine(model.Save(), 1);  // Conv and Relu straddle a submission boundary.
  EXPECT_NE(engine.GetGraph().FindTensor("features")->Buffer().buffer,
            engine.GetGraph().FindTensor("raw")->Buffer().buffer);
  for (int offset : {0, 7}) {
    const auto input = TestValues(batch * channels * height * width, .125F, offset);
    std::vector<float> expected(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
      const auto channel = (i / (height * width)) % channels;
      const float value = std::max(input[i], 0.F) * (channel % 2 ? -1.F : 1.F) - .25F;
      expected[i] = 1.F / (1.F + std::exp(-std::max(value, 0.F)));
    }
    engine.GetInputTensor()->PopulateTensor(input);
    engine.ExecuteGraph();
    ExpectNear(engine.GetOutputTensor()->Data(), expected);
    const auto result = engine.BenchmarkGraph(0, 2);
    EXPECT_EQ(result.submissions_per_run, 4U);
    ExpectNear(engine.GetOutputTensor()->Data(), expected);
  }
}
}  // namespace

TEST(EngineTest, ConvReluPreservesValuesAndLifetimesInGeneralKernel) { CheckConvRelu(3, 1); }
TEST(EngineTest, ConvReluPreservesValuesAndLifetimesInTiledKernel) { CheckConvRelu(17, 3); }

TEST(EngineTest, ConvReluPreservesUnactivatedBranchValues) {
  TestModel model({1, 1, 1, 2}, {1, 1, 1, 2});
  model.Weight("w", {1, 1, 1, 1}, {-1});
  model.Node("Conv", {"input", "w"}, "raw");
  model.Node("Relu", {"raw"}, "positive");
  model.Node("Add", {"raw", "positive"});
  model.Check({2, -3}, {-2, 6});
}

TEST(EngineTest, ConvReluPreservesUnactivatedGraphOutput) {
  TestModel model({1, 1, 1, 2}, {1, 1, 1, 2});
  model.Weight("w", {1, 1, 1, 1}, {-1});
  model.Node("Conv", {"input", "w"}, "raw");
  model.Node("Relu", {"raw"});
  TestModel::Value(model.Proto().mutable_graph()->add_output(), "raw", {1, 1, 1, 2});
  Engine engine(model.Save());
  engine.GetInputTensor()->PopulateTensor({2, -3});
  engine.ExecuteGraph();
  EXPECT_EQ(engine.GetOutputTensor()->Data(), (std::vector<float>{0, 3}));
  EXPECT_EQ(engine.GetGraph().Outputs().at(1)->Data(), (std::vector<float>{-2, 3}));
}

TEST(EngineTest, AllocatesBuffersAndBuildsTopologicalLayers) {
  TestModel model({2, 1, 2, 3}, {2, 3});
  MakeTestGraph(model);
  Engine engine(model.Save());
  const auto& graph = engine.GetGraph();
  ASSERT_EQ(graph.Inputs().size(), 1U);
  ASSERT_TRUE(graph.Inputs().front()->HasBuffer());
  EXPECT_EQ(graph.Inputs().front()->Buffer().Size(), 12U * sizeof(float));
  for (const auto& operation : graph.Operations())
    for (const auto& output : operation->Outputs())
      EXPECT_EQ(output->HasBuffer(), !output->HasInt64Data());
  const auto& layers = engine.GetLayers();
  ASSERT_EQ(layers.size(), 3U);
  EXPECT_FALSE(layers.contains("Constant1"));
  for (const auto* name : {"Reshape2", "Gemm3", "Relu4"}) {
    ASSERT_TRUE(layers.contains(name));
  }
  EXPECT_EQ(graph.FindTensor("flat")->Buffer().buffer, graph.Inputs().front()->Buffer().buffer);
  // Reshape aliases storage; only compute layers have a pipeline.
  for (const auto* name : {"Gemm3", "Relu4"}) {
    EXPECT_NE(layers.at(name)->pipeline, VK_NULL_HANDLE);
  }
  for (const auto* name : {"weights", "bias"}) {
    auto tensor = graph.FindTensor(name);
    ASSERT_NE(tensor, nullptr);
    EXPECT_FALSE(tensor->HasBuffer());
  }
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

TEST(EngineTest, ReusesLargerBufferAcrossReshapeAndReadsOnlyThePayload) {
  TestModel model({1, 1, 2, 4}, {1, 1, 1, 2});
  model.Node("Relu", {"input"}, "a");
  const auto pool = model.Node("MaxPool", {"a"}, "b");
  TestModel::Ints(pool, "kernel_shape", {2, 2});
  TestModel::Ints(pool, "strides", {2, 2});
  model.Node("Relu", {"b"}, "c");
  model.Node("Identity", {"c"}, "view");
  model.Node("Relu", {"view"});
  Engine engine(model.Save(), 1);
  const auto& graph = engine.GetGraph();
  const auto a = graph.FindTensor("a");
  const auto c = graph.FindTensor("c");
  EXPECT_EQ(a->Buffer().buffer, c->Buffer().buffer);
  EXPECT_EQ(c->Buffer().Size(), 8U * sizeof(float));
  EXPECT_EQ(c->ByteSize(), 2U * sizeof(float));
  EXPECT_EQ(c->Buffer().buffer, graph.FindTensor("view")->Buffer().buffer);
  engine.GetInputTensor()->PopulateTensor({1, 2, 3, 4, 5, 6, 7, 8});
  engine.ExecuteGraph();
  EXPECT_EQ(engine.GetOutputTensor()->Data(), (std::vector<float>{6, 8}));
  engine.BenchmarkGraph(1, 3);
  EXPECT_EQ(engine.GetOutputTensor()->Data(), (std::vector<float>{6, 8}));
  engine.GetInputTensor()->PopulateTensor({8, 7, 6, 5, 4, 3, 2, 1});
  engine.ExecuteGraph();
  EXPECT_EQ(engine.GetOutputTensor()->Data(), (std::vector<float>{8, 6}));
}

TEST(EngineTest, OutputViewPreservesItsShapeAndValues) {
  TestModel model({1, 1, 2, 4}, {1, 1, 1, 2});
  model.Node("Relu", {"input"}, "a");
  const auto pool = model.Node("MaxPool", {"a"}, "b");
  TestModel::Ints(pool, "kernel_shape", {2, 2});
  TestModel::Ints(pool, "strides", {2, 2});
  model.Node("Relu", {"b"}, "c");
  model.Node("Identity", {"c"});
  model.Check({1, 2, 3, 4, 5, 6, 7, 8}, {6, 8});
}

TEST(EngineTest, PreservesGpuConstantsAlsoUsedAsLayerParameters) {
  TestModel model({1, 2}, {1, 2});
  model.Weight("weights", {2, 2}, {1, 0, 0, 1});
  model.Weight("bias", {2}, {1, 2});
  const auto dense = model.Node("Gemm", {"input", "weights", "bias"}, "dense");
  TestModel::Int(dense, "transB", 1);
  model.Node("Identity", {"bias"}, "offset");
  model.Node("Add", {"dense", "offset"});
  Engine engine(model.Save());
  EXPECT_FALSE(engine.GetGraph().FindTensor("weights")->HasBuffer());
  EXPECT_TRUE(engine.GetGraph().FindTensor("bias")->HasBuffer());
  EXPECT_EQ(engine.GetGraph().FindTensor("bias")->Buffer().buffer,
            engine.GetGraph().FindTensor("offset")->Buffer().buffer);
  engine.GetInputTensor()->PopulateTensor({3, 4});
  engine.ExecuteGraph();
  EXPECT_EQ(engine.GetOutputTensor()->Data(), (std::vector<float>{5, 8}));
  engine.BenchmarkGraph(1, 3);
  EXPECT_EQ(engine.GetOutputTensor()->Data(), (std::vector<float>{5, 8}));
}

TEST(EngineTest, PreservesMultipleOutputViewsAndOriginalBranchData) {
  TestModel model({1, 2}, {1, 2});
  model.Node("Relu", {"input"}, "a");
  model.Node("Identity", {"a"}, "view");
  model.Node("Sigmoid", {"view"}, "b");
  model.Node("Relu", {"b"}, "c");
  model.Node("Add", {"view", "c"});
  TestModel::Value(model.Proto().mutable_graph()->add_output(), "view", {1, 2});
  Engine engine(model.Save(), 2);
  for (float value : {-1.F, 1.F}) {
    engine.GetInputTensor()->PopulateTensor({value, 0});
    engine.ExecuteGraph();
    const float activated = std::max(value, 0.F);
    EXPECT_EQ(engine.GetGraph().Outputs().at(1)->Data(), (std::vector<float>{activated, 0}));
    EXPECT_NEAR(engine.GetOutputTensor()->Data()[0], activated + 1.F / (1.F + std::exp(-activated)),
                1e-6F);
    EXPECT_FLOAT_EQ(engine.GetOutputTensor()->Data()[1], .5F);
  }
}

TEST(EngineTest, IdentityOnlyGraphHasNoGpuDispatchAndRetainsInputAcrossBenchmarkRuns) {
  TestModel model({1, 2}, {1, 2});
  model.Node("Identity", {"input"});
  Engine engine(model.Save());
  EXPECT_EQ(engine.GetInputTensor()->Buffer().buffer, engine.GetOutputTensor()->Buffer().buffer);
  engine.GetInputTensor()->PopulateTensor({-2, 3});
  engine.ExecuteGraph();
  EXPECT_EQ(engine.GetOutputTensor()->Data(), (std::vector<float>{-2, 3}));
  engine.BenchmarkGraph(1, 3);
  EXPECT_EQ(engine.GetOutputTensor()->Data(), (std::vector<float>{-2, 3}));
}

TEST(EngineTest, StandaloneReshapeCopiesLogicalBytesFromLargerStorage) {
  core::vulkan::VulkanContext context(false, core::vulkan::QueueFamilyType::Compute,
                                      VK_NULL_HANDLE);
  context.Init();
  constexpr auto usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                         VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  core::vulkan::VulkanBuffer input(&context, 4 * sizeof(float), usage, kHostVisibleMemory);
  core::vulkan::VulkanBuffer output(&context, 2 * sizeof(float), usage, kHostVisibleMemory);
  input.MapData([](void* mapped) {
    const float data[] = {2, 3, 99, 99};
    std::memcpy(mapped, data, sizeof(data));
  });
  Reshape reshape(&context, 2 * sizeof(float));
  auto command = core::vulkan::VulkanCommandBuffer::BeginOneTimeCommands(&context);
  reshape.Execute(command.buffer(), input, output);
  Synchronization::InsertHostReadBarrier(command.buffer(), output);
  command.EndOneTimeCommands();
  output.MapData([](void* mapped) {
    const auto* data = static_cast<float*>(mapped);
    EXPECT_FLOAT_EQ(data[0], 2);
    EXPECT_FLOAT_EQ(data[1], 3);
  });
  Reshape too_large(&context, 5 * sizeof(float));
  EXPECT_THROW(too_large.Execute(VK_NULL_HANDLE, input, output), std::invalid_argument);
}

TEST(EngineTest, RetainedOutputStorageOutlivesEngine) {
  TestModel model({1, 4}, {1, 4});
  model.Node("Relu", {"input"});
  std::shared_ptr<Tensor> output;
  {
    Engine engine(model.Save());
    engine.GetInputTensor()->PopulateTensor({-2, 3, -4, 5});
    engine.ExecuteGraph();
    output = engine.GetOutputTensor();
  }
  EXPECT_EQ(output->Data(), (std::vector<float>{0, 3, 0, 5}));
  output->Buffer().MapData([](void* mapped) {
    const auto* values = static_cast<const float*>(mapped);
    EXPECT_EQ(std::vector<float>(values, values + 4), (std::vector<float>{0, 3, 0, 5}));
  });
  std::weak_ptr<Tensor> weak_output = output;
  output.reset();
  EXPECT_TRUE(weak_output.expired());
}

TEST(EngineTest, RetainedViewsAndCopiedTensorShareLiveStorageAfterEngineDestruction) {
  TestModel model({1, 4}, {2, 2});
  model.IntWeight("shape", {2}, {2, 2});
  model.Node("Reshape", {"input", "shape"}, "view");
  model.Node("Identity", {"view"});
  for (int iteration = 0; iteration < 3; ++iteration) {
    std::shared_ptr<Tensor> input, output;
    std::unique_ptr<Tensor> copy;
    {
      Engine engine(model.Save());
      input = engine.GetInputTensor();
      input->PopulateTensor({1, 2, 3, 4});
      engine.ExecuteGraph();
      output = engine.GetOutputTensor();
      copy = std::make_unique<Tensor>(*engine.GetGraph().FindTensor("view"));
      EXPECT_EQ(input->Buffer().buffer, output->Buffer().buffer);
      EXPECT_EQ(copy->Buffer().buffer, output->Buffer().buffer);
    }
    input->Buffer().MapData([](void* mapped) {
      const float updated[] = {5, 6, 7, 8};
      std::memcpy(mapped, updated, sizeof(updated));
    });
    input.reset();
    output.reset();
    copy->Buffer().MapData([](void* mapped) {
      const auto* values = static_cast<const float*>(mapped);
      EXPECT_EQ(std::vector<float>(values, values + 4), (std::vector<float>{5, 6, 7, 8}));
    });
    copy.reset();
  }
}

TEST(EngineTest, CopiedGraphAndRetainedOperationKeepTheirBuffersAlive) {
  TestModel model({1, 2}, {1, 2});
  model.Weight("offset", {2}, {1, 2});
  model.Node("Add", {"input", "offset"}, "sum");
  model.Node("Relu", {"sum"});
  Graph graph;
  std::shared_ptr<Operation> operation;
  {
    Engine engine(model.Save(), 1);
    engine.GetInputTensor()->PopulateTensor({-3, 4});
    engine.ExecuteGraph();
    graph = engine.GetGraph();
    operation = graph.Operations().front();
    ASSERT_EQ(operation->Type(), OpType::Add);
  }
  EXPECT_EQ(graph.Outputs().front()->Data(), (std::vector<float>{0, 6}));
  graph.FindTensor("offset")->Buffer().MapData([](void* mapped) {
    const auto* values = static_cast<const float*>(mapped);
    EXPECT_FLOAT_EQ(values[0], 1);
    EXPECT_FLOAT_EQ(values[1], 2);
  });
  graph = Graph{};
  operation->Outputs().front()->Buffer().MapData([](void* mapped) {
    const auto* values = static_cast<const float*>(mapped);
    EXPECT_FLOAT_EQ(values[0], -2);
    EXPECT_FLOAT_EQ(values[1], 6);
  });
  operation.reset();
}
}  // namespace vkai::test
