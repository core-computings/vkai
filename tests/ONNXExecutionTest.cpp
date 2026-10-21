#include <gtest/gtest.h>
#include <onnx/onnx_pb.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

#include "OnnxTestModel.h"
#include "graph/Engine.h"
#include "utils/ONNXLoader.h"

namespace {

using Model = vkai::test::TestModel;

TEST(ONNXExecutionTest, ConvImportsStridesPaddingAndDilation) {
  Model model({1, 1, 3, 3}, {1, 1, 2, 2});
  model.Weight("w", {1, 1, 2, 2}, {1, 2, 3, 4});
  model.Weight("b", {1}, {.5F});
  auto* node = model.Node("Conv", {"input", "w", "b"});
  Model::Ints(node, "strides", {2, 2});
  Model::Ints(node, "pads", {1, 1, 1, 1});
  Model::Ints(node, "dilations", {1, 1});
  model.Check({1, 2, 3, 4, 5, 6, 7, 8, 9}, {4.5F, 18.5F, 36.5F, 77.5F});
}
TEST(ONNXExecutionTest, BatchNormalizationUsesRunningStatisticsAndEpsilon) {
  Model model({2, 1, 1, 2}, {2, 1, 1, 2});
  model.Weight("scale", {1}, {2});
  model.Weight("b", {1}, {-.5F});
  model.Weight("mean", {1}, {1});
  model.Weight("var", {1}, {.75F});
  auto* node = model.Node("BatchNormalization", {"input", "scale", "b", "mean", "var"});
  Model::Float(node, "epsilon", .25F);
  model.Check({1, 2, -1, 0}, {-.5F, 1.5F, -4.5F, -2.5F});
}
TEST(ONNXExecutionTest, ResidualBranchReluAndAdd) {
  Model model({1, 1, 2, 2}, {1, 1, 2, 2});
  model.Node("Relu", {"input"}, "positive");
  model.Node("Add", {"positive", "input"});
  model.Check({-2, 1, -3, 4}, {-2, 2, -3, 8});
}
TEST(ONNXExecutionTest, PaddedMaxPoolPreservesNegativeValues) {
  Model model({1, 1, 3, 3}, {1, 1, 2, 2});
  auto* node = model.Node("MaxPool", {"input"});
  Model::Ints(node, "kernel_shape", {3, 3});
  Model::Ints(node, "strides", {2, 2});
  Model::Ints(node, "pads", {1, 1, 1, 1});
  model.Check({-1, -2, -3, -4, -5, -6, -7, -8, -9}, {-1, -2, -4, -5});
}
TEST(ONNXExecutionTest, GlobalAveragePoolFlattenGemmAndIdentityParameters) {
  Model model({1, 2, 2, 2}, {1, 2});
  model.Weight("w", {2, 2}, {1, 2, 3, 4});
  model.Weight("b", {2}, {.5F, -.5F});
  model.Node("Identity", {"w"}, "alias");
  model.Node("GlobalAveragePool", {"input"}, "pool");
  auto* flatten = model.Node("Flatten", {"pool"}, "flat");
  Model::Int(flatten, "axis", 1);
  auto* dense = model.Node("Gemm", {"flat", "alias", "b"});
  Model::Int(dense, "transB", 1);
  model.Check({1, 2, 3, 4, 5, 6, 7, 8}, {16, 33});
}
TEST(ONNXExecutionTest, ConcatOrdersChannelsAcrossBatches) {
  Model model({2, 1, 1, 2}, {2, 3, 1, 2});
  model.Weight("skip", {2, 2, 1, 2}, {10, 11, 12, 13, 14, 15, 16, 17});
  auto* node = model.Node("Concat", {"input", "skip"});
  Model::Int(node, "axis", 1);
  model.Check({1, 2, 3, 4}, {1, 2, 10, 11, 12, 13, 3, 4, 14, 15, 16, 17});
}
TEST(ONNXExecutionTest, TransposedConvolutionAndSigmoid) {
  Model model({1, 1, 1, 2}, {1, 1, 2, 4});
  model.Weight("w", {1, 1, 2, 2}, {1, 2, 3, 4});
  auto* node = model.Node("ConvTranspose", {"input", "w"}, "upsampled");
  Model::Ints(node, "strides", {2, 2});
  model.Node("Sigmoid", {"upsampled"});
  std::vector<float> reference{1, 2, 2, 4, 3, 4, 6, 8};
  for (auto& f : reference) f = 1 / (1 + std::exp(-f));
  model.Check({1, 2}, reference);
}
TEST(ONNXExecutionTest, ConstantReshapeAndSoftmax) {
  Model model({1, 1, 1, 2}, {1, 2});
  auto* constant = model.Node("Constant", {}, "shape");
  auto* a = constant->add_attribute();
  a->set_name("value");
  a->set_type(onnx::AttributeProto::TENSOR);
  auto* tensor = a->mutable_t();
  tensor->set_data_type(onnx::TensorProto::INT64);
  tensor->add_dims(2);
  tensor->add_int64_data(1);
  tensor->add_int64_data(2);
  model.Node("Reshape", {"input", "shape"}, "reshaped");
  auto* softmax = model.Node("Softmax", {"reshaped"});
  Model::Int(softmax, "axis", 1);
  model.Check({0, 0}, {.5F, .5F});
}
TEST(ONNXExecutionTest, FloatConstantsContainActualData) {
  Model model({1, 2}, {1, 2});
  auto* constant = model.Node("Constant", {}, "offset");
  auto* a = constant->add_attribute();
  a->set_name("value");
  a->set_type(onnx::AttributeProto::TENSOR);
  auto* tensor = a->mutable_t();
  tensor->set_data_type(onnx::TensorProto::FLOAT);
  tensor->add_dims(1);
  tensor->add_dims(2);
  tensor->add_float_data(3);
  tensor->add_float_data(4);
  model.Node("Add", {"input", "offset"});
  model.Check({1, 2}, {4, 6});
}
TEST(ONNXExecutionTest, RejectsInvalidGroupsAndUnsupportedDomain) {
  Model model({1, 2, 2, 2}, {1, 2, 2, 2});
  model.Weight("w", {2, 1, 1, 1}, {1, 1});
  auto* node = model.Node("Conv", {"input", "w"});
  Model::Int(node, "group", 3);
  EXPECT_THROW(vkai::Engine(model.Save()), std::runtime_error);
  node->set_domain("custom");
  EXPECT_THROW(vkai::BuildGraphFromONNX(model.Save()), std::runtime_error);
}
TEST(ONNXExecutionTest, RejectsShortOrMissingInputsBeforeRecordingCommands) {
  Model model({1, 2}, {1, 2});
  model.Node("Sigmoid", {"input"});
  vkai::Engine engine(model.Save());
  EXPECT_THROW(engine.ExecuteGraph(), std::runtime_error);
  engine.GetInputTensor()->PopulateTensor({1});
  EXPECT_THROW(engine.ExecuteGraph(), std::runtime_error);
}

}  // namespace

TEST(ONNXExecutionTest, GpuBenchmarkPreservesGraphResultsAndValidatesCounts) {
  Model model({1, 2}, {1, 2});
  model.Node("Sigmoid", {"input"});
  vkai::Engine engine(model.Save());
  engine.GetInputTensor()->PopulateTensor({0, 0});
  const auto benchmark = engine.BenchmarkGraph(1, 3);
  ASSERT_EQ(benchmark.gpu_ms.size(), 3U);
  EXPECT_FALSE(benchmark.device_name.empty());
  EXPECT_GT(benchmark.timestamp_valid_bits, 0U);
  EXPECT_GT(benchmark.timestamp_period_ns, 0);
  for (auto ms : benchmark.gpu_ms) {
    EXPECT_TRUE(std::isfinite(ms));
    EXPECT_GE(ms, 0);
  }
  ASSERT_EQ(engine.GetOutputTensor()->Data().size(), 2U);
  EXPECT_FLOAT_EQ(engine.GetOutputTensor()->Data()[0], .5F);
  EXPECT_FLOAT_EQ(engine.GetOutputTensor()->Data()[1], .5F);
  engine.GetInputTensor()->PopulateTensor({1, -1});
  engine.ExecuteGraph();
  EXPECT_NEAR(engine.GetOutputTensor()->Data()[0], 1.0F / (1 + std::exp(-1.0F)), 1e-6F);
  EXPECT_NEAR(engine.GetOutputTensor()->Data()[1], 1.0F / (1 + std::exp(1.0F)), 1e-6F);
  EXPECT_THROW(engine.BenchmarkGraph(-1, 1), std::invalid_argument);
  EXPECT_THROW(engine.BenchmarkGraph(0, 0), std::invalid_argument);
}

namespace {
void CheckGroupedConvolution(int ic, int oc, int groups) {
  constexpr int n = 2, h = 5, w = 4, kh = 3, kw = 2, sh = 2, sw = 1, ph = 1, pw = 0, dh = 1, dw = 2;
  constexpr int oh = 3, ow = 2;
  std::vector<float> x(n * ic * h * w), weights(oc * (ic / groups) * kh * kw), bias(oc);
  for (size_t i = 0; i < x.size(); ++i) x[i] = (static_cast<int>(i % 19) - 9) * .125F;
  for (size_t i = 0; i < weights.size(); ++i) weights[i] = (static_cast<int>(i % 11) - 5) * .25F;
  for (int c = 0; c < oc; ++c) bias[c] = c * .5F;
  std::vector<float> reference(n * oc * oh * ow);
  for (int batch = 0; batch < n; ++batch)
    for (int out = 0; out < oc; ++out)
      for (int y = 0; y < oh; ++y)
        for (int z = 0; z < ow; ++z) {
          float sum = bias[out];
          for (int local = 0; local < ic / groups; ++local)
            for (int ky = 0; ky < kh; ++ky)
              for (int kx = 0; kx < kw; ++kx) {
                const int iy = y * sh - ph + ky * dh, ix = z * sw - pw + kx * dw;
                const int in = (out / (oc / groups)) * (ic / groups) + local;
                if (iy >= 0 && iy < h && ix >= 0 && ix < w)
                  sum += x[((batch * ic + in) * h + iy) * w + ix] *
                         weights[((out * (ic / groups) + local) * kh + ky) * kw + kx];
              }
          reference[((batch * oc + out) * oh + y) * ow + z] = sum;
        }
  Model model({n, ic, h, w}, {n, oc, oh, ow});
  model.Weight("weights", {oc, ic / groups, kh, kw}, weights);
  model.Weight("bias", {oc}, bias);
  auto* conv = model.Node("Conv", {"input", "weights", "bias"});
  Model::Int(conv, "group", groups);
  Model::Ints(conv, "strides", {sh, sw});
  Model::Ints(conv, "pads", {ph, pw, ph, pw});
  Model::Ints(conv, "dilations", {dh, dw});
  model.Check(x, reference);
}
TEST(GroupedConv2DTest, BatchedGroupsWithStrideDilationAndBias) {
  CheckGroupedConvolution(4, 6, 2);
}
TEST(GroupedConv2DTest, DepthwiseWithChannelMultiplier) { CheckGroupedConvolution(3, 6, 3); }
TEST(ONNXExecutionTest, ClipScalarBoundsAndSpatialReduceMean) {
  Model model({2, 1, 2, 2}, {2, 1});
  model.Weight("lower", {}, {0});
  auto* constant = model.Node("Constant", {}, "upper");
  auto* a = constant->add_attribute();
  a->set_name("value");
  a->set_type(onnx::AttributeProto::TENSOR);
  a->mutable_t()->set_data_type(onnx::TensorProto::FLOAT);
  a->mutable_t()->add_float_data(6);
  model.Node("Clip", {"input", "lower", "upper"}, "clipped");
  auto* reduce = model.Node("ReduceMean", {"clipped"});
  Model::Ints(reduce, "axes", {2, 3});
  Model::Int(reduce, "keepdims", 0);
  model.Check({-1, 0, 3, 10, 6, 7, 8, -1}, {2.25F, 4.5F});
}
TEST(ONNXExecutionTest, ClipAllowsOmittedMinimumAndSpatialMeanKeepsDimensions) {
  Model model({1, 1, 2, 2}, {1, 1, 1, 1});
  model.Weight("upper", {}, {2});
  model.Node("Clip", {"input", "", "upper"}, "clipped");
  auto* reduce = model.Node("ReduceMean", {"clipped"});
  Model::Ints(reduce, "axes", {-2, -1});
  Model::Int(reduce, "keepdims", 1);
  model.Check({-4, 0, 3, 10}, {0});
}
}  // namespace

namespace {
TEST(BinaryElementwiseTest, BroadcastsAllFourOperationsAcrossBatchesAndChannels) {
  for (const std::string type : {"Add", "Sub", "Mul", "Div"}) {
    SCOPED_TRACE(type);
    Model model({2, 1, 2, 3}, {2, 4, 2, 3});
    model.Weight("rhs", {1, 4, 1, 1}, {2, 4, 6, 8});
    model.Node(type, {"input", "rhs"});
    std::vector<float> x(12), expected;
    for (int i = 0; i < 12; ++i) x[i] = (i - 6) * .5F;
    for (int n = 0; n < 2; ++n)
      for (int c = 0; c < 4; ++c)
        for (int i = 0; i < 6; ++i) {
          float a = x[n * 6 + i], b = (c + 1) * 2;
          expected.push_back(type == "Add"   ? a + b
                             : type == "Sub" ? a - b
                             : type == "Mul" ? a * b
                                             : a / b);
        }
    model.Check(x, expected);
  }
}
TEST(BinaryElementwiseTest, BroadcastsScalarAndUnequalRanks) {
  Model model({2, 3}, {2, 3});
  model.Weight("scalar", {}, {2});
  model.Node("Sub", {"input", "scalar"});
  model.Check({0, 1, 2, 3, 4, 5}, {-2, -1, 0, 1, 2, 3});
}
TEST(TensorTransformTest, TransposesFourDimensionalDFLLayout) {
  Model model({2, 3, 2, 4}, {2, 2, 3, 4});
  auto* node = model.Node("Transpose", {"input"});
  Model::Ints(node, "perm", {0, 2, 1, 3});
  std::vector<float> x(48), expected;
  std::iota(x.begin(), x.end(), 0.F);
  for (int n = 0; n < 2; ++n)
    for (int h = 0; h < 2; ++h)
      for (int c = 0; c < 3; ++c)
        for (int w = 0; w < 4; ++w) expected.push_back(x[((n * 3 + c) * 2 + h) * 4 + w]);
  model.Check(x, expected);
}
TEST(TensorTransformTest, DefaultsToReversedPermutation) {
  Model model({2, 3}, {3, 2});
  model.Node("Transpose", {"input"});
  model.Check({0, 1, 2, 3, 4, 5}, {0, 3, 1, 4, 2, 5});
}
TEST(TensorTransformTest, SlicesNegativeAxisWithNegativeStepAndExtremeEnd) {
  Model model({2, 5}, {2, 3});
  model.IntWeight("starts", {1}, {-1});
  model.IntWeight("ends", {1}, {std::numeric_limits<int64_t>::min()});
  model.IntWeight("axes", {1}, {-1});
  model.IntWeight("steps", {1}, {-2});
  model.Node("Slice", {"input", "starts", "ends", "axes", "steps"});
  model.Check({0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, {4, 2, 0, 9, 7, 5});
}
TEST(TensorTransformTest, SlicesMultipleAxesWithPositiveSteps) {
  Model model({3, 5}, {2, 2});
  model.IntWeight("starts", {2}, {0, 1});
  model.IntWeight("ends", {2}, {3, 5});
  model.IntWeight("steps", {2}, {2, 2});
  model.Node("Slice", {"input", "starts", "ends", "", "steps"});
  model.Check({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14}, {1, 3, 11, 13});
}
TEST(TensorConcatTest, SupportsMoreThanFiveInputsAndNegativeAxis) {
  Model model({1, 2, 1}, {1, 2, 7});
  std::vector<std::string> inputs{"input"};
  for (int i = 1; i < 7; ++i) {
    std::string name = "part" + std::to_string(i);
    model.Weight(name, {1, 2, 1}, {float(i), float(100 + i)});
    inputs.push_back(name);
  }
  auto* concat = model.Node("Concat", inputs);
  Model::Int(concat, "axis", -1);
  model.Check({0, 100}, {0, 1, 2, 3, 4, 5, 6, 100, 101, 102, 103, 104, 105, 106});
}
TEST(TensorTransformTest, SplitLowersMultipleOutputsWithoutLosingOrder) {
  Model model({1, 6, 2}, {1, 6, 2});
  model.IntWeight("sizes", {2}, {2, 4});
  auto* split = model.Node("Split", {"input", "sizes"}, "left");
  split->add_output("right");
  Model::Int(split, "axis", 1);
  auto* concat = model.Node("Concat", {"right", "left"});
  Model::Int(concat, "axis", 1);
  model.Check({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}, {4, 5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3});
}
TEST(NearestResize2DTest, BatchedNonSquareUpsamplingWithScales) {
  Model model({2, 1, 2, 3}, {2, 1, 4, 6});
  model.Weight("scales", {4}, {1, 1, 2, 2});
  auto* node = model.Node("Resize", {"input", "", "scales"});
  Model::String(node, "mode", "nearest");
  Model::String(node, "coordinate_transformation_mode", "asymmetric");
  Model::String(node, "nearest_mode", "floor");
  std::vector<float> x(12), expected;
  std::iota(x.begin(), x.end(), 0.F);
  for (int n = 0; n < 2; ++n)
    for (int y = 0; y < 4; ++y)
      for (int z = 0; z < 6; ++z) expected.push_back(x[n * 6 + (y / 2) * 3 + z / 2]);
  model.Check(x, expected);
}
TEST(NearestResize2DTest, DownsamplesUsingSizes) {
  Model model({1, 1, 3, 5}, {1, 1, 2, 3});
  model.IntWeight("sizes", {4}, {1, 1, 2, 3});
  auto* node = model.Node("Resize", {"input", "", "", "sizes"});
  Model::String(node, "mode", "nearest");
  Model::String(node, "coordinate_transformation_mode", "asymmetric");
  Model::String(node, "nearest_mode", "floor");
  model.Check({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14}, {0, 1, 3, 5, 6, 8});
}
}  // namespace
