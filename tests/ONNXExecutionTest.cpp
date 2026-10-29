#include <gtest/gtest.h>
#include <onnx/onnx_pb.h>

#include <algorithm>
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
#include "TestUtils.h"
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
TEST(ONNXExecutionTest, PointwiseConvReluResidualPreservesChangedInputsAndBenchmarkTransitions) {
  constexpr int batch = 2, ic = 17, oc = 19, h = 7, w = 11, spatial = h * w;
  Model model({batch, ic, h, w}, {batch, oc, h, w});
  const auto weights = vkai::test::TestValues(oc * ic, .0625F, 4);
  const auto bias = vkai::test::TestValues(oc, .5F, 9);
  model.Weight("weights", {oc, ic, 1, 1}, weights);
  model.Weight("bias", {oc}, bias);
  model.Weight("residual", {}, {.25F});
  model.Node("Conv", {"input", "weights", "bias"}, "conv");
  model.Node("Relu", {"conv"}, "relu");
  model.Node("Add", {"relu", "residual"});
  vkai::Engine engine(model.Save());
  for (int iteration = 0; iteration < 3; ++iteration) {
    const auto input = vkai::test::TestValues(batch * ic * spatial, .125F, iteration * 7);
    std::vector<float> expected(batch * oc * spatial);
    for (int n = 0; n < batch; ++n)
      for (int out = 0; out < oc; ++out)
        for (int pos = 0; pos < spatial; ++pos) {
          double value = bias[out];
          for (int in = 0; in < ic; ++in)
            value += double(input[(n * ic + in) * spatial + pos]) * weights[out * ic + in];
          expected[(n * oc + out) * spatial + pos] = std::max(value, 0.0) + .25;
        }
    engine.GetInputTensor()->PopulateTensor(input);
    engine.ExecuteGraph();
    vkai::test::ExpectNear(engine.GetOutputTensor()->Data(), expected);
    if (iteration == 1) {
      const auto benchmark = engine.BenchmarkGraph(1, 2);
      EXPECT_EQ(benchmark.gpu_ms.size(), 2U);
      vkai::test::ExpectNear(engine.GetOutputTensor()->Data(), expected);
    }
  }
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
TEST(ONNXExecutionTest, CeilMaxPoolIncludesPartialBottomAndRightWindows) {
  Model model({1, 1, 4, 6}, {1, 1, 2, 3});
  auto* node = model.Node("MaxPool", {"input"}, "pooled");
  Model::Ints(node, "kernel_shape", {3, 3});
  Model::Ints(node, "strides", {2, 2});
  Model::Int(node, "ceil_mode", 1);
  model.Node("Relu", {"pooled"});
  // Positive maxima at the partial edges also test their inferred downstream shape.
  model.Check({1,  -2,  3,  -4,  5,  -6,  -7,  -8,  -9,  -10, -11, -12,
               13, -14, 15, -16, 17, -18, -19, -20, -21, -22, -23, -24},
              {15, 17, 17, 15, 17, 17});
}
TEST(ONNXExecutionTest, CeilMaxPoolSkipsRightPaddingForCurrentOnnxSchema) {
  Model model({1, 1, 3, 3}, {1, 1, 2, 2});
  model.Proto().mutable_opset_import(0)->set_version(22);
  auto* node = model.Node("MaxPool", {"input"});
  Model::Ints(node, "kernel_shape", {2, 2});
  Model::Ints(node, "strides", {2, 2});
  Model::Ints(node, "pads", {1, 1, 1, 1});
  Model::Int(node, "ceil_mode", 1);
  model.Check({-1, -2, -3, -4, -5, -6, -7, -8, -9}, {-1, -2, -4, -5});
}
TEST(ONNXExecutionTest, RejectsLegacyCeilShapesContainingRightPaddingOnlyWindows) {
  // Opset13 uses MaxPool12 shape inference, which infers 3x3 for this case.
  // Preserve that graph's contract by rejecting it, rather than silently changing
  // downstream tensor shapes to the corrected 2x2 windows used by PyTorch.
  Model model({1, 1, 3, 3}, {1, 1, 3, 3});
  auto* node = model.Node("MaxPool", {"input"});
  Model::Ints(node, "kernel_shape", {2, 2});
  Model::Ints(node, "strides", {2, 2});
  Model::Ints(node, "pads", {1, 1, 1, 1});
  Model::Int(node, "ceil_mode", 1);
  try {
    vkai::Engine engine(model.Save());
    FAIL() << "Expected legacy right-padding-only output shape to be rejected";
  } catch (const std::runtime_error& error) {
    EXPECT_NE(std::string(error.what()).find("inferred right-padding-only windows"),
              std::string::npos);
  }
}
TEST(ONNXExecutionTest, RejectsInvalidMaxPoolCeilMode) {
  for (int mode : {-1, 2}) {
    Model model({1, 1, 4, 4}, {1, 1, 2, 2});
    auto* node = model.Node("MaxPool", {"input"});
    Model::Ints(node, "kernel_shape", {2, 2});
    Model::Ints(node, "strides", {2, 2});
    Model::Int(node, "ceil_mode", mode);
    EXPECT_THROW(vkai::Engine(model.Save()), std::runtime_error);
  }
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
TEST(ONNXExecutionTest, FireModuleMatchesCpuAcrossBranchesRepeatedRunsAndChangedInput) {
  constexpr int batch = 2, height = 3, width = 5, spatial = height * width;
  constexpr int input_channels = 3, squeeze_channels = 2, expand1_channels = 3;
  constexpr int expand3_channels = 2, output_channels = expand1_channels + expand3_channels;
  const std::vector<float> squeeze_weights{1, .5F, -.25F, -.5F, .25F, 1};
  const std::vector<float> squeeze_bias{-.125F, .25F};
  const std::vector<float> expand1_weights{1, .5F, -1, .25F, .125F, -.5F};
  const std::vector<float> expand1_bias{-.25F, .25F, .5F};
  const auto expand3_weights = vkai::test::TestValues(expand3_channels * squeeze_channels * 9);
  const std::vector<float> expand3_bias{.125F, -.25F};
  Model model({batch, input_channels, height, width}, {batch, output_channels, height, width});
  model.Weight("squeeze_w", {squeeze_channels, input_channels, 1, 1}, squeeze_weights);
  model.Weight("squeeze_b", {squeeze_channels}, squeeze_bias);
  model.Weight("expand1_w", {expand1_channels, squeeze_channels, 1, 1}, expand1_weights);
  model.Weight("expand1_b", {expand1_channels}, expand1_bias);
  model.Weight("expand3_w", {expand3_channels, squeeze_channels, 3, 3}, expand3_weights);
  model.Weight("expand3_b", {expand3_channels}, expand3_bias);
  model.Node("Conv", {"input", "squeeze_w", "squeeze_b"}, "squeeze");
  model.Node("Relu", {"squeeze"}, "squeezed");
  model.Node("Conv", {"squeezed", "expand1_w", "expand1_b"}, "expand1");
  model.Node("Relu", {"expand1"}, "expanded1");
  auto* expand3 = model.Node("Conv", {"squeezed", "expand3_w", "expand3_b"}, "expand3");
  Model::Ints(expand3, "pads", {1, 1, 1, 1});
  model.Node("Relu", {"expand3"}, "expanded3");
  auto* concat = model.Node("Concat", {"expanded1", "expanded3"});
  Model::Int(concat, "axis", 1);

  // Direct NCHW CPU convolution and ReLU provide a reference independent of the
  // importer, graph planner, GPU kernels and concatenation implementation.
  auto convolution_relu = [&](const std::vector<float>& input, int ic, int oc, int kernel,
                              const std::vector<float>& weights, const std::vector<float>& bias) {
    std::vector<float> output(batch * oc * spatial);
    for (int n = 0; n < batch; ++n)
      for (int out = 0; out < oc; ++out)
        for (int y = 0; y < height; ++y)
          for (int x = 0; x < width; ++x) {
            float value = bias[out];
            for (int in = 0; in < ic; ++in)
              for (int ky = 0; ky < kernel; ++ky)
                for (int kx = 0; kx < kernel; ++kx) {
                  const int iy = y + ky - kernel / 2, ix = x + kx - kernel / 2;
                  if (iy >= 0 && iy < height && ix >= 0 && ix < width)
                    value += input[((n * ic + in) * height + iy) * width + ix] *
                             weights[((out * ic + in) * kernel + ky) * kernel + kx];
                }
            output[(n * oc + out) * spatial + y * width + x] = std::max(value, 0.F);
          }
    return output;
  };
  auto reference = [&](const std::vector<float>& input) {
    const auto squeezed =
        convolution_relu(input, input_channels, squeeze_channels, 1, squeeze_weights, squeeze_bias);
    const auto expanded1 = convolution_relu(squeezed, squeeze_channels, expand1_channels, 1,
                                            expand1_weights, expand1_bias);
    const auto expanded3 = convolution_relu(squeezed, squeeze_channels, expand3_channels, 3,
                                            expand3_weights, expand3_bias);
    std::vector<float> output;
    for (int n = 0; n < batch; ++n) {
      output.insert(output.end(), expanded1.begin() + n * expand1_channels * spatial,
                    expanded1.begin() + (n + 1) * expand1_channels * spatial);
      output.insert(output.end(), expanded3.begin() + n * expand3_channels * spatial,
                    expanded3.begin() + (n + 1) * expand3_channels * spatial);
    }
    return output;
  };
  const auto first = vkai::test::TestValues(batch * input_channels * spatial);
  const auto changed = vkai::test::TestValues(batch * input_channels * spatial, .25F, 7);
  const auto expected_first = reference(first), expected_changed = reference(changed);
  ASSERT_NE(expected_first, expected_changed);
  const auto path = model.Save();
  for (size_t submission_limit : {1U, 99U}) {
    SCOPED_TRACE(submission_limit);
    vkai::Engine engine(path, submission_limit);
    for (int iteration = 0; iteration < 3; ++iteration) {
      SCOPED_TRACE(iteration);
      engine.GetInputTensor()->PopulateTensor(iteration == 2 ? changed : first);
      engine.ExecuteGraph();
      vkai::test::ExpectNear(engine.GetOutputTensor()->Data(),
                             iteration == 2 ? expected_changed : expected_first);
    }
  }
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
TEST(ONNXExecutionTest, LegacySoftmaxFlattensDimensionsStartingAtAxis) {
  for (int version : {1, 11, 12}) {
    SCOPED_TRACE(version);
    Model model({2, 2, 2, 2}, {2, 2, 2, 2});
    model.Proto().mutable_opset_import(0)->set_version(version);
    auto* node = model.Node("Softmax", {"input"});
    Model::Int(node, "axis", 1);
    model.Check(std::vector<float>(16, 0.F), std::vector<float>(16, .125F));
  }
}

TEST(ONNXExecutionTest, SoftmaxDefaultAxisDependsOnOpset) {
  for (int version : {11, 12, 13, 21}) {
    SCOPED_TRACE(version);
    Model model({2, 2, 2, 4}, {2, 2, 2, 4});
    model.Proto().mutable_opset_import(0)->set_version(version);
    model.Node("Softmax", {"input"});
    model.Check(std::vector<float>(32, 0.F),
                std::vector<float>(32, version < 13 ? 1.F / 16 : 1.F / 4));
  }
}

TEST(ONNXExecutionTest, SoftmaxReducesSpecifiedAxisAcrossRanksAndOpsets) {
  const std::vector<std::vector<int64_t>> shapes{
      {5}, {2, 3}, {2, 3, 4}, {2, 3, 2, 5}, {1, 2, 3, 2, 1}};
  for (int version : {11, 13}) {
    for (const auto& shape : shapes) {
      const int rank = static_cast<int>(shape.size());
      for (int axis = 0; axis < rank; ++axis) {
        SCOPED_TRACE(::testing::Message()
                     << "opset=" << version << " rank=" << rank << " axis=" << axis);
        int count = 1, trailing = 1;
        for (int i = 0; i < rank; ++i) {
          count *= shape[i];
          if (i > axis) trailing *= shape[i];
        }
        const auto input = vkai::test::TestValues(count, 100.F);
        std::vector<float> expected(count);
        // Group coordinates directly by ONNX's reduction rule. Use double
        // precision and recompute the denominator independently for each output.
        for (int i = 0; i < count; ++i) {
          const int reduced = version < 13 ? shape[axis] * trailing : shape[axis];
          const int stride = version < 13 ? 1 : trailing;
          const int base = i - ((i / stride) % reduced) * stride;
          double maximum = -std::numeric_limits<double>::infinity();
          for (int j = 0; j < reduced; ++j)
            maximum = std::max(maximum, double(input[base + j * stride]));
          double sum = 0;
          for (int j = 0; j < reduced; ++j)
            sum += std::exp(double(input[base + j * stride]) - maximum);
          expected[i] = std::exp(double(input[i]) - maximum) / sum;
        }
        Model model(shape, shape);
        model.Proto().mutable_opset_import(0)->set_version(version);
        auto* node = model.Node("Softmax", {"input"}, "probabilities");
        Model::Int(node, "axis", axis - rank);
        // Exercise the probabilities in a downstream layer and memory planning.
        model.Node("Add", {"probabilities", "input"});
        for (int i = 0; i < count; ++i) expected[i] += input[i];
        model.Check(input, expected);
      }
    }
  }
}

TEST(ONNXExecutionTest, SoftmaxRejectsAxisOutsideTensorRank) {
  for (int version : {11, 13}) {
    for (int axis : {-4, 3}) {
      Model model({2, 3, 4}, {2, 3, 4});
      model.Proto().mutable_opset_import(0)->set_version(version);
      auto* node = model.Node("Softmax", {"input"});
      Model::Int(node, "axis", axis);
      EXPECT_THROW(vkai::Engine(model.Save()), std::runtime_error);
    }
  }
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

TEST(ONNXExecutionTest, SplitSubmissionsPreserveResidualDependenciesAndGpuTimings) {
  Model model({1, 2}, {1, 2});
  model.Weight("scale", {}, {2});
  model.Weight("offset", {1, 2}, {-.25F, .5F});
  model.Node("Mul", {"input", "scale"}, "scaled");
  model.Node("Relu", {"scaled"}, "activated");
  model.Node("Sub", {"input", "activated"}, "residual");
  model.Node("Add", {"residual", "offset"});
  const auto path = model.Save();
  for (size_t limit : {1U, 3U, 99U}) {
    SCOPED_TRACE(limit);
    vkai::Engine engine(path, limit);
    engine.GetInputTensor()->PopulateTensor({-1, 2});
    engine.ExecuteGraph();
    EXPECT_EQ(engine.GetOutputTensor()->Data(), (std::vector<float>{-1.25F, -1.5F}));
    const auto result = engine.BenchmarkGraph(1, 3);
    EXPECT_EQ(result.submissions_per_run, 1 + 3 / limit);
    ASSERT_EQ(result.gpu_ms.size(), 3U);
    for (double ms : result.gpu_ms) {
      EXPECT_TRUE(std::isfinite(ms));
      EXPECT_GT(ms, 0);
    }
    EXPECT_EQ(engine.GetOutputTensor()->Data(), (std::vector<float>{-1.25F, -1.5F}));
    engine.GetInputTensor()->PopulateTensor({.5F, -2});
    engine.ExecuteGraph();
    EXPECT_EQ(engine.GetOutputTensor()->Data(), (std::vector<float>{-.75F, -1.5F}));
  }
  EXPECT_THROW(vkai::Engine(path, 0), std::invalid_argument);
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

TEST(ONNXExecutionTest, HardSigmoidDefaultsAndCustomAttributes) {
  Model defaults({1, 7}, {1, 7});
  defaults.Node("HardSigmoid", {"input"});
  defaults.Check({-10, -2.5F, -1, 0, 1, 2.5F, 10}, {0, 0, .3F, .5F, .7F, 1, 1});
  Model custom({1, 7}, {1, 7});
  auto* node = custom.Node("HardSigmoid", {"input"}, "gate");
  Model::Float(node, "alpha", 1.F / 6);
  Model::Float(node, "beta", .5F);
  custom.Node("Mul", {"input", "gate"});
  custom.Check({-6, -3, -1.5F, 0, 1.5F, 3, 6}, {0, 0, -.375F, 0, 1.125F, 3, 6});
}

TEST(ONNXExecutionTest, HardSigmoidSupportsNegativeAndZeroAlpha) {
  for (float alpha : {-.5F, 0.F}) {
    Model model({1, 3}, {1, 3});
    auto* node = model.Node("HardSigmoid", {"input"});
    Model::Float(node, "alpha", alpha);
    Model::Float(node, "beta", .25F);
    model.Check({-2, 0, 2},
                alpha == 0 ? std::vector<float>{.25F, .25F, .25F} : std::vector<float>{1, .25F, 0});
  }
}

TEST(ONNXExecutionTest, BilinearResizeHalfPixelScalesAndSingletonOutput) {
  Model model({1, 1, 2, 2}, {1, 1, 4, 4});
  model.Weight("scales", {4}, {1, 1, 2, 2});
  auto* node = model.Node("Resize", {"input", "", "scales"});
  Model::String(node, "mode", "linear");
  model.Check({0, 4, 8, 12}, {0, 1, 3, 4, 2, 3, 5, 6, 6, 7, 9, 10, 8, 9, 11, 12});
  Model down({1, 1, 2, 2}, {1, 1, 1, 1});
  down.IntWeight("sizes", {4}, {1, 1, 1, 1});
  node = down.Node("Resize", {"input", "", "", "sizes"});
  Model::String(node, "mode", "linear");
  down.Check({0, 4, 8, 12}, {6});
}

TEST(ONNXExecutionTest, BilinearResizeSizesAcrossBatchesChannelsAndNonSquareShapes) {
  Model model({2, 2, 1, 2}, {2, 2, 3, 3});
  model.IntWeight("sizes", {4}, {2, 2, 3, 3});
  auto* node = model.Node("Resize", {"input", "", "", "sizes"});
  Model::String(node, "mode", "linear");
  Model::String(node, "coordinate_transformation_mode", "half_pixel");
  std::vector<float> expected;
  for (int plane = 0; plane < 4; ++plane)
    for (int y = 0; y < 3; ++y)
      for (float offset : {0.F, 1.F, 2.F}) expected.push_back(plane * 10 + offset);
  model.Check({0, 2, 10, 12, 20, 22, 30, 32}, expected);
}

TEST(ONNXExecutionTest, ResizeRejectsUnsupportedModesCoordinatesAndChannelResize) {
  for (const std::string coordinate : {"align_corners", "asymmetric", "tf_crop_and_resize"}) {
    Model model({1, 1, 2, 2}, {1, 1, 4, 4});
    model.IntWeight("sizes", {4}, {1, 1, 4, 4});
    auto* node = model.Node("Resize", {"input", "", "", "sizes"});
    Model::String(node, "mode", "linear");
    Model::String(node, "coordinate_transformation_mode", coordinate);
    EXPECT_THROW(vkai::Engine(model.Save()), std::runtime_error);
  }
  Model cubic({1, 1, 2, 2}, {1, 1, 4, 4});
  cubic.IntWeight("sizes", {4}, {1, 1, 4, 4});
  auto* node = cubic.Node("Resize", {"input", "", "", "sizes"});
  Model::String(node, "mode", "cubic");
  EXPECT_THROW(vkai::Engine(cubic.Save()), std::runtime_error);
  Model channels({1, 1, 2, 2}, {1, 2, 2, 2});
  channels.IntWeight("sizes", {4}, {1, 2, 2, 2});
  node = channels.Node("Resize", {"input", "", "", "sizes"});
  Model::String(node, "mode", "linear");
  EXPECT_THROW(vkai::Engine(channels.Save()), std::runtime_error);
}
}  // namespace
