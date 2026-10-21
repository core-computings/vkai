#include "TestUtils.h"
#include "layers/Conv2D.h"

namespace vkai::test {
using Conv2DTest = GpuTest;

void CheckConvolution(core::vulkan::VulkanContext& context, int batch, int ic, int oc, int h, int w,
                      int kh, int kw, int pad, int dilation, bool bias) {
  const int oh = h + 2 * pad - dilation * (kh - 1), ow = w + 2 * pad - dilation * (kw - 1);
  const auto input = TestValues(batch * ic * h * w);
  const auto weights = TestValues(oc * ic * kh * kw, .0625F, 4);
  const auto offsets = bias ? TestValues(oc, .5F, 9) : std::vector<float>{};
  std::vector<float> expected(batch * oc * oh * ow);
  for (int n = 0; n < batch; ++n)
    for (int out = 0; out < oc; ++out)
      for (int y = 0; y < oh; ++y)
        for (int x = 0; x < ow; ++x) {
          double value = bias ? offsets[out] : 0;
          for (int in = 0; in < ic; ++in)
            for (int ky = 0; ky < kh; ++ky)
              for (int kx = 0; kx < kw; ++kx) {
                const int iy = y - pad + ky * dilation, ix = x - pad + kx * dilation;
                if (iy >= 0 && iy < h && ix >= 0 && ix < w)
                  value += double(input[((n * ic + in) * h + iy) * w + ix]) *
                           weights[((out * ic + in) * kh + ky) * kw + kx];
              }
          expected[((n * oc + out) * oh + y) * ow + x] = value;
        }
  Conv2D layer(&context, ic, oc, h, w, kh, kw, 1, 1, pad, pad, PaddingType::Zero, batch, dilation,
               dilation);
  layer.MapWeights(weights, offsets);
  EXPECT_EQ(layer.OutputHeight(), oh);
  EXPECT_EQ(layer.OutputWidth(), ow);
  layer.Init();
  ExpectNear(RunLayer(context, input, expected.size(),
                      [&](auto cmd, auto& a, auto& b) { layer.Execute(cmd, a, b); }),
             expected);
}

TEST_F(Conv2DTest, ZeroPaddingWithMultipleChannelsAndBatches) {
  CheckConvolution(context, 2, 2, 3, 7, 9, 3, 3, 1, 1, false);
}
TEST_F(Conv2DTest, SupportsReflectPadding) {
  const std::vector<float> input{1, 2, 3, 4, 5, 6, 7, 8, 9};
  Conv2D layer(&context, 1, 1, 3, 3, 3, 3, 1, 1, 1, 1, PaddingType::Reflect);
  layer.MapWeights(std::vector<float>(9, 1));
  layer.Init();
  EXPECT_EQ(
      RunLayer(context, input, 9, [&](auto cmd, auto& a, auto& b) { layer.Execute(cmd, a, b); }),
      (std::vector<float>{33, 36, 39, 42, 45, 48, 51, 54, 57}));
}
TEST_F(Conv2DTest, MatchesDilatedConvolutionWithCpuReference) {
  CheckConvolution(context, 2, 3, 4, 7, 9, 3, 3, 2, 2, false);
}
TEST_F(Conv2DTest, AppliesBiasInPointwiseConvolution) {
  CheckConvolution(context, 2, 5, 3, 7, 11, 1, 1, 0, 1, true);
}
}  // namespace vkai::test

namespace {
void CheckRepeatedReflection(int height, int width) {
  constexpr int n = 2, ic = 2, oc = 3, kh = 3, kw = 2, ph = 4, pw = 3;
  constexpr int sh = 2, sw = 1, dh = 2, dw = 1;
  const int oh = (height + 2 * ph - dh * (kh - 1) - 1) / sh + 1;
  const int ow = (width + 2 * pw - dw * (kw - 1) - 1) / sw + 1;
  std::vector<float> x(n * ic * height * width), weights(oc * ic * kh * kw);
  for (size_t i = 0; i < x.size(); ++i) x[i] = static_cast<int>(i % 13) - 6;
  for (size_t i = 0; i < weights.size(); ++i) weights[i] = (static_cast<int>(i % 7) - 3) * .25F;
  auto reflect = [](int index, int size) {
    if (size == 1) return 0;
    while (index < 0 || index >= size) index = index < 0 ? -index : 2 * size - index - 2;
    return index;
  };
  std::vector<float> expected(n * oc * oh * ow, .5F);
  for (int batch = 0; batch < n; ++batch)
    for (int out = 0; out < oc; ++out)
      for (int y = 0; y < oh; ++y)
        for (int z = 0; z < ow; ++z)
          for (int in = 0; in < ic; ++in)
            for (int ky = 0; ky < kh; ++ky)
              for (int kx = 0; kx < kw; ++kx) {
                const int iy = reflect(y * sh - ph + ky * dh, height);
                const int ix = reflect(z * sw - pw + kx * dw, width);
                expected[((batch * oc + out) * oh + y) * ow + z] +=
                    x[((batch * ic + in) * height + iy) * width + ix] *
                    weights[((out * ic + in) * kh + ky) * kw + kx];
              }
  core::vulkan::VulkanContext context(false, core::vulkan::QueueFamilyType::Compute,
                                      VK_NULL_HANDLE);
  context.Init();
  core::vulkan::VulkanBuffer input(&context, x.size() * sizeof(float),
                                   VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, vkai::kHostVisibleMemory);
  core::vulkan::VulkanBuffer output(&context, expected.size() * sizeof(float),
                                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, vkai::kHostVisibleMemory);
  input.MapData([&](void* data) { std::memcpy(data, x.data(), x.size() * sizeof(float)); });
  vkai::Conv2D layer(&context, ic, oc, height, width, kh, kw, sh, sw, ph, pw,
                     vkai::PaddingType::Reflect, n, dh, dw);
  layer.MapWeights(weights, std::vector<float>(oc, .5F));
  layer.Init();
  auto command = core::vulkan::VulkanCommandBuffer::BeginOneTimeCommands(&context);
  layer.Execute(command.buffer(), input, output);
  vkai::Synchronization::InsertHostReadBarrier(command.buffer(), output);
  command.EndOneTimeCommands();
  std::vector<float> actual(expected.size());
  output.MapData(
      [&](void* data) { std::memcpy(actual.data(), data, actual.size() * sizeof(float)); });
  for (size_t i = 0; i < actual.size(); ++i) EXPECT_FLOAT_EQ(actual[i], expected[i]) << i;
}
TEST(Conv2DReflectionTest, ReflectsRepeatedPeriodsWithBatchesStridesAndDilation) {
  CheckRepeatedReflection(2, 3);
}
TEST(Conv2DReflectionTest, ReflectsSingletonDimensions) { CheckRepeatedReflection(1, 2); }
}  // namespace
