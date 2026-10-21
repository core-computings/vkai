#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <vector>

#include "VulkanCommandBuffer.h"
#include "layers/ConvTranspose2D.h"
#include "utils/BufferUtils.h"
#include "utils/Synchronization.h"

namespace {

// Independent CPU scatter reference, including overlapping kernels and IOHW weights.
void CheckTranspose(int n, int ic, int oc, int h, int w, int kh, int kw, int sh, int sw, int ph,
                    int pw, int dh, int dw, int oph, int opw, bool bias) {
  const int oh = (h - 1) * sh - 2 * ph + dh * (kh - 1) + oph + 1;
  const int ow = (w - 1) * sw - 2 * pw + dw * (kw - 1) + opw + 1;
  std::vector<float> x(n * ic * h * w), weights(ic * oc * kh * kw), b(bias ? oc : 0);
  for (size_t i = 0; i < x.size(); ++i) x[i] = (static_cast<int>(i % 17) - 8) * .125F;
  for (size_t i = 0; i < weights.size(); ++i) weights[i] = (static_cast<int>(i % 11) - 5) * .1F;
  for (int i = 0; i < static_cast<int>(b.size()); ++i) b[i] = i * .3F;
  std::vector<float> expected(n * oc * oh * ow);
  for (int batch = 0; batch < n; ++batch) {
    for (int c = 0; c < oc; ++c) {
      for (int y = 0; y < oh; ++y)
        for (int z = 0; z < ow; ++z)
          expected[((batch * oc + c) * oh + y) * ow + z] = bias ? b[c] : 0;
    }
    for (int in = 0; in < ic; ++in)
      for (int y = 0; y < h; ++y)
        for (int z = 0; z < w; ++z)
          for (int out = 0; out < oc; ++out)
            for (int ky = 0; ky < kh; ++ky)
              for (int kx = 0; kx < kw; ++kx) {
                const int oy = y * sh - ph + ky * dh, ox = z * sw - pw + kx * dw;
                if (oy >= 0 && oy < oh && ox >= 0 && ox < ow)
                  expected[((batch * oc + out) * oh + oy) * ow + ox] +=
                      x[((batch * ic + in) * h + y) * w + z] *
                      weights[((in * oc + out) * kh + ky) * kw + kx];
              }
  }
  core::vulkan::VulkanContext context(false, core::vulkan::QueueFamilyType::Compute,
                                      VK_NULL_HANDLE);
  context.Init();
  core::vulkan::VulkanBuffer input(&context, x.size() * sizeof(float),
                                   VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, vkai::kHostVisibleMemory);
  core::vulkan::VulkanBuffer output(&context, expected.size() * sizeof(float),
                                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, vkai::kHostVisibleMemory);
  input.MapData([&](void* data) { std::memcpy(data, x.data(), x.size() * sizeof(float)); });
  vkai::ConvTranspose2D layer(&context, ic, oc, h, w, kh, kw, sh, sw, ph, pw, n, dh, dw, oph, opw);
  EXPECT_EQ(layer.OutputHeight(), oh);
  EXPECT_EQ(layer.OutputWidth(), ow);
  layer.MapWeights(weights, b);
  layer.Init();
  auto command = core::vulkan::VulkanCommandBuffer::BeginOneTimeCommands(&context);
  layer.Execute(command.buffer(), input, output);
  vkai::Synchronization::InsertHostReadBarrier(command.buffer(), output);
  command.EndOneTimeCommands();
  std::vector<float> actual(expected.size());
  output.MapData(
      [&](void* data) { std::memcpy(actual.data(), data, actual.size() * sizeof(float)); });
  for (size_t i = 0; i < actual.size(); ++i) EXPECT_NEAR(actual[i], expected[i], 1e-5F) << i;
}

TEST(ConvTranspose2DTest, UNetUpsamplingWithBiasAndMultipleBatches) {
  CheckTranspose(2, 3, 2, 3, 5, 2, 2, 2, 2, 0, 0, 1, 1, 0, 0, true);
}
TEST(ConvTranspose2DTest, OverlappingNonSquareKernelsWithPaddingAndOutputPadding) {
  CheckTranspose(2, 2, 3, 3, 4, 3, 2, 2, 3, 1, 0, 1, 1, 1, 2, false);
}
TEST(ConvTranspose2DTest, DilatedKernelsAndStrideOne) {
  CheckTranspose(1, 2, 3, 4, 3, 3, 2, 1, 1, 1, 1, 2, 2, 0, 0, true);
}
TEST(ConvTranspose2DTest, RejectsInvalidWeightsAndOutputPadding) {
  core::vulkan::VulkanContext context(false, core::vulkan::QueueFamilyType::Compute,
                                      VK_NULL_HANDLE);
  context.Init();
  EXPECT_THROW((vkai::ConvTranspose2D(&context, 0, 3, 4, 4, 2, 2, 2, 2, 0, 0)),
               std::invalid_argument);
  EXPECT_THROW((vkai::ConvTranspose2D(&context, 2, 3, 4, 4, 2, 2, 0, 2, 0, 0)),
               std::invalid_argument);
  EXPECT_THROW((vkai::ConvTranspose2D(&context, 2, 3, 1, 1, 2, 2, 2, 2, 3, 3)),
               std::invalid_argument);
  EXPECT_THROW((vkai::ConvTranspose2D(&context, 2, 3, 4, 4, 2, 2, 2, 2, -1, 0)),
               std::invalid_argument);
  vkai::ConvTranspose2D layer(&context, 2, 3, 4, 4, 2, 2, 2, 2, 0, 0);
  EXPECT_THROW(layer.MapWeights({1}), std::invalid_argument);
  EXPECT_THROW(layer.MapWeights(std::vector<float>(24), {1}), std::invalid_argument);
  EXPECT_THROW(layer.Init(), std::logic_error);
  EXPECT_THROW((vkai::ConvTranspose2D(&context, 2, 3, 4, 4, 2, 2, 2, 2, 0, 0, 1, 1, 1, 2, 0)),
               std::invalid_argument);
}
}  // namespace
