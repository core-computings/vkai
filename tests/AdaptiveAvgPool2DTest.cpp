#include "TestUtils.h"
#include "layers/AdaptiveAvgPool2D.h"

namespace vkai::test {
using AdaptiveAvgPool2DTest = GpuTest;

void CheckAverage(core::vulkan::VulkanContext& context, int n, int c, int h, int w, int oh,
                  int ow) {
  const auto input = TestValues(n * c * h * w);
  std::vector<float> expected(n * c * oh * ow);
  // Independently average the continuous grid cells covered by each output bin.
  for (int plane = 0; plane < n * c; ++plane)
    for (int y = 0; y < oh; ++y)
      for (int x = 0; x < ow; ++x) {
        const int top = y * h / oh, bottom = ((y + 1) * h + oh - 1) / oh;
        const int left = x * w / ow, right = ((x + 1) * w + ow - 1) / ow;
        double total = 0;
        for (int iy = top; iy < bottom; ++iy)
          for (int ix = left; ix < right; ++ix) total += input[(plane * h + iy) * w + ix];
        expected[(plane * oh + y) * ow + x] = total / ((bottom - top) * (right - left));
      }
  AdaptiveAvgPool2D layer(&context, c, h, w, oh, ow, n);
  EXPECT_EQ(layer.OutputHeight(), oh);
  EXPECT_EQ(layer.OutputWidth(), ow);
  layer.Init();
  ExpectNear(RunLayer(context, input, expected.size(),
                      [&](auto cmd, auto& a, auto& b) { layer.Execute(cmd, a, b); }),
             expected);
}

TEST_F(AdaptiveAvgPool2DTest, SupportsGlobalAndMultipleOutputSizes) {
  for (int size : {1, 2, 3, 6}) CheckAverage(context, 1, 3, 6, 8, size, size);
}
TEST_F(AdaptiveAvgPool2DTest, HandlesBatchedNonDivisibleRegions) {
  CheckAverage(context, 2, 3, 7, 11, 3, 4);
}
}  // namespace vkai::test
