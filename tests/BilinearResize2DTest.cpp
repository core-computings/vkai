#include <algorithm>

#include "TestUtils.h"
#include "layers/BilinearResize2D.h"

namespace vkai::test {
using BilinearResize2DTest = GpuTest;

void CheckResize(core::vulkan::VulkanContext& context, int n, int c, int h, int w, int oh, int ow) {
  const auto input = TestValues(n * c * h * w);
  std::vector<float> expected(n * c * oh * ow);
  for (int plane = 0; plane < n * c; ++plane)
    for (int y = 0; y < oh; ++y)
      for (int x = 0; x < ow; ++x) {
        const double sy = std::clamp((y + .5) * h / oh - .5, 0., double(h - 1));
        const double sx = std::clamp((x + .5) * w / ow - .5, 0., double(w - 1));
        const int top = std::floor(sy), left = std::floor(sx);
        const int bottom = std::min(top + 1, h - 1), right = std::min(left + 1, w - 1);
        const double dy = sy - top, dx = sx - left;
        auto at = [&](int iy, int ix) { return input[(plane * h + iy) * w + ix]; };
        expected[(plane * oh + y) * ow + x] =
            at(top, left) * (1 - dx) * (1 - dy) + at(top, right) * dx * (1 - dy) +
            at(bottom, left) * (1 - dx) * dy + at(bottom, right) * dx * dy;
      }
  BilinearResize2D layer(&context, c, h, w, oh, ow, n);
  EXPECT_EQ(layer.OutputHeight(), oh);
  EXPECT_EQ(layer.OutputWidth(), ow);
  layer.Init();
  ExpectNear(RunLayer(context, input, expected.size(),
                      [&](auto cmd, auto& a, auto& b) { layer.Execute(cmd, a, b); }),
             expected, 2e-5F, 2e-5F);
}

TEST_F(BilinearResize2DTest, UpsamplesSingletonAndMultipleInputSizes) {
  for (int size : {1, 2, 3, 6}) CheckResize(context, 1, 3, size, size, 7, 9);
}
TEST_F(BilinearResize2DTest, DownsamplesWithHalfPixelCoordinates) {
  CheckResize(context, 1, 3, 11, 13, 4, 5);
}
TEST_F(BilinearResize2DTest, ResizesBatchedNonSquareInputs) {
  CheckResize(context, 2, 3, 3, 5, 7, 11);
}
}  // namespace vkai::test
