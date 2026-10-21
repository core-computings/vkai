#include <algorithm>
#include <limits>

#include "TestUtils.h"
#include "layers/MaxPool2D.h"

namespace vkai::test {
using MaxPool2DTest = GpuTest;

void CheckMaxPool(core::vulkan::VulkanContext& context, int batch, int h, int w, int kh, int kw,
                  int sh, int sw, int ph, int pw, bool negative) {
  constexpr int channels = 3;
  auto input = TestValues(batch * channels * h * w);
  if (negative)
    for (auto& value : input) value = -std::abs(value) - .25F;
  const int oh = (h + 2 * ph - kh) / sh + 1, ow = (w + 2 * pw - kw) / sw + 1;
  std::vector<float> expected(batch * channels * oh * ow, -std::numeric_limits<float>::infinity());
  for (int plane = 0; plane < batch * channels; ++plane)
    for (int y = 0; y < oh; ++y)
      for (int x = 0; x < ow; ++x)
        for (int iy = y * sh - ph; iy < y * sh - ph + kh; ++iy)
          for (int ix = x * sw - pw; ix < x * sw - pw + kw; ++ix)
            if (iy >= 0 && iy < h && ix >= 0 && ix < w) {
              auto& value = expected[(plane * oh + y) * ow + x];
              value = std::max(value, input[(plane * h + iy) * w + ix]);
            }
  MaxPool2D layer(&context, channels, h, w, kh, kw, sh, sw, batch, ph, pw);
  EXPECT_EQ(layer.OutputHeight(), oh);
  EXPECT_EQ(layer.OutputWidth(), ow);
  layer.Init();
  EXPECT_EQ(RunLayer(context, input, expected.size(),
                     [&](auto cmd, auto& a, auto& b) { layer.Execute(cmd, a, b); }),
            expected);
}

TEST_F(MaxPool2DTest, PoolsMultipleBatchesAndChannels) {
  CheckMaxPool(context, 2, 8, 10, 2, 2, 2, 2, 0, 0, false);
}
TEST_F(MaxPool2DTest, PaddingPreservesNegativeValues) {
  CheckMaxPool(context, 2, 5, 7, 3, 3, 2, 2, 1, 1, true);
}
TEST_F(MaxPool2DTest, PoolsNonSquareWindowsAndStrides) {
  CheckMaxPool(context, 2, 7, 11, 2, 3, 1, 2, 0, 1, false);
}
}  // namespace vkai::test
