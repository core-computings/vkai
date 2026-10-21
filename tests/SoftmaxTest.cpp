#include <algorithm>
#include <limits>

#include "TestUtils.h"
#include "layers/Softmax.h"

namespace vkai::test {
using SoftmaxTest = GpuTest;

void CheckSoftmax(core::vulkan::VulkanContext& context, bool channel_mode, bool extreme) {
  const int n = 2, c = 7, h = channel_mode ? 3 : 1, w = channel_mode ? 5 : 1;
  const int spatial = h * w;
  auto input = TestValues(n * c * spatial, extreme ? 100.F : .5F);
  std::vector<float> expected(input.size());
  for (int batch = 0; batch < n; ++batch)
    for (int pos = 0; pos < spatial; ++pos) {
      double maximum = -std::numeric_limits<double>::infinity();
      for (int channel = 0; channel < c; ++channel)
        maximum = std::max(maximum, double(input[(batch * c + channel) * spatial + pos]));
      double sum = 0;
      for (int channel = 0; channel < c; ++channel)
        sum += std::exp(input[(batch * c + channel) * spatial + pos] - maximum);
      for (int channel = 0; channel < c; ++channel) {
        const auto index = (batch * c + channel) * spatial + pos;
        expected[index] = std::exp(input[index] - maximum) / sum;
      }
    }
  auto layer = channel_mode
                   ? std::make_unique<Softmax>(&context, c, h, w, n, SoftmaxMode::kChannelNCHW)
                   : std::make_unique<Softmax>(&context, c, n);
  layer->Init();
  auto actual = RunLayer(context, input, expected.size(),
                         [&](auto cmd, auto& a, auto& b) { layer->Execute(cmd, a, b); });
  ExpectNear(actual, expected, 2e-6F, 2e-6F);
  for (int batch = 0; batch < n; ++batch)
    for (int pos = 0; pos < spatial; ++pos) {
      float sum = 0;
      for (int channel = 0; channel < c; ++channel)
        sum += actual[(batch * c + channel) * spatial + pos];
      EXPECT_NEAR(sum, 1.F, 2e-6F);
    }
}

TEST_F(SoftmaxTest, NormalizesContiguousBatchVectors) { CheckSoftmax(context, false, false); }
TEST_F(SoftmaxTest, NormalizesChannelsAtEachSpatialPosition) { CheckSoftmax(context, true, false); }
TEST_F(SoftmaxTest, HandlesBatchedExtremeChannelValues) { CheckSoftmax(context, true, true); }
}  // namespace vkai::test
