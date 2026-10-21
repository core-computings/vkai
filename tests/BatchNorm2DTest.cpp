#include <limits>
#include <stdexcept>

#include "TestUtils.h"
#include "layers/BatchNorm2D.h"

namespace vkai::test {
using BatchNorm2DTest = GpuTest;

void CheckNormalization(core::vulkan::VulkanContext& context, int batch, int spatial, float epsilon,
                        bool affine, bool zero_variance = false) {
  constexpr int channels = 3;
  auto input = TestValues(batch * channels * spatial);
  std::vector<float> mean{-.5F, 0, 1}, variance{.25F, 2, zero_variance ? 0.F : .75F};
  std::vector<float> weight = affine ? std::vector<float>{-2, 0, 1.5F} : std::vector<float>{};
  std::vector<float> bias = affine ? std::vector<float>{.25F, -1, .5F} : std::vector<float>{};
  std::vector<float> expected(input.size());
  for (size_t i = 0; i < input.size(); ++i) {
    const int channel = (i / spatial) % channels;
    const double normalized =
        (input[i] - mean[channel]) / std::sqrt(double(variance[channel]) + epsilon);
    expected[i] = normalized * (affine ? weight[channel] : 1) + (affine ? bias[channel] : 0);
  }
  BatchNorm2D layer(&context, channels, spatial, batch, epsilon);
  layer.MapParameters(mean, variance, weight, bias);
  layer.Init();
  ExpectNear(RunLayer(context, input, expected.size(),
                      [&](auto cmd, auto& a, auto& b) { layer.Execute(cmd, a, b); }),
             expected);
}

TEST_F(BatchNorm2DTest, PreservesNegativeValuesBeforeActivation) {
  CheckNormalization(context, 1, 257, 1e-5F, true);
}
TEST_F(BatchNorm2DTest, AppliesChannelParametersAcrossBatches) {
  CheckNormalization(context, 2, 17, 1e-5F, true);
}
TEST_F(BatchNorm2DTest, BatchedNonDefaultEpsilonAndZeroVariance) {
  CheckNormalization(context, 2, 259, .01F, true, true);
}
TEST_F(BatchNorm2DTest, SupportsNonAffineNormalization) {
  CheckNormalization(context, 2, 13, .001F, false);
}
TEST_F(BatchNorm2DTest, RejectsInvalidParameters) {
  BatchNorm2D layer(&context, 1, 1, 1, 1e-5F);
  EXPECT_THROW(layer.MapParameters({0}, {-1}), std::invalid_argument);
  EXPECT_THROW(layer.MapParameters({0}, {1}, {1, 2}), std::invalid_argument);
  EXPECT_THROW(BatchNorm2D(&context, 1, 1, 1, -1.F), std::invalid_argument);
  EXPECT_THROW(BatchNorm2D(&context, std::numeric_limits<int>::max(), 2, 1, 1e-5F),
               std::invalid_argument);
}
}  // namespace vkai::test
