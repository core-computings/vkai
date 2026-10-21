#include "TestUtils.h"
#include "layers/Linear.h"

namespace vkai::test {
using LinearTest = GpuTest;

void CheckLinear(core::vulkan::VulkanContext& context, int n, int in, int out) {
  const auto input = TestValues(n * in), weights = TestValues(out * in, .0625F, 3);
  const auto bias = TestValues(out, .25F, 7);
  std::vector<float> expected(n * out);
  for (int batch = 0; batch < n; ++batch)
    for (int row = 0; row < out; ++row) {
      double value = bias[row];
      for (int col = 0; col < in; ++col)
        value += double(input[batch * in + col]) * weights[row * in + col];
      expected[batch * out + row] = value;
    }
  Linear layer(&context, in, out, n);
  layer.MapWeights(weights, bias);
  layer.Init();
  ExpectNear(RunLayer(context, input, expected.size(),
                      [&](auto cmd, auto& a, auto& b) { layer.Execute(cmd, a, b); }),
             expected);
}

TEST_F(LinearTest, HandlesNonAlignedInputAndOutputSizes) { CheckLinear(context, 1, 37, 19); }
TEST_F(LinearTest, AppliesWeightsAndBiasAcrossBatches) { CheckLinear(context, 3, 17, 7); }
}  // namespace vkai::test
