#include <limits>

#include "TestUtils.h"
#include "layers/HardSigmoid.h"

namespace vkai::test {
using HardSigmoidTest = GpuTest;

TEST_F(HardSigmoidTest, SaturationBoundariesAndPartialWorkgroup) {
  std::vector<float> input(517);
  for (size_t i = 0; i < input.size(); ++i) input[i] = (int(i) - 258) * .125F;
  input[0] = -3;
  input[1] = 3;
  input[2] = -std::numeric_limits<float>::infinity();
  input[3] = std::numeric_limits<float>::infinity();
  HardSigmoid layer(&context, input.size(), 1.F / 6, .5F);
  layer.Init();
  const auto actual = RunLayer(context, input, input.size(),
                               [&](auto cmd, auto& a, auto& b) { layer.Execute(cmd, a, b); });
  std::vector<float> expected;
  for (float value : input) expected.push_back(std::clamp(value / 6 + .5F, 0.F, 1.F));
  ExpectNear(actual, expected, 1e-6F, 1e-6F);
}

TEST_F(HardSigmoidTest, RejectsInvalidSizeAndNonFiniteParameters) {
  EXPECT_THROW(HardSigmoid(&context, 0, .2F, .5F), std::invalid_argument);
  EXPECT_THROW(HardSigmoid(&context, 1, std::numeric_limits<float>::quiet_NaN(), .5F),
               std::invalid_argument);
  EXPECT_THROW(HardSigmoid(&context, 1, .2F, std::numeric_limits<float>::infinity()),
               std::invalid_argument);
}
}  // namespace vkai::test
