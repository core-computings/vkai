#include <algorithm>

#include "TestUtils.h"
#include "layers/Relu.h"

namespace vkai::test {
using ReluTest = GpuTest;
TEST_F(ReluTest, HandlesSignsZeroAndPartialWorkgroups) {
  auto input = TestValues(513);
  input[0] = 0;
  input[1] = -0.F;
  auto expected = input;
  for (auto& value : expected) value = std::max(value, 0.F);
  Relu layer(&context, input.size());
  layer.Init();
  EXPECT_EQ(RunLayer(context, input, expected.size(),
                     [&](auto cmd, auto& a, auto& b) { layer.Execute(cmd, a, b); }),
            expected);
}
}  // namespace vkai::test
