#include "TestUtils.h"
#include "layers/Add.h"

namespace vkai::test {
using AddTest = GpuTest;

void CheckAdd(core::vulkan::VulkanContext& context, AddMode mode) {
  constexpr int n = 2, c = 3, spatial = 259;
  auto input = TestValues(n * c * spatial);
  auto addend = TestValues(mode == AddMode::kChannelBias ? c : input.size(), .25F, 5);
  std::vector<float> expected(input.size());
  for (size_t i = 0; i < input.size(); ++i)
    expected[i] = input[i] + addend[mode == AddMode::kChannelBias ? (i / spatial) % c : i];
  auto second = Upload(context, addend);
  Add layer(&context, c, spatial, n, mode);
  layer.Init();
  auto actual = RunLayer(context, input, expected.size(),
                         [&](auto cmd, auto& x, auto& y) { layer.Execute(cmd, x, *second, y); });
  ExpectNear(actual, expected);
}

TEST_F(AddTest, BroadcastsChannelBiasAcrossBatches) { CheckAdd(context, AddMode::kChannelBias); }
TEST_F(AddTest, AddsElementwiseResiduals) { CheckAdd(context, AddMode::kElementwise); }
}  // namespace vkai::test
