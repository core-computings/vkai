#include <numeric>

#include "TestUtils.h"
#include "layers/ChannelConcat.h"

namespace vkai::test {
using ChannelConcatTest = GpuTest;

void CheckConcat(core::vulkan::VulkanContext& context, const std::vector<int>& channels,
                 int batch) {
  constexpr int h = 3, w = 5, spatial = h * w;
  const int total = std::accumulate(channels.begin(), channels.end(), 0);
  std::vector<std::vector<float>> values;
  std::vector<std::unique_ptr<core::vulkan::VulkanBuffer>> buffers;
  std::vector<const core::vulkan::VulkanBuffer*> pointers;
  for (size_t i = 0; i < channels.size(); ++i) {
    values.push_back(TestValues(batch * channels[i] * spatial, .25F, i * 3));
    buffers.push_back(Upload(context, values.back()));
    pointers.push_back(buffers.back().get());
  }
  std::vector<float> expected;
  for (int n = 0; n < batch; ++n)
    for (size_t i = 0; i < channels.size(); ++i) {
      const auto begin = values[i].begin() + n * channels[i] * spatial;
      expected.insert(expected.end(), begin, begin + channels[i] * spatial);
    }
  ChannelConcat layer(&context, channels, h, w, batch);
  EXPECT_EQ(layer.OutputChannels(), total);
  layer.Init();
  auto actual = RunLayer(context, values.front(), expected.size(),
                         [&](auto cmd, auto&, auto& out) { layer.Execute(cmd, pointers, out); });
  EXPECT_EQ(actual, expected);
}

TEST_F(ChannelConcatTest, ConcatenatesFiveInputsWithUnequalChannels) {
  CheckConcat(context, {3, 1, 2, 4, 1}, 1);
}
TEST_F(ChannelConcatTest, ConcatenatesTwoInputsAcrossBatches) { CheckConcat(context, {2, 3}, 2); }
}  // namespace vkai::test
