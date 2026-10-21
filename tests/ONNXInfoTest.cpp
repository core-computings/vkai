#include <onnx/shape_inference/implementation.h>

#include "OnnxTestModel.h"

namespace vkai::test {
TEST(ONNXInfoTest, InfersIntermediateShapesInGeneratedGraph) {
  TestModel model({2, 1, 2, 3}, {2, 3});
  MakeTestGraph(model);
  ASSERT_NO_THROW(onnx::shape_inference::InferShapes(model.Proto()));
  const auto& graph = model.Proto().graph();
  auto shape = [](const onnx::ValueInfoProto& value) {
    std::vector<int64_t> result;
    for (const auto& dimension : value.type().tensor_type().shape().dim())
      result.push_back(dimension.dim_value());
    return result;
  };
  EXPECT_EQ(shape(graph.output(0)), (std::vector<int64_t>{2, 3}));
  bool found_flat = false, found_dense = false;
  for (const auto& value : graph.value_info()) {
    if (value.name() == "flat") {
      found_flat = true;
      EXPECT_EQ(shape(value), (std::vector<int64_t>{2, 6}));
    } else if (value.name() == "dense") {
      found_dense = true;
      EXPECT_EQ(shape(value), (std::vector<int64_t>{2, 3}));
    }
  }
  EXPECT_TRUE(found_flat);
  EXPECT_TRUE(found_dense);
}
}  // namespace vkai::test
