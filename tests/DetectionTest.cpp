#include <gtest/gtest.h>

#include <limits>

#include "utils/Detection.h"
namespace {
std::vector<float> Predictions() {
  return {5,  5.1F, 5,  100, 5,   5.1F, 5,   100,  10,  10,  10,  10,
          10, 10,   10, 10,  .9F, .8F,  .1F, .25F, .1F, .1F, .7F, .1F};
}
}  // namespace
TEST(DetectionTest, SuppressesOverlappingSameClassAndPreservesDifferentClasses) {
  auto result = vkai::DecodeYOLODetections(Predictions(), 2, 4);
  ASSERT_EQ(result.size(), 3U);
  EXPECT_EQ(result[0].class_id, 0);
  EXPECT_EQ(result[1].class_id, 1);
  EXPECT_FLOAT_EQ(result[0].confidence, .9F);
  EXPECT_FLOAT_EQ(result[2].confidence, .25F);
  EXPECT_EQ(result[0].box_xyxy, (std::array<float, 4>{0, 0, 10, 10}));
  EXPECT_FLOAT_EQ(vkai::BoxIoU(result[0].box_xyxy, result[1].box_xyxy), 1);
  EXPECT_FLOAT_EQ(vkai::BoxIoU(result[0].box_xyxy, result[2].box_xyxy), 0);
}
TEST(DetectionTest, AppliesThresholdAndMaximumCount) {
  EXPECT_EQ(vkai::DecodeYOLODetections(Predictions(), 2, 4, .8F).size(), 1U);
  EXPECT_EQ(vkai::DecodeYOLODetections(Predictions(), 2, 4, .25F, .45F, 2).size(), 2U);
  EXPECT_TRUE(vkai::DecodeYOLODetections(Predictions(), 2, 4, 1).empty());
}
TEST(DetectionTest, RejectsInvalidShapesParametersAndNonFiniteOutput) {
  EXPECT_THROW(vkai::DecodeYOLODetections({}, 2, 4), std::invalid_argument);
  EXPECT_THROW(vkai::DecodeYOLODetections(Predictions(), 2, 4, -.1F), std::invalid_argument);
  EXPECT_THROW(vkai::DecodeYOLODetections(Predictions(), 2, 4, .25F, 2), std::invalid_argument);
  auto p = Predictions();
  p[0] = std::numeric_limits<float>::quiet_NaN();
  EXPECT_THROW(vkai::DecodeYOLODetections(p, 2, 4), std::invalid_argument);
}
