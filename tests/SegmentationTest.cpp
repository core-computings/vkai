#include <gtest/gtest.h>

#include <limits>

#include "utils/Segmentation.h"

TEST(SegmentationTest, ArgmaxUsesChannelPlanesNegativeLogitsAndFirstTie) {
  EXPECT_EQ(vkai::ArgmaxSegmentation({-2, 1, -5, 0, -1, 1, -6, 3, -3, 0, -4, 2}, 3, 2, 2),
            (std::vector<uint8_t>{1, 0, 2, 1}));
}

TEST(SegmentationTest, RejectsInvalidShapeClassCountAndNonFiniteLogits) {
  EXPECT_THROW(vkai::ArgmaxSegmentation({}, 0, 1, 1), std::invalid_argument);
  EXPECT_THROW(vkai::ArgmaxSegmentation({}, 257, 1, 1), std::invalid_argument);
  EXPECT_THROW(vkai::ArgmaxSegmentation({}, 1, -1, 1), std::invalid_argument);
  EXPECT_THROW(vkai::ArgmaxSegmentation({1}, 2, 1, 1), std::invalid_argument);
  EXPECT_THROW(vkai::ArgmaxSegmentation({1, std::numeric_limits<float>::quiet_NaN()}, 2, 1, 1),
               std::invalid_argument);
  EXPECT_THROW(vkai::ArgmaxSegmentation({std::numeric_limits<float>::infinity()}, 1, 1, 1),
               std::invalid_argument);
}

TEST(SegmentationTest, IoUIncludesMissingAndExtraClassesButSkipsAbsentClasses) {
  const auto comparison = vkai::CompareSegmentation({0, 1, 1, 2}, {0, 1, 2, 2}, 4);
  EXPECT_DOUBLE_EQ(comparison.pixel_agreement, .75);
  EXPECT_DOUBLE_EQ(comparison.mean_iou, 2. / 3);
  EXPECT_EQ(comparison.class_pixels, (std::vector<size_t>{1, 2, 1, 0}));
  const auto missing = vkai::CompareSegmentation({0, 0}, {0, 1}, 3);
  EXPECT_DOUBLE_EQ(missing.mean_iou, .25);
  const auto extra = vkai::CompareSegmentation({0, 2}, {0, 0}, 3);
  EXPECT_DOUBLE_EQ(extra.mean_iou, .25);
  EXPECT_DOUBLE_EQ(vkai::CompareSegmentation({1, 1}, {1, 1}, 3).mean_iou, 1);
}

TEST(SegmentationTest, RejectsInvalidMasks) {
  EXPECT_THROW(vkai::CompareSegmentation({}, {}, 2), std::invalid_argument);
  EXPECT_THROW(vkai::CompareSegmentation({0}, {0, 1}, 2), std::invalid_argument);
  EXPECT_THROW(vkai::CompareSegmentation({2}, {0}, 2), std::invalid_argument);
  EXPECT_THROW(vkai::CompareSegmentation({0}, {2}, 2), std::invalid_argument);
  EXPECT_THROW(vkai::CompareSegmentation({0}, {0}, 0), std::invalid_argument);
}
