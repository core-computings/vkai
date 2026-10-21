#include <gtest/gtest.h>

#include "ModelTestUtils.h"

TEST(ModelValidationTest, ClassificationTop5MatchesPyTorchOrder) {
  const std::vector<float> scores{.1F, .4F, .2F, .8F, .5F, .9F};
  EXPECT_EQ(model_test::CompareTop5(scores, scores), (std::vector<size_t>{5, 3, 4, 1, 2}));
}

TEST(ModelValidationTest, RejectsChangedTop5WithinElementwiseTolerance) {
  const std::vector<float> actual{.1F, .2F, .3F, .4F, .5F, .5001F};
  const std::vector<float> reference{.1F, .2F, .3F, .4F, .5001F, .5F};
  EXPECT_NO_THROW(model_test::CompareOutput(actual, reference));
  EXPECT_THROW(model_test::CompareTop5(actual, reference), std::runtime_error);
}

TEST(ModelValidationTest, RejectsIncompleteClassificationOutput) {
  EXPECT_THROW(model_test::Top5({.1F, .2F, .3F, .4F}), std::runtime_error);
}
