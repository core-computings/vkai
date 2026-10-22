#include <gtest/gtest.h>

#include <limits>

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

TEST(ModelValidationTest, DenoisingMeasuresPSNRGainAgainstCleanImage) {
  const auto quality = model_test::CompareDenoising({0.5F, -0.5F}, {0.25F, -0.25F}, {0, 0});
  EXPECT_NEAR(quality.noisy_psnr_db, 6.020599913, 1e-8);
  EXPECT_NEAR(quality.denoised_psnr_db, 12.041199827, 1e-8);
  const auto perfect = model_test::CompareDenoising({0.5F}, {0}, {0});
  EXPECT_DOUBLE_EQ(perfect.denoised_psnr_db, 120);
}

TEST(ModelValidationTest, RejectsNoOpAndDegradedDenoising) {
  EXPECT_THROW(model_test::CompareDenoising({0.5F}, {0.5F}, {0}), std::runtime_error);
  EXPECT_THROW(model_test::CompareDenoising({0.5F}, {1.F}, {0}), std::runtime_error);
}

TEST(ModelValidationTest, RejectsInvalidDenoisingReferences) {
  EXPECT_THROW(model_test::CompareDenoising({}, {}, {}), std::runtime_error);
  EXPECT_THROW(model_test::CompareDenoising({0.5F}, {0}, {0, 0}), std::runtime_error);
  EXPECT_THROW(model_test::CompareDenoising({0.5F}, {}, {0}), std::runtime_error);
  const float nan = std::numeric_limits<float>::quiet_NaN();
  EXPECT_THROW(model_test::CompareDenoising({nan}, {0}, {0}), std::runtime_error);
  EXPECT_THROW(model_test::CompareDenoising({0.5F}, {nan}, {0}), std::runtime_error);
  EXPECT_THROW(model_test::CompareDenoising({0.5F}, {0}, {nan}), std::runtime_error);
}
