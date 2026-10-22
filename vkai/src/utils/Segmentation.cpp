#include "utils/Segmentation.h"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace vkai {
std::vector<uint8_t> ArgmaxSegmentation(const std::vector<float>& logits, int classes, int height,
                                        int width) {
  if (classes <= 0 || classes > 256 || height <= 0 || width <= 0 ||
      size_t(height) > std::numeric_limits<size_t>::max() / size_t(width) / size_t(classes) ||
      logits.size() != size_t(classes) * height * width)
    throw std::invalid_argument("Segmentation requires CHW logits and 1..256 classes");
  const size_t pixels = size_t(height) * width;
  std::vector<uint8_t> mask(pixels, 0);
  for (int c = 0; c < classes; ++c) {
    for (size_t i = 0; i < pixels; ++i) {
      const float value = logits[size_t(c) * pixels + i];
      if (!std::isfinite(value)) throw std::invalid_argument("Non-finite segmentation logit");
      if (value > logits[size_t(mask[i]) * pixels + i]) mask[i] = static_cast<uint8_t>(c);
    }
  }
  return mask;
}

SegmentationComparison CompareSegmentation(const std::vector<uint8_t>& actual,
                                           const std::vector<uint8_t>& reference, int classes) {
  if (classes <= 0 || classes > 256 || actual.empty() || actual.size() != reference.size())
    throw std::invalid_argument("Segmentation mask shape/classes mismatch");
  std::vector<size_t> intersection(classes), unions(classes), counts(classes);
  size_t agreement = 0;
  for (size_t i = 0; i < actual.size(); ++i) {
    const int a = actual[i], b = reference[i];
    if (a >= classes || b >= classes) throw std::invalid_argument("Invalid segmentation class");
    ++counts[a];
    ++unions[a];
    if (a == b) {
      ++agreement;
      ++intersection[a];
    } else {
      ++unions[b];
    }
  }
  double total_iou = 0;
  int present = 0;
  for (int c = 0; c < classes; ++c) {
    if (unions[c] == 0) continue;
    total_iou += double(intersection[c]) / unions[c];
    ++present;
  }
  return {double(agreement) / actual.size(), total_iou / present, counts};
}
}  // namespace vkai
