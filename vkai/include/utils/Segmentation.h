#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace vkai {
// Batch=1, CHW logits. Ties select the lowest class index, as in torch.argmax.
std::vector<uint8_t> ArgmaxSegmentation(const std::vector<float>& logits, int classes, int height,
                                        int width);

struct SegmentationComparison {
  double pixel_agreement;
  // Mean over classes present in either mask, including background.
  double mean_iou;
  std::vector<size_t> class_pixels;
};

// Measures agreement with a reference prediction, not dataset ground-truth accuracy.
SegmentationComparison CompareSegmentation(const std::vector<uint8_t>& actual,
                                           const std::vector<uint8_t>& reference, int classes);
}  // namespace vkai
