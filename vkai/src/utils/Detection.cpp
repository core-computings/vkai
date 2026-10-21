#include "utils/Detection.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
namespace vkai {
float BoxIoU(const std::array<float, 4>& a, const std::array<float, 4>& b) {
  float intersection = std::max(0.F, std::min(a[2], b[2]) - std::max(a[0], b[0])) *
                       std::max(0.F, std::min(a[3], b[3]) - std::max(a[1], b[1]));
  float area_a = std::max(0.F, a[2] - a[0]) * std::max(0.F, a[3] - a[1]),
        area_b = std::max(0.F, b[2] - b[0]) * std::max(0.F, b[3] - b[1]);
  float total = area_a + area_b - intersection;
  return total > 0 ? intersection / total : 0;
}
std::vector<Detection> DecodeYOLODetections(const std::vector<float>& p, int classes, int anchors,
                                            float confidence_threshold, float iou_threshold,
                                            size_t max_detections) {
  if (classes <= 0 || anchors <= 0 || p.size() != size_t(int64_t(classes) + 4) * size_t(anchors) ||
      !std::isfinite(confidence_threshold) || !std::isfinite(iou_threshold) ||
      confidence_threshold < 0 || confidence_threshold > 1 || iou_threshold < 0 ||
      iou_threshold > 1 || !max_detections)
    throw std::invalid_argument("Invalid YOLO output shape or NMS parameters");
  for (float value : p)
    if (!std::isfinite(value)) throw std::invalid_argument("Non-finite YOLO predictions");
  std::vector<Detection> candidates;
  for (int i = 0; i < anchors; ++i) {
    int label = 0;
    float score = p[size_t(4) * anchors + i];
    for (int c = 1; c < classes; ++c)
      if (p[size_t(c + 4) * anchors + i] > score) {
        score = p[size_t(c + 4) * anchors + i];
        label = c;
      }
    if (score < confidence_threshold) continue;
    float x = p[i], y = p[size_t(anchors) + i], w = p[size_t(2) * anchors + i],
          h = p[size_t(3) * anchors + i];
    if (w <= 0 || h <= 0) continue;
    candidates.push_back({{x - w / 2, y - h / 2, x + w / 2, y + h / 2}, score, label});
  }
  std::stable_sort(candidates.begin(), candidates.end(),
                   [](const auto& a, const auto& b) { return a.confidence > b.confidence; });
  std::vector<Detection> result;
  for (const auto& candidate : candidates) {
    bool suppressed = false;
    for (const auto& kept : result)
      if (candidate.class_id == kept.class_id &&
          BoxIoU(candidate.box_xyxy, kept.box_xyxy) > iou_threshold) {
        suppressed = true;
        break;
      }
    if (!suppressed) {
      result.push_back(candidate);
      if (result.size() == max_detections) break;
    }
  }
  return result;
}
}  // namespace vkai
