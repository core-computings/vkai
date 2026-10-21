#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>
namespace vkai {
struct Detection {
  std::array<float, 4> box_xyxy;
  float confidence;
  int class_id;
};
// YOLOv8 decoded output layout: [1, 4 + classes, anchors], xywh + class scores.
// Performs confidence filtering and stable class-aware greedy NMS on the CPU.
std::vector<Detection> DecodeYOLODetections(const std::vector<float>& predictions, int classes,
                                            int anchors, float confidence_threshold = .25F,
                                            float iou_threshold = .45F,
                                            size_t max_detections = 300);
float BoxIoU(const std::array<float, 4>& a, const std::array<float, 4>& b);
}  // namespace vkai
