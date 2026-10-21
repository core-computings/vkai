#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>

#include "graph/Engine.h"
#include "utils/Detection.h"
namespace {
std::vector<float> ReadFloats(const std::string& path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) throw std::runtime_error("Cannot open " + path);
  auto bytes = file.tellg();
  if (bytes <= 0 || bytes % sizeof(float)) throw std::runtime_error("Invalid float32 input");
  std::vector<float> data(size_t(bytes) / sizeof(float));
  file.seekg(0);
  if (!file.read(reinterpret_cast<char*>(data.data()), bytes))
    throw std::runtime_error("Cannot read input");
  return data;
}
}  // namespace
int main(int argc, char** argv) {
  if (argc != 5) {
    std::cerr << "Usage: vkai_detect MODEL.onnx INPUT.bin RAW_OUTPUT.bin DETECTIONS.json\n";
    return 2;
  }
  try {
    vkai::Engine engine(argv[1]);
    engine.GetInputTensor()->PopulateTensor(ReadFloats(argv[2]));
    auto start = std::chrono::steady_clock::now();
    engine.ExecuteGraph();
    auto inferred = std::chrono::steady_clock::now();
    const auto& shape = engine.GetOutputTensor()->GetShape();
    if (shape.size() != 3 || shape[0] != 1 || shape[1] <= 4)
      throw std::runtime_error("Expected YOLO [1,4+classes,anchors] output");
    const auto& data = engine.GetOutputTensor()->Data();
    const auto detections = vkai::DecodeYOLODetections(data, shape[1] - 4, shape[2]);
    auto decoded = std::chrono::steady_clock::now();
    std::ofstream raw(argv[3], std::ios::binary);
    if (!raw.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(float)))
      throw std::runtime_error("Cannot write raw output");
    std::ofstream json(argv[4]);
    json << std::setprecision(9);
    json << "{\"coordinate_space\":\"letterboxed network input\",\"network_wall_ms\":"
         << std::chrono::duration<double, std::milli>(inferred - start).count()
         << ",\"postprocess_ms\":"
         << std::chrono::duration<double, std::milli>(decoded - inferred).count()
         << ",\"detections\":[";
    for (size_t i = 0; i < detections.size(); ++i) {
      const auto& d = detections[i];
      if (i) json << ',';
      json << "{\"class_id\":" << d.class_id << ",\"confidence\":" << d.confidence
           << ",\"box_xyxy\":[" << d.box_xyxy[0] << ',' << d.box_xyxy[1] << ',' << d.box_xyxy[2]
           << ',' << d.box_xyxy[3] << "]}";
    }
    json << "]}\n";
    if (!json) throw std::runtime_error("Cannot write detections");
    std::cout << "{\"detections\":" << detections.size() << "}\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "Detection failed: " << error.what() << '\n';
    return 1;
  }
}
