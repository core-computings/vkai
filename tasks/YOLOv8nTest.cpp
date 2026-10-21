#include <array>
#include <filesystem>
#include <memory>
#include <sstream>

#include "ModelTestUtils.h"
#include "utils/Detection.h"

#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_JPEG
#define STBI_ONLY_PNG
#include <stb_image.h>

#define STB_IMAGE_WRITE_STATIC
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_easy_font.h>
#include <stb_image_write.h>

namespace {
constexpr std::array<std::string_view, 80> kCategories{
    "person",        "bicycle",      "car",
    "motorcycle",    "airplane",     "bus",
    "train",         "truck",        "boat",
    "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench",        "bird",
    "cat",           "dog",          "horse",
    "sheep",         "cow",          "elephant",
    "bear",          "zebra",        "giraffe",
    "backpack",      "umbrella",     "handbag",
    "tie",           "suitcase",     "frisbee",
    "skis",          "snowboard",    "sports ball",
    "kite",          "baseball bat", "baseball glove",
    "skateboard",    "surfboard",    "tennis racket",
    "bottle",        "wine glass",   "cup",
    "fork",          "knife",        "spoon",
    "bowl",          "banana",       "apple",
    "sandwich",      "orange",       "broccoli",
    "carrot",        "hot dog",      "pizza",
    "donut",         "cake",         "chair",
    "couch",         "potted plant", "bed",
    "dining table",  "toilet",       "tv",
    "laptop",        "mouse",        "remote",
    "keyboard",      "cell phone",   "microwave",
    "oven",          "toaster",      "sink",
    "refrigerator",  "book",         "clock",
    "vase",          "scissors",     "teddy bear",
    "hair drier",    "toothbrush"};

using Color = std::array<unsigned char, 3>;

void FillRectangle(unsigned char* pixels, int width, int height, int x1, int y1, int x2, int y2,
                   const Color& color) {
  for (int y = std::clamp(y1, 0, height); y < std::clamp(y2, 0, height); ++y) {
    for (int x = std::clamp(x1, 0, width); x < std::clamp(x2, 0, width); ++x) {
      auto* pixel = pixels + (size_t(y) * width + x) * 3;
      std::copy(color.begin(), color.end(), pixel);
    }
  }
}

void DrawLabel(unsigned char* pixels, int width, int height, int x, int y, std::string text,
               const Color& color) {
  constexpr int scale = 2, padding = 4;
  const int label_width = stb_easy_font_width(text.data()) * scale + padding * 2;
  const int label_height = stb_easy_font_height(text.data()) * scale + padding * 2;
  x = std::clamp(x, 0, std::max(0, width - label_width));
  y = std::clamp(y >= label_height ? y - label_height : y, 0, std::max(0, height - label_height));
  FillRectangle(pixels, width, height, x, y, x + label_width, y + label_height, color);
  struct Vertex {
    float x, y, z;
    unsigned char color[4];
  };
  static_assert(sizeof(Vertex) == 16);
  // Each ASCII glyph has at most 16 horizontal + 16 vertical segments, four vertices each.
  std::vector<Vertex> vertices(text.size() * 128);
  const int quads = stb_easy_font_print(0, 0, text.data(), nullptr, vertices.data(),
                                        static_cast<int>(vertices.size() * sizeof(Vertex)));
  for (int i = 0; i < quads; ++i) {
    const auto& first = vertices[size_t(i) * 4];
    const auto& opposite = vertices[size_t(i) * 4 + 2];
    FillRectangle(pixels, width, height, x + padding + int(first.x * scale),
                  y + padding + int(first.y * scale), x + padding + int(opposite.x * scale),
                  y + padding + int(opposite.y * scale), {0, 0, 0});
  }
}

void SaveVisualization(const std::filesystem::path& source, const std::filesystem::path& output,
                       int network_width, int network_height,
                       const std::vector<vkai::Detection>& detections) {
  int width = 0, height = 0, channels = 0;
  std::unique_ptr<unsigned char, decltype(&stbi_image_free)> pixels(
      stbi_load(source.string().c_str(), &width, &height, &channels, 3), stbi_image_free);
  if (!pixels) throw std::runtime_error("Cannot read visualization input: " + source.string());
  const double scale = std::min(double(network_width) / width, double(network_height) / height);
  // Match the exporter's nearest-even resize rounding and integer letterbox padding.
  const int pad_left = (network_width - int(std::nearbyint(width * scale))) / 2;
  const int pad_top = (network_height - int(std::nearbyint(height * scale))) / 2;
  struct Label {
    int x, y;
    std::string text;
    Color color;
  };
  std::vector<Label> labels;
  for (const auto& detection : detections) {
    const auto& box = detection.box_xyxy;
    const int x1 =
        int(std::lround(std::clamp((box[0] - pad_left) / scale, 0.0, double(width - 1))));
    const int y1 =
        int(std::lround(std::clamp((box[1] - pad_top) / scale, 0.0, double(height - 1))));
    const int x2 =
        int(std::lround(std::clamp((box[2] - pad_left) / scale, 0.0, double(width - 1))));
    const int y2 =
        int(std::lround(std::clamp((box[3] - pad_top) / scale, 0.0, double(height - 1))));
    const Color color = detection.class_id == 5 ? Color{255, 160, 20} : Color{10, 180, 255};
    constexpr int thickness = 4;
    FillRectangle(pixels.get(), width, height, x1, y1, x2 + 1, y1 + thickness, color);
    FillRectangle(pixels.get(), width, height, x1, y2 - thickness + 1, x2 + 1, y2 + 1, color);
    FillRectangle(pixels.get(), width, height, x1, y1, x1 + thickness, y2 + 1, color);
    FillRectangle(pixels.get(), width, height, x2 - thickness + 1, y1, x2 + 1, y2 + 1, color);
    std::ostringstream label;
    label << kCategories.at(detection.class_id) << ' ' << std::fixed << std::setprecision(2)
          << detection.confidence;
    labels.push_back({x1, y1, label.str(), color});
  }
  // Keep later boxes from painting over text belonging to earlier detections.
  for (const auto& label : labels)
    DrawLabel(pixels.get(), width, height, label.x, label.y, label.text, label.color);
  if (!stbi_write_png(output.string().c_str(), width, height, 3, pixels.get(), width * 3))
    throw std::runtime_error("Cannot save detection visualization: " + output.string());
}

// Exported by torchvision NMS from the PyTorch output, independently of C++ NMS.
std::vector<vkai::Detection> ReadReferenceDetections(const std::string& path) {
  std::ifstream stream(path);
  size_t count = 0;
  if (!(stream >> count) || count == 0 || count > 300)
    throw std::runtime_error("Invalid PyTorch detection reference: " + path);
  std::vector<vkai::Detection> result(count);
  for (auto& detection : result) {
    auto& box = detection.box_xyxy;
    if (!(stream >> detection.class_id >> detection.confidence >> box[0] >> box[1] >> box[2] >>
          box[3]) ||
        detection.class_id < 0 || detection.class_id >= 80 ||
        !std::isfinite(detection.confidence) || detection.confidence < 0 ||
        detection.confidence > 1 ||
        !std::all_of(box.begin(), box.end(), [](float value) { return std::isfinite(value); }) ||
        box[2] <= box[0] || box[3] <= box[1]) {
      throw std::runtime_error("Invalid PyTorch detection entry: " + path);
    }
  }
  std::string extra;
  if (stream >> extra) throw std::runtime_error("Unexpected trailing reference data: " + path);
  return result;
}
}  // namespace

int main(int argc, char** argv) {
  const bool benchmark = argc > 1 && std::string_view(argv[1]) == "--benchmark";
  if (benchmark ? (argc != 5 && argc != 7) : (argc != 7 && argc != 9)) {
    std::cerr << "Usage: vkai_yolov8n_test MODEL.onnx INPUT.bin RAW_OUTPUT.bin DETECTIONS.json "
                 "REFERENCE.bin REFERENCE_DETECTIONS.txt [INPUT.jpg OUTPUT.png]\n"
                 "       vkai_yolov8n_test --benchmark MODEL.onnx INPUT.bin REFERENCE.bin "
                 "[WARMUP RUNS]\n";
    return 2;
  }
  const int first = benchmark ? 2 : 1;
  vkai::Engine engine(argv[first]);
  const auto& input_shape = engine.GetInputTensor()->GetShape();
  if (input_shape.size() != 4 || input_shape[0] != 1 || input_shape[1] != 3 ||
      input_shape[2] <= 0 || input_shape[2] != input_shape[3] || input_shape[2] % 32) {
    throw std::runtime_error("Expected YOLO [1,3,size,size] input, size a multiple of 32");
  }
  const int64_t grid = input_shape[2] / 32;
  model_test::RequireShape(*engine.GetOutputTensor(), {1, 84, 21 * grid * grid});
  engine.GetInputTensor()->PopulateTensor(model_test::ReadFloats(argv[first + 1]));
  if (benchmark) {
    model_test::RunBenchmark(engine, argv[4], argc == 7 ? std::stoi(argv[5]) : 5,
                             argc == 7 ? std::stoi(argv[6]) : 30);
    return 0;
  }
  const auto start = std::chrono::steady_clock::now();
  engine.ExecuteGraph();
  const auto inferred = std::chrono::steady_clock::now();
  const auto& shape = engine.GetOutputTensor()->GetShape();
  const auto& data = engine.GetOutputTensor()->Data();
  const auto detections = vkai::DecodeYOLODetections(data, shape[1] - 4, shape[2]);
  const auto decoded = std::chrono::steady_clock::now();
  const auto reference = model_test::ReadFloats(argv[5]);
  const auto comparison = model_test::CompareOutput(data, reference);
  const auto expected = ReadReferenceDetections(argv[6]);
  if (detections.size() != expected.size())
    throw std::runtime_error("Detection count differs from PyTorch reference");
  bool people = false, bus = false;
  float min_iou = 1, max_confidence_error = 0, class_score_error = 0;
  for (size_t i = 0; i < detections.size(); ++i) {
    const auto& detection = detections[i];
    if (detection.class_id != expected[i].class_id)
      throw std::runtime_error("Detection class/order differs from PyTorch reference");
    const float confidence_error = std::abs(detection.confidence - expected[i].confidence);
    const float iou = vkai::BoxIoU(detection.box_xyxy, expected[i].box_xyxy);
    if (confidence_error >= 1e-4F)
      throw std::runtime_error("Detection confidence differs from PyTorch reference");
    if (iou <= .999F) throw std::runtime_error("Detection box differs from PyTorch reference");
    min_iou = std::min(min_iou, iou);
    max_confidence_error = std::max(max_confidence_error, confidence_error);
    people |= detection.class_id == 0;
    bus |= detection.class_id == 5;
  }
  if (!people || !bus) throw std::runtime_error("Expected people and bus in the sample");
  for (size_t i = size_t(4) * shape[2]; i < data.size(); ++i)
    class_score_error = std::max(class_score_error, std::abs(data[i] - reference[i]));
  const auto source_image = argc == 9 ? std::filesystem::path(argv[7])
                                      : std::filesystem::path(argv[1]).parent_path() / "input.jpg";
  auto visualization = std::filesystem::path(argv[4]);
  visualization.replace_extension(".png");
  if (argc == 9) visualization = argv[8];
  SaveVisualization(source_image, visualization, input_shape[3], input_shape[2], detections);
  model_test::WriteFloats(argv[3], data);
  std::ofstream json(argv[4]);
  json << std::setprecision(9);
  json << "{\"coordinate_space\":\"letterboxed network input\",\"network_wall_ms\":"
       << std::chrono::duration<double, std::milli>(inferred - start).count()
       << ",\"postprocess_ms\":"
       << std::chrono::duration<double, std::milli>(decoded - inferred).count()
       << ",\"detections\":[";
  for (size_t i = 0; i < detections.size(); ++i) {
    const auto& detection = detections[i];
    if (i) json << ',';
    json << "{\"class_id\":" << detection.class_id << ",\"confidence\":" << detection.confidence
         << ",\"box_xyxy\":[" << detection.box_xyxy[0] << ',' << detection.box_xyxy[1] << ','
         << detection.box_xyxy[2] << ',' << detection.box_xyxy[3] << "]}";
  }
  json << "]}\n";
  if (!json) throw std::runtime_error("Cannot write detections");
  std::cout << std::setprecision(9) << "{\"model\":\"yolov8n\",\"network_wall_ms\":"
            << std::chrono::duration<double, std::milli>(inferred - start).count()
            << ",\"postprocess_ms\":"
            << std::chrono::duration<double, std::milli>(decoded - inferred).count() << ',';
  model_test::WriteComparison(std::cout, comparison, data.size());
  std::cout << ",\"detection_count\":" << detections.size()
            << ",\"min_box_iou_vs_pytorch\":" << min_iou
            << ",\"max_confidence_error\":" << max_confidence_error
            << ",\"class_score_max_abs_error\":" << class_score_error
            << ",\"visualization_file\":" << std::quoted(visualization.string()) << "}\n";
  return 0;
}
