#include <array>
#include <filesystem>
#include <memory>

#include "ModelTestUtils.h"
#include "utils/Segmentation.h"

#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_STATIC
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_easy_font.h>
#include <stb_image.h>
#include <stb_image_write.h>

namespace {
constexpr int kWidth = 320, kHeight = 320, kClasses = 21;
constexpr std::array<const char*, kClasses> kCategories{
    "background", "aeroplane", "bicycle",     "bird",  "boat",        "bottle", "bus",
    "car",        "cat",       "chair",       "cow",   "diningtable", "dog",    "horse",
    "motorbike",  "person",    "pottedplant", "sheep", "sofa",        "train",  "tvmonitor"};
using Color = std::array<unsigned char, 3>;

// Pascal VOC's bit-interleaved palette, with black background.
Color ClassColor(int label) {
  Color color{};
  for (int bit = 0; bit < 8; ++bit) {
    for (int c = 0; c < 3; ++c) color[c] |= ((label >> c) & 1) << (7 - bit);
    label >>= 3;
  }
  return color;
}

std::vector<uint8_t> ReadMask(const std::filesystem::path& path) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream || stream.tellg() != kWidth * kHeight)
    throw std::runtime_error("Expected a 320x320 uint8 PyTorch mask: " + path.string());
  std::vector<uint8_t> mask(kWidth * kHeight);
  stream.seekg(0);
  if (!stream.read(reinterpret_cast<char*>(mask.data()), mask.size()))
    throw std::runtime_error("Cannot read PyTorch mask: " + path.string());
  return mask;
}

void SavePNG(const std::filesystem::path& path, const std::vector<unsigned char>& pixels,
             int width = kWidth, int height = kHeight, int channels = 3) {
  if (!stbi_write_png(path.string().c_str(), width, height, channels, pixels.data(),
                      width * channels))
    throw std::runtime_error("Cannot save segmentation visualization: " + path.string());
}

void DrawText(std::vector<unsigned char>& pixels, int width, int height, int x, int y,
              std::string text, Color color = {240, 240, 240}) {
  struct Vertex {
    float x, y, z;
    unsigned char color[4];
  };
  static_assert(sizeof(Vertex) == 16);
  std::vector<Vertex> vertices(text.size() * 128);
  const int quads = stb_easy_font_print(0, 0, text.data(), nullptr, vertices.data(),
                                        static_cast<int>(vertices.size() * sizeof(Vertex)));
  for (int q = 0; q < quads; ++q) {
    const auto& a = vertices[size_t(q) * 4];
    const auto& b = vertices[size_t(q) * 4 + 2];
    for (int py = std::max(0, y + int(a.y)); py < std::min(height, y + int(b.y)); ++py)
      for (int px = std::max(0, x + int(a.x)); px < std::min(width, x + int(b.x)); ++px)
        std::copy(color.begin(), color.end(), pixels.data() + (size_t(py) * width + px) * 3);
  }
}

void SaveVisualizations(const std::filesystem::path& output, const std::filesystem::path& source,
                        const std::vector<uint8_t>& mask, const std::vector<uint8_t>& reference,
                        const vkai::SegmentationComparison& quality) {
  int width = 0, height = 0, channels = 0;
  std::unique_ptr<unsigned char, decltype(&stbi_image_free)> image(
      stbi_load(source.string().c_str(), &width, &height, &channels, 3), stbi_image_free);
  if (!image || width != kWidth || height != kHeight)
    throw std::runtime_error("Visualization requires the exporter's 320x320 input.png");
  std::array<std::vector<unsigned char>, 4> panels;
  panels[0].assign(image.get(), image.get() + size_t(width) * height * 3);
  for (size_t p = 1; p < panels.size(); ++p) panels[p].resize(panels[0].size());
  for (size_t i = 0; i < mask.size(); ++i) {
    const auto color = ClassColor(mask[i]), expected = ClassColor(reference[i]);
    for (int c = 0; c < 3; ++c) {
      panels[1][i * 3 + c] = expected[c];
      panels[2][i * 3 + c] = color[c];
      panels[3][i * 3 + c] =
          mask[i] == 0 ? panels[0][i * 3 + c] : (int(panels[0][i * 3 + c]) + int(color[c])) / 2;
    }
  }
  const auto prefix = (output.parent_path() / output.stem()).string();
  SavePNG(prefix + "_mask.png", mask, width, height, 1);
  SavePNG(prefix + "_color.png", panels[2]);
  SavePNG(prefix + "_overlay.png", panels[3]);
  constexpr int header = 24, footer = 80, canvas_width = kWidth * 4;
  constexpr int canvas_height = kHeight + header + footer;
  std::vector<unsigned char> comparison(size_t(canvas_width) * canvas_height * 3, 32);
  const std::array<std::string, 4> titles{"Input", "PyTorch argmax", "VKAI argmax", "VKAI overlay"};
  for (size_t panel = 0; panel < panels.size(); ++panel) {
    for (int y = 0; y < kHeight; ++y)
      std::copy_n(panels[panel].data() + size_t(y) * kWidth * 3, kWidth * 3,
                  comparison.data() + (size_t(y + header) * canvas_width + panel * kWidth) * 3);
    DrawText(comparison, canvas_width, canvas_height, int(panel) * kWidth + 8, 7, titles[panel]);
  }
  DrawText(comparison, canvas_width, canvas_height, 8, header + kHeight + 8,
           "Pascal VOC classes (background is black)");
  int legend = 0;
  for (int c = 0; c < kClasses; ++c) {
    if (quality.class_pixels[c] == 0) continue;
    const int x = 8 + (legend % 7) * 180, y = header + kHeight + 26 + (legend / 7) * 16;
    for (int py = y; py < y + 10; ++py)
      for (int px = x; px < x + 12; ++px) {
        const auto color = ClassColor(c);
        std::copy(color.begin(), color.end(),
                  comparison.data() + (size_t(py) * canvas_width + px) * 3);
      }
    DrawText(comparison, canvas_width, canvas_height, x + 18, y, kCategories[c]);
    ++legend;
  }
  SavePNG(prefix + "_comparison.png", comparison, canvas_width, canvas_height);
}
}  // namespace

int main(int argc, char** argv) {
  const bool benchmark = argc > 1 && std::string_view(argv[1]) == "--benchmark";
  if (argc != 5 && argc != 7) {
    std::cerr << "Usage: vkai_lraspp_test MODEL.onnx INPUT.bin OUTPUT.bin REFERENCE.bin "
                 "[REFERENCE_MASK.bin INPUT.png]\n"
                 "       vkai_lraspp_test --benchmark MODEL.onnx INPUT.bin REFERENCE.bin "
                 "[WARMUP RUNS]\n";
    return 2;
  }
  const int first = benchmark ? 2 : 1;
  vkai::Engine engine(argv[first]);
  model_test::RequireShape(*engine.GetInputTensor(), {1, 3, kHeight, kWidth});
  model_test::RequireShape(*engine.GetOutputTensor(), {1, kClasses, kHeight, kWidth});
  engine.GetInputTensor()->PopulateTensor(model_test::ReadFloats(argv[first + 1]));
  if (benchmark) {
    model_test::RunBenchmark(engine, argv[4], argc == 7 ? std::stoi(argv[5]) : 5,
                             argc == 7 ? std::stoi(argv[6]) : 30);
    return 0;
  }
  const auto elapsed = model_test::RunInference(engine);
  const auto& actual = engine.GetOutputTensor()->Data();
  const auto reference = model_test::ReadFloats(argv[4]);
  const auto comparison = model_test::CompareOutput(actual, reference);
  const auto folder = std::filesystem::path(argv[1]).parent_path();
  const auto expected =
      ReadMask(argc == 7 ? std::filesystem::path(argv[5]) : folder / "reference_mask.bin");
  if (expected != vkai::ArgmaxSegmentation(reference, kClasses, kHeight, kWidth))
    throw std::runtime_error("PyTorch reference logits and mask disagree");
  const auto mask = vkai::ArgmaxSegmentation(actual, kClasses, kHeight, kWidth);
  const auto quality = vkai::CompareSegmentation(mask, expected, kClasses);
  if (quality.pixel_agreement < .999 || quality.mean_iou < .999)
    throw std::runtime_error("Segmentation mask differs from PyTorch reference");
  if (quality.class_pixels[0] == 0 || quality.class_pixels[12] == 0)
    throw std::runtime_error("Expected background and dog in sample segmentation");
  model_test::WriteFloats(argv[3], actual);
  SaveVisualizations(argv[3], argc == 7 ? std::filesystem::path(argv[6]) : folder / "input.png",
                     mask, expected, quality);
  std::cout << std::setprecision(9)
            << "{\"model\":\"lraspp\",\"input_shape\":[1,3,320,320],\"inference_ms\":" << elapsed
            << ',';
  model_test::WriteComparison(std::cout, comparison, actual.size());
  std::cout << ",\"mask_pixel_agreement\":" << quality.pixel_agreement
            << ",\"mask_mean_iou_vs_pytorch\":" << quality.mean_iou << ",\"classes\":[";
  bool separator = false;
  for (int c = 0; c < kClasses; ++c) {
    if (quality.class_pixels[c] == 0) continue;
    if (separator) std::cout << ',';
    separator = true;
    std::cout << "{\"id\":" << c << ",\"name\":" << std::quoted(kCategories[c])
              << ",\"pixels\":" << quality.class_pixels[c] << '}';
  }
  const auto prefix =
      (std::filesystem::path(argv[3]).parent_path() / std::filesystem::path(argv[3]).stem())
          .string();
  std::cout << "],\"comparison_file\":" << std::quoted(prefix + "_comparison.png") << "}\n";
  return 0;
}
