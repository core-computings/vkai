#include <array>
#include <filesystem>

#include "ModelTestUtils.h"

#define STB_IMAGE_WRITE_STATIC
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_easy_font.h>
#include <stb_image_write.h>

namespace {
constexpr int kWidth = 256, kHeight = 256, kChannels = 3;

std::vector<unsigned char> Pixels(const std::vector<float>& chw) {
  std::vector<unsigned char> pixels(chw.size());
  for (int i = 0; i < kWidth * kHeight; ++i)
    for (int c = 0; c < kChannels; ++c)
      pixels[size_t(i) * kChannels + c] = static_cast<unsigned char>(
          std::lround(std::clamp(chw[size_t(c) * kWidth * kHeight + i], 0.F, 1.F) * 255));
  return pixels;
}

void SavePNG(const std::filesystem::path& path, int width, int height,
             const std::vector<unsigned char>& pixels) {
  if (!stbi_write_png(path.string().c_str(), width, height, 3, pixels.data(), width * 3))
    throw std::runtime_error("Cannot save denoising visualization: " + path.string());
}

void SaveVisualizations(const std::filesystem::path& output, const std::vector<float>& noisy,
                        const std::vector<float>& denoised, const std::vector<float>& clean) {
  const auto prefix = output.parent_path() / output.stem();
  const std::array images{Pixels(noisy), Pixels(denoised), Pixels(clean)};
  SavePNG(prefix.string() + "_noisy.png", kWidth, kHeight, images[0]);
  SavePNG(prefix.string() + "_denoised.png", kWidth, kHeight, images[1]);
  constexpr int header = 24, width = kWidth * 3;
  std::vector<unsigned char> comparison(size_t(width) * (kHeight + header) * 3, 32);
  std::array<std::string, 3> titles{"Noisy input", "VKAI denoised", "Clean reference"};
  for (size_t panel = 0; panel < images.size(); ++panel) {
    for (int y = 0; y < kHeight; ++y)
      std::copy_n(images[panel].data() + size_t(y) * kWidth * 3, kWidth * 3,
                  comparison.data() + (size_t(y + header) * width + panel * kWidth) * 3);
    struct Vertex {
      float x, y, z;
      unsigned char color[4];
    };
    static_assert(sizeof(Vertex) == 16);
    std::vector<Vertex> vertices(titles[panel].size() * 128);
    const int quads = stb_easy_font_print(0, 0, titles[panel].data(), nullptr, vertices.data(),
                                          static_cast<int>(vertices.size() * sizeof(Vertex)));
    for (int q = 0; q < quads; ++q) {
      const auto& a = vertices[size_t(q) * 4];
      const auto& b = vertices[size_t(q) * 4 + 2];
      for (int y = int(a.y) + 7; y < int(b.y) + 7; ++y)
        for (int x = int(a.x) + 8; x < int(b.x) + 8; ++x)
          std::fill_n(comparison.data() + (size_t(y) * width + panel * kWidth + x) * 3, 3, 240);
    }
  }
  SavePNG(prefix.string() + "_comparison.png", width, kHeight + header, comparison);
}
}  // namespace

int main(int argc, char** argv) {
  const bool benchmark = argc > 1 && std::string_view(argv[1]) == "--benchmark";
  if (benchmark ? (argc != 5 && argc != 7) : argc != 6) {
    std::cerr << "Usage: vkai_dncnn_test MODEL.onnx INPUT.bin OUTPUT.bin REFERENCE.bin CLEAN.bin\n"
                 "       vkai_dncnn_test --benchmark MODEL.onnx INPUT.bin REFERENCE.bin "
                 "[WARMUP RUNS]\n";
    return 2;
  }
  const int first = benchmark ? 2 : 1;
  // Keep each mobile GPU job below the driver's watchdog limit. Use the same
  // submission boundaries for validation and benchmarking on both platforms.
  vkai::Engine engine(argv[first], 8);
  model_test::RequireShape(*engine.GetInputTensor(), {1, kChannels, kHeight, kWidth});
  model_test::RequireShape(*engine.GetOutputTensor(), {1, kChannels, kHeight, kWidth});
  const auto noisy = model_test::ReadFloats(argv[first + 1]);
  engine.GetInputTensor()->PopulateTensor(noisy);
  if (benchmark) {
    model_test::RunBenchmark(engine, argv[4], argc == 7 ? std::stoi(argv[5]) : 5,
                             argc == 7 ? std::stoi(argv[6]) : 30);
    return 0;
  }
  const auto elapsed = model_test::RunInference(engine);
  const auto& actual = engine.GetOutputTensor()->Data();
  model_test::WriteFloats(argv[3], actual);
  const auto reference = model_test::ReadFloats(argv[4]);
  const auto comparison = model_test::CompareOutput(actual, reference);
  const auto clean = model_test::ReadFloats(argv[5]);
  const auto quality = model_test::CompareDenoising(noisy, actual, clean);
  SaveVisualizations(argv[3], noisy, actual, clean);
  std::cout << std::setprecision(9)
            << "{\"model\":\"dncnn\",\"input_shape\":[1,3,256,256],\"inference_ms\":" << elapsed
            << ',';
  model_test::WriteComparison(std::cout, comparison, actual.size());
  std::cout << ",\"noisy_psnr_db\":" << quality.noisy_psnr_db
            << ",\"denoised_psnr_db\":" << quality.denoised_psnr_db
            << ",\"psnr_gain_db\":" << quality.denoised_psnr_db - quality.noisy_psnr_db
            << ",\"comparison_file\":"
            << std::quoted((std::filesystem::path(argv[3]).parent_path() /
                            (std::filesystem::path(argv[3]).stem().string() + "_comparison.png"))
                               .string())
            << "}\n";
  return 0;
}
