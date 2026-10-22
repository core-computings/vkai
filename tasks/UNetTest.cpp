#include "ModelTestUtils.h"

int main(int argc, char** argv) {
  const bool benchmark = argc > 1 && std::string_view(argv[1]) == "--benchmark";
  if (benchmark ? (argc != 5 && argc != 7) : argc != 5) {
    std::cerr << "Usage: vkai_unet_test MODEL.onnx INPUT.bin OUTPUT.bin REFERENCE.bin\n"
                 "       vkai_unet_test --benchmark MODEL.onnx INPUT.bin REFERENCE.bin "
                 "[WARMUP RUNS]\n";
    return 2;
  }
  const int first = benchmark ? 2 : 1;
  vkai::Engine engine(argv[first]);
  model_test::RequireShape(*engine.GetInputTensor(), {1, 3, 256, 256});
  model_test::RequireShape(*engine.GetOutputTensor(), {1, 1, 256, 256});
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
  size_t foreground = 0, expected_foreground = 0, intersection = 0, agreement = 0;
  for (size_t i = 0; i < actual.size(); ++i) {
    const bool mask = actual[i] >= .5F, expected = reference[i] >= .5F;
    foreground += mask;
    expected_foreground += expected;
    intersection += mask && expected;
    agreement += mask == expected;
  }
  if (foreground == 0 || foreground == actual.size())
    throw std::runtime_error("Segmentation must contain foreground and background");
  const double dice = 2.0 * intersection / (foreground + expected_foreground);
  if (dice < .999) throw std::runtime_error("Segmentation mask differs from PyTorch reference");
  model_test::WriteFloats(argv[3], actual);
  std::cout << std::setprecision(9) << "{\"model\":\"unet\",\"inference_ms\":" << elapsed << ',';
  model_test::WriteComparison(std::cout, comparison, actual.size());
  std::cout << ",\"mask_dice_vs_pytorch\":" << dice
            << ",\"mask_pixel_agreement\":" << static_cast<double>(agreement) / actual.size()
            << ",\"foreground_pixels\":" << foreground << "}\n";
  return 0;
}
