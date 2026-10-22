#include "ModelTestUtils.h"

int main(int argc, char** argv) {
  const bool benchmark = argc > 1 && std::string_view(argv[1]) == "--benchmark";
  if (benchmark ? (argc != 5 && argc != 7) : argc != 5) {
    std::cerr << "Usage: vkai_mobilenet_v2_test MODEL.onnx INPUT.bin OUTPUT.bin REFERENCE.bin\n"
                 "       vkai_mobilenet_v2_test --benchmark MODEL.onnx INPUT.bin REFERENCE.bin "
                 "[WARMUP RUNS]\n";
    return 2;
  }
  const int first = benchmark ? 2 : 1;
  vkai::Engine engine(argv[first]);
  model_test::RequireShape(*engine.GetInputTensor(), {1, 3, 224, 224});
  model_test::RequireShape(*engine.GetOutputTensor(), {1, 1000});
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
  const auto top5 = model_test::CompareTop5(actual, reference);
  model_test::WriteFloats(argv[3], actual);
  std::cout << std::setprecision(9) << "{\"model\":\"mobilenet_v2\",\"inference_ms\":" << elapsed
            << ',';
  model_test::WriteComparison(std::cout, comparison, actual.size());
  model_test::WriteTop5(std::cout, actual, top5);
  std::cout << "}\n";
  return 0;
}
