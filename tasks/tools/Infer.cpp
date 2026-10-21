#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "graph/Engine.h"
#include "utils/TensorUtils.h"

namespace {
std::vector<float> ReadFloats(const std::string& path) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream) throw std::runtime_error("Cannot open " + path);
  const auto bytes = stream.tellg();
  if (bytes <= 0 || bytes % sizeof(float))
    throw std::runtime_error("Invalid float32 file: " + path);
  std::vector<float> data(static_cast<size_t>(bytes) / sizeof(float));
  stream.seekg(0);
  if (!stream.read(reinterpret_cast<char*>(data.data()), bytes))
    throw std::runtime_error("Cannot read " + path);
  return data;
}
}  // namespace

int main(int argc, char** argv) {
  if (argc != 4 && argc != 5) {
    std::cerr << "Usage: vkai_infer MODEL.onnx INPUT.bin OUTPUT.bin [REFERENCE.bin]\n";
    return 2;
  }
  try {
    vkai::Engine engine(argv[1]);
    engine.GetInputTensor()->PopulateTensor(ReadFloats(argv[2]));
    const auto start = std::chrono::steady_clock::now();
    engine.ExecuteGraph();
    const auto elapsed =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    const auto& actual = engine.GetOutputTensor()->Data();
    std::ofstream output(argv[3], std::ios::binary);
    if (!output.write(reinterpret_cast<const char*>(actual.data()),
                      actual.size() * sizeof(float))) {
      throw std::runtime_error("Cannot write output");
    }
    for (auto value : actual)
      if (!std::isfinite(value)) throw std::runtime_error("Non-finite model output");
    std::cout << "{\"elements\":" << actual.size() << ",\"inference_ms\":" << elapsed;
    size_t mismatches = 0;
    if (argc == 5) {
      const auto reference = ReadFloats(argv[4]);
      if (reference.size() != actual.size()) throw std::runtime_error("Reference shape mismatch");
      float maximum = 0;
      double squared = 0;
      for (size_t i = 0; i < actual.size(); ++i) {
        const float error = std::abs(actual[i] - reference[i]);
        maximum = std::max(maximum, error);
        squared += error * error;
        if (!std::isfinite(reference[i]) || error > 1e-3F + 1e-3F * std::abs(reference[i]))
          ++mismatches;
      }
      std::cout << ",\"max_abs_error\":" << maximum
                << ",\"rmse\":" << std::sqrt(squared / actual.size())
                << ",\"mismatches\":" << mismatches;
    }
    std::cout << "}\n";
    return mismatches == 0 ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << "Inference failed: " << error.what() << '\n';
    return 1;
  }
}
