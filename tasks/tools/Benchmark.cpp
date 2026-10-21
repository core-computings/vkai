#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "graph/Engine.h"

namespace {
std::vector<float> ReadFloats(const std::string& path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) throw std::runtime_error("Cannot open " + path);
  const auto bytes = file.tellg();
  if (bytes <= 0 || bytes % sizeof(float))
    throw std::runtime_error("Invalid float32 file: " + path);
  std::vector<float> values(static_cast<size_t>(bytes) / sizeof(float));
  file.seekg(0);
  if (!file.read(reinterpret_cast<char*>(values.data()), bytes))
    throw std::runtime_error("Cannot read " + path);
  return values;
}
}  // namespace

int main(int argc, char** argv) {
  if (argc != 4 && argc != 6) {
    std::cerr << "Usage: vkai_benchmark MODEL.onnx INPUT.bin REFERENCE.bin [WARMUP RUNS]\n";
    return 2;
  }
  try {
    const int warmup = argc == 6 ? std::stoi(argv[4]) : 5;
    const int runs = argc == 6 ? std::stoi(argv[5]) : 30;
    vkai::Engine engine(argv[1]);
    engine.GetInputTensor()->PopulateTensor(ReadFloats(argv[2]));
    const auto reference = ReadFloats(argv[3]);
    const auto result = engine.BenchmarkGraph(warmup, runs);
    const auto& actual = engine.GetOutputTensor()->Data();
    if (actual.size() != reference.size()) throw std::runtime_error("Reference shape mismatch");
    float max_error = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
      const float error = std::abs(actual[i] - reference[i]);
      if (!std::isfinite(actual[i]) || !std::isfinite(reference[i]) ||
          error > 1e-3F + 1e-3F * std::abs(reference[i])) {
        throw std::runtime_error("Benchmark output does not match reference");
      }
      max_error = std::max(max_error, error);
    }
    auto sorted = result.gpu_ms;
    std::sort(sorted.begin(), sorted.end());
    const double median = (sorted[(sorted.size() - 1) / 2] + sorted[sorted.size() / 2]) / 2;
    std::cout << std::setprecision(9) << "{\"device\":" << std::quoted(result.device_name)
              << ",\"timer\":\"GPU timestamps, complete graph\",\"warmup_runs\":" << warmup
              << ",\"measured_runs\":" << runs
              << ",\"timestamp_valid_bits\":" << result.timestamp_valid_bits
              << ",\"timestamp_period_ns\":" << result.timestamp_period_ns << ",\"mean_ms\":"
              << std::accumulate(sorted.begin(), sorted.end(), 0.0) / sorted.size()
              << ",\"median_ms\":" << median << ",\"min_ms\":" << sorted.front()
              << ",\"max_ms\":" << sorted.back() << ",\"max_abs_error\":" << max_error
              << ",\"samples_ms\":[";
    for (size_t i = 0; i < result.gpu_ms.size(); ++i) {
      if (i) std::cout << ',';
      std::cout << result.gpu_ms[i];
    }
    std::cout << "]}\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "Benchmark failed: " << error.what() << '\n';
    return 1;
  }
}
