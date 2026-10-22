#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "graph/Engine.h"

namespace model_test {

inline std::vector<float> ReadFloats(const std::string& path) {
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

inline void WriteFloats(const std::string& path, const std::vector<float>& data) {
  std::ofstream stream(path, std::ios::binary);
  if (!stream.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(float)))
    throw std::runtime_error("Cannot write " + path);
}

inline void RequireShape(const vkai::Tensor& tensor, const std::vector<int64_t>& expected) {
  if (tensor.GetShape() != expected)
    throw std::runtime_error("Unexpected model tensor shape: " + tensor.Name());
}

struct OutputComparison {
  float max_abs_error = 0;
  double rmse = 0;
};

inline OutputComparison CompareOutput(const std::vector<float>& actual,
                                      const std::vector<float>& reference) {
  if (actual.empty() || actual.size() != reference.size())
    throw std::runtime_error("Reference shape mismatch");
  OutputComparison result;
  double squared = 0;
  for (size_t i = 0; i < actual.size(); ++i) {
    const float error = std::abs(actual[i] - reference[i]);
    if (!std::isfinite(actual[i]) || !std::isfinite(reference[i]) ||
        error > 1e-3F + 1e-3F * std::abs(reference[i])) {
      throw std::runtime_error("Model output differs from reference at element " +
                               std::to_string(i) + ": actual=" + std::to_string(actual[i]) +
                               ", reference=" + std::to_string(reference[i]) +
                               ", abs_error=" + std::to_string(error));
    }
    result.max_abs_error = std::max(result.max_abs_error, error);
    squared += static_cast<double>(error) * error;
  }
  result.rmse = std::sqrt(squared / actual.size());
  return result;
}

struct DenoisingQuality {
  double noisy_psnr_db;
  double denoised_psnr_db;
};

// Measure the raw float32 RGB tensors (peak=1), before display clipping.
inline DenoisingQuality CompareDenoising(const std::vector<float>& noisy,
                                         const std::vector<float>& denoised,
                                         const std::vector<float>& clean) {
  if (clean.empty() || noisy.size() != clean.size() || denoised.size() != clean.size())
    throw std::runtime_error("Denoising reference shape mismatch");
  double noisy_squared = 0, denoised_squared = 0;
  for (size_t i = 0; i < clean.size(); ++i) {
    if (!std::isfinite(noisy[i]) || !std::isfinite(denoised[i]) || !std::isfinite(clean[i]))
      throw std::runtime_error("Non-finite denoising tensor");
    const double noise = static_cast<double>(noisy[i]) - clean[i];
    const double error = static_cast<double>(denoised[i]) - clean[i];
    noisy_squared += noise * noise;
    denoised_squared += error * error;
  }
  // Cap perfect reconstruction at 120 dB so reports remain valid JSON.
  const DenoisingQuality result{-10 * std::log10(std::max(noisy_squared / clean.size(), 1e-12)),
                                -10 * std::log10(std::max(denoised_squared / clean.size(), 1e-12))};
  if (result.denoised_psnr_db < result.noisy_psnr_db + 3)
    throw std::runtime_error("Denoising must improve PSNR by at least 3 dB");
  return result;
}

inline double RunInference(vkai::Engine& engine) {
  const auto start = std::chrono::steady_clock::now();
  engine.ExecuteGraph();
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
      .count();
}

inline void WriteComparison(std::ostream& stream, const OutputComparison& comparison,
                            size_t elements) {
  stream << std::setprecision(9) << "\"elements\":" << elements
         << ",\"max_abs_error\":" << comparison.max_abs_error << ",\"rmse\":" << comparison.rmse
         << ",\"mismatches\":0";
}

inline std::vector<size_t> Top5(const std::vector<float>& values) {
  if (values.size() < 5) throw std::runtime_error("Expected at least five classification scores");
  std::vector<size_t> indices(values.size());
  std::iota(indices.begin(), indices.end(), 0);
  std::partial_sort(indices.begin(), indices.begin() + 5, indices.end(),
                    [&values](size_t a, size_t b) {
                      return values[a] == values[b] ? a < b : values[a] > values[b];
                    });
  indices.resize(5);
  return indices;
}

inline std::vector<size_t> CompareTop5(const std::vector<float>& actual,
                                       const std::vector<float>& reference) {
  const auto indices = Top5(actual);
  if (indices != Top5(reference))
    throw std::runtime_error("Classification top-5 differs from PyTorch reference");
  return indices;
}

inline void WriteTop5(std::ostream& stream, const std::vector<float>& scores,
                      const std::vector<size_t>& indices) {
  const float maximum = *std::max_element(scores.begin(), scores.end());
  double total = 0;
  for (float score : scores) total += std::exp(static_cast<double>(score) - maximum);
  stream << ",\"top5\":[";
  for (size_t i = 0; i < indices.size(); ++i) {
    if (i) stream << ',';
    stream << "{\"index\":" << indices[i] << ",\"probability\":"
           << std::exp(static_cast<double>(scores[indices[i]]) - maximum) / total << '}';
  }
  stream << ']';
}

inline void RunBenchmark(vkai::Engine& engine, const char* reference_path, int warmup, int runs) {
  const auto reference = ReadFloats(reference_path);
  const auto result = engine.BenchmarkGraph(warmup, runs);
  const auto comparison = CompareOutput(engine.GetOutputTensor()->Data(), reference);
  auto sorted = result.gpu_ms;
  std::sort(sorted.begin(), sorted.end());
  const double median = (sorted[(sorted.size() - 1) / 2] + sorted[sorted.size() / 2]) / 2;
  std::cout << std::setprecision(9) << "{\"device\":" << std::quoted(result.device_name)
            << ",\"input_shape\":[";
  const auto& shape = engine.GetInputTensor()->GetShape();
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i) std::cout << ',';
    std::cout << shape[i];
  }
  std::cout << "],\"dtype\":\"float32\""
            << ",\"timer\":\"GPU timestamps, complete graph, summed submissions\""
            << ",\"submissions_per_run\":" << result.submissions_per_run
            << ",\"warmup_runs\":" << warmup << ",\"measured_runs\":" << runs
            << ",\"timestamp_valid_bits\":" << result.timestamp_valid_bits
            << ",\"timestamp_period_ns\":" << result.timestamp_period_ns
            << ",\"mean_ms\":" << std::accumulate(sorted.begin(), sorted.end(), 0.0) / sorted.size()
            << ",\"median_ms\":" << median << ",\"min_ms\":" << sorted.front()
            << ",\"max_ms\":" << sorted.back() << ",\"max_abs_error\":" << comparison.max_abs_error
            << ",\"samples_ms\":[";
  for (size_t i = 0; i < result.gpu_ms.size(); ++i) {
    if (i) std::cout << ',';
    std::cout << result.gpu_ms[i];
  }
  std::cout << "]}\n";
}

}  // namespace model_test
