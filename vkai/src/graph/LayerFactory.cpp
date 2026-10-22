#include "graph/LayerFactory.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

#include "layers/AdaptiveAvgPool2D.h"
#include "layers/BatchNorm2D.h"
#include "layers/BilinearResize2D.h"
#include "layers/BinaryElementwise.h"
#include "layers/Clip.h"
#include "layers/Conv2D.h"
#include "layers/ConvTranspose2D.h"
#include "layers/HardSigmoid.h"
#include "layers/Linear.h"
#include "layers/MaxPool2D.h"
#include "layers/NearestResize2D.h"
#include "layers/Relu.h"
#include "layers/Reshape.h"
#include "layers/Sigmoid.h"
#include "layers/Softmax.h"
#include "layers/TensorConcat.h"
#include "layers/TensorTransform.h"
#include "utils/ShapeUtils.h"
#include "utils/TensorUtils.h"

namespace vkai {
namespace {

template <typename T>
T Attribute(const Operation& op, const std::string& key, T fallback) {
  const auto* value = op.FindAttribute<T>(key);
  return value ? *value : fallback;
}

void Require(bool condition, const Operation& op, const std::string& message) {
  if (!condition) throw std::runtime_error(op.Name() + ": " + message);
}

std::vector<int> Shape(const Tensor& tensor) {
  std::vector<int> result;
  for (size_t i = 0; i < tensor.GetShape().size(); ++i) {
    const auto d = tensor.GetShape()[i];
    result.push_back(i == 0 && d == 0 ? 1 : ToLayerDimension(d, tensor.Name()));
  }
  return result;
}

std::vector<int64_t> Pair(const Operation& op, const std::string& name, int64_t fallback) {
  auto value = Attribute(op, name, std::vector<int64_t>{fallback, fallback});
  Require(value.size() == 2, op, name + " must have two entries");
  for (const auto d : value) Require(d >= 0 && d <= 65535, op, "Invalid " + name);
  return value;
}

std::vector<int64_t> Padding(const Operation& op) {
  const auto p = Attribute(op, "pads", std::vector<int64_t>{0, 0, 0, 0});
  Require(p.size() == 4 && p[0] == p[2] && p[1] == p[3], op,
          "Only symmetric 2D padding is supported");
  for (auto d : p) Require(d >= 0 && d <= 65535, op, "Invalid padding");
  Require(Attribute(op, "auto_pad", std::string("NOTSET")) == "NOTSET", op,
          "auto_pad is unsupported; export explicit pads");
  return p;
}

std::vector<float> Bias(const Operation& op) {
  return op.HasInput("bias") ? op.GetInput("bias")->Data() : std::vector<float>{};
}

}  // namespace

std::unique_ptr<Layer> CreateLayer(core::vulkan::VulkanContext* context, const Operation& op) {
  if (op.Type() == OpType::Constant) return nullptr;
  Require(op.Outputs().size() == 1, op, "Exactly one output is required");
  const auto input = op.GetInput("input");
  const auto s = Shape(*input);
  const auto out = Shape(*op.Outputs().front());
  std::unique_ptr<Layer> layer;
  switch (op.Type()) {
    case OpType::Reshape:
      Require(TensorElementCount(*input) == TensorElementCount(*op.Outputs().front()), op,
              "Reshape must preserve the element count");
      layer = std::make_unique<Reshape>(context, input->ByteSize());
      break;
    case OpType::Dense: {
      Require(s.size() == 2 && out.size() == 2, op, "Gemm requires rank 2");
      Require(Attribute<int64_t>(op, "transA", 0) == 0 &&
                  Attribute<int64_t>(op, "transB", 0) == 1 && Attribute(op, "alpha", 1.0F) == 1 &&
                  Attribute(op, "beta", 1.0F) == 1,
              op, "Gemm requires transA=0, transB=1, alpha=beta=1");
      Require(Shape(*op.GetInput("weights")) == std::vector<int>({out[1], s[1]}) && out[0] == s[0],
              op, "Gemm shape mismatch");
      auto linear = std::make_unique<Linear>(context, s[1], out[1], s[0]);
      auto bias = Bias(op);
      if (bias.empty()) bias.resize(out[1], 0.0F);
      Require(bias.size() == static_cast<size_t>(out[1]), op, "Gemm bias shape mismatch");
      linear->MapWeights(op.GetInput("weights")->Data(), bias);
      layer = std::move(linear);
      break;
    }
    case OpType::Conv2D:
    case OpType::ConvTranspose2D: {
      Require(s.size() == 4 && out.size() == 4, op, "Convolution requires NCHW");
      const auto w = Shape(*op.GetInput("weights"));
      Require(w.size() == 4, op, "Convolution weights must have rank 4");
      const auto groups = Attribute<int64_t>(op, "group", 1);
      Require(groups > 0 && groups <= s[1] && s[1] % groups == 0 && out[1] % groups == 0, op,
              "Invalid convolution groups");
      auto k = Pair(op, "kernel_shape", 0);
      if (!op.HasAttribute("kernel_shape")) k = {w[2], w[3]};
      Require(k[0] == w[2] && k[1] == w[3], op, "Kernel shape disagrees with weights");
      const auto strides = Pair(op, "strides", 1);
      const auto dilations = Pair(op, "dilations", 1);
      Require(strides[0] > 0 && strides[1] > 0 && dilations[0] > 0 && dilations[1] > 0, op,
              "Strides and dilations must be positive");
      const auto p = Padding(op);
      const bool transpose = op.Type() == OpType::ConvTranspose2D;
      Require(
          (transpose ? s[1] == w[0] && out[1] == w[1] : s[1] / groups == w[1] && out[1] == w[0]) &&
              out[0] == s[0],
          op, "Convolution channel or batch mismatch");
      if (transpose) {
        Require(groups == 1, op, "Grouped transposed convolution is unsupported");
        Require(!op.HasAttribute("output_shape"), op, "ConvTranspose output_shape is unsupported");
        const auto extra = Pair(op, "output_padding", 0);
        auto conv = std::make_unique<ConvTranspose2D>(
            context, s[1], out[1], s[2], s[3], k[0], k[1], strides[0], strides[1], p[0], p[1], s[0],
            dilations[0], dilations[1], extra[0], extra[1]);
        Require(out[2] == conv->OutputHeight() && out[3] == conv->OutputWidth(), op,
                "ConvTranspose output shape mismatch");
        conv->MapWeights(op.GetInput("weights")->Data(), Bias(op));
        layer = std::move(conv);
      } else {
        auto conv = std::make_unique<Conv2D>(context, s[1], out[1], s[2], s[3], k[0], k[1],
                                             strides[0], strides[1], p[0], p[1], PaddingType::Zero,
                                             s[0], dilations[0], dilations[1], groups);
        Require(out[2] == conv->OutputHeight() && out[3] == conv->OutputWidth(), op,
                "Conv output shape mismatch");
        conv->MapWeights(op.GetInput("weights")->Data(), Bias(op));
        layer = std::move(conv);
      }
      break;
    }
    case OpType::Relu:
      Require(s == out, op, "Relu shape mismatch");
      layer = std::make_unique<Relu>(context, TensorElementCount(*input));
      break;
    case OpType::Sigmoid:
      Require(s == out, op, "Sigmoid shape mismatch");
      layer = std::make_unique<Sigmoid>(context, TensorElementCount(*input));
      break;
    case OpType::HardSigmoid:
      Require(s == out, op, "HardSigmoid shape mismatch");
      layer =
          std::make_unique<HardSigmoid>(context, TensorElementCount(*input),
                                        Attribute(op, "alpha", .2F), Attribute(op, "beta", .5F));
      break;
    case OpType::MaxPool2D: {
      Require(s.size() == 4 && out.size() == 4, op, "MaxPool requires NCHW");
      Require(Attribute<int64_t>(op, "ceil_mode", 0) == 0 &&
                  Pair(op, "dilations", 1) == std::vector<int64_t>({1, 1}) &&
                  Attribute<int64_t>(op, "storage_order", 0) == 0,
              op, "MaxPool ceil mode, dilation and storage order are unsupported");
      const auto k = Pair(op, "kernel_shape", 0);
      const auto st = Pair(op, "strides", 1);
      const auto p = Padding(op);
      Require(k[0] > 0 && k[1] > 0 && st[0] > 0 && st[1] > 0, op, "Invalid pooling dimensions");
      auto pool = std::make_unique<MaxPool2D>(context, s[1], s[2], s[3], k[0], k[1], st[0], st[1],
                                              s[0], p[0], p[1]);
      Require(out == std::vector<int>({s[0], s[1], pool->OutputHeight(), pool->OutputWidth()}), op,
              "MaxPool output shape mismatch");
      layer = std::move(pool);
      break;
    }
    case OpType::AdaptiveAvgPool2D: {
      Require(s.size() == 4, op, "Spatial average requires NCHW");
      const bool reduce = Attribute(op, "onnx_type", std::string{}) == "ReduceMean";
      const bool keep = !reduce || Attribute<int64_t>(op, "keepdims", 1) == 1;
      if (reduce) {
        const auto axes = Attribute(op, "axes", std::vector<int64_t>{});
        Require(axes == std::vector<int64_t>({2, 3}) || axes == std::vector<int64_t>({-2, -1}), op,
                "ReduceMean supports the two NCHW spatial axes only");
      }
      Require(out == (keep ? std::vector<int>{s[0], s[1], 1, 1} : std::vector<int>{s[0], s[1]}), op,
              "Spatial average output shape mismatch");
      layer = std::make_unique<AdaptiveAvgPool2D>(context, s[1], s[2], s[3], 1, 1, s[0]);
      break;
    }
    case OpType::Clip: {
      Require(s == out, op, "Clip shape mismatch");
      auto bound = [&](const std::string& role, float fallback) {
        if (!op.HasInput(role)) return fallback;
        const auto tensor = op.GetInput(role);
        Require(tensor->HasData() && tensor->Data().size() == 1, op,
                "Clip bounds must be constant float scalars");
        return tensor->Data()[0];
      };
      layer = std::make_unique<Clip>(
          context, TensorElementCount(*input),
          bound("min", Attribute(op, "min", -std::numeric_limits<float>::max())),
          bound("max", Attribute(op, "max", std::numeric_limits<float>::max())));
      break;
    }
    case OpType::BatchNorm2D: {
      Require(s.size() == 4 && s == out, op, "BatchNormalization requires matching NCHW shapes");
      Require(Attribute<int64_t>(op, "training_mode", 0) == 0, op,
              "Training BatchNorm is unsupported");
      auto bn = std::make_unique<BatchNorm2D>(context, s[1], s[2] * s[3], s[0],
                                              Attribute(op, "epsilon", 1e-5F));
      bn->MapParameters(op.GetInput("mean")->Data(), op.GetInput("variance")->Data(),
                        op.GetInput("scale")->Data(), op.GetInput("bias")->Data());
      layer = std::move(bn);
      break;
    }
    case OpType::Add:
    case OpType::Sub:
    case OpType::Mul:
    case OpType::Div: {
      auto rhs = Shape(*op.GetInput(op.Type() == OpType::Add ? "addend" : "rhs"));
      Require(out == BroadcastShape(s, rhs), op, "Binary output shape mismatch");
      BinaryKind kind = op.Type() == OpType::Add   ? BinaryKind::Add
                        : op.Type() == OpType::Sub ? BinaryKind::Sub
                        : op.Type() == OpType::Mul ? BinaryKind::Mul
                                                   : BinaryKind::Div;
      layer = std::make_unique<BinaryElementwise>(context, s, rhs, kind);
      break;
    }
    case OpType::ChannelConcat: {
      const int rank = s.size();
      int64_t axis = Attribute<int64_t>(op, "axis", 0);
      if (axis < 0) axis += rank;
      Require(axis >= 0 && axis < rank, op, "Invalid Concat axis");
      const int count = op.GetAttribute<int>("input_count");
      Require(count >= 1, op, "Concat needs inputs");
      std::vector<std::vector<int>> shapes;
      auto expected = s;
      int64_t total = 0;
      for (int i = 0; i < count; ++i) {
        auto shape = Shape(*op.GetInput(i == 0 ? "input" : "input_" + std::to_string(i)));
        Require(shape.size() == s.size(), op, "Concat rank mismatch");
        total += shape[axis];
        shapes.push_back(shape);
      }
      Require(total <= std::numeric_limits<int>::max(), op, "Concat is too large");
      expected[axis] = total;
      Require(out == expected, op, "Concat output shape mismatch");
      layer = std::make_unique<TensorConcat>(context, shapes, axis);
      break;
    }
    case OpType::Transpose: {
      auto perm = Attribute(op, "perm", std::vector<int64_t>{});
      if (!op.HasAttribute("perm")) {
        perm.resize(s.size());
        for (size_t i = 0; i < s.size(); ++i) perm[i] = s.size() - 1 - i;
      }
      Require(perm.size() == s.size(), op, "Transpose permutation rank mismatch");
      auto strides = ContiguousStrides(s);
      std::vector<int> mapped, expected;
      std::vector<bool> used(s.size(), false);
      for (auto axis : perm) {
        Require(axis >= 0 && axis < static_cast<int64_t>(s.size()) && !used[axis], op,
                "Invalid Transpose permutation");
        used[axis] = true;
        mapped.push_back(strides[axis]);
        expected.push_back(s[axis]);
      }
      Require(out == expected, op, "Transpose output mismatch");
      layer = std::make_unique<TensorTransform>(context, ElementCount(s), out, mapped);
      break;
    }
    case OpType::Slice: {
      auto ints = [&](const std::string& role) {
        const auto tensor = op.GetInput(role);
        Require(tensor->HasInt64Data(), op, "Slice parameters must be constant int64 tensors");
        return tensor->Int64Data();
      };
      std::vector<int64_t> starts, ends, axes, steps;
      if (op.HasAttribute("slice_axis")) {
        axes = {op.GetAttribute<int64_t>("slice_axis")};
        starts = {op.GetAttribute<int64_t>("slice_start")};
        ends = {starts[0] + op.GetAttribute<int64_t>("slice_extent")};
        steps = {1};
      } else {
        starts = ints("starts");
        ends = ints("ends");
        axes = op.HasInput("axes") ? ints("axes") : std::vector<int64_t>(starts.size());
        if (!op.HasInput("axes")) std::iota(axes.begin(), axes.end(), 0);
        steps = op.HasInput("steps") ? ints("steps") : std::vector<int64_t>(starts.size(), 1);
      }
      Require(starts.size() == ends.size() && starts.size() == axes.size() &&
                  starts.size() == steps.size(),
              op, "Slice parameter lengths differ");
      auto expected = s, strides = ContiguousStrides(s);
      int64_t offset = 0;
      std::vector<bool> used(s.size(), false);
      for (size_t i = 0; i < axes.size(); ++i) {
        int64_t axis = axes[i];
        if (axis < 0) axis += s.size();
        Require(axis >= 0 && axis < static_cast<int64_t>(s.size()) && !used[axis] && steps[i] != 0,
                op, "Invalid Slice axes or step");
        used[axis] = true;
        int64_t start = starts[i], end = ends[i], extent = s[axis], step = steps[i];
        if (start < 0) start += extent;
        if (end < 0) end += extent;
        start = std::clamp<int64_t>(start, step > 0 ? 0 : -1, step > 0 ? extent : extent - 1);
        end = std::clamp<int64_t>(end, step > 0 ? 0 : -1, step > 0 ? extent : extent - 1);
        Require(step != std::numeric_limits<int64_t>::min(), op, "Slice step is too large");
        int64_t length = step > 0 ? (end > start ? (end - start - 1) / step + 1 : 0)
                                  : (start > end ? (start - end - 1) / (-step) + 1 : 0);
        Require(length > 0, op, "Empty Slice outputs are unsupported");
        offset += start * strides[axis];
        expected[axis] = length;
        Require(step >= std::numeric_limits<int>::min() / strides[axis] &&
                    step <= std::numeric_limits<int>::max() / strides[axis],
                op, "Slice stride too large");
        const int64_t stride = strides[axis] * step;
        Require(
            stride >= std::numeric_limits<int>::min() && stride <= std::numeric_limits<int>::max(),
            op, "Slice stride too large");
        strides[axis] = stride;
      }
      Require(expected == out, op, "Slice output shape mismatch");
      layer = std::make_unique<TensorTransform>(context, ElementCount(s), out, strides, offset);
      break;
    }
    case OpType::NearestResize2D:
    case OpType::BilinearResize2D: {
      Require(s.size() == 4 && out.size() == 4 && s[0] == out[0] && s[1] == out[1], op,
              "Resize requires matching NCHW batch/channels");
      const bool linear = op.Type() == OpType::BilinearResize2D;
      if (linear) {
        Require(Attribute(op, "mode", std::string("nearest")) == "linear" &&
                    Attribute(op, "coordinate_transformation_mode", std::string("half_pixel")) ==
                        "half_pixel",
                op, "Linear Resize supports half_pixel coordinates only");
      } else {
        Require(Attribute(op, "mode", std::string("nearest")) == "nearest" &&
                    Attribute(op, "coordinate_transformation_mode", std::string("half_pixel")) ==
                        "asymmetric" &&
                    Attribute(op, "nearest_mode", std::string("round_prefer_floor")) == "floor",
                op, "Resize supports nearest/asymmetric/floor");
      }
      Require(Attribute<int64_t>(op, "exclude_outside", 0) == 0, op,
              "Resize exclude_outside unsupported");
      if (op.HasInput("scales") && !op.GetInput("scales")->Data().empty()) {
        const auto& scales = op.GetInput("scales")->Data();
        Require(scales.size() == 4 && scales[0] == 1 && scales[1] == 1 && scales[2] > 0 &&
                    scales[3] > 0 && std::floor(s[2] * scales[2]) == out[2] &&
                    std::floor(s[3] * scales[3]) == out[3],
                op, "Resize scale mismatch");
        // Integer ratios use the same exact floor coordinate calculation in the shader.
        Require(scales[2] == float(out[2]) / s[2] && scales[3] == float(out[3]) / s[3], op,
                "Resize needs exact output/input scale ratios");
      } else {
        Require(op.HasInput("sizes") && op.GetInput("sizes")->HasInt64Data(), op,
                "Resize needs constant scales or sizes");
        const auto& sizes = op.GetInput("sizes")->Int64Data();
        Require(sizes == std::vector<int64_t>(out.begin(), out.end()), op, "Resize sizes mismatch");
      }
      if (linear)
        layer = std::make_unique<BilinearResize2D>(context, s[1], s[2], s[3], out[2], out[3], s[0]);
      else
        layer = std::make_unique<NearestResize2D>(context, s[1], s[2], s[3], out[2], out[3], s[0]);
      break;
    }
    case OpType::Softmax: {
      const auto axis = Attribute<int64_t>(op, "axis", -1);
      Require(s == out, op, "Softmax shape mismatch");
      if (s.size() == 2 && (axis == 1 || axis == -1)) {
        layer = std::make_unique<Softmax>(context, s[1], s[0]);
      } else {
        Require(s.size() == 4 && (axis == 1 || axis == -3), op, "Unsupported Softmax axis");
        layer =
            std::make_unique<Softmax>(context, s[1], s[2], s[3], s[0], SoftmaxMode::kChannelNCHW);
      }
      break;
    }
    default:
      throw std::runtime_error("Layer is not configured: " + op.Name());
  }
  layer->Init();
  return layer;
}

void RecordLayer(Layer& layer, const Operation& op, VkCommandBuffer command) {
  auto& output = op.Outputs().front()->Buffer();
  const auto& input = op.GetInput("input")->Buffer();
  if (op.Type() == OpType::Add || op.Type() == OpType::Mul || op.Type() == OpType::Sub ||
      op.Type() == OpType::Div) {
    static_cast<BinaryElementwise&>(layer).Execute(
        command, input, op.GetInput(op.Type() == OpType::Add ? "addend" : "rhs")->Buffer(), output);
  } else if (op.Type() == OpType::ChannelConcat) {
    std::vector<const core::vulkan::VulkanBuffer*> inputs{&input};
    for (int i = 1; i < op.GetAttribute<int>("input_count"); ++i) {
      inputs.push_back(&op.GetInput("input_" + std::to_string(i))->Buffer());
    }
    static_cast<TensorConcat&>(layer).Execute(command, inputs, output);
  } else {
    layer.Execute(command, input, output);
  }
}

}  // namespace vkai
