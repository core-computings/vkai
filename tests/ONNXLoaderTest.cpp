#include <limits>

#include "OnnxTestModel.h"
#include "utils/ONNXLoader.h"

namespace vkai::test {
TEST(ONNXLoaderTest, LoadsGeneratedGraphAttributesParametersAndConnections) {
  TestModel model({2, 1, 2, 3}, {2, 3});
  MakeTestGraph(model);
  Graph graph = BuildGraphFromONNX(model.Save());
  auto input = graph.FindTensor("input");
  ASSERT_NE(input, nullptr);
  EXPECT_EQ(input->GetShape(), (std::vector<int64_t>{2, 1, 2, 3}));
  auto weights = graph.FindTensor("weights");
  ASSERT_NE(weights, nullptr);
  EXPECT_TRUE(weights->HasData());
  EXPECT_EQ(weights->GetShape(), (std::vector<int64_t>{3, 6}));
  EXPECT_EQ(weights->Data().size(), 18U);
  auto dense = graph.FindOperation("Gemm3");
  ASSERT_NE(dense, nullptr);
  EXPECT_EQ(dense->GetAttribute<int>("input_size"), 6);
  EXPECT_EQ(dense->GetAttribute<int>("output_size"), 3);
  EXPECT_EQ(dense->GetAttribute<int>("batch_size"), 2);
  EXPECT_FLOAT_EQ(dense->GetAttribute<float>("alpha"), 1.F);
  EXPECT_FLOAT_EQ(dense->GetAttribute<float>("beta"), 1.F);
  EXPECT_EQ(dense->GetAttribute<int64_t>("transB"), 1);
  EXPECT_EQ(dense->GetInput("input"), graph.FindTensor("flat"));
  EXPECT_EQ(dense->GetInput("weights"), weights);
  EXPECT_EQ(dense->GetInput("bias"), graph.FindTensor("bias"));
  auto relu = graph.FindOperation("Relu4");
  ASSERT_NE(relu, nullptr);
  EXPECT_EQ(relu->GetAttribute<int>("element_count"), 6);
  EXPECT_EQ(relu->GetInput("input"), graph.FindTensor("dense"));
  EXPECT_FALSE(relu->HasInput("weights"));
  auto reshape = graph.FindOperation("Reshape2");
  ASSERT_NE(reshape, nullptr);
  EXPECT_EQ(reshape->GetInput("input"), input);
  EXPECT_EQ(reshape->GetInput("shape"), graph.FindTensor("shape"));
  ASSERT_EQ(graph.Operations().size(), 4U);
  const auto order = graph.TopologicalSort();
  ASSERT_EQ(order.size(), 4U);
  EXPECT_EQ(order[0]->Name(), "Constant1");
  EXPECT_EQ(order[1]->Name(), "Reshape2");
  EXPECT_EQ(order[2]->Name(), "Gemm3");
  EXPECT_EQ(order[3]->Name(), "Relu4");
}
}  // namespace vkai::test

namespace vkai::test {
namespace {

// A model bundle lives beside, rather than in, the process working directory.
// All files are generated locally so the same cases run on Android unchanged.
class ExternalBundle {
 public:
  TestModel model;
  std::filesystem::path directory;

  ExternalBundle(std::vector<int64_t> input = {2, 2}, std::vector<int64_t> output = {2, 2})
      : model(input, output), directory(model.Save() + ".bundle") {
    std::filesystem::create_directories(directory / "data");
  }
  ~ExternalBundle() { std::filesystem::remove_all(directory); }

  static void Metadata(onnx::TensorProto& tensor, const std::string& key,
                       const std::string& value) {
    for (auto& entry : *tensor.mutable_external_data()) {
      if (entry.key() == key) {
        entry.set_value(value);
        return;
      }
    }
    auto* entry = tensor.add_external_data();
    entry->set_key(key);
    entry->set_value(value);
  }

  template <typename T>
  void Store(onnx::TensorProto& tensor, const std::vector<T>& data,
             const std::string& file = "data/weights.bin", bool range = true) {
    const auto path = directory / file;
    const auto offset = std::filesystem::exists(path) ? std::filesystem::file_size(path) : 0;
    std::ofstream stream(path, std::ios::binary | std::ios::app);
    stream.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(T));
    tensor.clear_float_data();
    tensor.clear_int64_data();
    tensor.clear_raw_data();
    tensor.set_data_location(onnx::TensorProto::EXTERNAL);
    Metadata(tensor, "location", file);
    if (range) {
      Metadata(tensor, "offset", std::to_string(offset));
      Metadata(tensor, "length", std::to_string(data.size() * sizeof(T)));
    }
  }

  onnx::TensorProto& Weights() { return *model.Proto().mutable_graph()->mutable_initializer(0); }

  void Dense(const std::string& input = "input") {
    model.Weight("w", {2, 2}, {1, 2, 3, 4});
    model.Weight("b", {2}, {.5F, -.5F});
    auto* node = model.Node("Gemm", {input, "w", "b"});
    TestModel::Int(node, "transB", 1);
    // Nonzero, unaligned offsets and multiple tensors in one shared file.
    std::ofstream(directory / "data/weights.bin", std::ios::binary) << "a bundle header!!";
    Store(Weights(), std::vector<float>{1, 2, 3, 4});
    Store(*model.Proto().mutable_graph()->mutable_initializer(1), std::vector<float>{.5F, -.5F});
  }

  std::string Save() {
    const auto path = directory / "model.onnx";
    std::ofstream stream(path, std::ios::binary);
    if (!model.Proto().SerializeToOstream(&stream)) throw std::runtime_error("Cannot save bundle");
    return path.string();
  }

  void Check(const std::vector<float>& input, const std::vector<float>& reference) {
    Engine engine(Save());
    for (int iteration = 0; iteration < 2; ++iteration) {
      auto values = input;
      for (auto& value : values) value *= iteration + 1;
      engine.GetInputTensor()->PopulateTensor(values);
      engine.ExecuteGraph();
      const auto& actual = engine.GetOutputTensor()->Data();
      ASSERT_EQ(actual.size(), reference.size());
      for (size_t i = 0; i < actual.size(); ++i) {
        // All callers use this helper with a homogeneous linear transform.
        EXPECT_NEAR(actual[i], reference[i] * (iteration + 1), 1e-5F) << i;
      }
    }
  }
};

TEST(ONNXLoaderTest, ExecutesSharedExternalWeightsAndBiasWithChangingInputs) {
  ExternalBundle bundle;
  bundle.Dense();
  // External data wins even if a producer left a stale embedded payload.
  bundle.Weights().set_raw_data(std::string(16, '\0'));
  Engine engine(bundle.Save());
  for (int scale : {1, 2}) {
    engine.GetInputTensor()->PopulateTensor(
        {float(scale), float(2 * scale), float(3 * scale), float(4 * scale)});
    engine.ExecuteGraph();
    EXPECT_EQ(engine.GetOutputTensor()->Data(),
              (std::vector<float>{5.F * scale + .5F, 11.F * scale - .5F, 11.F * scale + .5F,
                                  25.F * scale - .5F}));
  }
}

TEST(ONNXLoaderTest, LoadsExternalInt64ShapeBeforeInference) {
  ExternalBundle bundle({1, 1, 2, 2});
  bundle.model.IntWeight("shape", {2}, {2, 2});
  bundle.Store(bundle.Weights(), std::vector<int64_t>{2, 2}, "shape.bin", false);
  bundle.model.Node("Reshape", {"input", "shape"});
  const auto graph = BuildGraphFromONNX(bundle.Save());
  EXPECT_EQ(graph.FindTensor("shape")->Int64Data(), (std::vector<int64_t>{2, 2}));
  EXPECT_EQ(graph.Outputs().front()->GetShape(), (std::vector<int64_t>{2, 2}));
  bundle.Check({1, 2, 3, 4}, {1, 2, 3, 4});
}

TEST(ONNXLoaderTest, ExecutesExternalTensorAttributesAndScalarWithDefaultRange) {
  ExternalBundle bundle;
  auto* constant = bundle.model.Node("Constant", {}, "factor");
  auto* attribute = constant->add_attribute();
  attribute->set_name("value");
  attribute->set_type(onnx::AttributeProto::TENSOR);
  attribute->mutable_t()->set_data_type(onnx::TensorProto::FLOAT);
  bundle.Store(*attribute->mutable_t(), std::vector<float>{2}, "factor.bin", false);
  bundle.model.Node("Mul", {"input", "factor"});
  bundle.Check({1, -2, 3, -4}, {2, -4, 6, -8});

  ExternalBundle shape_bundle({1, 1, 2, 2});
  constant = shape_bundle.model.Node("Constant", {}, "shape");
  attribute = constant->add_attribute();
  attribute->set_name("value");
  attribute->set_type(onnx::AttributeProto::TENSOR);
  auto* shape = attribute->mutable_t();
  shape->set_data_type(onnx::TensorProto::INT64);
  shape->add_dims(2);
  shape_bundle.Store(*shape, std::vector<int64_t>{2, 2});
  shape_bundle.model.Node("Reshape", {"input", "shape"});
  shape_bundle.Check({1, 2, 3, 4}, {1, 2, 3, 4});
}

TEST(ONNXLoaderTest, RejectsMissingTruncatedAndMismatchedExternalFiles) {
  for (const std::string failure : {"missing", "truncated", "length", "offset", "dimensions"}) {
    SCOPED_TRACE(failure);
    ExternalBundle bundle;
    bundle.Dense();
    if (failure == "missing") std::filesystem::remove(bundle.directory / "data/weights.bin");
    if (failure == "truncated")
      std::filesystem::resize_file(bundle.directory / "data/weights.bin", 18);
    if (failure == "length") ExternalBundle::Metadata(bundle.Weights(), "length", "4");
    if (failure == "offset") ExternalBundle::Metadata(bundle.Weights(), "offset", "9999");
    if (failure == "dimensions") bundle.Weights().set_dims(0, std::numeric_limits<int64_t>::max());
    EXPECT_THROW(BuildGraphFromONNX(bundle.Save()), std::runtime_error);
  }
}

TEST(ONNXLoaderTest, RejectsMalformedAndDuplicateExternalRanges) {
  for (const std::string key : {"offset", "length"}) {
    for (const std::string value : {"", "-1", "+1", "1junk", " 1", "18446744073709551616"}) {
      SCOPED_TRACE(key + "=" + value);
      ExternalBundle bundle;
      bundle.Dense();
      ExternalBundle::Metadata(bundle.Weights(), key, value);
      EXPECT_THROW(BuildGraphFromONNX(bundle.Save()), std::runtime_error);
    }
  }
  ExternalBundle bundle;
  bundle.Dense();
  auto* duplicate = bundle.Weights().add_external_data();
  duplicate->set_key("offset");
  duplicate->set_value("17");
  EXPECT_THROW(BuildGraphFromONNX(bundle.Save()), std::runtime_error);
}

TEST(ONNXLoaderTest, RejectsExternalLocationsOutsideBundle) {
  for (const std::string& location :
       std::vector<std::string>{"", "/tmp/weights.bin", "../weights.bin", "data/../../weights.bin",
                                std::string("data/weights.bin\0extra", 22)}) {
    SCOPED_TRACE(location);
    ExternalBundle bundle;
    bundle.Dense();
    ExternalBundle::Metadata(bundle.Weights(), "location", location);
    EXPECT_THROW(BuildGraphFromONNX(bundle.Save()), std::runtime_error);
  }
  ExternalBundle bundle;
  bundle.Dense();
  // A valid file outside the bundle must not be followed through a symlink.
  std::filesystem::create_symlink(bundle.model.Save(), bundle.directory / "escape.bin");
  ExternalBundle::Metadata(bundle.Weights(), "location", "escape.bin");
  EXPECT_THROW(BuildGraphFromONNX(bundle.Save()), std::runtime_error);
}

TEST(ONNXLoaderTest, RejectsUnsupportedExternalTensorType) {
  ExternalBundle bundle;
  bundle.Dense();
  bundle.Weights().set_data_type(onnx::TensorProto::DOUBLE);
  EXPECT_THROW(BuildGraphFromONNX(bundle.Save()), std::runtime_error);
}

}  // namespace
}  // namespace vkai::test
