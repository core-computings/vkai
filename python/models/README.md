# 四个预训练模型的 macOS / Android 验证

当前模型测试仅保留 ResNet-18、U-Net、MobileNetV2 和 YOLOv8n，均通过 `ONNX → Graph → LayerFactory → Vulkan` 执行。安卓端运行 C++，主机 Python 负责模型导出、参考结果比较和图像绘制。

| 模型 | 权重与输入 | 输出 |
| --- | --- | --- |
| ResNet-18 | [Torchvision IMAGENET1K_V1](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html)，官方狗图，224×224，ImageNet normalize | 1000 类 logits、top-5 |
| U-Net | [mateuszbuda brain MRI](https://pytorch.org/hub/mateuszbuda_brain-segmentation-pytorch_unet/)，256×256 MRI，各通道 z-score | 概率图、阈值 0.5 的 mask 和 overlay |
| MobileNetV2 | Torchvision IMAGENET1K_V2，官方狗图，224×224 | 1000 类 logits、top-5 |
| YOLOv8n | Ultralytics `yolov8n.pt`，公交车图，letterbox 到 640×640 | `[1,84,8400]` 检测头输出、C++ NMS 检测框和标注图 |

U-Net 上游固定在 `d45f8908ab2f0246ba204c702a6161c9eb25f902`，使用 `unet-e012d006.pt` 权重。YOLOv8n 使用 Ultralytics 8.3.200；导出时融合 Conv / BN，并用 onnxslim 折叠固定输入尺寸的形状常量。检测后处理在 C++ 中执行，置信度阈值为 0.25，同类框 NMS 的 IoU 阈值为 0.45，最多保留 300 个框。

## 准备模型

需要 CMake ≥3.26、C++20 编译器、Vulkan SDK（含 MoltenVK / glslc）及 Python 3.11。以下命令在仓库根目录执行：

```bash
git submodule update --init --recursive
# 如 SDK 尚未加入环境，先 source 本机 VulkanSDK/<版本>/setup-env.sh
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r python/models/requirements-yolo.txt
# 分别准备四个模型及其输入、参考结果：
.venv/bin/python python/models/export_resnet18.py
.venv/bin/python python/models/export_unet.py
.venv/bin/python python/models/export_mobilenet_v2.py
# 导出 YOLOv8n：
.venv/bin/python python/models/export_yolov8n.py
```

首次导出需要网络下载权重和输入图。`build.sh -r models` 使用已经导出的文件，不自动下载或导出。`artifacts/` 下缓存 ONNX、输入图、float32 输入/参考张量和 metadata；这些生成文件被 Git 忽略，两个平台共享同一份输入和参考结果。

每个脚本只准备对应的模型，默认输出到 `artifacts/<模型名>/`。可用 `--output-dir /path/to/model-folder` 指定该模型的输出目录。通用下载、ONNX 导出和参考结果校验位于 `export_common.py`。

导出时比较 PyTorch 与 ONNX Runtime；执行时逐元素比较 Vulkan 与 PyTorch，容差为 `atol=rtol=1e-3`。分类要求 top-5 索引一致；U-Net 要求前景/背景均存在且与 PyTorch mask 的 Dice ≥0.999；YOLO 要求类别、框数和参考一致，框 IoU >0.999。失败会返回非零退出码。

## 统一编译和测试入口

```bash
./scripts/build.sh -t macos -r unittests
./scripts/build.sh -t macos -r models
./scripts/build.sh -t android -r unittests
./scripts/build.sh -t android -r models
```

默认是 `-t macos -r unittests`，两个选项顺序不限。`-r 0` 只编译全部目标，不运行测试，也不需要安卓设备；`unittests` 运行全部通用算子、图执行和加载器测试；`models` 运行上述四个模型，并对 MobileNetV2 / YOLOv8n 做 GPU 计时。

全部单元测试均无需 Python 导出数据或下载模型：C++ 生成确定性输入与 CPU 参考结果，小型 ONNX 测试图在临时目录创建并清理，FlatBuffer 加载器的微型权重由 CMake 从 `assets/weights.json` 自动生成。C++ 测试执行不调用 Python；构建 ONNX 第三方依赖时仍需主机 Python 解释器做代码生成。

模型测试仍需主机 Python 做数值校验及图像绘制。结果位于 `artifacts/{macos,android}_validation.json`、`unet/{macos,android}_{mask,overlay}.png` 和 `yolov8n/{macos,android}_{detections.png,validation.json}`。

## Android arm64 真机

需要 NDK、adb 在线设备以及主机 Vulkan SDK / glslc。最低 API 为 28；设备还需满足 CORE 的 Vulkan 实例和 dynamic rendering 要求。已验证 Android 13（API 33）/ Adreno 740。

```bash
export ANDROID_NDK_ROOT="$HOME/Library/Android/sdk/ndk/28.2.13676358"
adb devices -l
./scripts/build.sh -t android -r unittests
./scripts/build.sh -t android -r models
```

脚本先构建主机 `flatc` / `protoc`，再交叉编译 arm64；shader 由主机 glslc 生成并嵌入程序，C++ runtime 静态链接。adb 自动上传所选测试组需要的程序和数据到 `/data/local/tmp/vkai` 并执行。多设备时设置标准 `ANDROID_SERIAL` 环境变量。

构建环境变量：`VKAI_JOBS`、`VKAI_BUILD_DIR`（macOS 输出）、`VKAI_HOST_BUILD_DIR`（主机生成器）、`VKAI_ANDROID_BUILD_DIR`（Android 输出）、`VKAI_PYTHON`（模型校验解释器）。`VKAI_TEST_BINARY_DIR` 指定单测在目标平台读取 CMake 生成权重的目录。

## 单独复跑

已构建后，可直接调用通用 C++ 入口：

```bash
build/tasks/vkai_infer \
  python/models/artifacts/resnet18/model.onnx \
  python/models/artifacts/resnet18/input.bin \
  build/resnet18_output.bin \
  python/models/artifacts/resnet18/reference.bin
```

参考文件参数可省略；张量采用连续、小端 float32。仅复跑部分模型或重新检查已有输出：

```bash
.venv/bin/python python/models/validate_models.py --models resnet18 unet mobilenet_v2
.venv/bin/python python/models/validate_yolov8n.py
.venv/bin/python python/models/validate_models.py --platform android \
  --runner build-android/tasks/vkai_infer --models resnet18 unet mobilenet_v2
.venv/bin/python python/models/validate_models.py --verify-only --models resnet18 unet mobilenet_v2
ctest --test-dir build --output-on-failure
```

## 纯网络 GPU inference 计时

`BenchmarkGraph()` / `vkai_benchmark` 用首尾两个 GPU timestamp 覆盖整张网络图，包含内部算子、同步和 Reshape / Identity 拷贝，排除模型初始化、CPU 提交、输入上传、输出下载及预后处理。输入上传一次，命令记录一次；batch=1、FP32，预热 5 次后串行测量 30 次，结束后仍核验输出。

2026-10-21 的测量记录：

| 平台 | ResNet-18 平均 | U-Net 平均 | MobileNetV2 平均 | YOLOv8n 平均 |
| --- | --- | --- | --- | --- |
| macOS / Apple M6 / MoltenVK | 18.23 ms | 138.80 ms | 5.24 ms | 46.98 ms |
| Android / Adreno 740 | 128.53 ms | 1013.17 ms | 38.39 ms | 355.27 ms |

设备使用默认功耗和频率。每次计时的原始采样、时间戳精度、设备名及输出误差保存在 `artifacts/<模型名>/{macos,android}_benchmark.json`，模型哈希位于对应的 `metadata.json`。`inference_ms` / `network_wall_ms` 是包含提交和输入输出拷贝的主机耗时，与 GPU timestamp 口径不同。

```bash
.venv/bin/python python/models/benchmark_models.py --models resnet18 unet mobilenet_v2 yolov8n
.venv/bin/python python/models/benchmark_models.py --platform android \
  --runner build-android/tasks/vkai_benchmark --models resnet18 unet mobilenet_v2 yolov8n
```

## 实现范围

`Engine` 负责缓冲区、调度和同步；`LayerFactory` 负责参数校验、layer 创建和多输入绑定。支持四个模型使用的静态 float32 ONNX 推理子集，符号 batch 默认为 1。卷积支持 group / depthwise、对称 padding；转置卷积支持 dilation 和小于 stride 的 output_padding；Concat 支持任意轴，Add / Sub / Mul / Div 支持多方向广播。Nearest Resize 支持 asymmetric / floor；Gemm 要求 transA=0、transB=1、alpha=beta=1。不支持的配置会明确报错。
