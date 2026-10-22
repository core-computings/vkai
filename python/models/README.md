# 六个预训练模型的 macOS / Android 验证

当前模型测试包含 ResNet-18、U-Net、MobileNetV2、YOLOv8n、DnCNN 和 LR-ASPP，均通过 `ONNX → Graph → LayerFactory → Vulkan` 执行。Python 提前导出模型和 PyTorch ground truth；推理、结果对比与 GPU 计时均在 C++ 中完成。YOLOv8n 标注图、DnCNN 去噪对比图和 LR-ASPP 分割图由 C++ 测试直接绘制并保存；Python 绘图脚本可按需单独运行。

每个网络在 `tasks/` 下有独立的 C++ 测试入口，不再使用 `tools/` 子目录或通用推理程序：

| 网络 | C++ 入口 | 可执行文件 |
| --- | --- | --- |
| ResNet-18 | `tasks/ResNet18Test.cpp` | `vkai_resnet18_test` |
| U-Net | `tasks/UNetTest.cpp` | `vkai_unet_test` |
| MobileNetV2 | `tasks/MobileNetV2Test.cpp` | `vkai_mobilenet_v2_test` |
| YOLOv8n | `tasks/YOLOv8nTest.cpp` | `vkai_yolov8n_test` |
| DnCNN | `tasks/DnCNNTest.cpp` | `vkai_dncnn_test` |
| LR-ASPP / MobileNetV3-Large | `tasks/LRASPPTest.cpp` | `vkai_lraspp_test` |

各入口校验对应网络的输入/输出形状，并逐元素比较推理输出与 `reference.bin`。ResNet-18 / MobileNetV2 进一步比较 top-5 索引及顺序；U-Net 比较阈值 mask 的 Dice；YOLOv8n 比较独立 PyTorch NMS 产生的参考检测框。公共文件读写、数值比较、分类辅助函数和 GPU 计时位于 `tasks/ModelTestUtils.h`；每个可执行文件均支持 `--benchmark`。测试失败直接 `throw`，入口没有 `try/catch`。

| 模型 | 权重与输入 | 输出 |
| --- | --- | --- |
| ResNet-18 | [Torchvision IMAGENET1K_V1](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html)，官方狗图，224×224，ImageNet normalize | 1000 类 logits、top-5 |
| U-Net | [mateuszbuda brain MRI](https://pytorch.org/hub/mateuszbuda_brain-segmentation-pytorch_unet/)，256×256 MRI，各通道 z-score | 概率图、阈值 0.5 的 mask 和 overlay |
| MobileNetV2 | Torchvision IMAGENET1K_V2，官方狗图，224×224 | 1000 类 logits、top-5 |
| YOLOv8n | Ultralytics `yolov8n.pt`，公交车图，letterbox 到 640×640 | `[1,84,8400]` 检测头输出、C++ NMS 检测框和标注图 |
| DnCNN | [KAIR 官方彩色盲去噪权重](https://github.com/cszn/KAIR/releases/tag/v1.0)，20 层、64 通道，Set5 baby 图缩放到 256×256 RGB，添加 σ=25/255 的高斯噪声 | 去噪 RGB 图、PyTorch 数值误差、去噪前后 PSNR、噪声与去噪对比图 |
| LR-ASPP | [Torchvision MobileNetV3-Large COCO_WITH_VOC_LABELS_V1](https://docs.pytorch.org/vision/0.23/models/generated/torchvision.models.segmentation.lraspp_mobilenet_v3_large.html)，狗图直接双线性缩放到 320×320，ImageNet normalize | `[1,21,320,320]` logits、逐像素类别 mask、彩色 mask、overlay 和四联对比图 |

U-Net 上游固定在 `d45f8908ab2f0246ba204c702a6161c9eb25f902`，使用 `unet-e012d006.pt` 权重。YOLOv8n 使用 Ultralytics 8.3.200；导出时融合 Conv / BN，并用 onnxslim 折叠固定输入尺寸的形状常量。检测后处理在 C++ 中执行，置信度阈值为 0.25，同类框 NMS 的 IoU 阈值为 0.45，最多保留 300 个框。

DnCNN 使用 `dncnn_color_blind.pth`，校验固定 SHA-256；网络结构与测试图来源固定到 KAIR 的 `fc1732f4a4514e42ce15e5b3a1e18c828af47a1e`。官方推理权重已经融合 BN，导出的 ONNX 仅含 Conv、Relu、Sub，均由现有算子执行。输入为 `[1,3,256,256]` float32 RGB，随机种子为 0；噪声添加到 `[0,1]` 原图后不裁剪、不量化。PSNR 使用未裁剪 RGB 浮点张量、峰值 1、包含边界的全部像素；只有显示用 PNG 裁剪到 `[0,1]`。

为避免 Adreno 驱动对长 GPU 任务的超时重置，DnCNN 通过 `Engine(model_path, 8)` 每 8 个算子提交一次，两端采用相同的 5 段划分。普通推理与 benchmark 共用相同划分；其余模型默认仍整图提交。ONNX 单元测试覆盖跨提交残差依赖、非整除分段、重复推理和 GPU 时间戳，提交失败会抛出异常。

## 准备模型

需要 CMake ≥3.26、C++20 编译器、Vulkan SDK（含 MoltenVK / glslc）及 Python 3.11。以下命令在仓库根目录执行：

```bash
git submodule update --init --recursive
# 如 SDK 尚未加入环境，先 source 本机 VulkanSDK/<版本>/setup-env.sh
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r python/models/requirements-yolo.txt
# 分别准备六个模型及其输入、参考结果：
.venv/bin/python python/models/export_resnet18.py
.venv/bin/python python/models/export_unet.py
.venv/bin/python python/models/export_mobilenet_v2.py
# 导出 YOLOv8n：
.venv/bin/python python/models/export_yolov8n.py
.venv/bin/python python/models/export_dncnn.py
.venv/bin/python python/models/export_lraspp.py
```

首次导出需要网络下载权重和输入图。`build.sh -r models` 使用已经导出的文件，不自动下载或导出。`artifacts/` 下缓存 ONNX、输入图、float32 输入/参考张量和 metadata；这些生成文件被 Git 忽略，两个平台共享同一份输入和参考结果。

每个脚本只准备对应的模型，默认输出到 `artifacts/<模型名>/`。可用 `--output-dir /path/to/model-folder` 指定该模型的输出目录。通用下载、ONNX 导出和参考结果校验位于 `export_common.py`。

`reference.bin` 是 PyTorch 的原始 float32 输出。YOLO 导出还会用 `torchvision.ops.batched_nms` 生成 `reference_detections.txt`：第一行为框数，之后每行依次为 `class_id confidence x1 y1 x2 y2`，坐标位于 letterbox 后的输入空间。C++ 直接读取这些独立参考框，不用自身的 NMS 生成 ground truth。已有 YOLO 数据需要重新运行 `export_yolov8n.py` 来生成此文件。

导出时比较 PyTorch 与 ONNX Runtime；运行模型测试时，由 C++ 逐元素比较 Vulkan 与 PyTorch，容差为 `atol=rtol=1e-3`。分类要求 top-5 索引及顺序一致；U-Net 要求前景/背景均存在且与 PyTorch mask 的 Dice ≥0.999；YOLO 要求类别、框数及顺序与参考一致，置信度差 <1e-4、框 IoU >0.999，且样例包含人和公交车。失败会返回非零退出码。

DnCNN 在同样的逐元素校验之外，还读取独立的 `clean.bin`，要求相对噪声输入的 PSNR 提升至少 3 dB，防止原样输出噪声图也被当作通过。单元测试覆盖 PSNR 数值、原样输出、质量退化、尺寸不符以及非有限值。导出器另保存 `clean.png`、`noisy.png`、`reference.png`。两端的运行结果与计时见 [DnCNN 运行报告](DNCNN_REPORT.md)。

LR-ASPP 固定使用 Torchvision 0.23.0 的预训练模型，导出时融合 Conv / BN 并通过 onnxslim 折叠形状计算，保留完整网络的双线性上采样。新增原生 `HardSigmoid` 支持 ONNX 的 alpha/beta（缺省为 0.2/0.5）；opset 13 的 Hardswish 由 `HardSigmoid(alpha=1/6) + Mul` 执行。ONNX `Resize(mode=linear, coordinate_transformation_mode=half_pixel)` 连接已有 BilinearResize2D 层，其他坐标模式明确拒绝。

LR-ASPP 导出器额外保存 `reference_mask.bin`（320×320 uint8，直接由 PyTorch argmax 生成），C++ 检查它与参考 logits 一致，再比较实际 Vulkan logits 和 mask。要求逐元素 `atol=rtol=1e-3`、mask 像素一致率及平均 IoU 均 ≥0.999，且存在背景和狗。IoU 针对两份预测 mask，按任一 mask 中出现的类别计算并包含背景，不代表数据集标注精度。C++ 保存 `{macos,android}_mask.png`（单通道 class ID）、`_color.png`（VOC 调色板）、`_overlay.png` 和 `_comparison.png`（输入 / PyTorch mask / Vulkan mask / Vulkan overlay，附类别图例）。具体结果和命令见 [LR-ASPP 运行报告](LRASPP_REPORT.md)。

## 统一编译和测试入口

```bash
./scripts/build.sh -t macos -r unittests
./scripts/build.sh -t macos -r models
./scripts/build.sh -t android -r unittests
./scripts/build.sh -t android -r models
```

默认是 `-t macos -r unittests`，两个选项顺序不限。`-r 0` 只编译全部目标，不运行测试，也不需要安卓设备；`unittests` 运行全部通用算子、图执行和加载器测试；`models` 运行上述六个模型，并对 MobileNetV2 / YOLOv8n / DnCNN / LR-ASPP 做 GPU 计时。

全部单元测试均无需 Python 导出数据或下载模型：C++ 生成确定性输入与 CPU 参考结果，小型 ONNX 测试图在临时目录创建并清理，FlatBuffer 加载器的微型权重由 CMake 从 `assets/weights.json` 自动生成。C++ 测试执行不调用 Python；构建 ONNX 第三方依赖时仍需主机 Python 解释器做代码生成。

`models` 通过 shell 直接运行六个 C++ 测试，不调用 Python。各模型的原始输出和 C++ 校验报告位于 `artifacts/<模型名>/{macos,android}.bin`、`{macos,android}_validation.json`；YOLO 另保存 `{macos,android}_detections.json` 和 `{macos,android}_detections.png`。DnCNN 另保存 `{macos,android}_noisy.png`、`{macos,android}_denoised.png`、`{macos,android}_comparison.png`，三联图从左到右为噪声输入、Vulkan 去噪结果、干净原图。报告含数值误差及对应的 top-5 / Dice / 检测框 IoU / PSNR。构建 ONNX 第三方依赖仍需主机 Python 做代码生成。

YOLO C++ 测试在校验通过后读取 `input.jpg`，把 letterbox 坐标还原到原图，绘制检测框、COCO 类别名和置信度，保存原始尺寸的 PNG。JPEG/PNG 读写及内置字体使用已有 stb 头文件，无需系统字体或 Python。绘图不计入网络和 GPU 计时；`--benchmark` 分支只计时及校验数值，不绘图。Android 的输入图随模型上传，在设备端绘图后自动拉回主机。

## Android arm64 真机

需要 NDK、adb 在线设备以及主机 Vulkan SDK / glslc。最低 API 为 28；设备还需满足 CORE 的 Vulkan 实例和 dynamic rendering 要求。已验证 Android 13（API 33）/ Adreno 740。

```bash
export ANDROID_NDK_ROOT="$HOME/Library/Android/sdk/ndk/28.2.13676358"
adb devices -l
./scripts/build.sh -t android -r unittests
./scripts/build.sh -t android -r models
```

脚本先构建主机 `flatc` / `protoc`，再交叉编译 arm64；shader 由主机 glslc 生成并嵌入程序，C++ runtime 静态链接。adb 自动上传所选测试组需要的程序和数据到 `/data/local/tmp/vkai` 并执行。多设备时设置标准 `ANDROID_SERIAL` 环境变量。

构建环境变量：`VKAI_JOBS`、`VKAI_BUILD_DIR`（macOS 输出）、`VKAI_HOST_BUILD_DIR`（主机生成器）、`VKAI_ANDROID_BUILD_DIR`（Android 输出）。`VKAI_TEST_BINARY_DIR` 指定单测在目标平台读取 CMake 生成权重的目录。

## 单独复跑

已构建后，可直接调用对应网络的 C++ 测试入口：

```bash
build/tasks/vkai_resnet18_test \
  python/models/artifacts/resnet18/model.onnx \
  python/models/artifacts/resnet18/input.bin \
  build/resnet18_output.bin \
  python/models/artifacts/resnet18/reference.bin
```

参考文件参数必填；张量采用连续、小端 float32。U-Net / MobileNetV2 使用相同参数顺序；YOLOv8n 需要额外指定检测框 JSON 输出和 PyTorch NMS 参考文件：

```bash
build/tasks/vkai_yolov8n_test \
  python/models/artifacts/yolov8n/model.onnx \
  python/models/artifacts/yolov8n/input.bin \
  build/yolov8n_output.bin \
  build/yolov8n_detections.json \
  python/models/artifacts/yolov8n/reference.bin \
  python/models/artifacts/yolov8n/reference_detections.txt \
  python/models/artifacts/yolov8n/input.jpg \
  build/yolov8n_detections.png
```

最后两个图像参数可以一起省略：输入图默认是模型所在目录的 `input.jpg`，PNG 默认与检测框 JSON 同名、扩展名改为 `.png`。输入图读取或 PNG 保存失败会直接 `throw`。

DnCNN 需要额外传入干净图的 float32 张量。PNG 直接从输入张量和实际 Vulkan 输出绘制，以 `OUTPUT.bin` 的文件名为前缀保存：

```bash
build/tasks/vkai_dncnn_test \
  python/models/artifacts/dncnn/model.onnx \
  python/models/artifacts/dncnn/input.bin \
  python/models/artifacts/dncnn/macos.bin \
  python/models/artifacts/dncnn/reference.bin \
  python/models/artifacts/dncnn/clean.bin
```

六个测试均直接读取预先导出的 PyTorch 参考结果。`scripts/build.sh` 负责 Android 文件上传、可执行文件调用和结果拉取；设备上不运行 Python。DnCNN 的三张 PNG、LR-ASPP 的四张 PNG 与 YOLO 标注图一样在目标设备生成，绘图不计入推理时间或 GPU benchmark。

## 可选图像绘制

YOLO 标注图已经由 C++ 自动生成。测试通过后，也可单独运行 Python 绘图脚本生成 U-Net mask / overlay，或重新绘制 YOLO 标注图。该脚本只读取 C++ 输出及输入图，不读取 ground truth，也不参与校验：

```bash
.venv/bin/python python/models/visualize_results.py --platform macos
.venv/bin/python python/models/visualize_results.py --platform android
```

## 纯网络 GPU inference 计时

各网络的 C++ 测试入口通过 `--benchmark` 调用 `BenchmarkGraph()`，在每段提交的首尾写入 GPU timestamp，并将各段耗时相加，覆盖全部内部算子、同步和 Reshape / Identity 拷贝。DnCNN 为 5 段，其余模型为 1 段；计时排除模型初始化、CPU 提交、输入上传、输出下载、分段间的主机等待间隙及预后处理。输入上传一次，各段命令记录一次；batch=1、FP32，预热 5 次后串行测量 30 次，结束后仍核验输出。报告中的 `submissions_per_run` 记录每次推理的提交次数。

2026-10-21 的测量记录：

| 平台 | ResNet-18 平均 | U-Net 平均 | MobileNetV2 平均 | YOLOv8n 平均 | DnCNN 平均 |
| --- | --- | --- | --- | --- | --- |
| macOS / Apple M6 / MoltenVK | 18.23 ms | 138.80 ms | 5.24 ms | 46.98 ms | 390.85 ms |
| Android / Adreno 740 | 128.53 ms | 1013.17 ms | 38.39 ms | 355.27 ms | 3156.68 ms |

设备使用默认功耗和频率。每次计时的原始采样、时间戳精度、设备名及输出误差保存在 `artifacts/<模型名>/{macos,android}_benchmark.json`，模型哈希位于对应的 `metadata.json`。`inference_ms` / `network_wall_ms` 是包含提交和输入输出拷贝的主机耗时，与 GPU timestamp 口径不同。

```bash
.venv/bin/python python/models/benchmark_models.py --models resnet18 unet mobilenet_v2 yolov8n dncnn lraspp
.venv/bin/python python/models/benchmark_models.py --platform android \
  --runner-dir build-android/tasks --models resnet18 unet mobilenet_v2 yolov8n dncnn lraspp

# 直接计时单个网络：
build/tasks/vkai_resnet18_test --benchmark \
  python/models/artifacts/resnet18/model.onnx \
  python/models/artifacts/resnet18/input.bin \
  python/models/artifacts/resnet18/reference.bin 5 30
```

## 实现范围

`Engine` 负责缓冲区、调度和同步；`LayerFactory` 负责参数校验、layer 创建和多输入绑定。支持六个模型使用的静态 float32 ONNX 推理子集，符号 batch 默认为 1。卷积支持 group / depthwise、对称 padding；转置卷积支持 dilation 和小于 stride 的 output_padding；Concat 支持任意轴，Add / Sub / Mul / Div 支持多方向广播。Nearest Resize 支持 asymmetric / floor，Bilinear Resize 支持 half_pixel 和固定 sizes / 精确输出输入比的 scales；HardSigmoid 支持有限 alpha/beta；Gemm 要求 transA=0、transB=1、alpha=beta=1。不支持的配置会明确报错。
