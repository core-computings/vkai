# LR-ASPP / MobileNetV3-Large 验证

2026-10-22 接入并在 macOS / Apple M6 / MoltenVK 和 Android 13 / Adreno 740 真机验证。两端全部 110 个单测、六个预训练模型验证通过。网络通过 `ONNX → Graph → LayerFactory → Vulkan` 执行，C++ 完成结果校验、后处理及绘图。

## 模型与算子

- Torchvision 0.23.0 的 `lraspp_mobilenet_v3_large`，权重为 `LRASPP_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1`，参数量 3,221,538。
- 输入为 `[1,3,320,320]` FP32 RGB；样例直接用 PIL bilinear 缩放到 320×320，除以 255 后按 ImageNet mean/std 归一化。该尺寸用于集成验证，区别于官方权重预处理的短边 520。
- 输出为 `[1,21,320,320]` logits，沿 channel 轴取 argmax，类别为 Pascal VOC 的 20 类物体加背景；相同分数取最小类别索引。
- 固定 opset 13，Conv / BN 融合并通过 onnxslim 折叠形状计算。实际 ONNX 包含 Conv×66、HardSigmoid×28、Mul×29、Relu×20、Add×11、GlobalAveragePool×9、Sigmoid×1、Resize×2。
- 新增原生 HardSigmoid shader：`clip(alpha*x+beta, 0, 1)`，支持 ONNX 的缺省 0.2/0.5 以及自定义参数。Hardswish 在此 opset 中通过 `HardSigmoid(alpha=1/6) + Mul` 执行。
- ONNX 双线性 Resize 接通已有 BilinearResize2D 层，保留网络的两次上采样，支持 `linear / half_pixel`，固定 NCHW sizes 或精确输出输入比的 scales。其他坐标模式明确报错。

## 数值与分割校验

导出器使用 PyTorch 2.8.0 生成原始 `reference.bin`，并用 `torch.argmax` 独立生成 uint8 `reference_mask.bin`。ONNX Runtime 1.23.1 与 PyTorch 比较通过，最大绝对误差为 `1.15037e-5`。

C++ 对所有 2,150,400 个 logits 逐元素比较，要求 `abs(actual-reference) ≤ 1e-3 + 1e-3*abs(reference)`。同时验证参考 mask 与参考 logits 一致，实际 mask 与 PyTorch mask 的像素一致率及平均 IoU 均须 ≥0.999，样例须含背景和狗。

| 平台 | 最大绝对误差 | RMSE | mask 像素一致率 | mask 平均 IoU |
| --- | ---: | ---: | ---: | ---: |
| macOS / Apple M6 | 1.43051e-5 | 2.58812e-6 | 100% | 1.0 |
| Android / Adreno 740 | 1.47820e-5 | 2.42900e-6 | 100% | 1.0 |

两端均得到背景 65,176 像素、狗 37,224 像素。平均 IoU 按任一 mask 中出现的类别计算并包含背景；这是与 PyTorch 预测的对齐指标，不是对标注数据集测得的准确率。

Release、FP32、batch=1、默认频率；预热 5 次后测量 30 次 GPU timestamp，整图一次提交。平均耗时为 macOS **12.56 ms**、Android **87.58 ms**。该计时包含网络内部算子和同步，不含初始化、输入上传、输出下载、CPU argmax 和绘图。

新增 11 个单测覆盖 HardSigmoid 饱和边界、非整工作组、非有限参数、ONNX 缺省与自定义参数、Hardswish 组合、双线性 Resize 的 scales/sizes、单像素尺寸、多 batch/channel、非方形图，以及分割 argmax 的负数/tie、IoU 的缺失类别和非法输入。单测由 C++ 生成输入及独立参考，不依赖模型下载。

## 可视化

`artifacts/lraspp/` 中的结果由实际 Vulkan 输出在 C++ 中绘制：

- `{macos,android}_mask.png`：单通道类别 ID（0..20）。
- `{macos,android}_color.png`：Pascal VOC 调色板，背景为黑色。
- `{macos,android}_overlay.png`：前景按 50% 混色叠加到输入图，背景保留原图。
- `{macos,android}_comparison.png`：输入、PyTorch mask、Vulkan mask、Vulkan overlay，附类别图例。

Android 在设备上绘图，`build.sh` 自动拉回主机。两端生成的四种 PNG 均逐字节一致。绘图不计入 GPU 计时。

![VKAI 语义分割对比](artifacts/lraspp/macos_comparison.png)

## 复现

从仓库根目录执行，依赖环境见 [模型说明](README.md)。基础 `requirements.txt` 已包含 LR-ASPP 所需的 onnxslim；运行已有 YOLO 示例需使用 `requirements-yolo.txt`。

```bash
.venv/bin/python python/models/export_lraspp.py
./scripts/build.sh -t macos -r unittests
./scripts/build.sh -t macos -r models
./scripts/build.sh -t android -r unittests
./scripts/build.sh -t android -r models
```

`-r models` 运行全部六个模型，需提前导出全部模型数据。只构建并验证 LR-ASPP：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target vkai_lraspp_test -j 10
build/tasks/vkai_lraspp_test \
  python/models/artifacts/lraspp/model.onnx \
  python/models/artifacts/lraspp/input.bin \
  python/models/artifacts/lraspp/macos.bin \
  python/models/artifacts/lraspp/reference.bin
```

最后还可显式传入 `REFERENCE_MASK.bin INPUT.png`；缺省读取模型目录的 `reference_mask.bin` 与 `input.png`。`--output-dir` 导出的目录同样适用。Python 只准备模型和参考，C++ 不调用 Python。

计时及复核输出：

```bash
.venv/bin/python python/models/benchmark_models.py --models lraspp
.venv/bin/python python/models/benchmark_models.py --platform android --models lraspp
```

原始校验和计时报告为 `artifacts/lraspp/{macos,android}_{validation,benchmark}.json`。ONNX、参考张量与 PNG 均为生成文件，按仓库惯例不提交 Git；`metadata.json` 记录模型、图像哈希及依赖版本。

本次 ONNX SHA-256：`82c15bdacf8553006370afa9ffad2ff012ed6f743a0205f593a29219b8c90c41`。输入源图 SHA-256：`f3f87bb8ab3c26c7ecfd3ac60421d7f32b0503d1d6c5baf8bac42ed93d86351a`。
