# DnCNN 运行报告

2026-10-21（America/Los_Angeles）。macOS 和 Android 真机均通过 87 项单元测试及全部五个模型验证；DnCNN 将样例图的 PSNR 从 20.19 dB 提升至 31.90 dB。

## 模型与输入

采用 [KAIR 官方 DnCNN 彩色盲去噪权重](https://github.com/cszn/KAIR/releases/tag/v1.0) `dncnn_color_blind.pth`，20 层、64 个特征通道，BN 已融合，参数量为 668,227，ONNX 文件为 2.68 MB。

两端共用相同的 ONNX、输入和 PyTorch 参考张量。Set5 baby 原图为 512×512，经 bicubic 缩放至 **256×256 RGB**；网络输入与输出均为 `[1,3,256,256]`，batch=1、FP32。图像归一化至 `[0,1]` 后添加高斯噪声，σ=25/255，NumPy RandomState 种子为 0；网络输入不裁剪、不量化。

## 运行时间

| 平台及设备 | 网络输入 | GPU 平均 | GPU 中位数 | 首次单次 ExecuteGraph 壁钟 |
| --- | --- | ---: | ---: | ---: |
| macOS 27.0.1 / Mac mini / Apple M6 / MoltenVK | 256×256 RGB | 390.85 ms | 390.86 ms | 411.40 ms |
| Android 13 / 2210132C / Adreno 740 | 256×256 RGB | 3156.68 ms | 3156.62 ms | 3235.34 ms |

Release 构建，设备默认频率。GPU 计时预热 5 次，再测量完整网络 30 次。为避免 Android 驱动的长任务超时，两端均每 8 个算子提交一次，共 5 次提交；GPU 时间为这 5 段时间戳耗时之和，包含全部网络算子及段内同步，排除模型加载、CPU 提交、输入输出传输、段间主机等待间隙及图片绘制。

单次 ExecuteGraph 壁钟包含输入上传、命令录制、提交与等待、输出读回；不含 Engine 初始化、模型加载、质量校验和 PNG 编码。该值是第一次执行的单次采样，与预热后的 GPU 平均值口径不同。

## 数值与去噪质量

| 平台 | 相对 PyTorch 最大绝对误差 | RMSE | 超容差元素 | 去噪后 PSNR | PSNR 提升 |
| --- | ---: | ---: | ---: | ---: | ---: |
| macOS | 0 | 0 | 0 | 31.8963 dB | 11.7079 dB |
| Android | 3.7e-06 | 7.67e-07 | 0 | 31.8963 dB | 11.7079 dB |

逐元素容差为 `abs(actual-reference) <= 1e-3 + 1e-3*abs(reference)`，比较全部 196,608 个 float32 元素。导出阶段 PyTorch 与 ONNX Runtime 的最大绝对误差为 5.45e-06。基准计时结束后也重新核验输出。

噪声输入 PSNR 为 20.1884 dB，PyTorch 输出为 31.8963 dB。PSNR 使用未裁剪的 RGB 浮点张量，峰值为 1，包含边界全部像素；C++ 还要求相对噪声输入至少提升 3 dB。显示用 PNG 才裁剪到 `[0,1]` 并转为 8 bit。质量数值对应这张固定高斯噪声样例。

## 测试结果

| 测试组 | macOS | Android |
| --- | --- | --- |
| 全部单元测试 | 87/87 PASS | 87/87 PASS |
| ResNet-18、U-Net、MobileNetV2、YOLOv8n、DnCNN | 5/5 PASS | 5/5 PASS |
| MobileNetV2、YOLOv8n、DnCNN GPU benchmark 及输出复核 | 3/3 PASS | 3/3 PASS |

DnCNN 导出图包含 20 个 Conv、19 个 Relu、1 个 Sub，均由现有 Vulkan 算子执行，无需新增算子。本次增加了 3 项去噪质量单元测试，覆盖已知 PSNR、原样输出、质量退化、尺寸错误和非有限值；增加了 1 项 ONNX 跨提交测试，覆盖残差依赖、非整除分段、更新输入后重复执行及 GPU 时间戳。Engine 对提交与等待失败明确报错。

## 整图单次提交复测

2026-10-21 再次在同一台 Android 13 / Adreno 740 真机上，用独立临时程序将全部 40 个算子一次提交，输入仍为 256×256 RGB。驱动再次记录 `gpu timeout ctx 4 ts 2`，第一个输出元素为 0，而 PyTorch 参考值为 0.978817；测试退出码为 134，未通过数值校验。

正式入口继续保留每 8 个算子一段、共 5 次提交的实现。超时后重新运行分段版本，数值校验通过，最大绝对误差为 3.70×10⁻⁶，PSNR 为 31.8963 dB；另以预热 1 次、测量 3 次复核 GPU 计时，亦通过。该短复核用于确认设备恢复及分段执行正常，上表仍采用原先 30 次测量结果。

复测记录：[配置与状态](artifacts/dncnn/single_submission_retry/result.json)、[驱动超时日志](artifacts/dncnn/single_submission_retry/android_gpu_timeout.txt)、[单次提交输出](artifacts/dncnn/single_submission_retry/android_stdout.txt)、[分段数值校验](artifacts/dncnn/single_submission_retry/android_segmented_validation.json)、[分段计时复核](artifacts/dncnn/single_submission_retry/android_segmented_benchmark.json)。

## GPU 超时原因分析

进一步的 ADB 日志与插桩对照实验表明，直接触发点是这台 Adreno 740 当前驱动的长 GPU 任务超时保护。约 2 秒的门槛来自实测推断；本机 watchdog 配置和 KGSL debugfs 读取受限，未取得固件中的配置值。

40 个算子按数据依赖顺序执行，整图模式把它们录入一个 command buffer、一次提交。每层的 GPU memory barrier 保证数据依赖，但没有把命令拆成独立的 GPU 提交。DnCNN 的 20 层卷积都保持输入的空间分辨率，整图 GPU 工作需要约 3.15 秒，超过了本次观察到的单批工作时间门槛。

### 输入尺寸与提交方式对照

所有配置均使用相同的 20 层官方权重、FP32 和数值容差。小尺寸输入是原始 256×256 噪声张量的左上裁剪，各自重新导出静态 ONNX 和 PyTorch 参考结果。成功配置先执行一次数值校验，再测一次 GPU 时间戳；以下为定位问题的单次采样，不替换上面的 30 次性能测量。

| 输入 | 提交次数 | 最长一段 GPU 时间 | 完整网络 GPU 时间 | 数值校验 |
| --- | ---: | ---: | ---: | --- |
| 128×128 RGB | 1 | 779.36 ms | 779.36 ms | PASS |
| 192×192 RGB | 1 | 1810.95 ms | 1810.95 ms | PASS |
| 224×224 RGB | 1 | 无有效测量 | 无有效测量 | GPU 超时，输出全 0 |
| 256×256 RGB | 1 | 无有效测量 | 无有效测量 | GPU 超时，输出全 0 |
| 256×256 RGB，每个算子单独提交 | 40 | 183.98 ms | 3179.14 ms | PASS |
| 256×256 RGB，每段 20 个算子 | 2 | 1576.83 ms | 3153.09 ms | PASS |
| 256×256 RGB，每段 8 个算子 | 5 | 697.75 ms | 3155.98 ms | PASS |

224×224 和 256×256 失败时，从 `vkQueueSubmit` 调用到 `vkWaitForFences` 返回，主机分别测得 2012.51 ms 和 2013.95 ms，同时驱动报出 `gpu timeout ctx 4 ts 2` 和 `Suspended GMU`。这两个时间是提交与等待的主机壁钟，不能当作有效的完整网络 GPU 时间。

相同的 40 个算子在较小输入下可以一次提交；相同的 256×256 模型在两段或五段下可以完成。分段仅改变提交边界，模型张量分配和卷积计算保持一致。这些对照支持单批任务耗时过长的解释，未显示算子数量上限、算子缺失或静态内存分配不足是本次直接原因。两段虽然在本次采样通过，但每段约 1.58 秒，余量比五段小，默认仍保留五段。

### 计算耗时来自卷积

逐算子计时中，20 个 Conv 共耗时 3150.62 ms，占完整网络 GPU 时间 3179.14 ms 的约 99.10%；19 个 Relu 共 28.42 ms，Sub 约 0.09 ms。单个最慢算子为 Conv，耗时 183.98 ms。

按卷积权重形状计算，256×256 输入的理论计算量约为 43.71G MAC，即约 87.43 GFLOP（一次 MAC 按两次浮点运算计算；边界零填充位置在当前 shader 中会跳过，实际略少）。668,227 个参数描述的是权重规模，空间位置上的重复计算使运算量大得多。

当前 `Conv2D.comp` 每个 invocation 计算一个输出元素，中间 64→64 的 3×3 卷积需要顺序累加 576 项，没有显式的 shared-memory tiling 或向量化多输出计算。结合逐算子测量，性能优化应优先集中到卷积；若希望整图一次提交，应降低整图耗时并为设备频率波动留出余量。

### 驱动日志与 Vulkan 返回值

同平台 Android 13 的[公开 Adreno Gen7 驱动实现](https://github.com/OnePlusOSS/android_kernel_modules_and_devicetree_oneplus_sm8550/blob/03d673ab5ca79f490296fdfa7021f7ba9bf956c7/vendor/qcom/opensource/graphics-kernel/adreno_gen7_hwsched_hfi.c#L420)将这条 `gpu timeout` 消息归类为 `GMU_GPU_SW_HANG`，并在启用 long-IB 检测时打开 `HFI_FEATURE_BAIL_OUT_TIMER`。这支持 GMU/驱动长任务保护的解释；该公开代码来自同平台设备，不能据此认定本机固件的精确配置值。

补充更正：返回值检查只能捕获显式报告的 Vulkan 错误。本次两次失败的插桩均记录 `vkQueueSubmit=VK_SUCCESS`、`vkWaitForFences=VK_SUCCESS`，但输出全 0、驱动同时报告超时。因此不能只凭 API 返回成功判断张量有效，PyTorch 数值与去噪质量检查仍然必要。[Vulkan 规范](https://docs.vulkan.org/refpages/latest/refpages/source/vkWaitForFences.html)也允许设备丢失后 `vkWaitForFences` 返回 `VK_SUCCESS` 或 `VK_ERROR_DEVICE_LOST`。主机等待参数已是 `UINT64_MAX`，增加这个等待值不能延长驱动的 watchdog 门槛。

原始证据：[对照实验 JSON](artifacts/dncnn/timeout_analysis/matrix.json)、[逐算子 GPU 计时及计算量](artifacts/dncnn/timeout_analysis/workload_profile.json)、[本次 GPU logcat](artifacts/dncnn/timeout_analysis/matrix_gpu_logcat.txt)、[224×224 插桩日志](artifacts/dncnn/timeout_analysis/224_single.stderr)、[256×256 插桩日志](artifacts/dncnn/timeout_analysis/256_single.stderr)、[五段提交日志](artifacts/dncnn/timeout_analysis/256_five.stderr)、[设备信息](artifacts/dncnn/timeout_analysis/device.txt)。诊断程序与插桩 Engine 源码保存在同一 artifacts 目录，正式推理代码未修改。

Perfetto 采集尝试中，所请求的 10 个 KGSL ftrace 事件全部启用失败，trace 仅有配置和统计，未获得 GPU 内核调度事件。相关限制记录在[采集状态](artifacts/dncnn/timeout_analysis/trace_capture_status.json)和[解码统计](artifacts/dncnn/timeout_analysis/perfetto_decode.txt)中；上述结论依据的是有效的驱动 logcat、主机插桩和 GPU 时间戳。

## 图片对比

每幅三联图从左到右依次为噪声输入、实际 Vulkan 去噪结果、干净原图。图像在各目标平台的 C++ 测试中生成；Android 的 PNG 由 adb 自动拉回。已核对 PNG 像素与实际输入、输出张量一致。

macOS：[噪声输入](artifacts/dncnn/macos_noisy.png) · [去噪结果](artifacts/dncnn/macos_denoised.png)

![macOS 噪声输入与 Vulkan 去噪结果及干净原图](artifacts/dncnn/macos_comparison.png)

Android：[噪声输入](artifacts/dncnn/android_noisy.png) · [去噪结果](artifacts/dncnn/android_denoised.png)

![Android 噪声输入与 Vulkan 去噪结果及干净原图](artifacts/dncnn/android_comparison.png)

## 复现与原始记录

先按[模型准备说明](README.md#准备模型)配置环境并准备五个模型，然后在仓库根目录运行：

```bash
.venv/bin/python python/models/export_dncnn.py
./scripts/build.sh -t macos -r unittests
./scripts/build.sh -t macos -r models
export ANDROID_NDK_ROOT="$HOME/Library/Android/sdk/ndk/28.2.13676358"
./scripts/build.sh -t android -r unittests
./scripts/build.sh -t android -r models
```

| 平台 | 数值校验 | GPU 原始采样 | 单元测试日志 | 模型测试日志 |
| --- | --- | --- | --- | --- |
| macOS | [JSON](artifacts/dncnn/macos_validation.json) | [JSON](artifacts/dncnn/macos_benchmark.json) | [日志](artifacts/dncnn/macos_unittests.log) | [日志](artifacts/dncnn/macos_models.log) |
| Android | [JSON](artifacts/dncnn/android_validation.json) | [JSON](artifacts/dncnn/android_benchmark.json) | [日志](artifacts/dncnn/android_unittests.log) | [日志](artifacts/dncnn/android_models.log) |

[导出元数据](artifacts/dncnn/metadata.json)记录模型来源、固定源代码版本、权重 SHA-256、输入形状和预处理；[导出日志](artifacts/dncnn/export.log)记录 ONNX Runtime 对照结果。ONNX SHA-256 为 `5b9480d9497d0563a6fd1e8e4f701067ec6044679bd2ecc74747e84ff060738e`。生成的模型、张量、图片和日志保存在被 Git 忽略的 `python/models/artifacts/dncnn/`。
