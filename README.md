# VKAI

Vulkan AI inference engine for any GPUs, any platforms.

## Setup

Requires CMake ≥3.26, a C++20 compiler, Python 3.11, uv, and the Vulkan SDK
(glslc and MoltenVK on macOS). Run commands from the repository root.

```bash
git submodule update --init --recursive
# Replace with your installed Vulkan SDK setup script:
source /path/to/VulkanSDK/setup-env.sh

uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r python/models/requirements-yolo.txt
```

## Prepare model tests

Unit tests generate their own data. For model tests, run these once to download
pretrained weights and sample images, then export ONNX models, inputs, and reference outputs:

```bash
.venv/bin/python python/models/export_resnet18.py
.venv/bin/python python/models/export_unet.py
.venv/bin/python python/models/export_mobilenet_v2.py
.venv/bin/python python/models/export_yolov8n.py
.venv/bin/python python/models/export_dncnn.py
```

Files are cached in `python/models/artifacts/` and shared by both platforms.

## Build and run tests

`build.sh` configures, compiles, and runs the selected tests. Android deployment
is automatic. Each C++ model test reads the exported PyTorch ground truth and
validates its own outputs. Python prepares models and optionally renders images;
model execution and comparison run entirely in C++.

Each network has its own C++ entry point directly in `tasks/`:
`ResNet18Test.cpp`, `UNetTest.cpp`, `MobileNetV2Test.cpp`, `YOLOv8nTest.cpp`, and `DnCNNTest.cpp`.
They build as `vkai_{resnet18,unet,mobilenet_v2,yolov8n,dncnn}_test`; each executable
also supports GPU timing with `--benchmark`.

The YOLOv8n C++ test also draws boxes, COCO class names, and confidence scores on
the original image and saves `python/models/artifacts/yolov8n/<platform>_detections.png`.
Android generates the PNG on the device and automatically pulls it back to the host.

DnCNN validates RGB denoising against PyTorch and requires a PSNR improvement of
at least 3 dB against the clean image. Its C++ test saves `<platform>_noisy.png`,
`<platform>_denoised.png`, and `<platform>_comparison.png` in
`python/models/artifacts/dncnn/`. The comparison shows noisy input, Vulkan output,
and clean reference. Android renders these images on the device and pulls them back.
See the [DnCNN run report](python/models/DNCNN_REPORT.md) for results and timings.

```bash
# macOS
./scripts/build.sh -t macos -r unittests
./scripts/build.sh -t macos -r models

# Android arm64: install NDK / adb and connect a device with USB debugging enabled.
export ANDROID_NDK_ROOT="/path/to/android-ndk"
adb devices
./scripts/build.sh -t android -r unittests
./scripts/build.sh -t android -r models
```

Default: `-t macos -r unittests`. `models` validates all five networks and benchmarks
MobileNetV2 / YOLOv8n / DnCNN. Both platforms pass all 87 unit tests and all five model tests.
See [model instructions](python/models/README.md) for output files and individual runs.

For compilation only (as in CI), use `./scripts/build.sh -t macos -r 0` or
`./scripts/build.sh -t android -r 0`. This builds all targets without running tests;
no model data or Android device is required.

## GPU inference time

Recorded on 2026-10-21: Release, FP32, batch=1, default clocks; averages of 30 runs
after 5 warm-up runs. GPU timestamps cover the complete network graph, excluding
model loading, CPU submission, input/output transfers, preprocessing, and postprocessing.
DnCNN submits eight operators at a time to stay below the Android GPU watchdog limit;
its reported GPU time sums all five submissions and excludes host gaps between them.

| Model | Input | macOS / Apple M6 / MoltenVK | Android 13 / Adreno 740 |
| --- | --- | ---: | ---: |
| ResNet-18 | 224×224 | 18.23 ms | 128.53 ms |
| U-Net | 256×256 | 138.80 ms | 1013.17 ms |
| MobileNetV2 | 224×224 | 5.41 ms | 38.31 ms |
| YOLOv8n | 640×640 | 48.37 ms | 358.68 ms |
| DnCNN | 256×256 RGB | 390.85 ms | 3156.68 ms |
