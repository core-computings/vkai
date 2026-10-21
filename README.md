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
```

Files are cached in `python/models/artifacts/` and shared by both platforms.

## Build and run tests

`build.sh` configures, compiles, and runs the selected tests. Android deployment
is automatic. Model tests use host Python for validation and visualization.

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

Default: `-t macos -r unittests`. `models` validates all four networks and benchmarks
MobileNetV2 / YOLOv8n. Both platforms have passed 80 unit tests and all four model tests.
See [model instructions](python/models/README.md) for output files and individual runs.

## GPU inference time

Recorded on 2026-10-21: Release, FP32, batch=1, default clocks; averages of 30 runs
after 5 warm-up runs. GPU timestamps cover the complete network graph, excluding
model loading, CPU submission, input/output transfers, preprocessing, and postprocessing.

| Model | Input | macOS / Apple M6 / MoltenVK | Android 13 / Adreno 740 |
| --- | --- | ---: | ---: |
| ResNet-18 | 224×224 | 18.23 ms | 128.53 ms |
| U-Net | 256×256 | 138.80 ms | 1013.17 ms |
| MobileNetV2 | 224×224 | 5.41 ms | 38.31 ms |
| YOLOv8n | 640×640 | 48.37 ms | 358.68 ms |
