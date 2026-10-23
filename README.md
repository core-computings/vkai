# VKAI

Vulkan AI inference engine for any GPUs, any platforms.

## Workflow

VKAI parses ONNX, infers tensor shapes, and builds a graph. It sorts operations
by dependency, initializes Vulkan, plans and allocates GPU storage, uploads
constants, and creates layers and compute pipelines. Inference uploads inputs,
records commands and barriers, submits work to the GPU, waits for completion,
and reads back outputs.

![Engine construction workflow](docs/ENGINE_BUILD_FLOW.svg)

The memory plan assigns tensor storage by lifetime. Intermediate tensors reuse
buffers after their last GPU consumer; reshape views share storage. CPU-only
parameters receive no graph buffer, while inputs, outputs, and GPU constants
keep dedicated storage.

![Tensor lifetimes and GPU buffer reuse](docs/MEMORY_PLAN.svg)

See [current model measurements](#model-parameters-memory-and-latency) for
validation results, memory use, and timings.

## Setup

Requires CMake ≥3.26, a C++20 compiler, Python 3.11, uv, and the Vulkan SDK
(glslc and MoltenVK on macOS). Run commands from the repository root.

```bash
git submodule update --init --recursive
# Replace with your installed Vulkan SDK setup script:
source /path/to/VulkanSDK/setup-env.sh
# macOS: expose the SDK loader libraries to test processes.
export DYLD_LIBRARY_PATH="$VULKAN_SDK/lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"

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
.venv/bin/python python/models/export_squeezenet.py
.venv/bin/python python/models/export_yolov8n.py
.venv/bin/python python/models/export_dncnn.py
.venv/bin/python python/models/export_lraspp.py
```

Files are cached in `python/models/artifacts/` and shared by both platforms.

## Build and run tests

`build.sh` builds and runs unit or model tests. Model outputs are checked against
exported PyTorch references; Android deployment is automatic.

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

Default: `-t macos -r unittests`; use `-r 0` to compile only.
Both platforms pass 119 unit tests and seven model tests; see
[model instructions](python/models/README.md) for details.

## Model parameters, memory, and latency

Measurements use Release, FP32, batch=1 on Apple M6 / MoltenVK and Android / Adreno 740.
GPU times are the mean of 30 runs after 5 warm-up runs.

| Model       | Input   | Parameters | Vulkan memory (MiB) | macOS GPU (ms) | Android GPU (ms) |
| ----------- | ------- | ----------: | -------------------: | --------------: | ----------------: |
| ResNet-18   | 224×224 | 11,689,512 | 52.08               | 18.17          | 135.73           |
| U-Net       | 256×256 | 7,763,041  | 85.13               | 138.82         | 1075.42          |
| MobileNetV2 | 224×224 | 3,504,872  | 26.28               | 7.46           | 36.43            |
| YOLOv8n     | 640×640 | 3,151,904  | 49.27               | 46.85          | 358.86           |
| DnCNN       | 256×256 | 668,227    | 36.05               | 393.90         | 3143.81          |
| LR-ASPP     | 320×320 | 3,221,538  | 38.82               | 12.56          | 87.58            |
| SqueezeNet  | 224×224 | 1,235,496  | 13.53               | 5.21           | 30.44            |
