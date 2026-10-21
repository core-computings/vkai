#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<EOF
Usage: $0 [-t {macos|android}] [-r {unittests|models}] [-b] [-h]

Build and run the selected tests. Defaults: -t macos -r unittests.
  unittests  Run all unit tests.
  models     Run ResNet-18, U-Net, MobileNetV2, YOLOv8n, then GPU benchmarks.
  -b         Build only; build all targets unless -r selects a group.
Android builds require ANDROID_NDK_ROOT and host glslc.
Running Android tests also requires adb and a connected arm64 device.
Running model tests requires exported ONNX models, inputs, and reference outputs.
EOF
}
platform=macos
test_mode=unittests
build_only=false
group_selected=false
while getopts ":t:r:bh" option; do
  case "$option" in
    t) platform="$OPTARG" ;;
    r) test_mode="$OPTARG"; group_selected=true ;;
    b) build_only=true ;;
    h) usage; exit 0 ;;
    :) echo "Option -$OPTARG requires a value." >&2; usage >&2; exit 2 ;;
    \?) echo "Unknown option: -$OPTARG" >&2; usage >&2; exit 2 ;;
  esac
done
shift "$((OPTIND - 1))"
if [[ $# -ne 0 ]]; then
  usage >&2
  exit 2
fi
case "$platform" in
  macos|android) ;;
  *) echo "Unsupported target platform: $platform" >&2; usage >&2; exit 2 ;;
esac
case "$test_mode" in
  unittests) targets=(vkai_unittests) ;;
  models) targets=(vkai_infer vkai_detect vkai_benchmark) ;;
  *) echo "Unknown test group: $test_mode" >&2; usage >&2; exit 2 ;;
esac
if [[ "$build_only" == true && "$group_selected" == false ]]; then
  targets=(vkai_unittests vkai_infer vkai_detect vkai_benchmark)
fi

project_root="$(cd "$(dirname "$0")/.." && pwd)"
remote=/data/local/tmp/vkai

run_cpp_tests() {
  local executable="$1"
  if [[ "$platform" == android ]]; then
    adb push "$executable" "$remote/"
    local name="${executable##*/}"
    adb shell "chmod 755 $remote/$name && $remote/$name"
  else
    "$executable"
  fi
}

if [[ "$platform" == android ]]; then
  : "${ANDROID_NDK_ROOT:?Set ANDROID_NDK_ROOT to an installed Android NDK}"
  if [[ "$build_only" == false ]]; then
    adb get-state >/dev/null
  fi
  host_build="${VKAI_HOST_BUILD_DIR:-$project_root/build}"
  build_dir="${VKAI_ANDROID_BUILD_DIR:-$project_root/build-android}"
  case "$(uname -s)" in
    Darwin) host_tag=darwin-x86_64 ;;
    Linux) host_tag=linux-x86_64 ;;
    *) echo "Unsupported Android build host: $(uname -s)" >&2; exit 1 ;;
  esac
  toolchain="$ANDROID_NDK_ROOT/toolchains/llvm/prebuilt/$host_tag"
  if ! glslc="$(command -v glslc)"; then
    echo "glslc was not found. Add the Vulkan SDK bin directory to PATH." >&2
    exit 1
  fi
  # Generate schemas with host-native flatc/protoc, then compile for the device.
  cmake -S "$project_root" -B "$host_build" -DCMAKE_BUILD_TYPE=Release
  cmake --build "$host_build" --target flatc protoc -j "${VKAI_JOBS:-10}"
  cmake -S "$project_root" -B "$build_dir" \
    -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK_ROOT/build/cmake/android.toolchain.cmake" \
    -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-28 -DANDROID_STL=c++_static \
    -DCMAKE_BUILD_TYPE=Release -DVKAI_HOST_BUILD_DIR="$host_build" \
    -DFLATBUFFERS_FLATC_EXECUTABLE="$host_build/external/flatbuffers/flatc" \
    -DONNX_CUSTOM_PROTOC_EXECUTABLE="$host_build/external/protobuf/protoc" \
    -DVulkan_INCLUDE_DIR="$toolchain/sysroot/usr/include" \
    -DVulkan_LIBRARY="$toolchain/sysroot/usr/lib/aarch64-linux-android/28/libvulkan.so" \
    -DGLSLC_EXECUTABLE="$glslc" \
    -DVKAI_TEST_BINARY_DIR=/data/local/tmp/vkai/build
else
  build_dir="${VKAI_BUILD_DIR:-$project_root/build}"
  cmake -S "$project_root" -B "$build_dir" -DCMAKE_BUILD_TYPE=Release
fi
cmake --build "$build_dir" --target "${targets[@]}" -j "${VKAI_JOBS:-10}"
if [[ "$build_only" == true ]]; then
  echo "Build complete: $platform ($build_dir)"
  exit 0
fi
echo "Running $test_mode on $platform"
case "$test_mode" in
  unittests)
    if [[ "$platform" == android ]]; then
      adb shell "mkdir -p $remote/build"
      adb push "$build_dir/weights.bin" "$remote/build/"
    fi
    run_cpp_tests "$build_dir/tests/vkai_unittests"
    ;;
  models)
    "${VKAI_PYTHON:-$project_root/.venv/bin/python}" "$project_root/python/models/validate_models.py" \
      --platform "$platform" --runner "$build_dir/tasks/vkai_infer" --models resnet18 unet mobilenet_v2
    "${VKAI_PYTHON:-$project_root/.venv/bin/python}" "$project_root/python/models/validate_yolov8n.py" \
      --platform "$platform" --runner "$build_dir/tasks/vkai_detect"
    "${VKAI_PYTHON:-$project_root/.venv/bin/python}" "$project_root/python/models/benchmark_models.py" \
      --platform "$platform" --runner "$build_dir/tasks/vkai_benchmark"
    ;;
esac
echo "PASS: $platform $test_mode"
