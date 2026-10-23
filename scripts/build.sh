#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<EOF
Usage: $0 [-t {macos|android}] [-r {0|unittests|models}] [-h]

Build and run the selected tests. Defaults: -t macos -r unittests.
  0          Build all targets without running tests or models.
  unittests  Run all unit tests.
  models     Run ResNet-18, U-Net, MobileNetV2, SqueezeNet, YOLOv8n, DnCNN, LR-ASPP,
             then GPU benchmarks.
Android builds require ANDROID_NDK_ROOT and host glslc.
Running Android tests also requires adb and a connected arm64 device.
Running model tests requires exported ONNX models, inputs, and reference outputs.
EOF
}
platform=macos
test_mode=unittests
while getopts ":t:r:h" option; do
  case "$option" in
    t) platform="$OPTARG" ;;
    r) test_mode="$OPTARG" ;;
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
model_targets=(vkai_resnet18_test vkai_unet_test vkai_mobilenet_v2_test
               vkai_squeezenet_test vkai_yolov8n_test
               vkai_dncnn_test vkai_lraspp_test)
case "$test_mode" in
  0) targets=(vkai_unittests "${model_targets[@]}") ;;
  unittests) targets=(vkai_unittests) ;;
  models) targets=("${model_targets[@]}") ;;
  *) echo "Unknown test group: $test_mode" >&2; usage >&2; exit 2 ;;
esac

project_root="$(cd "$(dirname "$0")/.." && pwd)"
remote=/data/local/tmp/vkai

run_cpp_tests() {
  local executable="$1"
  shift
  if [[ "$platform" == android ]]; then
    adb push "$executable" "$remote/" >&2
    local name="${executable##*/}"
    adb shell "chmod 755 $remote/$name"
    adb shell "$remote/$name" "$@"
  else
    "$executable" "$@"
  fi
}

run_model_test() {
  local model="$1"
  local folder="$project_root/python/models/artifacts/$model"
  local runtime_folder="$folder"
  local executable="$build_dir/tasks/vkai_${model}_test"
  local required=(model.onnx input.bin reference.bin)
  if [[ "$model" == yolov8n ]]; then
    required+=(reference_detections.txt input.jpg)
  elif [[ "$model" == dncnn ]]; then
    required+=(clean.bin)
  elif [[ "$model" == lraspp ]]; then
    required+=(reference_mask.bin input.png)
  fi
  local file
  for file in "${required[@]}"; do
    if [[ ! -f "$folder/$file" ]]; then
      echo "Missing $folder/$file. Run python/models/export_$model.py first." >&2
      return 1
    fi
  done
  if [[ "$platform" == android ]]; then
    runtime_folder="$remote/models/$model"
    adb shell "mkdir -p $runtime_folder"
    for file in "${required[@]}"; do
      adb push "$folder/$file" "$runtime_folder/" >&2
    done
  fi
  local args=("$runtime_folder/model.onnx" "$runtime_folder/input.bin"
              "$runtime_folder/$platform.bin")
  if [[ "$model" == yolov8n ]]; then
    args+=("$runtime_folder/${platform}_detections.json")
  fi
  args+=("$runtime_folder/reference.bin")
  if [[ "$model" == yolov8n ]]; then
    args+=("$runtime_folder/reference_detections.txt" "$runtime_folder/input.jpg"
           "$runtime_folder/${platform}_detections.png")
  elif [[ "$model" == dncnn ]]; then
    args+=("$runtime_folder/clean.bin")
  elif [[ "$model" == lraspp ]]; then
    args+=("$runtime_folder/reference_mask.bin" "$runtime_folder/input.png")
  fi
  run_cpp_tests "$executable" "${args[@]}" | tee "$folder/${platform}_validation.json"
  if [[ "$platform" == android ]]; then
    adb pull "$runtime_folder/$platform.bin" "$folder/$platform.bin" >&2
    if [[ "$model" == yolov8n ]]; then
      adb pull "$runtime_folder/${platform}_detections.json" \
        "$folder/${platform}_detections.json" >&2
      adb pull "$runtime_folder/${platform}_detections.png" \
        "$folder/${platform}_detections.png" >&2
    elif [[ "$model" == dncnn ]]; then
      for file in noisy denoised comparison; do
        adb pull "$runtime_folder/${platform}_$file.png" \
          "$folder/${platform}_$file.png" >&2
      done
    elif [[ "$model" == lraspp ]]; then
      for file in mask color overlay comparison; do
        adb pull "$runtime_folder/${platform}_$file.png" \
          "$folder/${platform}_$file.png" >&2
      done
    fi
  fi
}

run_model_benchmark() {
  local model="$1"
  local folder="$project_root/python/models/artifacts/$model"
  if [[ "$platform" == android ]]; then
    local runtime_folder="$remote/models/$model"
    adb shell "$remote/vkai_${model}_test" --benchmark "$runtime_folder/model.onnx" \
      "$runtime_folder/input.bin" "$runtime_folder/reference.bin" 5 30 \
      | tee "$folder/${platform}_benchmark.json"
  else
    "$build_dir/tasks/vkai_${model}_test" --benchmark "$folder/model.onnx" \
      "$folder/input.bin" "$folder/reference.bin" 5 30 \
      | tee "$folder/${platform}_benchmark.json"
  fi
}

if [[ "$platform" == android ]]; then
  : "${ANDROID_NDK_ROOT:?Set ANDROID_NDK_ROOT to an installed Android NDK}"
  if [[ "$test_mode" != 0 ]]; then
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
if [[ "$test_mode" == 0 ]]; then
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
    for model in resnet18 unet mobilenet_v2 squeezenet yolov8n dncnn lraspp; do
      run_model_test "$model"
    done
    for model in mobilenet_v2 squeezenet yolov8n dncnn lraspp; do
      run_model_benchmark "$model"
    done
    ;;
esac
echo "PASS: $platform $test_mode"
