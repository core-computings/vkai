"""Validate both builds and compare all Vulkan allocations and inference timings."""

import argparse
import array
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

MODELS = ["resnet18", "unet", "mobilenet_v2", "yolov8n", "dncnn"]


def run(command):
    result = subprocess.run([str(x) for x in command], text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(f"Command failed: {command}\n{result.stdout}\n{result.stderr}")
    return result.stdout


def save_json(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n")


def parse_result(output):
    return json.loads(next(line for line in reversed(output.splitlines()) if line.startswith("{")))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True, help="Baseline build directory")
    parser.add_argument("--after", type=Path, required=True, help="Updated build directory")
    parser.add_argument("--platform", choices=["macos", "android"], default="macos")
    parser.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    parser.add_argument("--artifacts", type=Path, default=Path("python/models/artifacts"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--gpu-runs", type=int, default=30)
    parser.add_argument("--e2e-runs", type=int, default=5)
    parser.add_argument("--cooldown", type=float, default=5)
    args = parser.parse_args()
    if min(args.repeats, args.gpu_runs, args.e2e_runs) <= 0 or args.warmup < 0 or args.cooldown < 0:
        parser.error("Invalid iteration counts or cooldown")
    args.output.mkdir(parents=True, exist_ok=True)
    builds = {"before": args.before.resolve(), "after": args.after.resolve()}
    remote = "/data/local/tmp/vkai-memory-comparison"
    android = args.platform == "android"
    if android:
        run(["adb", "shell", "mkdir", "-p", remote + "/models"])
    model_dirs = {}
    hashes = {}
    for model in args.models:
        folder = args.artifacts.resolve() / model
        files = ["model.onnx", "input.bin", "reference.bin"]
        if model == "yolov8n":
            files += ["reference_detections.txt", "input.jpg"]
        if model == "dncnn":
            files += ["clean.bin"]
        hashes[model] = {
            name: hashlib.sha256((folder / name).read_bytes()).hexdigest() for name in files
        }
        model_dirs[model] = str(folder) if not android else remote + "/models/" + model
        if android:
            run(["adb", "shell", "mkdir", "-p", model_dirs[model]])
            for name in files:
                run(["adb", "push", folder / name, model_dirs[model] + "/"])
    programs = {}
    for label, build in builds.items():
        names = [f"vkai_{model}_test" for model in args.models] + ["vkai_memory_benchmark"]
        programs[label] = {}
        for name in names:
            executable = build / "tasks" / name
            if not executable.exists():
                executable = build / name
            programs[label][name] = str(executable)
            if android:
                target = f"{remote}/{label}-{name}"
                run(["adb", "push", executable, target])
                run(["adb", "shell", "chmod", "755", target])
                programs[label][name] = target

    def execute(label, program, parameters):
        command = [programs[label][program], *parameters]
        return run(["adb", "shell", *command] if android else command)

    metadata = {
        "platform": args.platform,
        "models": args.models,
        "repeats": args.repeats,
        "warmup": args.warmup,
        "gpu_runs_per_repeat": args.gpu_runs,
        "e2e_runs_per_repeat": args.e2e_runs,
        "cooldown_seconds": args.cooldown,
        "model_sha256": hashes,
        "builds": {k: str(v) for k, v in builds.items()},
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "environment": {
            name: os.environ.get(name) for name in ["VK_INSTANCE_LAYERS", "VULKAN_SDK"]
        },
    }
    save_json(args.output / "metadata.json", metadata)
    for label in builds:
        for model in args.models:
            folder = model_dirs[model]
            output_path = (
                f"{folder}/{label}-validation.bin"
                if android
                else str(args.output / f"{label}-{model}.bin")
            )
            parameters = [folder + "/model.onnx", folder + "/input.bin", output_path]
            if model == "yolov8n":
                parameters += [output_path + ".json"]
            parameters += [folder + "/reference.bin"]
            if model == "yolov8n":
                parameters += [
                    folder + "/reference_detections.txt", folder + "/input.jpg", output_path + ".png"
                ]
            if model == "dncnn":
                parameters += [folder + "/clean.bin"]
            output = execute(label, f"vkai_{model}_test", parameters)
            (args.output / f"{label}-{model}-validation.log").write_text(output)
            save_json(args.output / f"{label}-{model}-validation.json", parse_result(output))
            print(f"{args.platform}: {label} {model} validation passed", flush=True)

    comparison = {}
    for model in args.models:
        payloads = []
        tensors = []
        for label in builds:
            output_path = args.output / f"{label}-{model}.bin"
            if android:
                run(["adb", "pull", f"{model_dirs[model]}/{label}-validation.bin", output_path])
            payload = output_path.read_bytes()
            values = array.array("f")
            values.frombytes(payload)
            payloads.append(payload)
            tensors.append(values)
        if len(tensors[0]) != len(tensors[1]):
            raise RuntimeError(f"Before/after output shape differs: {model}")
        comparison[model] = {
            "elements": len(tensors[0]),
            "max_abs_difference": max(abs(a - b) for a, b in zip(*tensors)),
            "different_elements": sum(a != b for a, b in zip(*tensors)),
            "byte_identical": payloads[0] == payloads[1],
        }
    save_json(args.output / "before_after_output_comparison.json", comparison)

    measurements = {model: {label: [] for label in builds} for model in args.models}
    for repeat in range(args.repeats):
        labels = ["before", "after"] if repeat % 2 == 0 else ["after", "before"]
        for model in args.models:
            folder = model_dirs[model]
            for label in labels:
                time.sleep(args.cooldown)
                parameters = [
                    folder + "/model.onnx", folder + "/input.bin", folder + "/reference.bin",
                    8 if model == "dncnn" else 2**63 - 1,
                    args.warmup, args.gpu_runs, args.e2e_runs,
                ]
                output = execute(label, "vkai_memory_benchmark", parameters)
                (args.output / f"{label}-{model}-{repeat + 1}.log").write_text(output)
                result = parse_result(output)
                measurements[model][label].append(result)
                save_json(args.output / f"{label}-{model}-{repeat + 1}.json", result)
                print(
                    f"{args.platform}: round {repeat + 1} {label} {model}: "
                    f"{result['allocated_bytes'] / 2**20:.2f} MiB, "
                    f"GPU median {statistics.median(result['gpu_ms']):.3f} ms", flush=True
                )
                save_json(args.output / "measurements.json", measurements)


if __name__ == "__main__":
    main()
