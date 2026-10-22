"""Benchmark selected models with GPU timestamps, warmup, and output checks."""

import argparse
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run(command):
    result = subprocess.run(
        [str(x) for x in command], check=False, text=True, capture_output=True
    )
    if result.returncode:
        raise RuntimeError(
            f"Command failed: {command}\n{result.stdout}\n{result.stderr}"
        )
    return result.stdout


def model_runner(name, platform, runner_dir):
    if runner_dir is None:
        build_dir = "build-android" if platform == "android" else "build"
        runner_dir = ROOT.parents[1] / build_dir / "tasks"
    return runner_dir / f"vkai_{name}_test"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", choices=["macos", "android"], default="macos")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["resnet18", "unet", "mobilenet_v2", "yolov8n", "dncnn"],
        default=["mobilenet_v2", "yolov8n", "dncnn"],
    )
    parser.add_argument(
        "--runner-dir", type=Path, help="Directory containing the per-network C++ tests"
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--artifacts", type=Path, default=ROOT / "artifacts")
    args = parser.parse_args()
    if args.platform == "android":
        run(["adb", "shell", "mkdir -p /data/local/tmp/vkai/models"])
    report = {}
    for name in args.models:
        folder = args.artifacts / name
        runner = model_runner(name, args.platform, args.runner_dir)
        if args.platform == "android":
            remote = f"/data/local/tmp/vkai/models/{name}"
            remote_runner = f"/data/local/tmp/vkai/{runner.name}"
            run(["adb", "push", runner, remote_runner])
            run(["adb", "shell", f"chmod 755 {remote_runner}"])
            run(["adb", "shell", f"mkdir -p {remote}"])
            for filename in ["model.onnx", "input.bin", "reference.bin"]:
                run(["adb", "push", folder / filename, remote + "/"])
            output = run(
                [
                    "adb",
                    "shell",
                    remote_runner,
                    "--benchmark",
                    remote + "/model.onnx",
                    remote + "/input.bin",
                    remote + "/reference.bin",
                    args.warmup,
                    args.runs,
                ]
            )
        else:
            output = run(
                [
                    runner,
                    "--benchmark",
                    folder / "model.onnx",
                    folder / "input.bin",
                    folder / "reference.bin",
                    args.warmup,
                    args.runs,
                ]
            )
        report[name] = json.loads(output.strip().splitlines()[-1])
        (folder / f"{args.platform}_benchmark.json").write_text(
            json.dumps(report[name], indent=2) + "\n"
        )
        print(
            f"{name}: mean={report[name]['mean_ms']:.3f} ms, median={report[name]['median_ms']:.3f} ms",
            flush=True,
        )


if __name__ == "__main__":
    main()
