"""Benchmark selected models with GPU timestamps, warmup, and output checks."""

import argparse
import json
from pathlib import Path
from validate_models import run

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", choices=["macos", "android"], default="macos")
    parser.add_argument("--models", nargs="+", default=["mobilenet_v2", "yolov8n"])
    parser.add_argument(
        "--runner", type=Path, default=ROOT.parents[1] / "build/tasks/vkai_benchmark"
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--artifacts", type=Path, default=ROOT / "artifacts")
    args = parser.parse_args()
    if args.platform == "android":
        run(["adb", "shell", "mkdir -p /data/local/tmp/vkai/models"])
        run(["adb", "push", args.runner, "/data/local/tmp/vkai/vkai_benchmark"])
        run(["adb", "shell", "chmod 755 /data/local/tmp/vkai/vkai_benchmark"])
    report = {}
    for name in args.models:
        folder = args.artifacts / name
        if args.platform == "android":
            remote = f"/data/local/tmp/vkai/models/{name}"
            run(["adb", "shell", f"mkdir -p {remote}"])
            for filename in ["model.onnx", "input.bin", "reference.bin"]:
                run(["adb", "push", folder / filename, remote + "/"])
            output = run(
                [
                    "adb",
                    "shell",
                    "/data/local/tmp/vkai/vkai_benchmark",
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
                    args.runner,
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
