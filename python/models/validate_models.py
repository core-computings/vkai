"""Run the Vulkan models, verify all outputs, and render segmentation masks."""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent


def run(command):
    result = subprocess.run([str(x) for x in command], text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(
            f"Command failed: {command}\n{result.stdout}\n{result.stderr}"
        )
    return result.stdout


def verify(folder, platform):
    meta = json.loads((folder / "metadata.json").read_text())
    reference = np.fromfile(folder / "reference.bin", dtype="<f4")
    actual = np.fromfile(folder / f"{platform}.bin", dtype="<f4")
    assert actual.size == int(np.prod(meta["output_shape"])), "Output shape mismatch"
    assert np.isfinite(actual).all(), "Non-finite Vulkan output"
    np.testing.assert_allclose(actual, reference, atol=1e-3, rtol=1e-3)
    result = dict(
        max_abs_error=float(np.abs(actual - reference).max()),
        rmse=float(np.sqrt(np.mean((actual - reference) ** 2, dtype=np.float64))),
    )
    if meta["model"] in ["resnet18", "mobilenet_v2"]:
        order = np.argsort(-actual)[:5]
        np.testing.assert_array_equal(order, np.argsort(-reference)[:5])
        probs = np.exp(actual - actual.max())
        probs /= probs.sum()
        result["top5"] = [
            dict(index=int(i), label=meta["categories"][i], probability=float(probs[i]))
            for i in order
        ]
    else:
        height, width = meta["output_shape"][-2:]
        mask = actual.reshape(height, width) >= 0.5
        expected = reference.reshape(height, width) >= 0.5
        assert mask.any() and not mask.all(), (
            "Segmentation must contain foreground and background"
        )
        dice = float(2 * (mask & expected).sum() / (mask.sum() + expected.sum()))
        assert dice >= 0.999, f"Segmentation mask differs from PyTorch: Dice={dice}"
        result.update(
            mask_dice_vs_pytorch=dice,
            mask_pixel_agreement=float((mask == expected).mean()),
            foreground_pixels=int(mask.sum()),
        )
        source = np.array(
            Image.open(folder / "input.png").convert("RGB").resize((width, height))
        )
        source[mask] = (source[mask] * 0.5 + np.array([255, 0, 0]) * 0.5).astype(
            np.uint8
        )
        Image.fromarray(source).save(folder / f"{platform}_overlay.png")
        Image.fromarray((mask * 255).astype(np.uint8)).save(
            folder / f"{platform}_mask.png"
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=ROOT / "artifacts")
    parser.add_argument(
        "--runner", type=Path, default=ROOT.parents[1] / "build/tasks/vkai_infer"
    )
    parser.add_argument("--platform", default="macos", choices=["macos", "android"])
    parser.add_argument("--models", nargs="+", default=["resnet18", "unet"])
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Check already downloaded output files",
    )
    args = parser.parse_args()
    report = {}
    if args.platform == "android" and not args.verify_only:
        run(["adb", "shell", "mkdir -p /data/local/tmp/vkai/models"])
        run(["adb", "push", args.runner, "/data/local/tmp/vkai/vkai_infer"])
        run(["adb", "shell", "chmod 755 /data/local/tmp/vkai/vkai_infer"])
    for name in args.models:
        folder = args.artifacts / name
        for filename in ["model.onnx", "input.bin", "reference.bin", "metadata.json"]:
            if not (folder / filename).is_file():
                parser.error(f"Missing {folder / filename}; run export_{name}.py first")
        metrics = {}
        if not args.verify_only:
            if args.platform == "android":
                remote = f"/data/local/tmp/vkai/models/{name}"
                run(["adb", "shell", f"mkdir -p {remote}"])
                for filename in ["model.onnx", "input.bin", "reference.bin"]:
                    run(["adb", "push", folder / filename, remote + "/"])
                stdout = run(
                    [
                        "adb",
                        "shell",
                        "/data/local/tmp/vkai/vkai_infer",
                        remote + "/model.onnx",
                        remote + "/input.bin",
                        remote + "/android.bin",
                        remote + "/reference.bin",
                    ]
                )
                run(["adb", "pull", remote + "/android.bin", folder / "android.bin"])
            else:
                stdout = run(
                    [
                        args.runner,
                        folder / "model.onnx",
                        folder / "input.bin",
                        folder / "macos.bin",
                        folder / "reference.bin",
                    ]
                )
            metrics = json.loads(stdout.strip().splitlines()[-1])
        report[name] = {**metrics, **verify(folder, args.platform)}
    path = args.artifacts / f"{args.platform}_validation.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
