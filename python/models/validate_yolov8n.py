"""Validate device-generated YOLO detections and render boxes on the input image."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.ops import batched_nms, box_iou
from validate_models import run

ROOT = Path(__file__).resolve().parent


def reference_detections(predictions, meta):
    output = torch.from_numpy(predictions.reshape(meta["output_shape"])[0].copy())
    scores, labels = output[4:].max(0)
    post = meta["postprocessing"]
    selected = scores >= post["confidence_threshold"]
    xywh = output[:4, selected].T
    boxes = torch.cat((xywh[:, :2] - xywh[:, 2:] / 2, xywh[:, :2] + xywh[:, 2:] / 2), 1)
    scores = scores[selected]
    labels = labels[selected]
    keep = batched_nms(boxes, scores, labels, post["iou_threshold"])[
        : post["max_detections"]
    ]
    return [
        dict(
            class_id=int(labels[i]),
            confidence=float(scores[i]),
            box_xyxy=boxes[i].tolist(),
        )
        for i in keep
    ]


def validate(folder, platform):
    meta = json.loads((folder / "metadata.json").read_text())
    reference = np.fromfile(folder / "reference.bin", dtype="<f4")
    actual = np.fromfile(folder / f"{platform}.bin", dtype="<f4")
    assert actual.size == reference.size == int(np.prod(meta["output_shape"]))
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, reference, atol=1e-3, rtol=1e-3)
    expected = reference_detections(reference, meta)
    (folder / "reference_detections.json").write_text(
        json.dumps(expected, indent=2) + "\n"
    )
    device = json.loads((folder / f"{platform}_detections.json").read_text())
    detected = device["detections"]
    assert len(detected) == len(expected) > 0, "Detection count differs from reference"
    assert {0, 5}.issubset({d["class_id"] for d in detected}), (
        "Expected people and bus in the sample"
    )
    ious = []
    for detection, ref in zip(detected, expected):
        assert detection["class_id"] == ref["class_id"], (
            "Detection class/order mismatch"
        )
        assert abs(detection["confidence"] - ref["confidence"]) < 1e-4, (
            "Detection confidence differs"
        )
        iou = float(
            box_iou(
                torch.tensor([detection["box_xyxy"]]), torch.tensor([ref["box_xyxy"]])
            )[0, 0]
        )
        assert iou > 0.999, f"Detection box differs from PyTorch reference: {iou}"
        ious.append(iou)
    image = Image.open(folder / "input.jpg").convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=22)
    p = meta["preprocessing"]
    mapped = []
    for detection in detected:
        x1, y1, x2, y2 = detection["box_xyxy"]
        box = [
            np.clip((x1 - p["pad_left"]) / p["scale"], 0, image.width),
            np.clip((y1 - p["pad_top"]) / p["scale"], 0, image.height),
            np.clip((x2 - p["pad_left"]) / p["scale"], 0, image.width),
            np.clip((y2 - p["pad_top"]) / p["scale"], 0, image.height),
        ]
        label = meta["categories"][detection["class_id"]]
        color = (255, 160, 20) if detection["class_id"] == 5 else (10, 180, 255)
        draw.rectangle(box, outline=color, width=4)
        text = f"{label} {detection['confidence']:.2f}"
        tx, ty = box[0], max(0, box[1] - 28)
        text_box = draw.textbbox((tx + 3, ty + 1), text, font=font)
        draw.rectangle((tx, ty, min(image.width, text_box[2] + 3), ty + 28), fill=color)
        draw.text((tx + 3, ty + 1), text, font=font, fill=(0, 0, 0))
        mapped.append(
            dict(**detection, label=label, box_original_xyxy=[float(v) for v in box])
        )
    image.save(folder / f"{platform}_detections.png")
    result = dict(
        platform=platform,
        network_wall_ms=device["network_wall_ms"],
        postprocess_ms=device["postprocess_ms"],
        max_abs_error=float(np.abs(actual - reference).max()),
        class_score_max_abs_error=float(
            np.abs(
                actual.reshape(meta["output_shape"])[:, 4:]
                - reference.reshape(meta["output_shape"])[:, 4:]
            ).max()
        ),
        detections=mapped,
        min_box_iou_vs_pytorch=min(ious),
    )
    (folder / f"{platform}_validation.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", choices=["macos", "android"], default="macos")
    parser.add_argument("--artifacts", type=Path, default=ROOT / "artifacts/yolov8n")
    parser.add_argument(
        "--runner", type=Path, default=ROOT.parents[1] / "build/tasks/vkai_detect"
    )
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    folder = args.artifacts
    if not args.verify_only:
        if args.platform == "android":
            remote = "/data/local/tmp/vkai/models/yolov8n"
            run(["adb", "shell", f"mkdir -p {remote}"])
            run(["adb", "push", args.runner, "/data/local/tmp/vkai/vkai_detect"])
            run(["adb", "shell", "chmod 755 /data/local/tmp/vkai/vkai_detect"])
            for name in ["model.onnx", "input.bin"]:
                run(["adb", "push", folder / name, remote + "/"])
            run(
                [
                    "adb",
                    "shell",
                    "/data/local/tmp/vkai/vkai_detect",
                    remote + "/model.onnx",
                    remote + "/input.bin",
                    remote + "/android.bin",
                    remote + "/android_detections.json",
                ]
            )
            for name in ["android.bin", "android_detections.json"]:
                run(["adb", "pull", remote + "/" + name, folder / name])
        else:
            run(
                [
                    args.runner,
                    folder / "model.onnx",
                    folder / "input.bin",
                    folder / "macos.bin",
                    folder / "macos_detections.json",
                ]
            )
    print(json.dumps(validate(folder, args.platform), indent=2))


if __name__ == "__main__":
    main()
