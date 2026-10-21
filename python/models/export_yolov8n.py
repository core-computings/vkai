"""Export the complete pretrained YOLOv8n detection graph at fixed input size."""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import onnx
import torch
from export_common import download, export
from PIL import Image
from torchvision.ops import batched_nms

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "artifacts/ultralytics_settings"))


def main():
    from ultralytics import YOLO

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=640)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/yolov8n")
    args = parser.parse_args()
    if args.size <= 0 or args.size % 32:
        parser.error("--size must be a positive multiple of 32")
    folder = args.output_dir
    folder.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    image_path = download(
        "https://raw.githubusercontent.com/ultralytics/ultralytics/v8.3.200/ultralytics/assets/bus.jpg",
        folder / "input.jpg",
    )
    weights = download(
        "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt",
        folder / "yolov8n.pt",
    )
    source = np.array(Image.open(image_path).convert("RGB"))
    h, w = source.shape[:2]
    ratio = min(args.size / h, args.size / w)
    rw, rh = round(w * ratio), round(h * ratio)
    left, top = (args.size - rw) // 2, (args.size - rh) // 2
    canvas = np.full((args.size, args.size, 3), 114, dtype=np.uint8)
    canvas[top : top + rh, left : left + rw] = cv2.resize(
        source, (rw, rh), interpolation=cv2.INTER_LINEAR
    )
    Image.fromarray(canvas).save(folder / "letterboxed.png")
    x = (
        torch.from_numpy(canvas.transpose(2, 0, 1).copy())
        .float()
        .div_(255)
        .unsqueeze(0)
    )
    model = YOLO(str(weights)).model.eval()
    model.fuse(verbose=False)
    head = model.model[-1]
    head.export = True
    head.format = "onnx"
    head.dynamic = False
    postprocessing = dict(
        confidence_threshold=0.25, iou_threshold=0.45, max_detections=300
    )
    reference = export(
        "yolov8n",
        model,
        x,
        folder,
        dict(
            ultralytics_version="8.3.200",
            weights_url="https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt",
            categories=[model.names[i] for i in range(80)],
            preprocessing=dict(
                original_width=w,
                original_height=h,
                scale=ratio,
                pad_left=left,
                pad_top=top,
                resize="OpenCV INTER_LINEAR",
                fill=114,
                normalization="RGB / 255",
            ),
            postprocessing=postprocessing,
            output_layout="N x 84 x anchors: center_x, center_y, width, height, 80 class scores",
        ),
        optimize=True,
    )
    # Save independent PyTorch postprocessing ground truth for the C++ test.
    output = torch.from_numpy(reference[0].copy())
    scores, labels = output[4:].max(0)
    selected = scores >= postprocessing["confidence_threshold"]
    xywh = output[:4, selected].T
    boxes = torch.cat((xywh[:, :2] - xywh[:, 2:] / 2, xywh[:, :2] + xywh[:, 2:] / 2), 1)
    scores, labels = scores[selected], labels[selected]
    keep = batched_nms(boxes, scores, labels, postprocessing["iou_threshold"])[
        : postprocessing["max_detections"]
    ]
    with (folder / "reference_detections.txt").open("w") as stream:
        stream.write(f"{len(keep)}\n")
        for i in keep:
            values = [float(scores[i]), *boxes[i].tolist()]
            stream.write(
                f"{int(labels[i])} " + " ".join(f"{v:.9g}" for v in values) + "\n"
            )
    m = onnx.load(folder / "model.onnx")
    from collections import Counter

    print(
        "ONNX operator counts:",
        dict(Counter(n.op_type for n in m.graph.node)),
        flush=True,
    )


if __name__ == "__main__":
    main()
