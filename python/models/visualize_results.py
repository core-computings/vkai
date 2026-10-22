"""Render masks and detection boxes from existing C++ model-test outputs."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent


def render_unet(folder, platform):
    meta = json.loads((folder / "metadata.json").read_text())
    height, width = meta["output_shape"][-2:]
    output = np.fromfile(folder / f"{platform}.bin", dtype="<f4")
    mask = output.reshape(height, width) >= 0.5
    source = np.array(
        Image.open(folder / "input.png").convert("RGB").resize((width, height))
    )
    source[mask] = (source[mask] * 0.5 + np.array([255, 0, 0]) * 0.5).astype(np.uint8)
    Image.fromarray(source).save(folder / f"{platform}_overlay.png")
    Image.fromarray((mask * 255).astype(np.uint8)).save(folder / f"{platform}_mask.png")


def render_yolo(folder, platform):
    meta = json.loads((folder / "metadata.json").read_text())
    result = json.loads((folder / f"{platform}_detections.json").read_text())
    image = Image.open(folder / "input.jpg").convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=22)
    p = meta["preprocessing"]
    for detection in result["detections"]:
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
    image.save(folder / f"{platform}_detections.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", choices=["macos", "android"], default="macos")
    parser.add_argument("--artifacts", type=Path, default=ROOT / "artifacts")
    parser.add_argument(
        "--models", nargs="+", choices=["unet", "yolov8n"], default=["unet", "yolov8n"]
    )
    args = parser.parse_args()
    for name in args.models:
        render = render_unet if name == "unet" else render_yolo
        render(args.artifacts / name, args.platform)
        print(f"Rendered {name} ({args.platform})")


if __name__ == "__main__":
    main()
