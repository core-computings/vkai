"""Export pretrained MobileNetV2 and its image classification reference data."""

import argparse
from pathlib import Path

import torch
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

from export_common import export_classifier

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/mobilenet_v2")
    args = parser.parse_args()
    torch.set_num_threads(4)
    weights = MobileNet_V2_Weights.IMAGENET1K_V2
    model = mobilenet_v2(weights=weights).eval()
    export_classifier("mobilenet_v2", model, weights, args.output_dir)


if __name__ == "__main__":
    main()
