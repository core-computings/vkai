"""Export pretrained ResNet-18 and its image classification reference data."""

import argparse
from pathlib import Path

import torch
from torchvision.models import ResNet18_Weights, resnet18

from export_common import export_classifier

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/resnet18")
    args = parser.parse_args()
    torch.set_num_threads(4)
    weights = ResNet18_Weights.IMAGENET1K_V1
    model = resnet18(weights=weights).eval()
    export_classifier("resnet18", model, weights, args.output_dir)


if __name__ == "__main__":
    main()
