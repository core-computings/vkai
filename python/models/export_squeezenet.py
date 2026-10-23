"""Export pretrained SqueezeNet and its image classification reference data."""

import argparse
from pathlib import Path

import torch
from torchvision.models import (
    SqueezeNet1_1_Weights as SqueezeNetWeights,
    squeezenet1_1 as squeezenet,
)

from export_common import export_classifier

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/squeezenet")
    args = parser.parse_args()
    torch.set_num_threads(4)
    weights = SqueezeNetWeights.IMAGENET1K_V1
    model = squeezenet(weights=weights).eval()
    export_classifier("squeezenet", model, weights, args.output_dir)


if __name__ == "__main__":
    main()
