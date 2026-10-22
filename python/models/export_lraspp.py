"""Export pretrained LR-ASPP MobileNetV3-Large and semantic segmentation references."""

import argparse
import hashlib
from pathlib import Path

import numpy as np
import torch
import torchvision
from export_common import download, export
from PIL import Image
from torchvision.models.segmentation import (
    LRASPP_MobileNet_V3_Large_Weights,
    lraspp_mobilenet_v3_large,
)
from torchvision.transforms.functional import normalize, pil_to_tensor

ROOT = Path(__file__).resolve().parent


class Logits(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)["out"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/lraspp")
    args = parser.parse_args()
    torch.set_num_threads(4)
    folder = args.output_dir
    folder.mkdir(parents=True, exist_ok=True)
    # Use the same public sample as the existing classification tests.
    image_url = "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg"
    source = download(image_url, folder / "source.jpg")
    image = (
        Image.open(source).convert("RGB").resize((320, 320), Image.Resampling.BILINEAR)
    )
    image.save(folder / "input.png")
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    x = normalize(pil_to_tensor(image).float() / 255, mean, std).unsqueeze(0)
    weights = LRASPP_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1
    model = Logits(lraspp_mobilenet_v3_large(weights=weights)).eval()
    reference = export(
        "lraspp",
        model,
        x,
        folder,
        {
            "weights_url": weights.url,
            "weights": "LRASPP_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1",
            "torchvision_version": torchvision.__version__,
            "categories": weights.meta["categories"],
            "parameters": sum(p.numel() for p in model.parameters()),
            "image_url": image_url,
            "image_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "preprocessing": {
                "resize": [320, 320],
                "interpolation": "PIL bilinear",
                "mean": mean,
                "std": std,
            },
            "output_layout": "NCHW logits; class argmax along C, including background",
        },
        optimize=True,
    )
    mask = torch.from_numpy(reference).argmax(dim=1)[0].numpy().astype(np.uint8)
    if not np.any(mask == 0) or not np.any(mask == 12):
        raise RuntimeError("Expected background and dog in the sample segmentation")
    # Independent PyTorch argmax reference; C++ must match this as well as raw logits.
    mask.tofile(folder / "reference_mask.bin")
    Image.fromarray(mask).save(folder / "reference_mask.png")


if __name__ == "__main__":
    main()
