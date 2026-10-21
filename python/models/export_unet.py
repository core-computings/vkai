"""Export pretrained U-Net and its MRI segmentation reference data."""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from export_common import download, export

ROOT = Path(__file__).resolve().parent
UNET_REV = "d45f8908ab2f0246ba204c702a6161c9eb25f902"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/unet")
    args = parser.parse_args()
    torch.set_num_threads(4)
    folder = args.output_dir
    folder.mkdir(parents=True, exist_ok=True)
    source = f"https://raw.githubusercontent.com/mateuszbuda/brain-segmentation-pytorch/{UNET_REV}"
    image = download(source + "/assets/TCGA_CS_4944.png", folder / "input.png")
    model = torch.hub.load(
        f"mateuszbuda/brain-segmentation-pytorch:{UNET_REV}",
        "unet",
        in_channels=3,
        out_channels=1,
        init_features=32,
        pretrained=True,
        trust_repo=True,
    ).eval()
    # Normalize MRI channels in their original intensity range (per image).
    array = np.asarray(
        Image.open(image).convert("RGB").resize((256, 256)), dtype=np.float32
    )
    normalized = (array - array.mean((0, 1))) / np.maximum(array.std((0, 1)), 1e-6)
    x = torch.from_numpy(normalized.transpose(2, 0, 1).copy()).unsqueeze(0)
    export(
        "unet",
        model,
        x,
        folder,
        dict(
            source_revision=UNET_REV,
            weights_url="https://github.com/mateuszbuda/brain-segmentation-pytorch/releases/download/v1.0/unet-e012d006.pt",
            preprocessing="RGB MRI channels: resize 256x256; per-channel (x-mean)/std",
        ),
    )
    with torch.inference_mode():
        mask = model(x)[0, 0].numpy() >= 0.5
    Image.fromarray((mask * 255).astype(np.uint8)).save(
        folder / "reference_mask.png"
    )


if __name__ == "__main__":
    main()
