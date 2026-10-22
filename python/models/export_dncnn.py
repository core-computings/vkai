"""Export pretrained color DnCNN, deterministic Gaussian noise, and PyTorch references."""

import argparse
import hashlib
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn

from export_common import download, export

ROOT = Path(__file__).resolve().parent
KAIR_REV = "fc1732f4a4514e42ce15e5b3a1e18c828af47a1e"
WEIGHTS_URL = "https://github.com/cszn/KAIR/releases/download/v1.0/dncnn_color_blind.pth"
WEIGHTS_SHA256 = "dfe11a6b703304df82cdf236f431fc6492766c7b9119d97e943edeb392c710a3"
IMAGE_URL = f"https://raw.githubusercontent.com/cszn/KAIR/{KAIR_REV}/testsets/set5/baby.bmp"


class DnCNN(nn.Module):
    """KAIR's 20-layer RGB DnCNN, using the official BN-fused inference weights."""

    def __init__(self):
        super().__init__()
        layers = [nn.Conv2d(3, 64, 3, padding=1), nn.ReLU()]
        for _ in range(18):
            layers.extend([nn.Conv2d(64, 64, 3, padding=1), nn.ReLU()])
        layers.append(nn.Conv2d(64, 3, 3, padding=1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return x - self.model(x)


def save_image(chw, path):
    pixels = np.rint(np.clip(chw.transpose(1, 2, 0), 0, 1) * 255).astype(np.uint8)
    Image.fromarray(pixels).save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/dncnn")
    args = parser.parse_args()
    torch.set_num_threads(4)
    folder = args.output_dir
    folder.mkdir(parents=True, exist_ok=True)
    weights = download(WEIGHTS_URL, folder / "dncnn_color_blind.pth")
    if hashlib.sha256(weights.read_bytes()).hexdigest() != WEIGHTS_SHA256:
        raise RuntimeError("DnCNN checkpoint SHA-256 differs from the pinned official weights")
    model = DnCNN().eval()
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True), strict=True)
    source = download(IMAGE_URL, folder / "source.bmp")
    image = Image.open(source).convert("RGB").resize((256, 256), Image.Resampling.BICUBIC)
    clean = np.asarray(image, dtype=np.float32).transpose(2, 0, 1).copy() / 255
    # Match the author's evaluation: floating-point AWGN, without clipping or
    # quantizing the network input. Clip only the PNG visualizations.
    sigma, seed = 25, 0
    noise = np.random.RandomState(seed).normal(0, sigma / 255, (256, 256, 3))
    noisy = clean + noise.transpose(2, 0, 1).astype(np.float32)
    x = torch.from_numpy(noisy.copy()).unsqueeze(0)
    with torch.inference_mode():
        denoised = model(x)[0].numpy()
    noisy_psnr = -10 * np.log10(np.mean((noisy.astype(np.float64) - clean) ** 2))
    denoised_psnr = -10 * np.log10(np.mean((denoised.astype(np.float64) - clean) ** 2))
    if denoised_psnr < noisy_psnr + 3:
        raise RuntimeError("Pretrained DnCNN must improve PSNR by at least 3 dB")
    export(
        "dncnn",
        model,
        x,
        folder,
        dict(
            variant="dncnn_color_blind; 20 layers, 64 features; BN fused",
            source_revision=KAIR_REV,
            architecture_url=f"https://github.com/cszn/KAIR/blob/{KAIR_REV}/models/network_dncnn.py",
            weights_url=WEIGHTS_URL,
            weights_sha256=hashlib.sha256(weights.read_bytes()).hexdigest(),
            image_url=IMAGE_URL,
            parameters=sum(p.numel() for p in model.parameters()),
            noise_sigma_255=sigma,
            noise_seed=seed,
            preprocessing="RGB; bicubic resize 256x256; [0,1]; add unclipped float32 AWGN",
            psnr_domain="unclipped float32 RGB, peak=1, all pixels including borders",
            noisy_psnr_db=float(noisy_psnr),
            pytorch_psnr_db=float(denoised_psnr),
            min_psnr_gain_db=3,
        ),
    )
    clean.astype("<f4").tofile(folder / "clean.bin")
    save_image(clean, folder / "clean.png")
    save_image(noisy, folder / "noisy.png")
    save_image(denoised, folder / "reference.png")


if __name__ == "__main__":
    main()
