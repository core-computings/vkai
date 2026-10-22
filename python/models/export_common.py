"""Shared download, ONNX export, and reference verification utilities."""

import hashlib
import json
from urllib.request import urlretrieve

import numpy as np
import onnx
import onnxruntime as ort
import torch
from PIL import Image


def download(url, path):
    if not path.exists():
        urlretrieve(url, path)
    return path


def export(name, model, x, folder, extra, optimize=False):
    folder.mkdir(parents=True, exist_ok=True)
    model.eval()
    with torch.inference_mode():
        y = model(x).contiguous().numpy()
    path = folder / "model.onnx"
    torch.onnx.export(
        model,
        x,
        path,
        input_names=["input"],
        output_names=["output"],
        opset_version=13,
        do_constant_folding=optimize,
        dynamo=False,
    )
    if optimize:
        from onnxslim import slim

        onnx.save(slim(onnx.load(path)), path)
    onnx.checker.check_model(str(path))
    actual = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"]).run(
        None, {"input": x.numpy()}
    )[0]
    np.testing.assert_allclose(actual, y, atol=2e-4, rtol=2e-4)
    x.numpy().astype("<f4").tofile(folder / "input.bin")
    y.astype("<f4").tofile(folder / "reference.bin")
    meta = dict(
        model=name,
        input_shape=list(x.shape),
        output_shape=list(y.shape),
        onnx_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        torch_version=torch.__version__,
        ort_max_error=float(np.abs(actual - y).max()),
        operators=sorted({node.op_type for node in onnx.load(path).graph.node}),
        **extra,
    )
    (folder / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2), flush=True)
    return y


def export_classifier(name, model, weights, folder):
    """Prepare the shared classification image and export logits with top-5 metadata."""
    folder.mkdir(parents=True, exist_ok=True)
    image = download(
        "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg",
        folder / "input.jpg",
    )
    x = weights.transforms()(Image.open(image).convert("RGB")).unsqueeze(0)
    model.eval()
    with torch.inference_mode():
        probabilities = model(x).softmax(1)[0]
    top = probabilities.topk(5)
    export(
        name,
        model,
        x,
        folder,
        dict(
            weights_url=weights.url,
            categories=weights.meta["categories"],
            top5=[
                dict(index=i, label=weights.meta["categories"][i], probability=p)
                for p, i in zip(top.values.tolist(), top.indices.tolist())
            ],
        ),
    )
