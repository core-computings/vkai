# MNIST Python workflow

This directory contains all MNIST-specific Python code, downloaded dataset
files, trained artifacts, and PyTorch reference tensors consumed by the C++
unit tests.

Run commands from any working directory; default paths are resolved relative to
this directory.

```bash
# From the repository root, activate the existing environment if needed.
source python/vkai_py/bin/activate

# Train and create mnist_model.pth and mnist_weights.bin.
python python/mnist/train.py

# Also export mnist_model.onnx.
python python/mnist/train.py --export-onnx

# Generate test_data/*.bin for the C++ layer and end-to-end tests.
python python/mnist/test.py --index 0

# Build and run the C++ tests.
cmake --build build
ctest --test-dir build --output-on-failure
```

`train.py` downloads MNIST into `data/` when needed. `test.py` reads the trained
model and exports raw little-endian float32 tensors plus
`test_data/metadata.json`.

Pass `--export-onnx` to `train.py` to additionally write `mnist_model.onnx`.

The generated directories and model binaries are intentionally ignored by Git.
They can always be recreated with the two commands above.
