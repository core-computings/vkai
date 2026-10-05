"""
Train a simple CNN for MNIST digit classification and export inference artifacts.
This demonstrates the training workflow described in Chapter 5.

Usage:
    python python/mnist/train.py
    python python/mnist/train.py --export-onnx

Requirements:
    pip install torch torchvision flatbuffers onnx
"""

import argparse
import sys
from pathlib import Path

import flatbuffers
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


SCRIPT_DIR = Path(__file__).resolve().parent
PYTHON_DIR = SCRIPT_DIR.parent
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from vkai.fbs import Model, Tensor

DATA_DIR = SCRIPT_DIR / "data"
MODEL_PATH = SCRIPT_DIR / "mnist_model.pth"
WEIGHTS_PATH = SCRIPT_DIR / "mnist_weights.bin"
ONNX_PATH = SCRIPT_DIR / "mnist_model.onnx"


class MNISTNet(nn.Module):
    """Simplest possible network for MNIST - single hidden layer.

    This is a "hello world" intro to ML inference with Vulkan.
    Architecture: 784 -> 128 -> 10 (just input->hidden->output)
    """

    def __init__(self):
        super(MNISTNet, self).__init__()

        # Single hidden layer - as simple as it gets!
        self.fc1 = nn.Linear(28 * 28, 128)  # 784 -> 128
        self.fc2 = nn.Linear(128, 10)       # 128 -> 10
        self.relu = nn.ReLU()

    def forward(self, x):
        # Flatten input
        x = x.view(-1, 28 * 28)

        # Single hidden layer with ReLU
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x


def train_mnist(
    num_epochs=5, batch_size=64, learning_rate=0.001, data_dir=DATA_DIR
):
    """Train the MNIST model."""

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Prepare data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
    ])

    train_dataset = datasets.MNIST(
        root=str(data_dir),
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.MNIST(
        root=str(data_dir),
        train=False,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    # Create model
    model = MNISTNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print(f"\nModel architecture:")
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters())}")

    # Training loop
    print(f"\nTraining for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)

            if batch_idx % 100 == 0:
                print(f'Epoch {epoch + 1}/{num_epochs}, '
                      f'Batch {batch_idx}/{len(train_loader)}, '
                      f'Loss: {loss.item():.4f}')

        train_accuracy = 100. * correct / total
        avg_loss = running_loss / len(train_loader)

        # Evaluate on test set
        model.eval()
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1)
                test_correct += (pred == target).sum().item()
                test_total += target.size(0)

        test_accuracy = 100. * test_correct / test_total

        print(f'\nEpoch {epoch + 1} Summary:')
        print(f'  Train Loss: {avg_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%')
        print(f'  Test Accuracy: {test_accuracy:.2f}%\n')

    return model


def export_weights_binary(model, filename=WEIGHTS_PATH):
    """Export model state_dict tensors as a FlatBuffers weights artifact."""

    model.eval()

    filename = Path(filename).expanduser().resolve()
    filename.parent.mkdir(parents=True, exist_ok=True)

    builder = flatbuffers.Builder(0)
    tensor_offsets = []
    for name, tensor in model.state_dict().items():
        array = tensor.detach().cpu().contiguous().numpy().astype('<f4', copy=False)
        shape = builder.CreateNumpyVector(np.asarray(array.shape, dtype='<u4'))
        data = builder.CreateNumpyVector(array.reshape(-1))
        tensor_name = builder.CreateString(name)
        Tensor.TensorStart(builder)
        Tensor.TensorAddName(builder, tensor_name)
        Tensor.TensorAddShape(builder, shape)
        Tensor.TensorAddData(builder, data)
        tensor_offsets.append(Tensor.TensorEnd(builder))
        print(f"  {name}: {array.shape}, {array.size} float32 values")

    Model.ModelStartWeightsVector(builder, len(tensor_offsets))
    for tensor_offset in reversed(tensor_offsets):
        builder.PrependUOffsetTRelative(tensor_offset)
    weights = builder.EndVector()
    model_name = builder.CreateString(model.__class__.__name__)
    Model.ModelStart(builder)
    Model.ModelAddName(builder, model_name)
    Model.ModelAddWeights(builder, weights)
    model_offset = Model.ModelEnd(builder)
    builder.Finish(model_offset, file_identifier=b"VKAI")
    filename.write_bytes(builder.Output())

    print(f"\nWeights exported to {filename}")


def export_onnx(model, filename=ONNX_PATH):
    """Export the trained model as an ONNX graph with a dynamic batch dimension."""

    model.eval()
    filename = Path(filename).expanduser().resolve()
    filename.parent.mkdir(parents=True, exist_ok=True)
    device = next(model.parameters()).device
    example_input = torch.zeros((1, 1, 28, 28), dtype=torch.float32, device=device)

    with torch.no_grad():
        torch.onnx.export(
            model,
            example_input,
            filename,
            dynamo=False,
            opset_version=13,
            input_names=["input"],
            output_names=["logits"],
            dynamic_axes={
                "input": {0: "batch_size"},
                "logits": {0: "batch_size"},
            },
        )

    print(f"\nONNX model exported to {filename}")


def test_inference(model, data_dir=DATA_DIR):
    """Test a single inference to verify the model works."""

    model.eval()

    # Get a single test image
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    test_dataset = datasets.MNIST(
        root=str(data_dir), train=False, download=False, transform=transform
    )
    test_image, test_label = test_dataset[0]

    # Run inference
    with torch.no_grad():
        output = model(test_image.unsqueeze(0))
        probabilities = torch.softmax(output, dim=1)
        predicted_class = output.argmax(dim=1).item()

    print(f"\nTest Inference:")
    print(f"  True label: {test_label}")
    print(f"  Predicted: {predicted_class}")
    print(f"  Confidence: {probabilities[0][predicted_class].item() * 100:.2f}%")
    print(f"\nAll class probabilities:")
    for i, prob in enumerate(probabilities[0]):
        print(f"  {i}: {prob.item() * 100:.2f}%")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train MNIST and export artifacts for the C++ inference tests."
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--weights", type=Path, default=WEIGHTS_PATH)
    parser.add_argument(
        "--export-onnx",
        action="store_true",
        help=f"Export an ONNX model to {ONNX_PATH}",
    )
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    return args


def main():
    args = parse_args()
    print("=" * 60)
    print("MNIST Training and Export")
    print("=" * 60)

    # Train model
    model = train_mnist(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        data_dir=args.data_dir,
    )

    # Save PyTorch model
    model_path = args.model.expanduser().resolve()
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"\nPyTorch model saved to {model_path}")

    # Test inference
    test_inference(model, args.data_dir)

    # Export weights for C++ inference engine
    print("\nExporting weights for C++ inference engine:")
    export_weights_binary(model, args.weights)

    if args.export_onnx:
        export_onnx(model)

    print("\n" + "=" * 60)
    print("Training and export complete!")
    print("=" * 60)
    print("\nYou can now use:")
    print(f"  - {args.weights.expanduser().resolve()} for the C++ unit tests")
    if args.export_onnx:
        print(f"  - {ONNX_PATH.resolve()} for ONNX-compatible runtimes")


if __name__ == "__main__":
    main()
