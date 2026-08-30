#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from common import (
    ATOL,
    RTOL,
    ResidualModel,
    linear_model,
    reference_input,
    state_data,
    tensor_data,
    write_json,
)


def stepped(optimizer: str) -> dict[str, object]:
    model = linear_model()
    if optimizer == "sgd":
        implementation = torch.optim.SGD(model.parameters(), lr=0.05)
    elif optimizer == "adam":
        implementation = torch.optim.Adam(model.parameters(), lr=0.01)
    else:
        raise ValueError(f"unknown optimizer: {optimizer}")
    loss = torch.nn.functional.cross_entropy(model(reference_input()), torch.tensor([2, 0]))
    implementation.zero_grad()
    loss.backward()
    implementation.step()
    return {"loss": loss.item(), "state": state_data(model)}


def generate(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)

    model = linear_model()
    value = reference_input(requires_grad=True)
    output = model(value)
    linear_loss = output.square().mean()
    linear_loss.backward()

    logits = model(reference_input()).detach()
    cross_entropy = torch.nn.functional.cross_entropy(logits, torch.tensor([2, 0]))
    mse_target = torch.tensor([[0.0, 0.5, -0.5], [1.0, -1.0, 0.25]])
    mse = torch.nn.functional.mse_loss(logits, mse_target)

    residual = ResidualModel()
    residual_input = reference_input(requires_grad=True)
    residual_output = residual(residual_input)
    residual_loss = residual_output.square().mean()
    residual_loss.backward()

    safetensors_path = output_dir / "pytorch_linear.safetensors"
    save_file(model.state_dict(), safetensors_path)
    loaded = load_file(safetensors_path)
    for name, expected in model.state_dict().items():
        torch.testing.assert_close(loaded[name], expected, rtol=0, atol=0)

    reference = {
        "metadata": {
            "schema": 1,
            "torch": torch.__version__,
            "dtype": "float32",
            "rtol": RTOL,
            "atol": ATOL,
        },
        "input": tensor_data(reference_input()),
        "linear": {
            "forward": tensor_data(output),
            "loss": linear_loss.item(),
            "input_grad": tensor_data(value.grad),
            "weight_grad": tensor_data(model.weight.grad),
            "bias_grad": tensor_data(model.bias.grad),
        },
        "losses": {"cross_entropy": cross_entropy.item(), "mse": mse.item()},
        "sgd": stepped("sgd"),
        "adam": stepped("adam"),
        "residual": {
            "forward": tensor_data(residual_output),
            "loss": residual_loss.item(),
            "input_grad": tensor_data(residual_input.grad),
            "parameter_grads": {
                name: tensor_data(parameter.grad) for name, parameter in residual.named_parameters()
            },
        },
        "safetensors": {"file": safetensors_path.name, "keys": sorted(loaded)},
    }
    write_json(output_dir / "reference.json", reference)
    write_json(
        output_dir / "linear_io.json",
        {"input": reference["input"], "output": reference["linear"]["forward"]},
    )
    print(output_dir / "reference.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate deterministic PyTorch parity assets")
    parser.add_argument("output_dir", type=Path)
    generate(parser.parse_args().output_dir)
