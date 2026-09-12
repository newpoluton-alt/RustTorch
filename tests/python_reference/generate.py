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
    elif optimizer == "adamw":
        implementation = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.1, amsgrad=True)
    elif optimizer == "rmsprop":
        implementation = torch.optim.RMSprop(model.parameters(), lr=0.01, alpha=0.9, weight_decay=0.1, momentum=0.5, centered=True)
    else:
        raise ValueError(f"unknown optimizer: {optimizer}")
    for _ in range(3 if optimizer in ("adamw", "rmsprop") else 1):
        loss = torch.nn.functional.cross_entropy(model(reference_input()), torch.tensor([2, 0]))
        implementation.zero_grad()
        loss.backward()
        implementation.step()
    return {"loss": loss.item(), "state": state_data(model)}



def layer_result(model: torch.nn.Module, value: torch.Tensor) -> dict[str, object]:
    output = model(value)
    output.square().mean().backward()
    result = {
        "forward": tensor_data(output),
        "state": state_data(model),
        "parameter_grads": {
            name: tensor_data(parameter.grad.to_dense() if parameter.grad.is_sparse else parameter.grad)
            for name, parameter in model.named_parameters()
        },
    }
    if value.requires_grad:
        result["input_grad"] = tensor_data(value.grad)
    return result


def layer_examples() -> dict[str, object]:
    result = {}
    for dimensions, shape, layer_type in (
        (1, (2, 2, 7), torch.nn.Conv1d),
        (2, (2, 2, 5, 6), torch.nn.Conv2d),
        (3, (1, 2, 4, 4, 4), torch.nn.Conv3d),
    ):
        torch.manual_seed(900 + dimensions)
        model = layer_type(2, 4, (2,) * dimensions, stride=2, padding=1, dilation=2, groups=2)
        initial = state_data(model)
        with torch.no_grad():
            model.weight.copy_(torch.arange(model.weight.numel()).reshape_as(model.weight) / 10 - 0.2)
            model.bias.copy_(torch.arange(4) / 10)
        value = (torch.arange(torch.tensor(shape).prod().item()).float().reshape(shape) / 20 - 0.5).requires_grad_()
        result[f"conv{dimensions}d"] = {"initial": initial, **layer_result(model, value)}

    model = torch.nn.LayerNorm((2, 3), eps=1e-4, bias=False)
    initial = state_data(model)
    with torch.no_grad():
        model.weight.copy_((torch.arange(6).float() / 10 + 0.5).reshape(2, 3))
    value = (torch.arange(12).float().reshape(2, 2, 3) / 5 - 0.8).requires_grad_()
    result["layer_norm"] = {"initial": initial, **layer_result(model, value)}

    for sparse in (False, True):
        torch.manual_seed(902)
        model = torch.nn.Embedding(5, 3, padding_idx=-1, scale_grad_by_freq=not sparse, sparse=sparse)
        initial = state_data(model)
        with torch.no_grad():
            model.weight.copy_((torch.arange(15).float() / 10).reshape(5, 3))
        value = torch.tensor([[0, 1, 1], [4, 2, 4]])
        result["embedding_sparse" if sparse else "embedding"] = {"initial": initial, **layer_result(model, value)}
    return result


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
        "adamw": stepped("adamw"),
        "rmsprop": stepped("rmsprop"),
        "layers": layer_examples(),
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
