from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch


RTOL = 1e-5
ATOL = 1e-6


def linear_model() -> torch.nn.Linear:
    model = torch.nn.Linear(4, 3)
    with torch.no_grad():
        model.weight.copy_(
            torch.tensor(
                [
                    [0.1, -0.2, 0.3, -0.4],
                    [0.5, 0.6, -0.7, 0.8],
                    [-0.9, 1.0, 0.2, -0.1],
                ],
                dtype=torch.float32,
            )
        )
        model.bias.copy_(torch.tensor([0.05, -0.15, 0.25], dtype=torch.float32))
    return model


class ResidualModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.main = torch.nn.Linear(4, 4, bias=False)
        self.skip = torch.nn.Linear(4, 4, bias=False)
        with torch.no_grad():
            self.main.weight.copy_(torch.eye(4, dtype=torch.float32) * 0.5)
            self.skip.weight.copy_(
                torch.tensor(
                    [
                        [0.0, 0.1, 0.0, -0.2],
                        [0.3, 0.0, 0.2, 0.0],
                        [0.0, -0.4, 0.0, 0.1],
                        [0.2, 0.0, -0.3, 0.0],
                    ],
                    dtype=torch.float32,
                )
            )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.main(value)) + self.skip(value)


def reference_input(*, requires_grad: bool = False) -> torch.Tensor:
    return torch.tensor(
        [[0.25, -0.5, 1.0, 2.0], [-1.0, 0.75, 0.5, -0.25]],
        dtype=torch.float32,
        requires_grad=requires_grad,
    )


def tensor_data(value: torch.Tensor) -> Any:
    return value.detach().cpu().tolist()


def state_data(model: torch.nn.Module) -> dict[str, Any]:
    return {name: tensor_data(value) for name, value in model.state_dict().items()}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def assert_close(actual: Any, expected: Any, path: str = "root") -> None:
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or actual.keys() != expected.keys():
            raise AssertionError(f"{path}: key mismatch")
        for key in expected:
            assert_close(actual[key], expected[key], f"{path}.{key}")
        return
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise AssertionError(f"{path}: length mismatch")
        for index, (left, right) in enumerate(zip(actual, expected, strict=True)):
            assert_close(left, right, f"{path}[{index}]")
        return
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        if not math.isclose(float(actual), float(expected), rel_tol=RTOL, abs_tol=ATOL):
            raise AssertionError(f"{path}: {actual} != {expected}")
        return
    if actual != expected:
        raise AssertionError(f"{path}: {actual!r} != {expected!r}")
