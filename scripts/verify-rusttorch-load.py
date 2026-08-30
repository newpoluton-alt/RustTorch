#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests" / "python_reference"))
from common import assert_close  # noqa: E402

DEFAULT_FIELDS = (
    "linear.forward",
    "linear.input_grad",
    "linear.weight_grad",
    "linear.bias_grad",
    "losses.cross_entropy",
    "losses.mse",
    "sgd.state",
    "adam.state",
    "residual.forward",
    "residual.input_grad",
    "residual.parameter_grads",
)


def lookup(value: object, field: str) -> object:
    for component in field.split("."):
        if not isinstance(value, dict) or component not in value:
            raise AssertionError(f"missing result field: {field}")
        value = value[component]
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare RustTorch JSON with PyTorch references")
    parser.add_argument("rust_results", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("--field", action="append", dest="fields")
    args = parser.parse_args()

    rust = json.loads(args.rust_results.read_text(encoding="utf-8"))
    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    try:
        for field in args.fields or DEFAULT_FIELDS:
            assert_close(lookup(rust, field), lookup(reference, field), field)
    except AssertionError as error:
        raise SystemExit(f"mismatch: {error}") from error
    print("RustTorch load and numerical parity passed")


if __name__ == "__main__":
    main()
