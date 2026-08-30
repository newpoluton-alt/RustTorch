#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from safetensors.torch import load_file
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests" / "python_reference"))
from common import assert_close, linear_model, tensor_data  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify RustTorch weights in equivalent PyTorch Linear")
    parser.add_argument("weights", type=Path)
    parser.add_argument("rust_results", type=Path, help="JSON with input and output arrays")
    args = parser.parse_args()

    payload = json.loads(args.rust_results.read_text(encoding="utf-8"))
    if set(payload) != {"input", "output"}:
        raise SystemExit("Rust result JSON must contain exactly input and output")

    model = linear_model()
    model.load_state_dict(load_file(args.weights, device="cpu"), strict=True)
    actual = tensor_data(model(torch.tensor(payload["input"], dtype=torch.float32)))
    try:
        assert_close(actual, payload["output"], "forward")
    except AssertionError as error:
        raise SystemExit(f"mismatch: {error}") from error
    print("PyTorch strict load and forward parity passed")


if __name__ == "__main__":
    main()
