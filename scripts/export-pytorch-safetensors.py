#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from safetensors.torch import load_file, save_file
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests" / "python_reference"))
from common import linear_model  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the deterministic PyTorch parity model")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    model = linear_model()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(model.state_dict(), args.output)

    loaded = load_file(args.output)
    if loaded.keys() != model.state_dict().keys():
        raise SystemExit("SafeTensors key mismatch after export")
    for name, expected in model.state_dict().items():
        torch.testing.assert_close(loaded[name], expected, rtol=0, atol=0)
    print(args.output)


if __name__ == "__main__":
    main()
