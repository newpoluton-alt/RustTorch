#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from safetensors.torch import load_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Safely inspect RustTorch SafeTensors state")
    parser.add_argument("weights", type=Path)
    args = parser.parse_args()

    state = load_file(args.weights, device="cpu")
    if not state:
        raise SystemExit("SafeTensors file contains no tensors")
    summary = {
        name: {"shape": list(value.shape), "dtype": str(value.dtype)}
        for name, value in sorted(state.items())
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
