#!/bin/sh
set -eu

RUSTTORCH_ROOT=${RUSTTORCH_ROOT:-"$(pwd)"}
export RUSTTORCH_ROOT
if [ ! -x "$RUSTTORCH_ROOT/.venv/bin/python3" ]; then
    echo "error: run from the repository root after creating .venv" >&2
    exit 1
fi

LIBTORCH_USE_PYTORCH=1
export LIBTORCH_USE_PYTORCH

"$RUSTTORCH_ROOT/.venv/bin/python3" <<'PY'
import json
import platform
import sys

import torch


def exercise(device: str) -> dict[str, object]:
    try:
        value = torch.tensor([2.0], device=device, requires_grad=True)
        (value.square().sum()).backward()
        return {"usable": True, "value": value.item(), "gradient": value.grad.item()}
    except Exception as error:  # backend error text is useful diagnostic output
        return {"usable": False, "reason": f"{type(error).__name__}: {error}"}


cpu = exercise("cpu")
cuda_reported = torch.cuda.is_available()
mps_built = torch.backends.mps.is_built()
mps_reported = torch.backends.mps.is_available()
cuda = exercise("cuda:0") if cuda_reported else {"usable": False, "reason": "not reported available"}
mps = exercise("mps") if mps_reported else {"usable": False, "reason": "not reported available"}

report = {
    "python": platform.python_version(),
    "torch": torch.__version__,
    "platform": platform.platform(),
    "cpu": cpu,
    "cuda": {
        "reported_available": cuda_reported,
        "device_count": torch.cuda.device_count(),
        "cudnn": torch.backends.cudnn.is_available(),
        **cuda,
    },
    "mps": {"built": mps_built, "reported_available": mps_reported, **mps},
    "note": "Python/LibTorch probe only; Rust backend tests are authoritative",
}
print(json.dumps(report, indent=2, sort_keys=True))

inconsistent = (
    not cpu["usable"]
    or (cuda_reported and not cuda["usable"])
    or (mps_reported and not mps["usable"])
)
sys.exit(1 if inconsistent else 0)
PY
