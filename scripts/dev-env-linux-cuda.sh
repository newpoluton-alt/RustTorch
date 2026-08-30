#!/bin/sh
# Source from the repository root: . scripts/dev-env-linux-cuda.sh

if [ "$(uname -s)" != "Linux" ]; then
    echo "error: dev-env-linux-cuda.sh requires Linux" >&2
    return 1 2>/dev/null || exit 1
fi

export RUSTTORCH_ROOT=${RUSTTORCH_ROOT:-"$(pwd)"}
. "$RUSTTORCH_ROOT/scripts/dev-env.sh" || return 1 2>/dev/null || exit 1

"$RUSTTORCH_ROOT/.venv/bin/python3" -c '
import sys
import torch
if not torch.cuda.is_available():
    sys.exit("error: project PyTorch/LibTorch has no usable CUDA backend")
print(f"Python CUDA devices={torch.cuda.device_count()} cuDNN={torch.backends.cudnn.is_available()}")
' || return 1 2>/dev/null || exit 1
