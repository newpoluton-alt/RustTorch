#!/bin/sh
# Source from the repository root: . scripts/dev-env-macos.sh

if [ "$(uname -s)" != "Darwin" ]; then
    echo "error: dev-env-macos.sh requires macOS" >&2
    return 1 2>/dev/null || exit 1
fi

export RUSTTORCH_ROOT=${RUSTTORCH_ROOT:-"$(pwd)"}
. "$RUSTTORCH_ROOT/scripts/dev-env.sh" || return 1 2>/dev/null || exit 1

"$RUSTTORCH_ROOT/.venv/bin/python3" -c '
import torch
print(f"Python MPS built={torch.backends.mps.is_built()} available={torch.backends.mps.is_available()}")
'
