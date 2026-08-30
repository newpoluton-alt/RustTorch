#!/bin/sh
# Source from the repository root: . scripts/dev-env-linux-cpu.sh

if [ "$(uname -s)" != "Linux" ]; then
    echo "error: dev-env-linux-cpu.sh requires Linux" >&2
    return 1 2>/dev/null || exit 1
fi

export RUSTTORCH_ROOT=${RUSTTORCH_ROOT:-"$(pwd)"}
export CUDA_VISIBLE_DEVICES=""
. "$RUSTTORCH_ROOT/scripts/dev-env.sh" || return 1 2>/dev/null || exit 1
echo "CUDA devices hidden for the Linux CPU environment"
