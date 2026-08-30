#!/bin/sh
set -eu

RUSTTORCH_ROOT=${RUSTTORCH_ROOT:-"$(pwd)"}
export RUSTTORCH_ROOT
if [ ! -f "$RUSTTORCH_ROOT/Cargo.toml" ]; then
    echo "error: run this script from the RustTorch repository root" >&2
    exit 1
fi

. "$RUSTTORCH_ROOT/scripts/dev-env.sh"

RUSTTORCH_PYTHON_REFERENCE_DIR=${RUSTTORCH_PYTHON_REFERENCE_DIR:-"$RUSTTORCH_ROOT/target/python-reference"}
export RUSTTORCH_PYTHON_REFERENCE_DIR

python "$RUSTTORCH_ROOT/tests/python_reference/generate.py" "$RUSTTORCH_PYTHON_REFERENCE_DIR"
cargo test --test python_parity -- --ignored --nocapture
