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
python "$RUSTTORCH_ROOT/tests/python_reference/spatial.py" > "$RUSTTORCH_PYTHON_REFERENCE_DIR/spatial.json"
python "$RUSTTORCH_ROOT/tests/python_reference/sequence.py" "$RUSTTORCH_PYTHON_REFERENCE_DIR"
python "$RUSTTORCH_ROOT/tests/python_reference/training.py" "$RUSTTORCH_PYTHON_REFERENCE_DIR"
python "$RUSTTORCH_ROOT/tests/python_reference/tensor_workflows.py" "$RUSTTORCH_PYTHON_REFERENCE_DIR"
python "$RUSTTORCH_ROOT/tests/python_reference/differentiation.py" "$RUSTTORCH_PYTHON_REFERENCE_DIR"
cargo test --locked --test python_parity --test nn_spatial --test nn_sequence \
    --test training_losses --test amp --test optim_state --test tensor_workflows \
    --test autograd --test distributions -- --ignored --nocapture
