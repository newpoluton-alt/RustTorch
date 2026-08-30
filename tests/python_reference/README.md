# Python reference assets

`generate.py` creates deterministic PyTorch 2.13.0 references for Linear
forward/backward, cross-entropy, MSE, one SGD step, one Adam step, residual
forward/backward, and SafeTensors round-trip.

```sh
. scripts/dev-env.sh
python tests/python_reference/generate.py target/python-reference
```

Generated files live under `target/` and are not committed. Rust parity tests
read the directory named by `RUSTTORCH_PYTHON_REFERENCE_DIR`. `linear_io.json`
is the small interchange protocol for the Python strict-load verifier. Values
use float32 with the tolerances recorded in `reference.json`; weights and
inputs are assigned explicitly rather than relying on matching random streams.
