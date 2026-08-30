# Backend parity

CPU is the reference backend. The same eager model and explicit graph
definitions run without structural changes on CPU, CUDA:0, and MPS when each
backend is genuinely available.

Deterministic tests assign weights and inputs directly. They compare:

- forward output and scalar loss;
- input and parameter gradients;
- one SGD update and one Adam update;
- output and gradient devices;
- state saved on one device and loaded on another.

For float32, the starting comparison policy is `rtol=1e-5, atol=1e-5` for
operations whose backend implementations are expected to align. MPS may use
`rtol=1e-4, atol=1e-4` for reductions or optimizer updates that show measured
backend-ordering differences. Any wider or operation-specific tolerance needs
a recorded failing value and rationale; tolerance must not hide shape, dtype,
device, or logical errors.

Random backend streams are not compared. Dropout is tested behaviorally in
train/eval modes rather than by requiring identical masks.

An unavailable CUDA or MPS backend prints a skip reason. A skip is not a pass,
and Python availability alone is not Rust backend evidence. Final reports must
name the exact host, linked versions, executed checks, and backend-specific
differences.

On the current macOS 26.5.2 arm64 development host, the Rust CPU and MPS checks
above passed with PyTorch/LibTorch 2.13.0. CUDA was unavailable and was skipped;
no CUDA pass is claimed.
