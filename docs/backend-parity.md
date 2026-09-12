# Backend parity

CPU is the reference backend. The same eager model and explicit graph
definitions run without structural changes on CPU, CUDA:0, and MPS when each
backend is available in the linked runtime.

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

An earlier macOS 26.5.2 arm64 development run passed the Rust CPU and MPS
checks above with LibTorch 2.13.0; CUDA was skipped.

The 2026-09-12 run on macOS 26.6.2 arm64 with LibTorch 2.13.0 passed CPU
workspace tests and cross-language convolution, layer-normalization, embedding,
AdamW, and RMSprop comparisons. Running the existing Rust backend tests with
host hardware access also passed all four CPU/MPS test cases: eager execution,
residual graphs, Adam/SGD updates, and weight transfer. CUDA was unavailable and
skipped. These MPS results do not establish accelerator parity for the newly
added layer and optimizer families.
