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

The 0.4.0 deployment regression exposed a native biased-linear failure on
GitHub's macOS 26.6.2 Apple M1 (Virtual), `VirtualMac2,1`, with LibTorch 2.13.0.
For a fixed two-row classifier, both the portable executor and a direct native
linear call produced `[28, 0, 64, 0]` instead of `[29, 0, 65, 0]`: the native
operation omitted its bias. Separate matrix multiplication and addition
produced the exact expected values on that runner. The
[diagnostic CI run](https://github.com/newpoluton-alt/RustTorch/actions/runs/34774599653)
records inputs, state, intermediate values and hardware. This agrees with the
[upstream MPS biased-linear report](https://github.com/pytorch/pytorch/issues/188438).

RustTorch routes its functional/layer, attention-projection and portable-executor
affine operations through one MPS correction using existing fallible tensor
operations. The correction preserves autograd and device placement; comparison
tolerances are unchanged. Native recurrent kernels, raw re-exported tensor
operations and previously saved TorchScript artifacts have separate runtime
behavior and are not intercepted by this correction.

On the local Apple M4 host, the correction passes exact Float/Half/BFloat16
affine outputs and input/weight/bias gradients for vector, matrix, batched and
strided inputs. A Float attention fixture checks all projection gradients
against CPU. The existing four eager/graph/optimizer/weight-transfer backend
tests and the portable classifier regression also pass. CUDA remains unavailable.
