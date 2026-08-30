# API coverage

This is the declared 0.1.0 MVP surface. A listed API is not a hardware pass
claim; final status comes from the Rust and Python test output.

The deterministic CPU Python parity suite passes in both directions, including
strict SafeTensors load, Linear and residual forward/backward, cross-entropy,
MSE, and one Adam and SGD step. Rust MPS backend parity passes on the current
macOS arm64 host; CUDA remains conditional and was unavailable there.

## Core and eager modules

| PyTorch concept | RustTorch surface | Classification | MVP |
|---|---|---|---|
| Tensor, Device, dtype, reduction | root `Tensor`, `Device`, `Kind`, `Reduction` | reexport | yes |
| manual seed | `manual_seed` | delegate | yes |
| Module forward/train/eval | `nn::Module` and model methods | composite_port | yes |
| Linear | `nn::Linear`, `LinearConfig`, `linear` | delegate/composite_port | yes |
| Identity, ReLU, GELU | `nn` modules and functional forms | delegate | yes |
| Dropout | module and functional form | delegate/composite_port | yes |
| Flatten | module and functional form | delegate | yes |
| Sequential | `nn::Sequential` builder | composite_port | yes |
| MSE, cross-entropy | `nn::functional` | delegate | yes |
| Adam, SGD | `optim` builders | delegate | yes |

## Devices, state, and graph

| Capability | Support |
|---|---|
| CPU | required |
| CUDA | conditional on CUDA-enabled LibTorch and NVIDIA runtime |
| MPS | conditional on linked LibTorch and supported macOS hardware |
| automatic device selection | CUDA:0, then MPS, then CPU |
| SafeTensors state save/load | required, strict by default |
| explicit state-key mapping | required; no fuzzy or implicit shape transforms |
| `.pt`/`.bin` state dict | not implemented in 0.1 |
| Graph IR and eager executor | required optional API |
| graph branching/residuals | required |
| graph validation, summary, DOT | required |

Graph operations are Input, Output, Identity, Linear, ReLU, GELU, Dropout,
Flatten, Add, Subtract, Multiply, Concatenate, MSE loss, and cross-entropy.

## Partial or unsupported

- TorchScript is not exposed by RustTorch 0.1; callers can use `tch::CModule`
  directly for opaque legacy inference.
- Dedicated persistent-buffer registration and tied-parameter alias
  preservation are not implemented or tested.
- RustTorch-native training-checkpoint serialization and optimizer-state
  interchange are not implemented.
- Mixed precision, PT2 import/export, and compiled execution are roadmap items,
  not implemented features.
- Arbitrary Python-pickled whole models, cross-language optimizer resume,
  replacement autograd, `torch.compile`, custom kernels, distributed training,
  and full PyTorch layer/operator coverage are unsupported.

Known differences are recorded in
[`pytorch-compatibility.md`](pytorch-compatibility.md) and the machine-readable
compatibility manifest.
