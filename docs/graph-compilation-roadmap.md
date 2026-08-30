# Graph compilation roadmap

## Current boundary

The MVP executes eager Rust code and explicit Graph IR through normal
`tch::Tensor` calls. LibTorch creates and runs the dynamic autograd graph.
Graph IR supplies static connectivity and metadata; it does not differentiate
or compile operations.

TorchDynamo observes Python bytecode and Python runtime behavior, so it cannot
directly trace arbitrary Rust functions. RustTorch therefore does not advertise
`torch.compile` compatibility and does not route eager users through a fake
compiler abstraction.

## Work needed for compilation

1. Stabilize an internal operator schema and explicit graph version.
2. Add shape/range constraints and runtime guards without recreating the full
   PyTorch symbolic-shape engine.
3. Represent supported control flow explicitly rather than tracing Rust
   branches implicitly.
4. Define operator decomposition into a versioned ATen-oriented subset.
5. Preserve parameter/buffer identity, aliasing, dtype, device, and autograd
   boundaries through lowering.
6. Add executor-specific validation and parity tests for CPU, CUDA, and MPS.

Possible targets are PyTorch Export-like IR, AOTInductor artifacts, ONNX, or a
future Rust-native compiler. Each target needs an honest capability check;
unsupported operations stay eager or fail explicitly.

PT2 integration begins with a read-only importer for a small versioned operator
subset, followed by round-trip and numerical parity. No timeline or compiled
executor is promised by the MVP.
