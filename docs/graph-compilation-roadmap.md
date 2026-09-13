# Graph compilation and deployment

The [deployment guide](deployment.md) provides executable Rust examples and
format selection. The original `graph` executor remains eager. Its supported
evaluation subset lowers through `deployment::Model::from_graph` into a
version-one portable tensor program with named state, calling trees, dtype/device
guards, shared symbolic dimensions and inclusive ranges.

Portable conditional programs execute only the chosen branch and retain
autograd. A supported branch-free single-output program can be traced into a
real LibTorch TorchScript module, saved, loaded and executed with retained
guards. The artifact freezes state. Tracing rejects conditionals and does not
capture arbitrary Rust branches or Python bytecode. No performance claim is made.

PT2 and ONNX import/export use explicit versioned operator subsets. Export
applies algebraic shape rules and rejects unrepresentable symbolic expressions.
State storage bounds and aliases are validated before tensor construction.
Imported interchange graphs execute through tensor operations independently of
native TorchScript tracing.

Remaining backend work includes:

- Native lowering of conditional/loop graphs and wider operator decompositions.
- Richer symbolic expressions, stride guards and mutation/alias semantics.
- AOTInductor binary execution and platform/toolchain compatibility validation.
- Additional interchange versions and larger streamed artifacts.
- Numerical/backend evidence for newly advertised CUDA or MPS capabilities.

TorchDynamo observes Python bytecode and runtime semantics. It is not Rust
function capture; `torch.compile` parity is not claimed. `tests/deployment.rs`
records the delivered graph, artifact, guard, conditional and malformed-input
contracts, including reference-runtime numerical round trips.
