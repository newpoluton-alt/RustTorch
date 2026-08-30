# Graph system

RustTorch Graph IR is an optional, backend-independent DAG for callers who need
explicit connectivity, validation, inspection, transformation, or a future
compilation boundary. Eager modules remain the primary API.

## Model

The graph contains stable `NodeId` and `ValueId` identities, unique readable
names, named inputs and outputs, tensor specifications, operations, and edges.
`TensorSpec` may describe known, dynamic, or symbolic dimensions, rank, dtype,
and optional device metadata. Runtime tensors remain authoritative; the MVP is
not a complete symbolic-shape solver.

Supported operations are Input, Output, Identity, Linear, ReLU, GELU, Dropout,
Flatten, Add, Subtract, Multiply, Concatenate, MSE loss, and cross-entropy.
Linear owns parameters in the model's single `tch::nn::VarStore`; parameter-free
operations do not create state.

## Build and validation

`GraphBuilder` returns value handles that can feed multiple later nodes, so a
branch or residual edge needs no closure-managed tensor storage. Before
execution, validation checks:

- at least one output, valid references, unique names, and connected inputs;
- operation arity, acyclicity, topological order, and reachable outputs;
- obvious shape/dtype incompatibility and duplicate parameter names;
- supported operations.

The pass surface includes validation, topological sorting, reachability,
dead-node elimination, and shape propagation for supported operations. A pass
must not change autograd semantics.

## Eager execution

`EagerExecutor` validates named inputs, stores tensors by value ID, traverses
topological order, dispatches ordinary `tch::Tensor` operations, and returns
named outputs. It performs no hidden host/device movement and no manual
backward pass. Branch and residual gradients are recorded and accumulated by
LibTorch.

`train()` and `eval()` select training-sensitive behavior such as Dropout;
evaluation mode does not disable gradients.

## Inspection

`summary()` reports names, operations, inputs, outputs, known specs, dtypes, and
parameter counts. `to_dot()` returns Graphviz DOT text and does not require a
Graphviz installation.

Compiled executors and stable graph serialization are roadmap items. There is
no placeholder `CompiledExecutor` in the MVP.
