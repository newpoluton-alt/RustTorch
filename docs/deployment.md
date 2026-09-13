# Deploy a RustTorch model

Use `rusttorch::deployment` to load a model without rebuilding its layers,
exchange an explicit tensor program, or save a native inference artifact.
A `Model` combines a versioned `Program` with named parameters, buffers and
constants. Every operation uses LibTorch; ordinary execution retains autograd.

| Format | Use case | Execution |
|---|---|---|
| Portable JSON | Save graph, input guards, calling trees and shared state | RustTorch tensor executor |
| PT2 | Exchange the supported ExportedProgram subset and parameter names | RustTorch or the pinned exporter runtime |
| ONNX | Exchange common inference operators | RustTorch or an ONNX runtime |
| TorchScript | Save a native traced inference graph | LibTorch JIT |

All operations return `rusttorch::Result`. Importing PT2 or ONNX does not compile
a graph. `Model::trace` creates a real TorchScript artifact from a branch-free,
single-output program and freezes a copy of its state.

## Convert an existing graph

Build an evaluation graph with explicit input dimensions and dtype. Conversion
preserves parameter names such as `head.weight`.

```rust
use rusttorch::{DeviceSpec, Kind, Tensor, Result};
use rusttorch::graph::{GraphBuilder, TensorSpec};
use rusttorch::deployment::Model;

fn main() -> Result<()> {
    let mut builder = GraphBuilder::new();
    let input = builder.input("features",
        TensorSpec::new().known_dimensions([2, 4]).kind(Kind::Float))?;
    let head = builder.linear("head", input, 4, 3)?;
    let scores = builder.relu("scores", head)?;
    let mut graph = builder.output("prediction", scores)?.build(DeviceSpec::Cpu)?;
    graph.eval();

    let portable = Model::from_graph(&graph)?;
    let features = Tensor::ones([2, 4], (Kind::Float, portable.device()));
    assert_eq!(portable.run(&[features])?[0].size(), [2, 3]);
    assert!(portable.state().contains_key("head.weight"));
    Ok(())
}
```

Conversion supports linear, identity, ReLU, flatten, add/subtract/multiply and
concatenation. Evaluation dropout becomes identity. Training mode and unsupported
operations return an error; conversion never silently removes a layer.

## Accept a bounded dynamic batch

This scorer accepts one to eight rows of three features and produces two scores
per row. Reusing the same symbol in multiple inputs requires those dimensions
to agree within each call.

```rust
use rusttorch::{Device, Kind, Tensor, Result};
use rusttorch::deployment::{DType, Dimension, Model, Operation, Operator,
    Program, StateRole, Tree, ValueSpec};

fn main() -> Result<()> {
    let batch = Dimension::Symbol { name: "batch".into(), min: 1, max: Some(8) };
    let program = Program::new(
        vec![ValueSpec::new("features", DType::Float,
            vec![batch, Dimension::Known(3)])],
        vec![Operation::new("projection", Operator::Linear,
                  ["features", "head.weight"]),
             Operation::new("scores", Operator::Relu, ["projection"])],
        Tree::Tensor("scores".into()),
    );
    let weights = Tensor::from_slice(&[1_f32, 0., 0., 0., 1., 0.]).reshape([2, 3]);
    let model = Model::new(program,
        [("head.weight".into(), StateRole::Parameter, weights)])?;
    assert_eq!(model.run(&[Tensor::ones([4, 3], (Kind::Float, Device::Cpu))])?[0]
        .size(), [4, 2]);
    assert!(model.run(&[Tensor::zeros([9, 3], (Kind::Float, Device::Cpu))]).is_err());

    let restored = Model::from_json(&model.to_json()?)?;
    assert_eq!(restored.program(), model.program());
    Ok(())
}
```

`Dimension::Known` enforces an exact size. Dtype, rank and device are also
checked before execution. `Model::new` copies state into owned CPU storage.
Call `to_device(Device::Cuda(0))` or `to_device(Device::Mps)` when that backend
is available, then supply inputs on the same device. CPU is the numerical
interchange reference; hardware-specific evidence is recorded separately.
For MPS affine operations, RustTorch uses separate matrix multiplication and
bias addition while retaining GPU outputs and gradients. This avoids the
biased-linear defect on affected Mac runtimes. Imported native TorchScript
artifacts keep their own operator graph; check the
[MPS guide](https://github.com/newpoluton-alt/RustTorch/blob/main/docs/mps-support.md)
when choosing an execution path.

## Retain structured inputs and outputs

`Program::input_tree` and `output_tree` describe tensors within tuples, lists and
dictionaries. `run` receives and returns flat tensor vectors in leaf order.

```rust
use rusttorch::deployment::Tree;
let outputs = Tree::Dict(vec![
    ("prediction".into(), Tree::Tensor("scores".into())),
    ("features".into(), Tree::List(vec![Tree::Tensor("hidden".into())])),
]);
assert_eq!(outputs.leaves()?, ["scores", "hidden"]);
# Ok::<(), rusttorch::RustTorchError>(())
```

Dictionary keys must be unique. Imported PT2 input trees retain the exporter’s
`(positional arguments, keyword arguments)` structure. Rust callers use the flat
order listed in `program.inputs`; no Python objects are needed.

## Select a branch explicitly

`Operator::If` takes a scalar Boolean followed by the branch operands. Both
branches have the same input signature and exactly one output. Its common
result spec checks the selected output. Only the selected branch executes;
its tensor operations retain their gradients.

```rust
use rusttorch::{Tensor, Result};
use rusttorch::deployment::{DType, Dimension, Model, Operation, Operator,
    Program, Tree, ValueSpec};

fn main() -> Result<()> {
    let operand = ValueSpec::new("operand", DType::Float, vec![Dimension::Known(2)]);
    let square = Program::new(vec![operand.clone()],
        vec![Operation::new("squared", Operator::Multiply, ["operand", "operand"])],
        Tree::Tensor("squared".into()));
    let double = Program::new(vec![operand.clone()],
        vec![Operation::new("doubled", Operator::Add, ["operand", "operand"])],
        Tree::Tensor("doubled".into()));
    let program = Program::new(
        vec![ValueSpec::new("choose_square", DType::Bool, vec![]),
             ValueSpec::new("x", DType::Float, vec![Dimension::Known(2)])],
        vec![Operation::new("result", Operator::If {
            then_branch: Box::new(square), else_branch: Box::new(double),
            result: operand,
        }, ["choose_square", "x"])],
        Tree::Tensor("result".into()),
    );
    let model = Model::new(program, [])?;
    let output = model.run(&[Tensor::from(true), Tensor::from_slice(&[2_f32, 3.])])?;
    assert_eq!(Vec::<f32>::try_from(&output[0])?, [4., 9.]);
    Ok(())
}
```

Conditional programs save to portable JSON. PT2/ONNX control-flow conversion,
loops and conditional-to-TorchScript tracing return explicit unsupported errors.
A sampled Rust branch is never presented as a compiled conditional.

## Exchange artifacts and save native inference

```rust,no_run
use rusttorch::{Device, Result, Tensor};
use rusttorch::deployment::{Model, TracedModel};

fn save_model(model: &Model, example: Tensor) -> Result<()> {
    model.save_pt2("classifier.pt2")?;
    let exported = Model::load_pt2("classifier.pt2")?;
    model.save_onnx("classifier.onnx")?;
    let inference = Model::load_onnx("classifier.onnx")?;

    let traced = model.trace(&[example])?;
    traced.save("classifier.pt")?; // also classifier.pt.guards.json
    let loaded = TracedModel::load("classifier.pt", Device::Cpu)?;
    # let _ = (exported, inference, loaded);
    Ok(())
}
```

Trace again after training changes: updating the original model does not update
the frozen artifact. Keep the `.guards.json` sidecar with its TorchScript file.
Native artifacts contain executable operations and require a trusted producer.

PT2 uses PyTorch 2.13.0, archive version 0, serialization version 6, export
schema 8.20, ATen opset 10 and tree version 1. Float, Double, Int64 and Bool
parameters, persistent/non-persistent buffers and tensor constants are supported.
State names, parameter roles and persistent buffer roles are retained;
non-persistent buffers are represented as constants. Checked shared storage and non-overlapping
strided views preserve aliases; identical tied parameter views share one gradient
leaf. Internally overlapping state views and storage shared across PT2 weights
and constant directories are rejected.

The PT2 operator subset is alias, linear, add/subtract/multiply (alpha one),
matrix multiplication, ReLU, sigmoid, tanh, fixed reshape, flatten, transpose,
permutation and concatenate. Dimensions may be named symbols with ranges.
Export derives shapes algebraically. Unsupported symbolic sums/products/scales,
arbitrary expressions, executable guards, mutation and custom objects fail
explicitly. Pickle is never loaded; exported sample-input payloads are empty.

ONNX uses IR10/default-domain opset18 and supports Identity, Relu, Sigmoid, Tanh,
Add, Sub, Mul, MatMul, Transpose, Concat, axis-one Flatten, constant-shape Reshape,
dense Constant and Gemm with alpha/beta one, transA zero and transB one. Linear
exports as Transpose/MatMul/Add. ONNX initializers are inference constants;
trainable parameter roles are not preserved. Shared-state export is rejected.
Input ranges and calling trees are stored in versioned RustTorch metadata;
other ONNX runtimes must enforce those additional range constraints themselves.

External ONNX tensor files, sparse/quantized tensors, custom domains, functions,
training graphs and device configuration are unsupported. Interchange export
requires CPU state and matching operand dtypes. Supported artifact execution
does not imply `torch.compile`, AOTInductor binary execution or Python capture.

Imports limit files and decompressed storage to 256 MiB, PT2 JSON members to
16 MiB, program nodes to 10,000, tensor rank to 64 and tree depth to 32. Protobuf
field counts are checked before decoding repeated messages. ZIP and ZIP64
directory counts are checked before the archive parser allocates its file table. Larger streamed
artifacts and wider operator/backend support require separate implementations.
