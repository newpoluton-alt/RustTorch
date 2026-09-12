# Build and inspect computation graphs

Use a RustTorch graph when a model has branches, residual connections, multiple
named inputs or outputs, or when you want to inspect its connectivity. A graph
records operations and edges explicitly, then executes ordinary tensor
operations with automatic differentiation. For a simple chain of layers,
`nn::Sequential` provides a shorter API.

## Build a residual block

`GraphBuilder` returns a value handle from each operation. Reuse a handle to
branch the computation; pass two handles to `add` to form a residual connection.
Names identify inputs, operations, and outputs, so each must be unique.

This block computes `ReLU(projection(features)) + features`. The projection
keeps the same feature count as the shortcut, and a symbolic dimension lets
the block accept different batch sizes.

```rust
use rusttorch::{DeviceSpec, Kind, Result, Tensor, no_grad};
use rusttorch::graph::{Dim, GraphBuilder, GraphInputs, TensorSpec};

fn main() -> Result<()> {
    let mut builder = GraphBuilder::new();
    let features = builder.input(
        "features",
        TensorSpec::new()
            .dimensions(vec![Dim::Symbol("batch".into()), Dim::Known(4)])
            .kind(Kind::Float),
    )?;
    let projection = builder.linear("projection", features, 4, 4)?;
    let activated = builder.relu("activation", projection)?;
    let residual = builder.add("residual", activated, features)?;
    let mut model = builder.output("embedding", residual)?.build(DeviceSpec::Cpu)?;

    model.eval();
    let batch = Tensor::f_ones([3, 4], (Kind::Float, model.device()))?;
    let inputs = GraphInputs::new().with("features", batch)?;
    let outputs = no_grad(|| model.forward(inputs))?;
    assert_eq!(outputs.get("embedding")?.size(), [3, 4]);

    println!("{}", model.summary());
    println!("{}", model.to_dot());
    Ok(())
}
```

Use `add_output` to expose additional values before building the model. For
example, an application can return an embedding and a prediction together, or
inspect an intermediate activation without changing the model's tensor flow.
Read outputs by name with `GraphOutputs::get` or enumerate them with `iter`.

## Train using a graph loss

A loss can be a named graph output. Attach an optimizer to the graph's parameter
store and backpropagate that output just as you would for a sequential model.
The classifier below returns both its scores and a scalar cross-entropy loss.
Labels are `Int64` class indices; scores remain unnormalized logits.

```rust
use rusttorch::{DeviceSpec, Kind, Result, Tensor, optim::Adam};
use rusttorch::graph::{Dim, GraphBuilder, GraphInputs, TensorSpec};

fn main() -> Result<()> {
    let mut builder = GraphBuilder::new();
    let batch = Dim::Symbol("batch".into());
    let features = builder.input(
        "features",
        TensorSpec::new()
            .dimensions(vec![batch.clone(), Dim::Known(4)])
            .kind(Kind::Float),
    )?;
    let labels = builder.input(
        "labels",
        TensorSpec::new().dimensions(vec![batch]).kind(Kind::Int64),
    )?;
    let logits = builder.linear("classifier", features, 4, 3)?;
    let objective = builder.cross_entropy_loss("cross_entropy", logits, labels)?;
    builder.add_output("scores", logits)?;
    let mut model = builder.output("loss", objective)?.build(DeviceSpec::Cpu)?;
    let mut optimizer = Adam::builder().build(model.var_store())?;

    model.train();
    let features = Tensor::f_ones([2, 4], (Kind::Float, model.device()))?;
    let labels = Tensor::f_from_slice(&[0_i64, 2])?;
    let outputs = model.forward(
        GraphInputs::new().with("features", features)?.with("labels", labels)?,
    )?;
    optimizer.backward_step(outputs.get("loss")?)?;
    assert_eq!(outputs.get("scores")?.size(), [2, 3]);
    Ok(())
}
```

All declared input names must be supplied on every forward call. A graph that
declares labels for its loss therefore needs labels even in evaluation mode.
For label-free inference, build a graph whose outputs are the scores, omit its
loss and label input, and restore the same named classifier parameters. See
[saving and loading weights](model-interoperability.md).

`train()` and `eval()` control training-sensitive operations such as dropout.
Evaluation mode leaves gradient recording enabled; wrap inference in `no_grad`
when derivatives are unnecessary. `forward_t(inputs, training)` overrides the
mode for one call without changing the model's stored mode.

## Describe and validate inputs

`TensorSpec` describes constraints rather than allocating a tensor:

| Constraint | Meaning |
| --- | --- |
| `known_dimensions([8, 4])` | Require exactly eight samples with four features. |
| `Dim::Known(4)` | Require this dimension to have size four. |
| `Dim::Dynamic` | Accept any size for this dimension. |
| `Dim::Symbol("batch".into())` | Require every input dimension using this symbol to agree within a call. |
| `rank(2)` | Require two axes without fixing individual sizes. |
| `kind(Kind::Float)` | Require this tensor dtype. |
| `device(DeviceSpec::Cpu)` | Require the resolved device in addition to the model's device. |

Supply tensors on `model.device()`; graph execution does not move them
implicitly. Construction rejects invalid names and references, missing outputs,
invalid operation settings, and shape or dtype conflicts it can establish from
the available specs. Forward calls validate input names and tensor constraints;
the tensor operations check dimensions whose sizes were unknown during building.

## Inspect a graph before allocating parameters

Call `finish()` to obtain a graph definition without allocating model weights.
It validates the graph, marks nodes that do not contribute to outputs as
inactive, and propagates supported shape information. Call `build(device)` on
the resulting graph when it is ready to execute.

```rust
use rusttorch::{DeviceSpec, Kind, Result};
use rusttorch::graph::{GraphBuilder, TensorSpec};

fn main() -> Result<()> {
    let mut builder = GraphBuilder::new();
    let input = builder.input(
        "input",
        TensorSpec::new().known_dimensions([2, 4]).kind(Kind::Float),
    )?;
    let _unused = builder.relu("unused_activation", input)?;
    let projected = builder.linear("projection", input, 4, 2)?;
    let graph = builder.output("result", projected)?.finish()?;

    assert!(graph.nodes().iter().any(|node| node.name == "unused_activation" && !node.active));
    assert!(!graph.summary().contains("unused_activation"));
    assert!(graph.to_dot().starts_with("digraph rusttorch"));
    let model = graph.build(DeviceSpec::Cpu)?;
    assert_eq!(model.var_store().variables().len(), 2); // Weight and bias.
    Ok(())
}
```

`summary()` displays active nodes, input value IDs, inferred shapes, dtypes, and
parameter counts. `to_dot()` produces Graphviz DOT text without requiring
Graphviz to be installed. Write that text to a `.dot` file if you want to render
the topology with Graphviz.

For programmatic analysis, inspect `nodes()`, `inputs()`, `outputs()`,
`topological_order()`, and `reachable_nodes()`. The `Validation`,
`DeadNodeElimination`, and `ShapePropagation` passes are also available through
`GraphPass`.

## Choose operations

Graph builders support linear projections, identity, ReLU, GELU, dropout,
flattening, addition, subtraction, multiplication, concatenation, mean squared
error, and cross-entropy. Linear nodes register weights and optional bias under
their node names. Tensor calculations preserve their gradient history, including
gradient accumulation across branches.

The graph operation set is separate from the eager layer API; an eager layer
does not automatically become a graph operation. Use ordinary Rust modules for
architectures needing other layers or data-dependent control flow. Graphs execute
eagerly and do not provide compilation or a stable serialized architecture
format. See the [graph API](https://docs.rs/rusttorch/latest/rusttorch/graph/)
for the full builder, model, and inspection reference.
