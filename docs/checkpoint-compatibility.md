# Choose a checkpoint for the next run

| What the next process needs | What to save | How to restore it |
|---|---|---|
| Predictions from a model whose architecture you can rebuild | Named weights and persistent buffers in SafeTensors | Construct the model on its destination device and call `load_weights`. Use `StateDictMapping` when layer names differ. |
| The next optimizer update in the same training job | A version-one `.rtckpt` archive | Rebuild the model and optimizer, read `TrainingCheckpoint<Progress>`, and restore the saved state. |
| An executable model without reconstructing each layer | A portable tensor program, a supported PT2/ONNX model, or a traced TorchScript model with its guards | Load through `deployment::Model` or `deployment::TracedModel`. |

## Resume from a completed update

Capture a checkpoint after an optimizer update. Save to a fresh path each time.
`checkpoint::save` atomically publishes the model's SafeTensors weights, typed
optimizer state and your JSON-serializable progress in one file. Recreate the
same parameter names, shapes, dtypes and optimizer groups when restoring.

```rust
use rusttorch::{
    DeviceSpec, Tensor,
    checkpoint::{self, CheckpointLimits, TrainingCheckpoint},
    nn::Sequential, optim::Adam,
};

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let build = || Sequential::builder().linear(2, 1).build(DeviceSpec::Cpu);
    let model = build()?;
    let mut optimizer = Adam::builder().build(model.var_store())?;
    let input = Tensor::from_slice(&[1.0_f32, 2.0]).reshape([1, 2]);
    let loss = model.forward(&input)?.square().mean(rusttorch::Kind::Float);
    optimizer.backward_step(&loss)?;
    let expected = model.forward(&input)?;

    let path = std::env::temp_dir().join(format!("rusttorch-resume-{}.rtckpt", std::process::id()));
    let limits = CheckpointLimits::default();
    checkpoint::save(&path, model.var_store(), &mut optimizer, &1_u64, limits)?;

    let resumed = build()?;
    let mut optimizer = Adam::builder().build(resumed.var_store())?;
    let checkpoint = TrainingCheckpoint::<u64>::read(&path, limits)?;
    let completed_updates = checkpoint.restore(resumed.var_store(), &mut optimizer)?;
    assert_eq!(completed_updates, 1);
    assert!(expected.f_equal(&resumed.forward(&input)?)?);
    std::fs::remove_file(path)?;
    Ok(())
}
```

A stochastic job needs more than the update counter in this example. Put the
scheduler, gradient scaler, dataset identity, resumable loader position and
application-owned RNG state in a typed progress struct. Capture them at the
same update boundary. Validate the dataset and restore the input stream before
requesting the next batch. Select training or evaluation mode when rebuilding
the model. The [training-state recipe](https://docs.rs/rusttorch/latest/rusttorch/tutorials/tools/#save-model-optimizer-and-application-state-together)
shows how to include an isolated tensor generator.

## Share a model with another application

SafeTensors exchanges parameter values and persistent buffers across languages
when names or explicit mappings, shapes and dtypes agree. A standalone weight
file requires equivalent preprocessing and architecture at the destination.
The RustTorch training archive additionally uses RustTorch's optimizer schema
and your progress type. Importing arbitrary Python optimizer checkpoints would
require a separate conversion contract. Use the [weight exchange guide](https://github.com/newpoluton-alt/RustTorch/blob/main/docs/model-interoperability.md)
for a worked key mapping and the [deployment guide](https://docs.rs/rusttorch/latest/rusttorch/tutorials/deployment/)
to exchange a program with its weights.
