# Save, load, and exchange model weights

Use SafeTensors to keep trained RustTorch parameters between runs, deploy a model
on another device, or exchange weights with another machine-learning application.
The file stores named tensors. Keep the model architecture and preprocessing
configuration alongside it so an inference application can rebuild the model.

## Save a model and restore it in Rust

Build the same architecture before loading its weights. This example saves a
two-layer classifier, restores it into a new model, and checks that both produce
the same scores. In a training application, call `save_weights` after updating
the model with an optimizer.

```rust
use rusttorch::{DeviceSpec, Kind, Result, Tensor, nn::Sequential, no_grad};

fn classifier() -> Result<Sequential> {
    Sequential::builder()
        .linear(4, 8)
        .relu()
        .linear(8, 3)
        .build(DeviceSpec::Cpu)
}

fn main() -> Result<()> {
    let mut original = classifier()?;
    original.eval();
    original.save_weights("classifier.safetensors")?;

    let mut restored = classifier()?;
    let report = restored.load_weights("classifier.safetensors")?;
    restored.eval();
    assert!(report.missing.is_empty());
    assert!(report.unexpected.is_empty());

    let features = Tensor::f_ones([2, 4], (Kind::Float, restored.device()))?;
    let expected = no_grad(|| original.forward(&features))?;
    let actual = no_grad(|| restored.forward(&features))?;
    assert!(actual.f_allclose(&expected, 1e-6, 1e-6, false)?);
    Ok(())
}
```

`eval()` selects evaluation behavior for layers such as dropout. Use `no_grad`
as well to avoid recording gradients while predicting. The weight file does
not save either execution mode, optimizer moments, random-number state, or the
data-loader position. It is suitable for restoring predictions; a complete
training resume needs those additional states.

`GraphModule` provides the same `save_weights` and `load_weights` methods. Rebuild
the graph with the same parameter-bearing node names before loading it.

## Move weights between devices

Saving copies each named tensor to contiguous CPU storage. Loading copies values
into the destination model's existing tensors, so the destination device is
chosen when building the model. To deploy on a GPU, build with
`DeviceSpec::Cuda(0)` or `DeviceSpec::Mps`, then load the same file. An explicit
unavailable device returns an error; `DeviceSpec::Auto` allows CPU fallback.

Input tensors must also be on the model's device. Move them with
`input.f_to_device(model.device())?` before calling `forward`.

## Match parameter names explicitly

Sequential layers use their position as the parameter prefix. The classifier
above has `0.weight`, `0.bias`, `2.weight`, and `2.bias`; the ReLU at position one
has no parameters. Graph layers instead use their node names, for example
`encoder.weight`.

When an imported file uses different names, supply an explicit mapping. This
function loads a file whose `encoder.*` tensors belong to the first linear layer
and whose `head.*` tensors belong to the second:

```rust
use std::path::Path;
use rusttorch::{DeviceSpec, Result, nn::Sequential};
use rusttorch::interop::{LoadOptions, StateDictMapping};

fn load_classifier(path: &Path) -> Result<Sequential> {
    let mut model = Sequential::builder()
        .linear(4, 8)
        .relu()
        .linear(8, 3)
        .build(DeviceSpec::Cpu)?;
    let mapping = StateDictMapping::new()
        .map_prefix("encoder.", "0.")
        .map_prefix("head.", "2.");

    let preview = model.load_weights_with_mapping(
        path,
        &mapping,
        LoadOptions::strict().dry_run(true),
    )?;
    println!("Validated {} tensors", preview.loaded.len());

    model.load_weights_with_mapping(path, &mapping, LoadOptions::strict())?;
    model.eval();
    Ok(model)
}
```

`map(source, destination)` renames one exact key. Exact mappings take precedence
over prefixes; otherwise the longest matching prefix wins. Unmapped keys retain
their names. A dry run validates and reports a load without modifying weights.

## Diagnose a rejected load

| Result | Meaning and next step |
| --- | --- |
| Missing or unexpected keys | Check the architecture and names; add explicit mappings where appropriate. |
| Shape mismatch | Rebuild with matching layer dimensions; loading does not transpose or reshape tensors. |
| Dtype mismatch | Match the model and file dtypes; loading does not cast values. |
| Duplicate mapped destination | Correct mappings so each destination receives at most one source tensor. |
| Unsupported file extension | Provide a file ending in `.safetensors`. |

Strict loading checks every expected and supplied key. For intentional partial
loading, use `LoadOptions::non_strict()` and inspect the report's `loaded`,
`missing`, `unexpected`, and `remapped` fields. Matching tensors must still have
the correct shape and dtype. Key, shape, dtype, and mapping validation happen
before model tensors are changed.

The model state consists of variables registered in its `VarStore`. SafeTensors
stores values by name; loading does not reconstruct shared-storage relationships
between tied parameters.

## Exchange weights with PyTorch

PyTorch is relevant here when another application produced the weights or will
consume them. RustTorch inference itself does not need Python. Both applications
must agree on the architecture, layer options, parameter names, dtypes, and input
preprocessing; a matching weight file alone does not establish those choices.

For the classifier above, the equivalent Python architecture is:

```python
import torch
from safetensors.torch import load_file, save_file

model = torch.nn.Sequential(
    torch.nn.Linear(4, 8),
    torch.nn.ReLU(),
    torch.nn.Linear(8, 3),
)

# Import a file saved by RustTorch.
model.load_state_dict(load_file("classifier.safetensors"), strict=True)
model.eval()

# Export independent, contiguous CPU tensors for RustTorch to load.
state = {
    name: tensor.detach().cpu().contiguous().clone()
    for name, tensor in model.state_dict().items()
}
save_file(state, "exported-classifier.safetensors")
```

Install the Python `safetensors` package to use these import/export helpers.
For custom module names, apply `StateDictMapping` in Rust or explicitly rename
the exported keys.

Python pickle-based `.pt` and `.bin` files, whole Python model objects,
TorchScript files, and PT2 archives cannot be loaded through `load_weights`.
The [interoperability API](https://docs.rs/rusttorch/latest/rusttorch/interop/)
documents the accepted format and error types. Refer to the
[compatibility reference](api-coverage.md) for the scope of cross-language
behavior checks.
