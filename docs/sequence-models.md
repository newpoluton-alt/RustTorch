# Sequence models

RustTorch provides recurrent networks for ordered observations and attention
models for token or feature sequences. Layers register their parameters in a
`VarStore`, so a model can combine embeddings, sequence processing, prediction
heads, optimizers, and checkpoints without copying weights between components.

## Classify a time series with an LSTM

Each observation below contains three measurements. A unidirectional LSTM reads
five observations and a linear head predicts one of two classes from the final
output. The example performs a complete training update.

```rust
use rusttorch::{
    Device, Kind, Tensor,
    nn::{LinearConfig, RnnConfig, VarStore, functional},
    optim::Adam,
};

fn main() -> rusttorch::Result<()> {
    let store = VarStore::new(Device::Cpu);
    let encoder = RnnConfig::new(3, 8)
        .num_layers(2)
        .dropout(0.1)
        .batch_first(true)
        .build_lstm(&(store.root() / "encoder"))?;
    let classifier = LinearConfig::new(8, 2).build(&(store.root() / "classifier"))?;
    let mut optimizer = Adam::builder().learning_rate(0.001).build(&store)?;

    let observations = Tensor::arange(30, (Kind::Float, Device::Cpu)).reshape([2, 5, 3]) / 30.;
    let labels = Tensor::from_slice(&[0_i64, 1]);
    let (features, (_hidden, _cell)) = encoder.forward_t(&observations, None, true)?;
    let logits = classifier.forward(&features.select(1, 4))?;
    let loss = functional::cross_entropy(&logits, &labels)?;
    optimizer.backward_step(&loss)?;
    assert_eq!(logits.size(), [2, 2]);
    assert!(loss.double_value(&[]).is_finite());
    Ok(())
}
```

`RnnConfig::build_gru` creates a smaller gated encoder with one hidden-state
tensor. `build_rnn` creates a vanilla network with `RnnActivation::Tanh` or
`RnnActivation::Relu`. All three families support stacked layers, optional
biases, reverse directions, and dropout between layers. A single-layer model
has no intervening layer to which recurrent dropout can apply.

Input defaults to `[time, batch, features]`; `batch_first(true)` switches to
`[batch, time, features]`. Unbatched input is always `[time, features]`.
Hidden states remain `[layers * directions, batch, hidden_width]` in either
batched layout. For unbatched input the batch dimension is omitted.
Bidirectional output concatenates the two directions along the feature axis.
LSTM `projection(width)` reduces the output and hidden-state widths; its cell
state retains the original `hidden_size`.

## Continue a stream with explicit recurrent state

Carry the final state into the next chunk when observations belong to the same
stream. Reset to `None` at a new independent sequence. This example uses a GRU
in evaluation mode and detaches the state, which also prevents retaining the
previous chunk's autograd graph during truncated sequence training.

```rust
use rusttorch::{
    Device, Kind, Tensor,
    nn::{RnnConfig, VarStore},
};

fn main() -> rusttorch::Result<()> {
    let store = VarStore::new(Device::Cpu);
    let encoder = RnnConfig::new(3, 6)
        .batch_first(true)
        .build_gru(&store.root())?;
    let first_chunk = Tensor::ones([2, 4, 3], (Kind::Float, Device::Cpu));
    let next_chunk = Tensor::zeros([2, 2, 3], (Kind::Float, Device::Cpu));
    let (_first_features, state) = encoder.forward_t(&first_chunk, None, false)?;
    let state = state.f_detach()?;
    let (next_features, final_state) = encoder.forward_t(&next_chunk, Some(&state), false)?;
    assert_eq!(next_features.size(), [2, 2, 6]);
    assert_eq!(final_state.size(), [1, 2, 6]);
    Ok(())
}
```

The same protocol uses `(hidden, cell)` for an LSTM. Continuous forward-only
streams require a unidirectional model: a reverse direction needs future
observations and changes its result when chunk boundaries change. Packed
variable-length recurrent sequences and standalone recurrent cells are not
exposed yet; padding a recurrent sequence does not automatically stop its
state updates.

## Encode token features with attention

An encoder lets each position use information from the other visible positions.
Use a padding mask to exclude padded keys and a causal mask when predicting
later tokens must not reveal their contents. `true` means a position is blocked;
floating-point masks instead add directly to attention scores.

```rust
use rusttorch::{
    Device, Kind, Tensor,
    nn::{AttentionMask, TransformerActivation, TransformerConfig, VarStore},
};

fn main() -> rusttorch::Result<()> {
    let store = VarStore::new(Device::Cpu);
    let encoder = TransformerConfig::new(8, 2)
        .dim_feedforward(16)
        .activation(TransformerActivation::Gelu)
        .norm_first(true)
        .batch_first(true)
        .build_encoder(&(store.root() / "encoder"), 2, true)?;

    // Supply embeddings and positional information from your model.
    let token_features = Tensor::randn([2, 4, 8], (Kind::Float, Device::Cpu));
    let padding =
        Tensor::from_slice(&[false, false, false, true, false, false, true, true]).reshape([2, 4]);
    let encoded = encoder.forward_t(
        &token_features,
        AttentionMask {
            key_padding: Some(&padding),
            ..Default::default()
        },
        false,
    )?;
    assert_eq!(encoded.size(), [2, 4, 8]);
    Ok(())
}
```

Padding masks exclude keys, not query positions. Exclude padded targets from
the training loss separately. `MultiheadAttentionConfig` provides attention
without feed-forward or normalization layers; `forward_t` returns the output
and averaged attention maps, `forward_per_head_t` returns maps for each head,
and `forward_without_weights_t` computes only the output using native scaled
dot-product attention. Output-only attention contributes zero features for a
fully blocked row before its output projection. Returned attention maps use
ordinary softmax, which yields NaNs for a fully blocked row.

## Translate source features into a target sequence

The complete `Transformer` first encodes the source and then lets every decoder
layer attend to both the target prefix and the encoder's memory. Its result is
a feature vector per target position; add a linear vocabulary head and a token
loss for a language task, or a regression head for predicted measurements.

```rust
use rusttorch::{
    Device, Kind, Tensor,
    nn::{AttentionMask, TransformerConfig, TransformerMasks, VarStore},
};

fn main() -> rusttorch::Result<()> {
    let store = VarStore::new(Device::Cpu);
    let transformer = TransformerConfig::new(8, 2)
        .dim_feedforward(16)
        .num_encoder_layers(2)
        .num_decoder_layers(2)
        .batch_first(true)
        .build(&store.root())?;

    let source = Tensor::randn([2, 5, 8], (Kind::Float, Device::Cpu));
    let target_prefix = Tensor::randn([2, 3, 8], (Kind::Float, Device::Cpu));
    let source_padding = Tensor::from_slice(&[
        false, false, false, false, true, false, false, false, true, true,
    ])
    .reshape([2, 5]);
    let source_mask = AttentionMask {
        key_padding: Some(&source_padding),
        ..Default::default()
    };
    let decoded = transformer.forward_t(
        &source,
        &target_prefix,
        TransformerMasks {
            source: source_mask,
            target: AttentionMask {
                causal: true,
                ..Default::default()
            },
            memory: source_mask,
        },
        false,
    )?;
    assert_eq!(decoded.size(), [2, 3, 8]);
    Ok(())
}
```

Pass source padding to both the encoder and the decoder's memory attention.
The target's causal restriction only controls decoder self attention. During
teacher-forced next-token training, supply the target sequence shifted right
as decoder input and compare predictions with the unshifted targets.

Use `forward_t(..., true)` to enable dropout during training and `false` for
deterministic evaluation. Recurrent modules, multihead attention, encoder
layers, and encoder stacks also implement `Module`, so they can be constructed
with `Sequential::builder().layer(...)`. That interface uses zero recurrent
state, unmasked self attention, and output features only; use the explicit
methods for state, masks, or decoder memory. `Module::forward` defaults to
evaluation, while `Sequential` passes its current training mode.

## Save and inspect a sequence model

Use `rusttorch::interop::save_state_dict` and `load_state_dict` with the shared
`VarStore`, and rebuild the same configuration before loading. Recurrent names
include `encoder.weight_ih_l0`, `encoder.weight_hh_l0`, and `_reverse` suffixes.
Transformer stacks use `encoder.layers.0.self_attn.in_proj_weight`,
`decoder.layers.0.multihead_attn.in_proj_weight`, and named `linear`/`norm`
parameters. All layers, directions, and normalization parameters participate
in optimizer updates and state dictionaries.

The implemented attention interface uses equal query, key, and value feature
widths. Separate key/value widths, appended bias or zero tokens, nested tensors,
and key/value caches remain outside this API. Transformer stacks operate on
dense tensors and supply neither token embeddings nor positional encodings.
Tests currently provide CPU numerical evidence; accelerator support must be
verified on the intended hardware before making backend-specific claims.
