# Core model and training delivery contracts

Tracking issue: [#10](https://github.com/newpoluton-alt/RustTorch/issues/10).

The requested quarter is the first two of the eight workstreams in
[the roadmap](../../roadmap.md): core model layers and training controls.
These workstreams have different sizes, so this is a delivery count rather
than a percentage of PyTorch APIs or implementation effort. The contracts below
are fixed before implementation and must have code, examples, validation and
numerical evidence before being marked delivered.

## Core model contracts

- [x] BatchNorm1d/2d/3d: affine parameters, running statistics and counter,
  training/evaluation, exponential or cumulative statistics, persistence.
- [x] InstanceNorm1d/2d/3d and GroupNorm: supported input dimensions,
  normalization, affine settings, running statistics where configured.
- [x] Max, average and adaptive pooling in one to three dimensions, including
  validated sizes/strides/padding and native gradients.
- [x] Transposed convolution in one to three dimensions with groups, dilation,
  output padding, weight initialization and gradients.
- [x] Sigmoid, tanh, SiLU, softmax/log-softmax, leaky ReLU and ELU modules.
- [x] RNN, LSTM and GRU sequence execution with explicit state, multiple layers,
  bidirectionality, batch-first ordering, train/eval and gradients.
- [x] Multihead attention with attention/padding masks, causal execution,
  dropout, output weights and gradients.
- [x] Transformer encoder/decoder layers, stacks and encoder-decoder model,
  including masking, residual/norm behavior and parameter persistence.

## Training contracts

- [x] Configurable MSE, L1, smooth L1, Huber, BCE, BCE-with-logits, NLL and
  cross-entropy: native reduction and applicable weights/ignore/smoothing.
- [x] Validated parameter-group learning-rate and decay controls across the
  implemented optimizer families, preserving existing call patterns.
- [x] Versioned full optimizer algorithm/moment/counter state with validated
  restore and exact continuation after separately restoring model parameters.
- [x] Step, exponential, multistep, cosine and metric/plateau schedules with
  explicit ordering and serializable replayable state.
- [x] Native CUDA autocast through the safe runtime boundary, explicit device
  limitations and unwinding-safe state restoration.
- [x] Gradient scaling/unscaling, nonfinite update skipping, growth/backoff,
  clipping integration, and scaler-state restore.
- [x] End-to-end classification/sequence training and model/optimizer/scheduler/
  scaler checkpoint examples that compile and run.
- [x] Compatibility scopes, Rust API and GitHub guides, provenance, signed
  commits, workspace/MSRV/docs/package checks, CPU parity and hardware evidence.

## Boundaries

The implementation must describe every supported option precisely. Completing
these contracts does not close the exhaustive nn/optim inventory: specialized
flavors such as packed recurrent inputs, recurrent cells, distributed batch
normalization, fused/capturable optimizers, closure/line-search solvers,
non-CUDA autocast policies and graph compilation require separate evidence.
They remain explicitly listed as future variants rather than hidden under a
blanket support statement. The other six roadmap workstreams stay open.


## Delivery evidence

The first two core scopes are implemented in this contribution. Their specialized
extension backlogs above remain open; none of the other six workstreams is marked
delivered. Source availability is separate from a published crates.io release.

| Contract family | Executable evidence |
|---|---|
| Normalization, spatial layers, pooling, activations | `tests/nn_spatial.rs`; pinned `tests/python_reference/spatial.py`; registered-buffer SafeTensors round trip |
| Recurrent models, attention, Transformers | `tests/nn_sequence.rs`; pinned `tests/python_reference/sequence.py`; state/layout/mask/dropout, gradients and sequence-model guide examples |
| Configurable losses | `tests/training_losses.rs`; `tests/python_reference/training.py` values and gradients |
| Seven optimizer families, groups, moment state | `tests/optim_state.rs`; pinned `tests/python_reference/optim_state.py`; exact resume at every tested step boundary and corruption rejection |
| Five learning-rate schedules | `tests/optim_state.rs`; multi-epoch numerical comparison and exact serde resume, including plateau cooldown and duplicate milestones |
| Gradient scaler and autocast boundary | `tests/amp.rs`; default/custom scale CPU parity, skipped nonfinite updates, clipping, state restore, nested/unwinding scopes |
| Complete training checkpoint | `cargo run --example training_checkpoint`; next epoch's weights, moments, schedule and scaler match uninterrupted execution exactly |
| Public examples | `cargo test --workspace --doc`; GitHub training/sequence recipes compiled as documentation tests |

The contribution gates also include formatting, workspace check and Clippy,
Python policy tests, generated ledger validation, docs.rs configuration with
warnings denied, Rust 1.88 checking, workspace package verification, DCO sign-off,
and push to the review branch. CI runs the composite checkpoint example and all
six numerical parity test targets.

Backend evidence: CPU runs the new numerical fixtures. Existing CPU/MPS backend
checks pass on the development Mac. CUDA and CUDA autocast runtime branches are
explicitly skipped on this host because no CUDA device is available. Their
conditional tests remain runnable on CUDA hardware; no CUDA numerical pass is
claimed for this delivery.
