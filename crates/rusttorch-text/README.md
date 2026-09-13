# RustTorch Text

Prepare local text for embedding, classification and Transformer models using
Hugging Face tokenizers, explicit length policies and typed loader batches.
Nothing downloads a vocabulary or model implicitly.

## Installation

Enable `rusttorch = { version = "0.4", features = ["text"] }` to use
`rusttorch::text`. Direct consumers can use `rusttorch-text` with
`download-libtorch`. Advanced tokenizer types are available through
`rusttorch_text::tokenizers`.

## Features

The `text` facade feature enables local Hugging Face tokenization, padding and
token-budget batch sampling. The direct package has no default features.
`download-libtorch` obtains the tensor runtime; `doc-only` builds documentation
without native linking and cannot execute tensor operations.

## Native runtime

Tensor operations require **LibTorch 2.13.0**. Enable `download-libtorch`, or set
`LIBTORCH` to an extracted distribution and add its library directory to `PATH`
on Windows or the platform's shared-library search path. All RustTorch packages
in one application must share the same runtime. Documentation can be checked
with `cargo doc -p rusttorch-text --no-default-features --features doc-only`.

## Example: Tokenize, budget and batch

```no_run
use rusttorch_text::{HfTokenizer, Padding, TextCollator, TextDataset,
    TokenBudgetConfig, TokenBudgetSampler, Truncation};
use rusttorch_data::{DataLoader, ResourceLimits};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let limits = ResourceLimits::default();
    let tokenizer = HfTokenizer::from_file(
        "tokenizer.json", Truncation::Reject(512), true, limits,
    )?;
    println!("vocabulary entries: {}", tokenizer.vocabulary_size());
    let data = TextDataset::new(vec![
        "RustTorch prepares model inputs.".into(),
        "A shorter sentence.".into(),
    ], tokenizer)?;
    let batches = TokenBudgetSampler::new(data.lengths()?, TokenBudgetConfig {
        max_tokens: 2048, max_samples: 16, shuffle: true, seed: 42,
        ..Default::default()
    }, limits)?;
    let collator = TextCollator::new(Padding::Longest { pad_id: 0 }, limits)?;
    let mut loader = DataLoader::builder(data)
        .batch_sampler(batches)
        .collate(collator)
        .workers(2)
        .build()?;
    for batch in loader.iter() {
        let batch = batch?;
        // All four model-input tensors are Int64 [batch, padded_length].
        println!("{:?}", batch.input_ids.size());
        assert_eq!(batch.attention_mask.size(), batch.input_ids.size());
    }
    Ok(())
}
```

Use the padding ID from your model's vocabulary; zero is an example, not a
universal token ID. `attention_mask` is one for real tokens and zero for padding.
`token_type_ids` preserves the tokenizer's segment IDs, and
`special_tokens_mask` marks original special tokens. Original lengths remain
available separately. Padding values have zero type/special-token masks.

The budget accounts for padding: a batch costs **number of samples × longest
sequence length**. Every sequence must fit by itself, and `max_samples` also
bounds batches of empty sequences. Measured lengths must use the same tokenizer
and policy as loading; stochastic tokenizer configurations are inappropriate
for fixed-length plans and exact replay.

## Choose truncation and padding deliberately

```rust
use rusttorch_text::{EncodedSequence, Padding, TextCollator};
use rusttorch_data::Collate;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut collate = TextCollator::new(
        Padding::Fixed { pad_id: 9, length: 4 }, Default::default(),
    )?;
    let batch = collate.collate(vec![
        EncodedSequence::from_ids(vec![2, 3]),
        EncodedSequence::from_ids(vec![4]),
    ])?;
    assert_eq!(Vec::<Vec<i64>>::try_from(&batch.input_ids)?,
        [[2, 3, 9, 9], [4, 9, 9, 9]]);
    Ok(())
}
```

`Truncation::Reject(n)` returns an error above `n`; `Right(n)` keeps the first
`n` tokens **after** special-token insertion. Right truncation can remove a
trailing separator. Models that require preserving closing tokens should use a
suitable tokenizer/postprocessor policy and reject unexpected overlength input.
The adapter disables native padding/truncation to avoid applying two competing
length policies.

`Padding::Longest` selects each batch's maximum width. `Padding::Fixed` rejects
longer input; it never silently truncates. An all-empty longest-padded batch has
width zero, so check whether your model accepts it.

## Distributed sampling and resume

`TokenBudgetConfig` includes replicas, rank, epoch-based seeded shuffle and
rank-tail policy. It delegates index partitioning to the shared distributed
sampler. Equal sample counts do **not** guarantee equal batch counts when
sequence lengths differ. Synchronized training must handle uneven iteration
explicitly.

`TokenBudgetSampler` implements the shared batch-source checkpoint contract.
Its versioned state records length/configuration identity and the committed
sample/batch boundary; incompatible restoration fails. This does not make an
arbitrary external text source or stochastic tokenizer replay-safe. Exact
loader resume additionally requires the existing source/transform contracts.

## Limits and evidence

Text/vocabulary files, input strings, sequence lengths and padded tensor bytes
have finite limits. Large outputs are rejected; the upstream tokenizer still
owns temporary encoding allocations. Tests cover actual WordLevel tokenization,
worker batches, padding errors, budget overflow, disjoint ranks and restored
remaining batches in `tests/pipelines.rs`. The package adds no HTTP downloader,
plugin registry or separate loader.

| Operation | Executable evidence |
|---|---|
| Local WordLevel tokenization, workers and masks | `tokenizer_to_worker_loader_produces_ids_masks_and_budget_batches` |
| Explicit padding and invalid sequence rejection | `padding_policies_and_limits_reject_invalid_sequences` |
| Resume a nonzero epoch into a fresh sampler | `token_budget_resume_replays_only_remaining_complete_batches` |
| Disjoint distributed rank indices | `distributed_token_batches_preserve_disjoint_rank_indices` |
