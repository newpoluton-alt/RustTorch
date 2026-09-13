# Text fixture provenance

`pipelines.rs` constructs an original, tiny WordLevel vocabulary and whitespace pre-tokenizer through the public tokenizers API. All text strings, token IDs and length vectors are synthetic source literals. No model, vocabulary or dataset is downloaded. The tests cover actual encoding, worker batching, token masks, padding errors, distributed padded-cost batches and fresh-sampler restoration at a nonzero epoch. These fixtures are contributed under MIT OR Apache-2.0.
