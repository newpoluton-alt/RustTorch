//! Load typed training batches from indexed datasets or lazy streams.
//!
//! Start with [`DataLoader::builder`] when your application owns its dataset and
//! trains for multiple epochs. Use [`DataLoader::new`] for a single pass borrowing
//! an existing dataset, or [`batches`] to group an ordinary fallible iterator.
//! The loader returns `Result` for each batch, so decoding and collation failures
//! can be handled at the training boundary.
//!
//! # Train from paired features and labels
//!
//! A dataset can return `(Tensor, i64)` and let the default collator produce
//! `(Tensor, Tensor)`: feature rows gain a batch dimension, and integer labels
//! become an `Int64` tensor. This example keeps ordinary Rust values in the
//! dataset and creates each tensor when fetched, so two workers can safely share
//! it while preparing batches for a small classifier.
//!
//! ```
//! use std::convert::Infallible;
//! use rusttorch::{
//!     DeviceSpec, Tensor,
//!     data::{DataLoader, Dataset},
//!     nn::{Sequential, functional},
//!     optim::Adam,
//! };
//!
//! struct Examples(Vec<([f32; 2], i64)>);
//! impl Dataset for Examples {
//!     type Sample = (Tensor, i64);
//!     type Error = Infallible;
//!     fn len(&self) -> usize { self.0.len() }
//!     fn get(&self, index: usize) -> Result<Self::Sample, Infallible> {
//!         let (features, label) = &self.0[index];
//!         Ok((Tensor::from_slice(features), *label))
//!     }
//! }
//!
//! fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let examples = Examples(vec![([-1., 0.], 0), ([1., 0.], 1),
//!                                 ([-2., 1.], 0), ([2., 1.], 1)]);
//!     let mut loader = DataLoader::builder(examples)
//!         .batch_size(2).shuffle(42)?.workers(2).prefetch_factor(2).build()?;
//!     let model = Sequential::builder().linear(2, 2).build(DeviceSpec::Cpu)?;
//!     let mut optimizer = Adam::builder().learning_rate(0.01).build(model.var_store())?;
//!
//!     for epoch in 0..3 {
//!         loader.set_epoch(epoch);
//!         for batch in loader.iter() {
//!             let (features, labels) = batch?;
//!             assert_eq!(features.size(), [2, 2]);
//!             assert_eq!(labels.size(), [2]);
//!             let logits = model.forward_t(&features, true)?;
//!             let loss = functional::cross_entropy(&logits, &labels)?;
//!             optimizer.backward_step(&loss)?;
//!         }
//!     }
//!     Ok(())
//! }
//! ```
//!
//! `set_epoch` changes the seeded shuffle for the next pass; `iter()` alone
//! repeats the selected epoch. A loader keeps its last short batch unless
//! [`DataLoaderBuilder::drop_last`] is enabled. For tensors already resident in
//! memory, [`TensorDataset`] avoids a custom dataset implementation; its matching
//! fields are returned as a vector of batched tensors.
//!
//! # Choose the output and execution model
//!
//! | Use case | Entry point | Output |
//! | --- | --- | --- |
//! | Borrow a dataset for inspection | [`DataLoader::new`] | A `Vec` of samples per batch; iterate directly |
//! | Train over repeated epochs | [`DataLoader::builder`] | Default recursive collation; call `build()`, then `iter()` |
//! | Keep records as Rust values | [`DataLoaderBuilder::collate`] with [`VecCollate`] | `Vec<Sample>` |
//! | Pad variable-length sequences | [`FnCollate`] | Your own batch type |
//! | Yield individual samples | [`DataLoaderBuilder::without_batching`] | Converted sample, with no added batch dimension |
//! | Group an existing iterator | [`batches`] or [`batches_with_collate`] | Lazy batches without a dataset implementation |
//! | Read separately owned stream shards | [`StreamDataLoaderBuilder`] | Globally merged batches |
//! | Resume the next unconsumed batch | [`LoaderState`] or [`ExactStreamDataLoader`] | Validated checkpoint state |
//!
//! Workers prepare samples; collation runs on the calling thread. Start with the
//! default serial builder when debugging or using a dataset that is not
//! `Send + Sync`. [`DataLoaderBuilder::workers`] requires a thread-safe dataset,
//! even when passed zero. Prefetch bounds pending work, while
//! [`DataLoaderBuilder::pin_memory`] prepares host tensors for CUDA transfer;
//! it does not move them to the model's device.
//!
//! See [`DataLoaderBuilder`] for padding, unbatched loading and epoch examples,
//! [`StreamDataLoaderBuilder`] for lazy sharding, and [`LoaderState`] for a complete
//! map-loader restart. All items below are available through `rusttorch::data`.

#[doc(inline)]
pub use rusttorch_data::*;
