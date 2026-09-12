//! Turn samples into batches for training and inference.
//!
//! Use [`TensorDataset`] for features and labels stored in tensors, implement
//! [`Dataset`] for indexed records, or pass an existing fallible iterator to
//! [`batches`]. [`DataLoader::builder`] adds sampling, collation, optional workers,
//! and repeated iteration. The [data crate guide](rusttorch_data) includes
//! tensor batching, custom datasets, and links to resumable-loading APIs.
//!
//! # Batch an iterator
//!
//! Parsing and batching can remain lazy, so a large input need not be loaded
//! into memory at once. The final short batch is retained when `drop_last`
//! is `false`; parse errors propagate through the batch iterator.
//!
//! ```
//! use rusttorch::data::batches;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let records = ["10", "20", "30"].into_iter().map(str::parse::<i64>);
//! let loader = batches(records, 2, false)?;
//! let batches = loader.collect::<Result<Vec<_>, _>>()?;
//! assert_eq!(batches, [vec![10, 20], vec![30]]);
//! # Ok(())
//! # }
//! ```

pub use rusttorch_data::*;
