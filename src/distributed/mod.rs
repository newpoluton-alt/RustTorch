//! Coordinate CPU training processes and save portable distributed state.
//!
//! [`ProcessGroup`] provides blocking TCP collectives on dense CPU `Float`,
//! `Double`, and `Int64` tensors. [`DistributedDataParallel`] synchronizes an
//! ordinary model explicitly; [`ShardedTrainer`] keeps parameter and optimizer
//! shards between functional training steps. Neither requires Python at runtime.
//!
//! Connections are unencrypted and intended for mutually trusted workers. Use
//! an isolated network, a unique session name, identical operation order, and a
//! finite timeout. A failed communication poisons the group; restart all ranks
//! from their last completed checkpoint instead of retrying a partial step.
//!
//! ```
//! use rusttorch::{Tensor, distributed::{ProcessGroup, GroupOptions, ReduceOp}};
//! use std::net::TcpListener;
//! // A one-process group is useful for testing the same training program locally.
//! let listener = TcpListener::bind("127.0.0.1:0").unwrap();
//! let mut group = ProcessGroup::accept(listener, 1, GroupOptions::new("example"))?;
//! let total = group.all_reduce(&Tensor::from_slice(&[2_f64, 3.]), ReduceOp::Sum)?;
//! assert_eq!(Vec::<f64>::try_from(total)?, [2., 3.]);
//! # Ok::<(), rusttorch::RustTorchError>(())
//! ```
//!
//! See the [distributed training guide](crate::tutorials) for complete launch,
//! accumulation, checkpoint and resharding workflows. CUDA/NCCL, asynchronous
//! work handles, automatic backward hooks and communication overlap are absent.

mod process;
mod training;

pub use process::{GroupOptions, ProcessGroup, ReduceOp};
pub use training::{
    ConsolidatedCheckpoint, DistributedDataParallel, ShardedCheckpoint, ShardedTrainer,
};

use crate::{Result, RustTorchError};

fn invalid(reason: impl Into<String>) -> RustTorchError {
    RustTorchError::InvalidConfiguration {
        field: "distributed",
        reason: reason.into(),
    }
}

fn json<T: serde::Serialize>(value: &T) -> Result<Vec<u8>> {
    serde_json::to_vec(value).map_err(|e| invalid(format!("cannot encode distributed state: {e}")))
}

fn decode<T: serde::de::DeserializeOwned>(bytes: &[u8]) -> Result<T> {
    serde_json::from_slice(bytes).map_err(|e| invalid(format!("invalid distributed state: {e}")))
}
