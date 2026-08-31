use rusttorch_core::RustTorchError;

/// A typed failure produced while an owned data loader is active.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum LoaderError<E> {
    /// A dataset, transform, or collation stage failed.
    #[error("data pipeline failed (batch {batch:?}, worker {worker:?}): {source}")]
    Pipeline {
        /// Logical batch when known.
        batch: Option<u64>,
        /// Worker identifier, or `None` for coordinator work.
        worker: Option<usize>,
        /// Preserved pipeline error.
        #[source]
        source: E,
    },
    /// A worker thread panicked.
    #[error("worker {worker} panicked while loading batch {batch:?}")]
    WorkerPanic {
        /// Worker identifier.
        worker: usize,
        /// Logical batch when known.
        batch: Option<u64>,
    },
    /// Waiting for a batch exceeded the configured timeout.
    #[error("timed out waiting for batch {batch}")]
    Timeout {
        /// Logical batch being awaited.
        batch: u64,
    },
    /// Loading was cancelled.
    #[error("loader was cancelled")]
    Cancelled,
    /// A loader channel closed before the expected batch arrived.
    #[error("loader channel closed before batch {batch}")]
    ChannelClosed {
        /// Logical batch being awaited.
        batch: u64,
    },
    /// Batched dataset fetch returned the wrong number of samples.
    #[error(
        "batch {batch} on worker {worker:?} requested {expected} samples but the dataset returned {actual}"
    )]
    InvalidBatchCardinality {
        /// Logical batch that failed.
        batch: u64,
        /// Worker identifier, or `None` for serial loading.
        worker: Option<usize>,
        /// Number of requested samples.
        expected: usize,
        /// Number of returned samples.
        actual: usize,
    },
    /// A streaming source violated its sequence protocol.
    #[error("stream protocol failed at sequence {sequence:?}: {reason}")]
    StreamProtocol {
        /// Sequence identifier when known.
        sequence: Option<u64>,
        /// Human-readable protocol violation.
        reason: String,
    },
    /// A sample or batch exceeded the configured byte budget.
    #[error(
        "prefetch item exceeded the {limit} byte budget with {actual} bytes (batch {batch:?}, worker {worker:?}, sequence {sequence:?}, logical sample {logical_id:?})"
    )]
    MemoryLimit {
        /// Logical batch when known.
        batch: Option<u64>,
        /// Worker identifier when known.
        worker: Option<usize>,
        /// Stream sequence when known.
        sequence: Option<u64>,
        /// Logical sample identifier when known.
        logical_id: Option<u64>,
        /// Configured byte limit.
        limit: usize,
        /// Actual byte footprint.
        actual: usize,
    },
    /// Pinning a completed batch failed.
    #[error("pinning batch {batch} failed: {source}")]
    PinMemory {
        /// Logical batch that failed.
        batch: u64,
        /// Preserved backend failure.
        #[source]
        source: RustTorchError,
    },
    /// Checkpoint state was unavailable, corrupt, or incompatible.
    #[error("checkpoint is incompatible or unavailable: {reason}")]
    Checkpoint {
        /// Human-readable checkpoint rejection.
        reason: String,
    },
    /// Loader configuration is invalid or temporarily unsupported.
    #[error(transparent)]
    Configuration(#[from] RustTorchError),
}
