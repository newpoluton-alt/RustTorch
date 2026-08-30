//! Errors returned by the fallible RustTorch API.

use std::path::PathBuf;

use tch::{Device, Kind};

/// Result type used by RustTorch APIs that can fail recoverably.
pub type Result<T> = std::result::Result<T, RustTorchError>;

/// Recoverable configuration, graph, state, and backend failures.
///
/// New variants may be added in future releases, so downstream matches must
/// include a wildcard arm.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum RustTorchError {
    /// A named configuration field has an invalid value.
    #[error("invalid configuration for {field}: {reason}")]
    InvalidConfiguration {
        /// Configuration field that failed validation.
        field: &'static str,
        /// Human-readable validation failure.
        reason: String,
    },

    /// Tensor dimensions do not satisfy an operation's contract.
    #[error("invalid dimensions for {context}: expected {expected}, got {actual}")]
    InvalidDimensions {
        /// Operation or value being validated.
        context: String,
        /// Required dimensions or rank.
        expected: String,
        /// Dimensions or rank that were supplied.
        actual: String,
    },

    /// A requested compatibility option is not implemented by the backend.
    #[error("{component} does not support option `{option}` in the MVP")]
    UnsupportedOption {
        /// Component that rejected the option.
        component: &'static str,
        /// Unsupported option name.
        option: &'static str,
    },

    /// An explicitly requested execution backend cannot be used.
    #[error("{backend} backend is unavailable: {reason}")]
    BackendUnavailable {
        /// Backend name.
        backend: &'static str,
        /// Reason the backend cannot be used.
        reason: String,
    },

    /// A tensor is located on a different device than required.
    #[error("device mismatch in {context}: expected {expected:?}, got {actual:?}")]
    DeviceMismatch {
        /// Operation or value being validated.
        context: String,
        /// Required tensor device.
        expected: Device,
        /// Actual tensor device.
        actual: Device,
    },

    /// A state tensor's shape differs from the model tensor's shape.
    #[error("shape mismatch for `{name}`: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        /// Destination state key.
        name: String,
        /// Shape required by the model.
        expected: Vec<i64>,
        /// Shape found in the input state.
        actual: Vec<i64>,
    },

    /// A state tensor's dtype differs from the model tensor's dtype.
    #[error("dtype mismatch for `{name}`: expected {expected:?}, got {actual:?}")]
    DtypeMismatch {
        /// Destination state key.
        name: String,
        /// Dtype required by the model.
        expected: Kind,
        /// Dtype found in the input state.
        actual: Kind,
    },

    /// Strict state loading found missing or unexpected keys.
    #[error("incompatible model state; missing={missing:?}, unexpected={unexpected:?}")]
    IncompatibleModelState {
        /// Model keys absent from the input state.
        missing: Vec<String>,
        /// Input keys that do not map to model state.
        unexpected: Vec<String>,
    },

    /// Multiple source keys map to the same destination key.
    #[error("duplicate state mapping target `{0}`")]
    DuplicateMappedKey(String),

    /// A graph or input collection contains the same name more than once.
    #[error("duplicate name `{0}`")]
    DuplicateName(String),

    /// A required named graph input was not provided.
    #[error("missing graph input `{0}`")]
    MissingGraphInput(String),

    /// A graph input was provided but is not declared by the graph.
    #[error("unexpected graph input `{0}`")]
    UnexpectedGraphInput(String),

    /// A requested named graph output does not exist.
    #[error("missing graph output `{0}`")]
    MissingGraphOutput(String),

    /// A graph violates structural or tensor-spec constraints.
    #[error("graph validation failed: {0}")]
    GraphValidation(String),

    /// A graph operation cannot be executed by the selected executor.
    #[error("unsupported graph operation: {0}")]
    UnsupportedGraphOperation(String),

    /// A model file format is not supported by the state APIs.
    #[error("unsupported model file `{path}`: {reason}")]
    UnsupportedModelFile {
        /// Rejected file path.
        path: PathBuf,
        /// Required format or rejection reason.
        reason: String,
    },

    /// An error returned by `tch` or LibTorch.
    #[error(transparent)]
    Backend(#[from] tch::TchError),
}
