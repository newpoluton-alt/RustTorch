//! Save a consistent training boundary in one atomic, non-overwriting artifact.
//!
//! A checkpoint contains SafeTensors weights, the optimizer's typed state and
//! application-defined JSON state. Put scheduler/scaler state, exact loader
//! position and [`crate::reproducibility::TensorRngState`] in that application
//! state. Capture them at the same completed-update boundary. No Python pickle
//! or model code is executed while loading; rebuild the architecture in Rust.
//!
//! ```no_run
//! use rusttorch::{DeviceSpec, checkpoint::{save, TrainingCheckpoint, CheckpointLimits},
//!     nn::Sequential, optim::Adam};
//! let model = Sequential::builder().linear(2, 1).build(DeviceSpec::Cpu)?;
//! let mut optimizer = Adam::builder().build(model.var_store())?;
//! let limits = CheckpointLimits::default();
//! save("epoch-3.rtckpt", model.var_store(), &mut optimizer, &3_u64, limits)?;
//! let saved = TrainingCheckpoint::<u64>::read("epoch-3.rtckpt", limits)?;
//! let completed_epoch = saved.restore(model.var_store(), &mut optimizer)?;
//! assert_eq!(completed_epoch, 3);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use crate::{
    Device, RustTorchError, Tensor,
    nn::VarStore,
    optim::{Optimizer, OptimizerState},
};
use safetensors::{SafeTensors, tensor::TensorView};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    fs::{self, File, OpenOptions},
    io::{self, Read, Write},
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};
use zip::{ZipArchive, ZipWriter, write::FileOptions};

const VERSION: u32 = 1;
const WEIGHTS: &str = "model.safetensors";
const OPTIMIZER: &str = "optimizer.json";
const METADATA: &str = "metadata.json";
static TEMPORARY_ID: AtomicU64 = AtomicU64::new(0);

/// Checkpoint failures preserving filesystem, serialization and backend causes.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum CheckpointError {
    /// A checkpoint violates its format, resource or destination contract.
    #[error("invalid training checkpoint: {0}")]
    Invalid(String),
    /// A filesystem operation failed.
    #[error(transparent)]
    Io(#[from] io::Error),
    /// Application or optimizer JSON could not be encoded or decoded.
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    /// The ZIP envelope could not be read or written.
    #[error(transparent)]
    Archive(#[from] zip::result::ZipError),
    /// SafeTensors metadata or payload was malformed.
    #[error(transparent)]
    Weights(#[from] safetensors::SafeTensorError),
    /// Model or optimizer validation failed.
    #[error(transparent)]
    Model(#[from] RustTorchError),
    /// A native tensor operation failed.
    #[error(transparent)]
    Backend(#[from] tch::TchError),
}

/// Result from checkpoint operations.
pub type CheckpointResult<T> = std::result::Result<T, CheckpointError>;

/// Finite per-artifact resource ceilings used for both saving and loading.
///
/// These bound serialized payloads, not total process RSS: loading also stages
/// tensors and optimizer moments before changing model state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CheckpointLimits {
    /// Maximum total ZIP file bytes; default 1 GiB.
    pub archive_bytes: usize,
    /// Maximum SafeTensors payload including header; default 512 MiB.
    pub weight_bytes: usize,
    /// Maximum bytes in each JSON member; default 256 MiB.
    pub json_bytes: usize,
    /// Maximum number of named parameters/buffers; default 100,000.
    pub tensors: usize,
}

impl Default for CheckpointLimits {
    fn default() -> Self {
        Self {
            archive_bytes: 1024 * 1024 * 1024,
            weight_bytes: 512 * 1024 * 1024,
            json_bytes: 256 * 1024 * 1024,
            tensors: 100_000,
        }
    }
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Metadata<S> {
    version: u32,
    state: S,
}

/// Validated checkpoint contents, ready to restore into a matching architecture.
///
/// Application state stays typed. [`Self::restore`] validates names, dtypes,
/// shapes, tensor aliasing and optimizer compatibility, then prepares device
/// copies and moments before changing any live weights or settings.
#[derive(Debug)]
pub struct TrainingCheckpoint<S> {
    weights: BTreeMap<String, Tensor>,
    optimizer: OptimizerState,
    state: S,
}

impl<S: DeserializeOwned> TrainingCheckpoint<S> {
    /// Reads one bounded archive containing exactly the three expected members.
    ///
    /// Unsupported versions, unknown/duplicate entries, malformed shapes, CRC
    /// failures and excessive sizes return errors. Nothing is extracted to disk.
    /// Only regular local files are accepted; symlink entry points are rejected.
    pub fn read(path: impl AsRef<Path>, limits: CheckpointLimits) -> CheckpointResult<Self> {
        let path = path.as_ref();
        if !fs::symlink_metadata(path)?.file_type().is_file() {
            return Err(invalid("checkpoint must be a regular file"));
        }
        let file = File::open(path)?;
        check_size("archive", file.metadata()?.len(), limits.archive_bytes)?;
        // A bounded copy also protects against a file growing after metadata was read.
        let bytes = read_bounded(file, limits.archive_bytes)?;
        // zip 0.6 reserves its file table from the directory count before
        // parsing entries, so validate both ordinary and ZIP64 counts first.
        crate::deployment::validate_zip_envelope(&bytes, 3, limits.archive_bytes)?;
        let mut archive = ZipArchive::new(io::Cursor::new(bytes))?;
        if archive.len() != 3 {
            return Err(invalid("archive must contain exactly three members"));
        }
        let mut members = BTreeMap::new();
        let mut total = 0_usize;
        for index in 0..archive.len() {
            let mut entry = archive.by_index(index)?;
            let name = entry.name().to_owned();
            let limit = match name.as_str() {
                WEIGHTS => limits.weight_bytes,
                OPTIMIZER | METADATA => limits.json_bytes,
                _ => return Err(invalid("unexpected archive member")),
            };
            if entry.is_dir()
                || entry
                    .unix_mode()
                    .is_some_and(|mode| mode & 0o170000 != 0 && mode & 0o170000 != 0o100000)
            {
                return Err(invalid("archive members must be ordinary files"));
            }
            if members.contains_key(&name) {
                return Err(invalid("duplicate archive member"));
            }
            check_size(&name, entry.size(), limit)?;
            total = total
                .checked_add(
                    usize::try_from(entry.size())
                        .map_err(|_| invalid("entry size exceeds usize"))?,
                )
                .filter(|&total| total <= limits.archive_bytes)
                .ok_or_else(|| invalid("combined member payload exceeds archive limit"))?;
            let content = read_bounded(&mut entry, limit)?;
            if content.len() as u64 != entry.size() {
                return Err(invalid(
                    "archive member's decoded size differs from its directory",
                ));
            }
            members.insert(name, content);
        }
        let metadata: Metadata<S> = serde_json::from_slice(&members[METADATA])?;
        if metadata.version != VERSION {
            return Err(invalid("unsupported checkpoint schema version"));
        }
        let optimizer = serde_json::from_slice(&members[OPTIMIZER])?;
        let tensors = SafeTensors::deserialize(&members[WEIGHTS])?;
        if tensors.len() > limits.tensors {
            return Err(invalid("too many named tensors"));
        }
        let mut weights = BTreeMap::new();
        for (name, view) in tensors.tensors() {
            if name.is_empty() || name.len() > 4096 || view.shape().len() > 64 {
                return Err(invalid("invalid tensor name or rank"));
            }
            let shape = view
                .shape()
                .iter()
                .map(|&dimension| {
                    i64::try_from(dimension).map_err(|_| invalid("tensor dimension exceeds i64"))
                })
                .collect::<CheckpointResult<Vec<_>>>()?;
            let kind = tch::Kind::try_from(view.dtype())?;
            let count = view
                .shape()
                .iter()
                .try_fold(1usize, |n, &d| n.checked_mul(d))
                .ok_or_else(|| invalid("tensor shape product overflow"))?;
            if count.checked_mul(kind.elt_size_in_bytes()) != Some(view.data().len()) {
                return Err(invalid("tensor byte count mismatch"));
            }
            validate_tensor_bytes(kind, view.data())?;
            let mut bytes = view.data().to_vec();
            if cfg!(target_endian = "big") {
                for element in bytes.chunks_exact_mut(kind.elt_size_in_bytes()) {
                    element.reverse();
                }
            }
            weights.insert(name, Tensor::f_from_data_size(&bytes, &shape, kind)?);
        }
        Ok(Self {
            weights,
            optimizer,
            state: metadata.state,
        })
    }
}

impl<S> TrainingCheckpoint<S> {
    /// Borrows application state for validation before restoring model state.
    ///
    /// Validate the dataset identity and build the resumed exact loader from
    /// this state before calling [`Self::restore`]. This keeps incompatible data
    /// configuration from changing the model.
    pub const fn state(&self) -> &S {
        &self.state
    }

    /// Restores weights and optimizer state, returning the application state.
    ///
    /// Build the optimizer from this exact store. Destination tensors must be
    /// contiguous; partial storage overlap is rejected. Exact aliases require
    /// identical saved values. Existing tensor identities are preserved and
    /// gradients are not cleared. Call at a completed update boundary, then
    /// restore the scheduler, scaler, loader and local RNG before the next step.
    /// As with other native model operations, an execution failure during the
    /// final device copies can leave partial values; restore into a fresh model
    /// when retaining a previously running model is required.
    pub fn restore(self, store: &VarStore, optimizer: &mut Optimizer) -> CheckpointResult<S> {
        let destination = store.variables().into_iter().collect::<BTreeMap<_, _>>();
        if destination.keys().collect::<BTreeSet<_>>() != self.weights.keys().collect() {
            return Err(invalid("model parameter/buffer names differ"));
        }
        ensure_optimizer_store(store, optimizer)?;
        let mut prepared = Vec::new();
        let mut ranges: HashMap<Device, Vec<(usize, usize, &str)>> = HashMap::new();
        for (name, target) in &destination {
            let saved = &self.weights[name];
            if target.size() != saved.size()
                || target.kind() != saved.kind()
                || !target.is_contiguous()
            {
                return Err(invalid(&format!(
                    "model tensor metadata differs for {name}"
                )));
            }
            let bytes = target
                .numel()
                .checked_mul(target.kind().elt_size_in_bytes())
                .ok_or_else(|| invalid("model byte count overflow"))?;
            if bytes != 0 {
                let start = target.data_ptr() as usize;
                let end = start
                    .checked_add(bytes)
                    .ok_or_else(|| invalid("model storage range overflow"))?;
                ranges
                    .entry(target.device())
                    .or_default()
                    .push((start, end, name));
            }
            prepared.push((target.shallow_clone(), saved.f_to_device(target.device())?));
        }
        for device_ranges in ranges.values_mut() {
            device_ranges.sort_unstable();
            for pair in device_ranges.windows(2) {
                let (start, end, name) = pair[1];
                let (previous_start, previous_end, previous_name) = pair[0];
                let saved = &self.weights[name];
                let previous = &self.weights[previous_name];
                if start < previous_end
                    && (start != previous_start
                        || end != previous_end
                        || saved.kind() != previous.kind()
                        || saved.size() != previous.size()
                        || !saved.f_equal(previous)?)
                {
                    return Err(invalid(
                        "overlapping model tensors have incompatible checkpoint values",
                    ));
                }
            }
        }
        let prepared_optimizer = optimizer.prepare_state_dict(&self.optimizer)?;
        crate::no_grad(|| -> CheckpointResult<()> {
            for (mut target, value) in prepared {
                target.f_copy_(&value)?;
            }
            Ok(())
        })?;
        optimizer.apply_prepared_state(prepared_optimizer);
        Ok(self.state)
    }
}

/// Saves one training boundary without overwriting an existing checkpoint.
///
/// Capture application state at the same completed step as the model/optimizer.
/// All bytes are written and flushed to a temporary sibling file, then published
/// with an atomic hard link that fails if the destination exists. This requires
/// a filesystem supporting same-directory hard links (including normal NTFS,
/// APFS and ext4 setups). Failed saves remove their temporary file and preserve
/// any prior destination. No cloud/network upload is performed.
pub fn save<S: Serialize>(
    path: impl AsRef<Path>,
    store: &VarStore,
    optimizer: &mut Optimizer,
    state: &S,
    limits: CheckpointLimits,
) -> CheckpointResult<()> {
    let path = path.as_ref();
    if path.file_name().is_none() {
        return Err(invalid("checkpoint destination needs a file name"));
    }
    if fs::symlink_metadata(path).is_ok() {
        return Err(invalid("checkpoint destination already exists"));
    }
    let variables = store.variables().into_iter().collect::<BTreeMap<_, _>>();
    if variables.len() > limits.tensors {
        return Err(invalid("too many named tensors"));
    }
    let mut size = 0_usize;
    let mut payloads = Vec::new();
    for (name, tensor) in variables {
        if name.is_empty() || name.len() > 4096 || tensor.size().len() > 64 {
            return Err(invalid("invalid tensor name or rank"));
        }
        let count = tensor.numel();
        let bytes = count
            .checked_mul(tensor.kind().elt_size_in_bytes())
            .ok_or_else(|| invalid("tensor byte size overflow"))?;
        size = size
            .checked_add(bytes)
            .filter(|&n| n <= limits.weight_bytes)
            .ok_or_else(|| invalid("model exceeds checkpoint weight limit"))?;
        let tensor = tensor
            .f_detach()?
            .f_to_device(Device::Cpu)?
            .f_contiguous()?;
        let mut data = Vec::new();
        data.try_reserve_exact(bytes)
            .map_err(|_| invalid("cannot reserve model payload"))?;
        data.resize(bytes, 0);
        tensor.f_copy_data_u8(&mut data, count)?;
        validate_tensor_bytes(tensor.kind(), &data)?;
        if cfg!(target_endian = "big") {
            for element in data.chunks_exact_mut(tensor.kind().elt_size_in_bytes()) {
                element.reverse();
            }
        }
        payloads.push((
            name,
            tensor.kind().try_into()?,
            tensor
                .size()
                .into_iter()
                .map(|d| d as usize)
                .collect::<Vec<_>>(),
            data,
        ));
    }
    let views = payloads
        .iter()
        .map(|(name, dtype, shape, bytes)| {
            Ok((
                name.as_str(),
                TensorView::new(*dtype, shape.clone(), bytes)?,
            ))
        })
        .collect::<std::result::Result<Vec<_>, safetensors::SafeTensorError>>()?;
    let weights = safetensors::tensor::serialize(views, &None)?;
    check_size(WEIGHTS, weights.len() as u64, limits.weight_bytes)?;
    let optimizer_state = optimizer.state_dict()?;
    ensure_optimizer_store(store, optimizer)?;
    let optimizer_json = json_bounded(&optimizer_state, limits.json_bytes)?;
    let metadata = json_bounded(
        &Metadata {
            version: VERSION,
            state,
        },
        limits.json_bytes,
    )?;
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let (temporary, file) = create_temporary(parent)?;
    let cleanup = Temporary(temporary.clone());
    let mut zip = ZipWriter::new(file);
    let options = FileOptions::default()
        .compression_method(zip::CompressionMethod::Stored)
        .unix_permissions(0o600);
    for (name, bytes) in [
        (WEIGHTS, weights.as_slice()),
        (OPTIMIZER, optimizer_json.as_slice()),
        (METADATA, metadata.as_slice()),
    ] {
        zip.start_file(name, options)?;
        zip.write_all(bytes)?;
    }
    let file = zip.finish()?;
    check_size("archive", file.metadata()?.len(), limits.archive_bytes)?;
    file.sync_all()?;
    drop(file);
    fs::hard_link(&temporary, path)?;
    drop(cleanup);
    Ok(())
}

fn ensure_optimizer_store(store: &VarStore, optimizer: &Optimizer) -> CheckpointResult<()> {
    if optimizer.belongs_to(store) {
        Ok(())
    } else {
        Err(invalid("optimizer belongs to a different model store"))
    }
}

fn validate_tensor_bytes(kind: tch::Kind, bytes: &[u8]) -> CheckpointResult<()> {
    // SafeTensors validates byte counts, but BOOL's valid values also matter:
    // the native constructor copies bytes directly into bool tensor storage.
    if kind == tch::Kind::Bool && bytes.iter().any(|&value| value > 1) {
        return Err(invalid("boolean tensor payload must contain only 0 or 1"));
    }
    Ok(())
}

fn create_temporary(parent: &Path) -> io::Result<(PathBuf, File)> {
    for _ in 0..100 {
        let id = TEMPORARY_ID.fetch_add(1, Ordering::Relaxed);
        let path = parent.join(format!(
            ".rusttorch-checkpoint-{}-{id}.tmp",
            std::process::id()
        ));
        match OpenOptions::new().write(true).create_new(true).open(&path) {
            Ok(file) => return Ok((path, file)),
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error),
        }
    }
    Err(io::Error::new(
        io::ErrorKind::AlreadyExists,
        "checkpoint temporary-name collision limit reached",
    ))
}

struct Temporary(PathBuf);
impl Drop for Temporary {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}

struct BoundedJson {
    bytes: Vec<u8>,
    limit: usize,
}
impl Write for BoundedJson {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if self
            .bytes
            .len()
            .checked_add(bytes.len())
            .is_none_or(|len| len > self.limit)
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "checkpoint JSON exceeds resource limit",
            ));
        }
        self.bytes
            .try_reserve(bytes.len())
            .map_err(io::Error::other)?;
        self.bytes.extend_from_slice(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
fn json_bounded(value: &impl Serialize, limit: usize) -> CheckpointResult<Vec<u8>> {
    let mut writer = BoundedJson {
        bytes: Vec::new(),
        limit,
    };
    serde_json::to_writer(&mut writer, value)?;
    Ok(writer.bytes)
}
fn read_bounded(reader: impl Read, limit: usize) -> CheckpointResult<Vec<u8>> {
    let mut bytes = Vec::new();
    reader
        .take((limit as u64).saturating_add(1))
        .read_to_end(&mut bytes)?;
    if bytes.len() > limit {
        return Err(invalid("checkpoint payload exceeds resource limit"));
    }
    Ok(bytes)
}
fn check_size(name: &str, size: u64, limit: usize) -> CheckpointResult<()> {
    if size > limit as u64 {
        Err(invalid(&format!("{name} exceeds resource limit")))
    } else {
        Ok(())
    }
}
fn invalid(reason: &str) -> CheckpointError {
    CheckpointError::Invalid(reason.into())
}
