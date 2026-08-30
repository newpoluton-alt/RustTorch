//! Model-state interchange helpers.
//!
//! SafeTensors is the preferred portable format. Python pickle files are not
//! accepted by these APIs.

use std::{
    collections::{BTreeMap, BTreeSet},
    path::{Path, PathBuf},
};

use tch::{Device, Tensor, nn::VarStore, no_grad};

use crate::{Result, RustTorchError};

/// Explicit source-key to destination-key mappings for a state dictionary.
#[derive(Debug, Clone, Default)]
pub struct StateDictMapping {
    exact: BTreeMap<String, String>,
    prefixes: BTreeMap<String, String>,
}

impl StateDictMapping {
    /// Creates an empty identity mapping.
    pub fn new() -> Self {
        Self::default()
    }

    /// Maps one file key to one model key.
    #[must_use]
    pub fn map(mut self, source: impl Into<String>, destination: impl Into<String>) -> Self {
        self.exact.insert(source.into(), destination.into());
        self
    }

    /// Maps a source prefix to a destination prefix. Exact mappings win.
    #[must_use]
    pub fn map_prefix(mut self, source: impl Into<String>, destination: impl Into<String>) -> Self {
        self.prefixes.insert(source.into(), destination.into());
        self
    }

    fn destination(&self, source: &str) -> String {
        if let Some(destination) = self.exact.get(source) {
            return destination.clone();
        }
        self.prefixes
            .iter()
            .filter(|(prefix, _)| source.starts_with(prefix.as_str()))
            .max_by_key(|(prefix, _)| prefix.len())
            .map_or_else(
                || source.to_owned(),
                |(prefix, destination)| format!("{destination}{}", &source[prefix.len()..]),
            )
    }
}

/// State-loading policy.
///
/// Use [`LoadOptions::strict`] or [`LoadOptions::non_strict`] to construct this
/// value; future releases may add policy fields.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct LoadOptions {
    /// Whether missing or unexpected keys fail the load.
    pub strict: bool,
    /// Whether to validate and report without changing model tensors.
    pub dry_run: bool,
}

impl LoadOptions {
    /// Creates options that reject missing and unexpected keys.
    pub const fn strict() -> Self {
        Self {
            strict: true,
            dry_run: false,
        }
    }

    /// Creates options that load matching keys and report unmatched keys.
    pub const fn non_strict() -> Self {
        Self {
            strict: false,
            dry_run: false,
        }
    }

    #[must_use]
    /// Enables or disables validation-only execution.
    pub const fn dry_run(mut self, dry_run: bool) -> Self {
        self.dry_run = dry_run;
        self
    }
}

impl Default for LoadOptions {
    fn default() -> Self {
        Self::strict()
    }
}

/// Deterministic report for a state-dictionary load.
///
/// Future releases may add diagnostic fields.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
#[non_exhaustive]
pub struct LoadReport {
    /// Destination model keys that matched and were validated.
    pub loaded: Vec<String>,
    /// Destination model keys absent from the input file.
    pub missing: Vec<String>,
    /// Source file keys that did not map to the model.
    pub unexpected: Vec<String>,
    /// Applied `(source, destination)` key mappings.
    pub remapped: Vec<(String, String)>,
}

/// Saves every named variable and persistent buffer in a `VarStore`.
pub fn save_state_dict(path: impl AsRef<Path>, var_store: &VarStore) -> Result<()> {
    require_safetensors(path.as_ref())?;
    let tensors = var_store
        .variables()
        .into_iter()
        .collect::<BTreeMap<_, _>>()
        .into_iter()
        .map(|(name, tensor)| {
            let tensor = tensor.f_to_device(Device::Cpu)?.f_contiguous()?;
            Ok((name, tensor))
        })
        .collect::<Result<Vec<_>>>()?;
    Tensor::write_safetensors(&tensors, path).map_err(Into::into)
}

/// Loads a SafeTensors state dictionary strictly, without key remapping.
pub fn load_state_dict(path: impl AsRef<Path>, var_store: &VarStore) -> Result<LoadReport> {
    load_state_dict_with_mapping(
        path,
        var_store,
        &StateDictMapping::new(),
        LoadOptions::strict(),
    )
}

/// Validates and loads a SafeTensors state dictionary.
///
/// All keys, shapes, dtypes, and mapping destinations are checked before any
/// model tensor is changed.
pub fn load_state_dict_with_mapping(
    path: impl AsRef<Path>,
    var_store: &VarStore,
    mapping: &StateDictMapping,
    options: LoadOptions,
) -> Result<LoadReport> {
    let path = path.as_ref();
    require_safetensors(path)?;

    let model = var_store
        .variables()
        .into_iter()
        .collect::<BTreeMap<_, _>>();
    let mut file = Tensor::read_safetensors(path)?;
    file.sort_by(|(left, _), (right, _)| left.cmp(right));

    let mut report = LoadReport::default();
    let mut destinations = BTreeSet::new();
    let mut copies = Vec::new();

    for (source_name, source) in file {
        let destination = mapping.destination(&source_name);
        if !destinations.insert(destination.clone()) {
            return Err(RustTorchError::DuplicateMappedKey(destination));
        }
        if source_name != destination {
            report
                .remapped
                .push((source_name.clone(), destination.clone()));
        }

        let Some(target) = model.get(&destination) else {
            report.unexpected.push(source_name);
            continue;
        };
        if source.size() != target.size() {
            return Err(RustTorchError::ShapeMismatch {
                name: destination,
                expected: target.size(),
                actual: source.size(),
            });
        }
        if source.kind() != target.kind() {
            return Err(RustTorchError::DtypeMismatch {
                name: destination,
                expected: target.kind(),
                actual: source.kind(),
            });
        }
        report.loaded.push(destination.clone());
        copies.push((destination, source));
    }

    let loaded = report.loaded.iter().cloned().collect::<BTreeSet<_>>();
    report.missing = model
        .keys()
        .filter(|name| !loaded.contains(*name))
        .cloned()
        .collect();
    report.loaded.sort();
    report.missing.sort();
    report.unexpected.sort();
    report.remapped.sort();

    if options.strict && (!report.missing.is_empty() || !report.unexpected.is_empty()) {
        return Err(RustTorchError::IncompatibleModelState {
            missing: report.missing,
            unexpected: report.unexpected,
        });
    }

    if !options.dry_run {
        no_grad(|| -> Result<()> {
            for (name, source) in copies {
                let mut target = model[&name].shallow_clone();
                target.f_copy_(&source)?;
            }
            Ok(())
        })?;
    }

    Ok(report)
}

fn require_safetensors(path: &Path) -> Result<()> {
    if path
        .extension()
        .is_some_and(|extension| extension == "safetensors")
    {
        Ok(())
    } else {
        Err(RustTorchError::UnsupportedModelFile {
            path: PathBuf::from(path),
            reason: "expected a .safetensors weight file".to_owned(),
        })
    }
}
