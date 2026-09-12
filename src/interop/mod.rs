//! Save trained weights, restore an inference model, and match parameter names.
//!
//! Use a `.safetensors` file to keep a RustTorch model between runs or deploy its
//! weights on another device. These functions work with any [`crate::nn::VarStore`],
//! including custom models assembled from individual layers. For a sequential
//! model, [`crate::nn::Sequential::save_weights`] and
//! [`crate::nn::Sequential::load_weights`] provide the same format directly.
//!
//! # Restore weights after renaming a layer
//!
//! A classifier exported under `head.weight` and `head.bias` can be loaded into
//! a model that calls the same layer `classifier`. Build matching dimensions
//! and dtypes first, then map file names to the destination model's names.
//! This complete example writes its own input file, previews the mapping,
//! restores the weights, and verifies identical predictions.
//!
//! ```
//! use rusttorch::{
//!     Device, Kind, Tensor, no_grad,
//!     nn::{LinearConfig, VarStore},
//!     interop::{LoadOptions, StateDictMapping, save_state_dict,
//!         load_state_dict_with_mapping},
//! };
//!
//! fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let original = VarStore::new(Device::Cpu);
//!     let head = LinearConfig::new(4, 3).build(&(original.root() / "head"))?;
//!     let file = std::env::temp_dir().join(format!(
//!         "rusttorch-mapped-classifier-{}.safetensors", std::process::id(),
//!     ));
//!     save_state_dict(&file, &original)?;
//!
//!     let restored = VarStore::new(Device::Cpu);
//!     let classifier = LinearConfig::new(4, 3)
//!         .build(&(restored.root() / "classifier"))?;
//!     let mapping = StateDictMapping::new()
//!         .map("head.weight", "classifier.weight")
//!         .map("head.bias", "classifier.bias");
//!     let preview = load_state_dict_with_mapping(
//!         &file, &restored, &mapping, LoadOptions::strict().dry_run(true),
//!     )?;
//!     assert_eq!(preview.loaded, ["classifier.bias", "classifier.weight"]);
//!     load_state_dict_with_mapping(&file, &restored, &mapping, LoadOptions::strict())?;
//!
//!     let features = Tensor::f_ones([2, 4], (Kind::Float, Device::Cpu))?;
//!     let expected = no_grad(|| head.forward(&features))?;
//!     let actual = no_grad(|| classifier.forward(&features))?;
//!     assert_eq!(actual.size(), [2, 3]);
//!     assert!(actual.f_allclose(&expected, 0., 0., false)?);
//!     std::fs::remove_file(file)?;
//!     Ok(())
//! }
//! ```
//!
//! Use [`load_state_dict`] when names already agree. For a whole renamed
//! submodule, [`StateDictMapping::map_prefix`] maps `"head."` to `"classifier."`.
//! Exact mappings take precedence; otherwise the longest matching prefix wins.
//! Unmapped keys keep their original names.
//!
//! # Validate an imported model
//!
//! Strict loading rejects missing or unexpected keys. For an intentional partial
//! load, choose [`LoadOptions::non_strict`] and inspect [`LoadReport::missing`]
//! and [`LoadReport::unexpected`]. Matching tensors still need identical shapes
//! and dtypes; loading neither reshapes nor casts them. All key, shape, dtype,
//! and mapping checks precede changes to model tensors. A dry run performs those
//! checks and returns the report without copying any values.
//!
//! Saving includes every registered parameter and persistent buffer, using
//! contiguous CPU tensor data. Loading copies into the existing destination
//! tensors on their current device. Build or move the model onto its intended
//! device first, then provide inputs on that same device for inference.
//!
//! # Keep architecture and training state alongside weights
//!
//! A weight file contains named tensor values. Rebuild the model architecture
//! and preserve its preprocessing configuration separately. Choose evaluation
//! mode for inference and use [`crate::no_grad`] to disable gradient recording.
//! The file does not store execution mode, optimizer moments, scheduler/scaler
//! state, random-number state, or input position. See the
//! [training guide](crate::tutorials::training) for a complete training checkpoint.
//!
//! These APIs accept SafeTensors only; they do not deserialize Python pickle,
//! whole model objects, or compiled model archives. The
//! [weight exchange guide](https://github.com/newpoluton-alt/RustTorch/blob/main/docs/model-interoperability.md)
//! also covers importing weights from another application.

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
