use std::{
    fs,
    path::{Path, PathBuf},
    process,
    sync::atomic::{AtomicU64, Ordering},
};

use rusttorch::{
    Device, RustTorchError, Tensor,
    interop::{
        LoadOptions, StateDictMapping, load_state_dict, load_state_dict_with_mapping,
        save_state_dict,
    },
    no_grad,
};
use tch::nn::{Init, VarStore};

struct TempSafetensors(PathBuf);

impl TempSafetensors {
    fn new(label: &str) -> Self {
        static NEXT_ID: AtomicU64 = AtomicU64::new(0);
        let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
        Self(std::env::temp_dir().join(format!(
            "rusttorch-{label}-{}-{id}.safetensors",
            process::id(),
        )))
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempSafetensors {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}

fn add_var(var_store: &VarStore, name: &str, shape: &[i64], values: &[f32]) -> Tensor {
    let mut variable = var_store.root().var(name, shape, Init::Const(0.0));
    let value = Tensor::from_slice(values).reshape(shape);
    no_grad(|| variable.f_copy_(&value)).expect("deterministic test values must copy");
    variable
}

fn values(tensor: &Tensor) -> Vec<f32> {
    Vec::<f32>::try_from(&tensor.reshape([-1])).expect("test tensor must convert to f32 values")
}

#[test]
fn strict_safetensors_round_trip_restores_all_variables() {
    let file = TempSafetensors::new("strict-round-trip");
    let source = VarStore::new(Device::Cpu);
    let _source_weight = add_var(&source, "weight", &[2, 2], &[1.0, -2.0, 3.5, 4.0]);
    let _source_bias = add_var(&source, "bias", &[2], &[0.25, -0.5]);
    save_state_dict(file.path(), &source).expect("state dictionary must save");

    let target = VarStore::new(Device::Cpu);
    let target_weight = add_var(&target, "weight", &[2, 2], &[0.0; 4]);
    let target_bias = add_var(&target, "bias", &[2], &[0.0; 2]);

    let report = load_state_dict(file.path(), &target).expect("strict load must succeed");

    assert_eq!(report.loaded, ["bias", "weight"]);
    assert!(report.missing.is_empty());
    assert!(report.unexpected.is_empty());
    assert!(report.remapped.is_empty());
    assert_eq!(values(&target_weight), [1.0, -2.0, 3.5, 4.0]);
    assert_eq!(values(&target_bias), [0.25, -0.5]);
}

#[test]
fn non_strict_load_returns_a_sorted_complete_report() {
    let file = TempSafetensors::new("non-strict-report");
    let source = VarStore::new(Device::Cpu);
    let _z_loaded = add_var(&source, "z_loaded", &[1], &[9.0]);
    let _a_loaded = add_var(&source, "a_loaded", &[1], &[1.0]);
    let _x_unexpected = add_var(&source, "x_unexpected", &[1], &[8.0]);
    let _m_unexpected = add_var(&source, "m_unexpected", &[1], &[7.0]);
    save_state_dict(file.path(), &source).expect("state dictionary must save");

    let target = VarStore::new(Device::Cpu);
    let target_z = add_var(&target, "z_loaded", &[1], &[0.0]);
    let target_a = add_var(&target, "a_loaded", &[1], &[0.0]);
    let _y_missing = add_var(&target, "y_missing", &[1], &[0.0]);
    let _b_missing = add_var(&target, "b_missing", &[1], &[0.0]);

    let report = load_state_dict_with_mapping(
        file.path(),
        &target,
        &StateDictMapping::new(),
        LoadOptions::non_strict(),
    )
    .expect("non-strict load must return a report");

    assert_eq!(report.loaded, ["a_loaded", "z_loaded"]);
    assert_eq!(report.missing, ["b_missing", "y_missing"]);
    assert_eq!(report.unexpected, ["m_unexpected", "x_unexpected"]);
    assert!(report.remapped.is_empty());
    assert_eq!(values(&target_a), [1.0]);
    assert_eq!(values(&target_z), [9.0]);
}

#[test]
fn exact_mapping_loads_the_named_destination() {
    let file = TempSafetensors::new("exact-mapping");
    let source = VarStore::new(Device::Cpu);
    let _source_weight = add_var(&source, "python_weight", &[2], &[3.0, 4.0]);
    save_state_dict(file.path(), &source).expect("state dictionary must save");

    let target = VarStore::new(Device::Cpu);
    let target_weight = add_var(&target, "rust_weight", &[2], &[0.0, 0.0]);
    let mapping = StateDictMapping::new().map("python_weight", "rust_weight");

    let report =
        load_state_dict_with_mapping(file.path(), &target, &mapping, LoadOptions::strict())
            .expect("exact mapping must load strictly");

    assert_eq!(report.loaded, ["rust_weight"]);
    assert_eq!(
        report.remapped,
        [("python_weight".to_owned(), "rust_weight".to_owned())],
    );
    assert!(report.missing.is_empty());
    assert!(report.unexpected.is_empty());
    assert_eq!(values(&target_weight), [3.0, 4.0]);
}

#[test]
fn dry_run_reports_a_load_without_mutating_values() {
    let file = TempSafetensors::new("dry-run");
    let source = VarStore::new(Device::Cpu);
    let _source_weight = add_var(&source, "weight", &[2], &[8.0, -3.0]);
    save_state_dict(file.path(), &source).expect("state dictionary must save");

    let target = VarStore::new(Device::Cpu);
    let target_weight = add_var(&target, "weight", &[2], &[1.0, 2.0]);

    let report = load_state_dict_with_mapping(
        file.path(),
        &target,
        &StateDictMapping::new(),
        LoadOptions::strict().dry_run(true),
    )
    .expect("dry-run validation must succeed");

    assert_eq!(report.loaded, ["weight"]);
    assert_eq!(values(&target_weight), [1.0, 2.0]);
}

#[test]
fn shape_validation_happens_before_any_variable_is_mutated() {
    let file = TempSafetensors::new("shape-atomicity");
    let source = VarStore::new(Device::Cpu);
    let _source_good = add_var(&source, "a_good", &[2], &[9.0, 8.0]);
    let _source_bad = add_var(&source, "z_bad", &[3], &[7.0, 6.0, 5.0]);
    save_state_dict(file.path(), &source).expect("state dictionary must save");

    let target = VarStore::new(Device::Cpu);
    let target_good = add_var(&target, "a_good", &[2], &[1.0, 2.0]);
    let target_bad = add_var(&target, "z_bad", &[2], &[3.0, 4.0]);

    match load_state_dict(file.path(), &target) {
        Err(RustTorchError::ShapeMismatch {
            name,
            expected,
            actual,
        }) => {
            assert_eq!(name, "z_bad");
            assert_eq!(expected, [2]);
            assert_eq!(actual, [3]);
        }
        other => panic!("expected a shape mismatch, got {other:?}"),
    }

    assert_eq!(values(&target_good), [1.0, 2.0]);
    assert_eq!(values(&target_bad), [3.0, 4.0]);
}
