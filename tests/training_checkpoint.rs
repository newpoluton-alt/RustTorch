use rusttorch::{
    DeviceSpec, Result, Tensor,
    amp::{GradScaler, GradScalerState},
    checkpoint::{self, CheckpointLimits, TrainingCheckpoint},
    data::{DataLoader, Dataset, ReplaySafeDataset, ReplaySafeMap, VecCollate},
    nn::{Sequential, functional},
    optim::{AdamW, Optimizer, StepLr},
    reproducibility::{TensorRng, TensorRngState},
};
use serde::{Deserialize, Serialize};
use std::{
    convert::Infallible,
    fs,
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
};

static NEXT: AtomicU64 = AtomicU64::new(0);
struct Directory(PathBuf);
impl Directory {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "rusttorch-checkpoint-test-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
}
impl Drop for Directory {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

struct Rows;
impl Dataset for Rows {
    type Sample = i64;
    type Error = Infallible;
    fn len(&self) -> usize {
        6
    }
    fn get(&self, index: usize) -> std::result::Result<i64, Infallible> {
        Ok(index as i64)
    }
}
impl ReplaySafeDataset for Rows {}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Progress {
    steps: u64,
    schedule: StepLr,
    scaler: GradScalerState,
    rng: TensorRngState,
    loader: serde_json::Value,
}

fn model() -> Result<(Sequential, Optimizer)> {
    let model = Sequential::builder().linear(1, 1).build(DeviceSpec::Cpu)?;
    let optimizer = AdamW::builder()
        .learning_rate(0.01)
        .build(model.var_store())?;
    Ok((model, optimizer))
}
fn update(
    model: &Sequential,
    optimizer: &mut Optimizer,
    schedule: &mut StepLr,
    scaler: &mut GradScaler,
    rng: &mut TensorRng,
    batch: &[i64],
) -> Result<()> {
    let values = batch.iter().map(|&v| v as f32).collect::<Vec<_>>();
    let x = Tensor::from_slice(&values).reshape([-1, 1]);
    let target = x.f_mul_scalar(2.)?.f_add_scalar(1.)?;
    let noise = rng
        .normal(&x.size(), (x.kind(), x.device()))?
        .f_mul_scalar(0.01)?;
    let loss = functional::mse_loss(&model.forward(&x.f_add(&noise)?)?, &target)?;
    optimizer.try_zero_grad()?;
    scaler.scale(&loss)?.f_backward()?;
    if scaler.step(optimizer)? {
        schedule.step(optimizer)?;
    }
    scaler.update()?;
    Ok(())
}

#[test]
fn atomic_checkpoint_reproduces_next_loader_noise_optimizer_and_schedule_update()
-> std::result::Result<(), Box<dyn std::error::Error>> {
    let directory = Directory::new();
    let path = directory.0.join("step-1.rtckpt");
    let build = || {
        DataLoader::builder(ReplaySafeMap::new(Rows))
            .batch_size(2)
            .workers(2)
            .collate(VecCollate)
            .dataset_identity("rows-v1".to_owned())
    };
    let mut loader = build().build()?;
    let mut iter = loader.iter();
    let (original, mut optimizer) = model()?;
    let mut schedule = StepLr::new(&optimizer, 2, 0.5)?;
    let mut scaler = GradScaler::default();
    let mut rng = TensorRng::new(42);
    update(
        &original,
        &mut optimizer,
        &mut schedule,
        &mut scaler,
        &mut rng,
        &iter.next().unwrap()?,
    )?;
    let progress = Progress {
        steps: 1,
        schedule: schedule.clone(),
        scaler: scaler.state_dict()?,
        rng: rng.state_dict(),
        loader: serde_json::to_value(iter.checkpoint()?)?,
    };
    let limits = CheckpointLimits::default();
    checkpoint::save(
        &path,
        original.var_store(),
        &mut optimizer,
        &progress,
        limits,
    )?;
    let next = iter.next().unwrap()?;
    update(
        &original,
        &mut optimizer,
        &mut schedule,
        &mut scaler,
        &mut rng,
        &next,
    )?;
    let saved = TrainingCheckpoint::<Progress>::read(&path, limits)?;
    let mut resumed_loader = build()
        .resume_from(serde_json::from_value(saved.state().loader.clone())?)
        .build()?;
    let (restored, mut restored_optimizer) = model()?;
    let restored_progress = saved.restore(restored.var_store(), &mut restored_optimizer)?;
    assert_eq!(restored_progress.steps, 1);
    let mut restored_rng = TensorRng::from_state(restored_progress.rng)?;
    let mut restored_scaler = GradScaler::default();
    restored_scaler.load_state_dict(&restored_progress.scaler)?;
    let mut restored_schedule = restored_progress.schedule;
    let mut resumed_iter = resumed_loader.iter();
    let resumed_batch = resumed_iter.next().unwrap()?;
    assert_eq!(next, resumed_batch);
    update(
        &restored,
        &mut restored_optimizer,
        &mut restored_schedule,
        &mut restored_scaler,
        &mut restored_rng,
        &resumed_batch,
    )?;
    for (name, tensor) in original.var_store().variables() {
        assert!(
            tensor.f_equal(&restored.var_store().variables()[&name])?,
            "{name}"
        );
    }
    assert_eq!(optimizer.state_dict()?, restored_optimizer.state_dict()?);
    assert_eq!(schedule, restored_schedule);
    assert_eq!(rng.state_dict(), restored_rng.state_dict());
    assert_eq!(scaler.state_dict()?, restored_scaler.state_dict()?);
    assert_eq!(
        iter.collect::<std::result::Result<Vec<_>, _>>()?,
        resumed_iter.collect::<std::result::Result<Vec<_>, _>>()?
    );
    Ok(())
}

#[test]
fn corrupt_mismatched_or_oversized_checkpoints_preserve_existing_models_and_files()
-> std::result::Result<(), Box<dyn std::error::Error>> {
    use std::io::{Read, Write};
    let directory = Directory::new();
    let path = directory.0.join("saved.rtckpt");
    let (original, mut optimizer) = model()?;
    let limits = CheckpointLimits::default();
    checkpoint::save(&path, original.var_store(), &mut optimizer, &1_u64, limits)?;
    let bytes = fs::read(&path)?;
    assert!(checkpoint::save(&path, original.var_store(), &mut optimizer, &2_u64, limits).is_err());
    assert_eq!(fs::read(&path)?, bytes);
    let tiny = CheckpointLimits {
        weight_bytes: 1,
        ..limits
    };
    let denied = directory.0.join("denied.rtckpt");
    assert!(checkpoint::save(&denied, original.var_store(), &mut optimizer, &1_u64, tiny).is_err());
    assert!(!denied.exists());
    assert!(TrainingCheckpoint::<u64>::read(&path, tiny).is_err());
    let (different, mut different_optimizer) = model()?;
    assert!(
        checkpoint::save(
            &denied,
            different.var_store(),
            &mut optimizer,
            &1_u64,
            limits
        )
        .is_err()
    );
    let before = different
        .var_store()
        .variables()
        .into_iter()
        .map(|(k, v)| (k, v.copy()))
        .collect::<std::collections::BTreeMap<_, _>>();
    let optimizer_before = different_optimizer.state_dict()?;
    let saved = TrainingCheckpoint::<u64>::read(&path, limits)?;
    assert!(
        saved
            .restore(different.var_store(), &mut optimizer)
            .is_err()
    );
    for (key, tensor) in different.var_store().variables() {
        assert!(tensor.f_equal(&before[&key])?);
    }
    assert_eq!(optimizer_before, different_optimizer.state_dict()?);
    let malformed = directory.0.join("future-version.rtckpt");
    let mut archive = zip::ZipArchive::new(std::io::Cursor::new(&bytes))?;
    let mut output = zip::ZipWriter::new(fs::File::create(&malformed)?);
    for index in 0..archive.len() {
        let mut entry = archive.by_index(index)?;
        let name = entry.name().to_owned();
        let mut content = Vec::new();
        entry.read_to_end(&mut content)?;
        if name == "metadata.json" {
            content = br#"{"version":999,"state":1}"#.to_vec();
        }
        output.start_file(name, zip::write::FileOptions::default())?;
        output.write_all(&content)?;
    }
    output.finish()?;
    assert!(TrainingCheckpoint::<u64>::read(&malformed, limits).is_err());
    let truncated = directory.0.join("truncated.rtckpt");
    fs::write(&truncated, &bytes[..bytes.len() / 2])?;
    assert!(TrainingCheckpoint::<u64>::read(&truncated, limits).is_err());
    assert!(fs::read_dir(&directory.0)?.all(|entry| {
        !entry
            .unwrap()
            .file_name()
            .to_string_lossy()
            .ends_with(".tmp")
    }));
    Ok(())
}

#[test]
fn malformed_boolean_bytes_and_false_member_sizes_fail_before_native_restore()
-> std::result::Result<(), Box<dyn std::error::Error>> {
    use std::io::{Cursor, Read, Write};
    let directory = Directory::new();
    let path = directory.0.join("original.rtckpt");
    let (original, mut optimizer) = model()?;
    let limits = CheckpointLimits::default();
    checkpoint::save(&path, original.var_store(), &mut optimizer, &1_u64, limits)?;
    let bytes = fs::read(&path)?;
    let rewrite_weights =
        |weights: &[u8]| -> std::result::Result<Vec<u8>, Box<dyn std::error::Error>> {
            let mut archive = zip::ZipArchive::new(Cursor::new(&bytes))?;
            let mut output = zip::ZipWriter::new(Cursor::new(Vec::new()));
            for index in 0..archive.len() {
                let mut entry = archive.by_index(index)?;
                let name = entry.name().to_owned();
                let mut content = Vec::new();
                entry.read_to_end(&mut content)?;
                output.start_file(
                    &name,
                    zip::write::FileOptions::default()
                        .compression_method(zip::CompressionMethod::Stored),
                )?;
                output.write_all(if name == "model.safetensors" {
                    weights
                } else {
                    &content
                })?;
            }
            Ok(output.finish()?.into_inner())
        };

    for (payload, accepted) in [([0_u8, 1], true), ([0_u8, 2], false), ([255_u8, 1], false)] {
        let view =
            safetensors::tensor::TensorView::new(safetensors::Dtype::BOOL, vec![2], &payload)?;
        let weights = safetensors::tensor::serialize([("mask", view)], &None)?;
        // A valid SafeTensors header and CRC do not validate BOOL representations.
        assert!(safetensors::SafeTensors::deserialize(&weights).is_ok());
        let candidate = directory
            .0
            .join(format!("bool-{}-{}.rtckpt", payload[0], payload[1]));
        fs::write(&candidate, rewrite_weights(&weights)?)?;
        let result = TrainingCheckpoint::<u64>::read(&candidate, limits);
        if accepted {
            assert!(result.is_ok());
        } else {
            assert!(
                result
                    .unwrap_err()
                    .to_string()
                    .contains("boolean tensor payload")
            );
        }
    }

    // Keep each compressed length, payload and CRC correct while lying about
    // the decoded lengths in the central directory, which zip does not check.
    let mut malformed = bytes;
    let footer = malformed.len() - 22;
    assert_eq!(&malformed[footer..footer + 4], b"PK\x05\x06");
    let mut entry = u32::from_le_bytes(malformed[footer + 16..footer + 20].try_into()?) as usize;
    for _ in 0..3 {
        assert_eq!(&malformed[entry..entry + 4], b"PK\x01\x02");
        malformed[entry + 24..entry + 28].copy_from_slice(&0_u32.to_le_bytes());
        let variable_bytes = [28, 30, 32]
            .into_iter()
            .map(|offset| {
                u16::from_le_bytes(
                    malformed[entry + offset..entry + offset + 2]
                        .try_into()
                        .unwrap(),
                ) as usize
            })
            .sum::<usize>();
        entry += 46 + variable_bytes;
    }
    let candidate = directory.0.join("false-sizes.rtckpt");
    fs::write(&candidate, malformed)?;
    let error = TrainingCheckpoint::<u64>::read(&candidate, limits).unwrap_err();
    assert!(
        error.to_string().contains("decoded size differs"),
        "{error}"
    );
    Ok(())
}

#[test]
fn restore_validation_preserves_weights_and_requires_the_exact_empty_store()
-> std::result::Result<(), Box<dyn std::error::Error>> {
    let directory = Directory::new();
    let path = directory.0.join("adamw.rtckpt");
    let limits = CheckpointLimits::default();
    let (original, mut original_optimizer) = model()?;
    checkpoint::save(
        &path,
        original.var_store(),
        &mut original_optimizer,
        &1_u64,
        limits,
    )?;
    let (destination, _) = model()?;
    let mut different_algorithm =
        rusttorch::optim::Sgd::builder().build(destination.var_store())?;
    let weights_before = destination
        .var_store()
        .variables()
        .into_iter()
        .map(|(name, tensor)| (name, tensor.copy()))
        .collect::<std::collections::BTreeMap<_, _>>();
    let optimizer_before = different_algorithm.state_dict()?;
    let error = TrainingCheckpoint::<u64>::read(&path, limits)?
        .restore(destination.var_store(), &mut different_algorithm)
        .unwrap_err();
    assert!(error.to_string().contains("algorithm mismatch"), "{error}");
    for (name, tensor) in destination.var_store().variables() {
        assert!(tensor.f_equal(&weights_before[&name])?);
    }
    assert_eq!(different_algorithm.state_dict()?, optimizer_before);

    for zero_element_parameter in [false, true] {
        let source = rusttorch::nn::VarStore::new(rusttorch::Device::Cpu);
        let other = rusttorch::nn::VarStore::new(rusttorch::Device::Cpu);
        if zero_element_parameter {
            let _ = source
                .root()
                .f_var_copy("empty", &Tensor::from_slice::<f32>(&[]))?;
            let _ = other
                .root()
                .f_var_copy("empty", &Tensor::from_slice::<f32>(&[]))?;
        }
        let mut optimizer = AdamW::builder().build(&source)?;
        let path = directory
            .0
            .join(format!("empty-{zero_element_parameter}.rtckpt"));
        assert!(checkpoint::save(&path, &other, &mut optimizer, &1_u64, limits).is_err());
        assert!(!path.exists());
        checkpoint::save(&path, &source, &mut optimizer, &1_u64, limits)?;
        assert!(
            TrainingCheckpoint::<u64>::read(&path, limits)?
                .restore(&other, &mut optimizer)
                .is_err()
        );
        assert_eq!(
            TrainingCheckpoint::<u64>::read(&path, limits)?.restore(&source, &mut optimizer)?,
            1
        );
    }
    Ok(())
}

#[test]
fn restore_rejects_conflicting_buffer_aliases_and_accepts_exact_equal_aliases()
-> std::result::Result<(), Box<dyn std::error::Error>> {
    let directory = Directory::new();
    let limits = CheckpointLimits::default();
    for (partial_overlap, equal_values) in [(true, false), (false, false), (false, true)] {
        let source = rusttorch::nn::VarStore::new(rusttorch::Device::Cpu);
        let _ = source
            .root()
            .add("a", Tensor::from_slice(&[1_f32, 2.]), false);
        let _ = source.root().add(
            "b",
            Tensor::from_slice(if equal_values {
                &[1_f32, 2.]
            } else {
                &[3_f32, 4.]
            }),
            false,
        );
        let mut source_optimizer = AdamW::builder().build(&source)?;
        let path = directory
            .0
            .join(format!("aliases-{partial_overlap}-{equal_values}.rtckpt"));
        checkpoint::save(&path, &source, &mut source_optimizer, &1_u64, limits)?;
        let destination = rusttorch::nn::VarStore::new(rusttorch::Device::Cpu);
        let storage = Tensor::from_slice(&[0_f32, 0., 0.]);
        let _ = destination.root().add("a", storage.narrow(0, 0, 2), false);
        let _ =
            destination
                .root()
                .add("b", storage.narrow(0, i64::from(partial_overlap), 2), false);
        let mut optimizer = AdamW::builder().build(&destination)?;
        let before = optimizer.state_dict()?;
        let result =
            TrainingCheckpoint::<u64>::read(&path, limits)?.restore(&destination, &mut optimizer);
        if equal_values {
            assert_eq!(result?, 1);
            assert_eq!(Vec::<f32>::try_from(&storage)?, vec![1., 2., 0.]);
        } else {
            let error = result.unwrap_err();
            assert!(
                error.to_string().contains("overlapping model tensors"),
                "{error}"
            );
            assert_eq!(Vec::<f32>::try_from(&storage)?, vec![0., 0., 0.]);
        }
        assert_eq!(optimizer.state_dict()?, before);
    }
    Ok(())
}

#[test]
fn zip_and_zip64_directory_counts_are_bounded_before_archive_allocation()
-> std::result::Result<(), Box<dyn std::error::Error>> {
    let directory = Directory::new();
    let path = directory.0.join("original.rtckpt");
    let limits = CheckpointLimits::default();
    let (original, mut optimizer) = model()?;
    checkpoint::save(&path, original.var_store(), &mut optimizer, &1_u64, limits)?;
    let bytes = fs::read(&path)?;
    let footer = bytes.len() - 22;
    let size = u32::from_le_bytes(bytes[footer + 12..footer + 16].try_into()?) as u64;
    let offset = u32::from_le_bytes(bytes[footer + 16..footer + 20].try_into()?) as u64;
    let mut zip64 = bytes[..footer].to_vec();
    zip64.extend_from_slice(b"PK\x06\x06");
    zip64.extend_from_slice(&44_u64.to_le_bytes());
    for version in [45_u16, 45] {
        zip64.extend_from_slice(&version.to_le_bytes());
    }
    for disk in [0_u32, 0] {
        zip64.extend_from_slice(&disk.to_le_bytes());
    }
    for value in [3_u64, 3, size, offset] {
        zip64.extend_from_slice(&value.to_le_bytes());
    }
    zip64.extend_from_slice(b"PK\x06\x07");
    zip64.extend_from_slice(&0_u32.to_le_bytes());
    zip64.extend_from_slice(&(footer as u64).to_le_bytes());
    zip64.extend_from_slice(&1_u32.to_le_bytes());
    let zip64_footer = zip64.len();
    zip64.extend_from_slice(&bytes[footer..]);
    let candidate = directory.0.join("valid-zip64.rtckpt");
    fs::write(&candidate, &zip64)?;
    assert_eq!(
        *TrainingCheckpoint::<u64>::read(&candidate, limits)?.state(),
        1
    );

    let mut ordinary = bytes;
    ordinary[footer + 8..footer + 12].copy_from_slice(&[0xff; 4]);
    let candidate = directory.0.join("unbounded-zip.rtckpt");
    fs::write(&candidate, ordinary)?;
    let error = TrainingCheckpoint::<u64>::read(&candidate, limits).unwrap_err();
    assert!(
        error.to_string().contains("ZIP64 directory metadata"),
        "{error}"
    );

    zip64[zip64_footer + 8..zip64_footer + 12].copy_from_slice(&[0xff; 4]);
    zip64[footer + 24..footer + 40].copy_from_slice(&[0xff; 16]);
    let candidate = directory.0.join("unbounded-zip64.rtckpt");
    fs::write(&candidate, zip64)?;
    let error = TrainingCheckpoint::<u64>::read(&candidate, limits).unwrap_err();
    assert!(
        error.to_string().contains("ZIP directory count or bounds"),
        "{error}"
    );
    Ok(())
}
