//! Train a classifier, save a completed epoch, and verify its next update after restore.
//! Run with `cargo run --example training_checkpoint` in the configured checkout.
use std::{
    fs,
    time::{SystemTime, UNIX_EPOCH},
};

use rusttorch::{
    DeviceSpec, Result, Tensor,
    amp::{GradScaler, GradScalerState},
    nn::{Sequential, functional},
    optim::{AdamW, Optimizer, OptimizerState, StepLr},
};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Checkpoint {
    completed_epochs: u64,
    optimizer: OptimizerState,
    scheduler: StepLr,
    scaler: GradScalerState,
}

fn classifier() -> Result<(Sequential, Optimizer)> {
    let model = Sequential::builder()
        .linear(2, 8)
        .relu()
        .linear(8, 2)
        .build(DeviceSpec::Cpu)?;
    let optimizer = AdamW::builder()
        .learning_rate(0.01)
        .build(model.var_store())?;
    Ok((model, optimizer))
}

fn epoch(
    model: &Sequential,
    optimizer: &mut Optimizer,
    scaler: &mut GradScaler,
    scheduler: &mut StepLr,
) -> Result<()> {
    let features =
        Tensor::from_slice(&[-1_f32, -1., -1., 1., 1., -1., 1., 1.]).f_reshape([4, 2])?;
    let labels = Tensor::from_slice(&[0_i64, 0, 1, 1]);
    optimizer.try_zero_grad()?;
    let loss = functional::cross_entropy(&model.forward(&features)?, &labels)?;
    scaler.scale(&loss)?.f_backward()?;
    scaler.unscale(optimizer)?;
    optimizer.clip_grad_norm(1.0)?;
    let applied = scaler.step(optimizer)?;
    scaler.update()?;
    if applied {
        scheduler.step(optimizer)?;
    }
    Ok(())
}

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let (model, mut optimizer) = classifier()?;
    let mut scaler = GradScaler::default();
    let mut scheduler = StepLr::new(&optimizer, 2, 0.5)?;
    for _ in 0..2 {
        epoch(&model, &mut optimizer, &mut scaler, &mut scheduler)?;
    }

    // Use a fresh directory for each checkpoint. Write metadata after the weights.
    // Production jobs retain this directory and record their data/RNG position too.
    let directory = std::env::temp_dir().join(format!(
        "rusttorch-training-{}-{}",
        std::process::id(),
        SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos()
    ));
    fs::create_dir(&directory)?;
    model.save_weights(directory.join("model.safetensors"))?;
    let checkpoint = Checkpoint {
        completed_epochs: 2,
        optimizer: optimizer.state_dict()?,
        scheduler: scheduler.clone(),
        scaler: scaler.state_dict()?,
    };
    fs::write(
        directory.join("training.json"),
        serde_json::to_vec(&checkpoint)?,
    )?;

    // Advance the original run, then reproduce that update from the saved boundary.
    epoch(&model, &mut optimizer, &mut scaler, &mut scheduler)?;
    let (restored, mut restored_optimizer) = classifier()?;
    restored.load_weights(directory.join("model.safetensors"))?;
    let checkpoint: Checkpoint =
        serde_json::from_slice(&fs::read(directory.join("training.json"))?)?;
    assert_eq!(checkpoint.completed_epochs, 2);
    restored_optimizer.load_state_dict(&checkpoint.optimizer)?;
    let mut restored_scheduler = checkpoint.scheduler;
    let mut restored_scaler = GradScaler::default();
    restored_scaler.load_state_dict(&checkpoint.scaler)?;
    epoch(
        &restored,
        &mut restored_optimizer,
        &mut restored_scaler,
        &mut restored_scheduler,
    )?;
    for (name, value) in model.var_store().variables() {
        assert!(
            value.f_equal(&restored.var_store().variables()[&name])?,
            "{name}"
        );
    }
    assert_eq!(optimizer.state_dict()?, restored_optimizer.state_dict()?);
    assert_eq!(scheduler, restored_scheduler);
    assert_eq!(scaler.state_dict()?, restored_scaler.state_dict()?);
    fs::remove_dir_all(directory)?;
    println!("Restored weights, optimizer moments, schedule, and scale reproduce epoch 3 exactly.");
    Ok(())
}
