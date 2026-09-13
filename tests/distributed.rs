use rusttorch::{
    Device, Kind, Result, Tensor,
    distributed::{
        ConsolidatedCheckpoint, DistributedDataParallel, GroupOptions, ProcessGroup, ReduceOp,
        ShardedCheckpoint, ShardedTrainer,
    },
    nn::VarStore,
    no_grad,
    optim::{self, Optimizer},
};
use std::{
    collections::BTreeMap,
    fs,
    io::Write,
    net::{TcpListener, TcpStream},
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    sync::atomic::{AtomicU64, Ordering},
    thread,
    time::{Duration, Instant},
};

static NEXT: AtomicU64 = AtomicU64::new(0);
struct Children(Vec<Child>);
impl Drop for Children {
    fn drop(&mut self) {
        for child in &mut self.0 {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}
impl Children {
    fn finish(&mut self) {
        let deadline = Instant::now() + Duration::from_secs(45);
        for child in &mut self.0 {
            loop {
                if let Some(status) = child.try_wait().unwrap() {
                    assert!(status.success(), "distributed child failed: {status}");
                    break;
                }
                assert!(
                    Instant::now() < deadline,
                    "distributed child exceeded timeout"
                );
                thread::sleep(Duration::from_millis(10));
            }
        }
    }
}
struct Temp(PathBuf);
impl Temp {
    fn new() -> Self {
        Self(std::env::temp_dir().join(format!(
            "rusttorch-distributed-{}-{}.json",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        )))
    }
}
impl Drop for Temp {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}
fn options(session: &str, _scenario: &str) -> GroupOptions {
    let mut options = GroupOptions::new(session);
    options.timeout = Duration::from_secs(20);
    options
}

fn run(world: usize, scenario: &str, path: Option<&Path>) -> Result<()> {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let session = format!(
        "test-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    );
    let mut children = Children(Vec::new());
    for rank in 1..world {
        let mut command = Command::new(std::env::current_exe().unwrap());
        command
            .args(["--exact", "distributed_subprocess_worker", "--nocapture"])
            .env("RUSTTORCH_DIST_ADDRESS", address.to_string())
            .env("RUSTTORCH_DIST_RANK", rank.to_string())
            .env("RUSTTORCH_DIST_WORLD", world.to_string())
            .env("RUSTTORCH_DIST_SESSION", &session)
            .env("RUSTTORCH_DIST_SCENARIO", scenario)
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit());
        if let Some(path) = path {
            command.env("RUSTTORCH_DIST_STATE", path);
        }
        children.0.push(command.spawn().unwrap());
    }
    let mut group = ProcessGroup::accept(listener, world, options(&session, scenario))?;
    worker(&mut group, scenario, path)?;
    children.finish();
    Ok(())
}
#[test]
fn distributed_subprocess_worker() -> Result<()> {
    let Ok(address) = std::env::var("RUSTTORCH_DIST_ADDRESS") else {
        return Ok(());
    };
    let rank = std::env::var("RUSTTORCH_DIST_RANK")
        .unwrap()
        .parse()
        .unwrap();
    let world = std::env::var("RUSTTORCH_DIST_WORLD")
        .unwrap()
        .parse()
        .unwrap();
    let session = std::env::var("RUSTTORCH_DIST_SESSION").unwrap();
    let scenario = std::env::var("RUSTTORCH_DIST_SCENARIO").unwrap();
    let path = std::env::var_os("RUSTTORCH_DIST_STATE").map(PathBuf::from);
    let mut group = ProcessGroup::connect(
        address.parse().unwrap(),
        rank,
        world,
        options(&session, &scenario),
    )?;
    worker(&mut group, &scenario, path.as_deref())
}
fn close(tensor: &Tensor, expected: &[f64]) {
    let actual = Vec::<f64>::try_from(tensor.to_kind(Kind::Double).reshape([-1])).unwrap();
    assert_eq!(actual.len(), expected.len());
    for (a, e) in actual.iter().zip(expected) {
        assert!((a - e).abs() < 1e-10, "{a} != {e}");
    }
}
fn close_tensor(a: &Tensor, b: &Tensor) {
    close(
        a,
        &Vec::<f64>::try_from(b.to_kind(Kind::Double).reshape([-1])).unwrap(),
    );
}
fn values(map: &BTreeMap<String, Tensor>) -> BTreeMap<String, Vec<f64>> {
    map.iter()
        .map(|(name, t)| {
            (
                name.clone(),
                Vec::<f64>::try_from(t.to_kind(Kind::Double).reshape([-1])).unwrap(),
            )
        })
        .collect()
}
fn worker(group: &mut ProcessGroup, scenario: &str, path: Option<&Path>) -> Result<()> {
    match scenario {
        "collectives" => collectives(group),
        "parity_collectives" => parity_collectives(group, path.expect("fixture path")),
        "parity_training" => parity_training(group, path.expect("fixture path")),
        "failure_layout" => {
            let (store, _, _) = local_model("adam")?;
            let replica = DistributedDataParallel::new(group, &store)?;
            if group.rank() == 1 {
                let _ = store.root().f_zeros_no_train("late", &[1])?;
            }
            assert!(replica.sync_buffers(group).is_err());
            assert!(group.is_failed());
            Ok(())
        }
        "failure_shard_layout" => {
            let mut trainer = trainer(group, "adam")?;
            trainer.optimizer_mut().trainable_variables()[0]
                .shallow_clone()
                .f_set_data(&Tensor::zeros([4], (Kind::Double, Device::Cpu)))?;
            let rank = group.rank();
            assert!(trainer.train_step(group, |p| loss(p, rank, 2, 0)).is_err());
            assert!(group.is_failed());
            Ok(())
        }
        "failure_config" => {
            let mut trainer = trainer(group, "adam")?;
            if group.rank() == 1 {
                trainer.optimizer_mut().set_learning_rate(0.2)?;
            }
            let rank = group.rank();
            assert!(trainer.train_step(group, |p| loss(p, rank, 2, 0)).is_err());
            assert!(group.is_failed());
            Ok(())
        }
        "ddp" => ddp(group),
        "failure_order" => {
            let result = if group.rank() == 0 {
                group
                    .all_reduce(&Tensor::from(1_f64), ReduceOp::Sum)
                    .map(|_| ())
            } else {
                group.barrier()
            };
            assert!(result.is_err());
            assert!(group.is_failed());
            assert!(group.barrier().is_err());
            Ok(())
        }
        "failure_shape" => {
            let tensor = Tensor::zeros([group.rank() as i64 + 1], (Kind::Float, Device::Cpu));
            assert!(group.all_gather(&tensor).is_err());
            assert!(group.is_failed());
            Ok(())
        }
        "failure_root" => {
            assert!(group.broadcast(&Tensor::from(1_f64), group.rank()).is_err());
            assert!(group.is_failed());
            Ok(())
        }
        "failure_local" => {
            let tensor = if group.rank() == 0 {
                Tensor::from(1_f64)
            } else {
                Tensor::from(1_f32).to_kind(Kind::Half)
            };
            assert!(group.all_gather(&tensor).is_err());
            assert!(group.is_failed());
            Ok(())
        }
        "failure_disconnect" => {
            if group.rank() == 1 {
                group.abort();
            } else {
                assert!(group.barrier().is_err());
                assert!(group.is_failed());
            }
            Ok(())
        }
        "failure_timeout" => {
            group.set_timeout(Duration::from_millis(300))?;
            if group.rank() == 1 {
                thread::sleep(Duration::from_millis(600));
            } else {
                let start = Instant::now();
                assert!(group.barrier().is_err());
                assert!(start.elapsed() < Duration::from_secs(10));
            }
            Ok(())
        }
        "failure_tag" => {
            if group.rank() == 0 {
                assert!(group.send(&Tensor::from(1_f64), 1, 7).is_err());
            } else {
                assert!(group.recv(0, 8).is_err());
            }
            assert!(group.is_failed());
            Ok(())
        }
        "failure_grad" => gradient_mismatch(group),
        "failure_closure" => {
            let mut trainer = trainer(group, "adam")?;
            let rank = group.rank();
            assert!(
                trainer
                    .train_step(group, |p| if rank == 0 {
                        loss(p, rank, 2, 0)
                    } else {
                        Err(rusttorch::RustTorchError::GraphValidation(
                            "bad batch".into(),
                        ))
                    })
                    .is_err()
            );
            assert!(group.is_failed());
            Ok(())
        }
        "reshard" => reshard(group, path.expect("state path")),
        name if name.starts_with("sharded_") => {
            sharded(group, name.trim_start_matches("sharded_"), path)
        }
        _ => panic!("unknown scenario"),
    }
}
fn collectives(group: &mut ProcessGroup) -> Result<()> {
    let rank = group.rank();
    let value = Tensor::from_slice(&[rank as f64 + 1., rank as f64 + 2.]);
    for (op, expected) in [
        (ReduceOp::Sum, [6., 9.]),
        (ReduceOp::Mean, [2., 3.]),
        (ReduceOp::Min, [1., 2.]),
        (ReduceOp::Max, [3., 4.]),
        (ReduceOp::Product, [6., 24.]),
    ] {
        let result = group.all_reduce(&value, op)?;
        close(&result, &expected);
        assert!(!result.requires_grad());
    }
    close(
        &group.all_reduce(&Tensor::from(1_i64 + rank as i64), ReduceOp::Sum)?,
        &[6.],
    );
    close(&group.broadcast(&value, 2)?, &[3., 4.]);
    let reduced = group.reduce(&value, ReduceOp::Sum, 1)?;
    if rank == 1 {
        close(&reduced.unwrap(), &[6., 9.]);
    } else {
        assert!(reduced.is_none());
    }
    let gathered = group.all_gather(&value)?;
    for (r, t) in gathered.iter().enumerate() {
        close(t, &[r as f64 + 1., r as f64 + 2.]);
    }
    let gathered = group.gather(&value, 2)?;
    assert_eq!(gathered.len(), if rank == 2 { 3 } else { 0 });
    let scattered = if rank == 1 {
        (0..3)
            .map(|r| Tensor::from_slice(&[10_f64 + r as f64]))
            .collect()
    } else {
        vec![]
    };
    close(&group.scatter(&scattered, 1)?, &[10. + rank as f64]);
    let inputs = (0..3)
        .map(|r| Tensor::from_slice(&[rank as f64 + 1. + r as f64]))
        .collect::<Vec<_>>();
    close(
        &group.reduce_scatter(&inputs, ReduceOp::Sum)?,
        &[6. + 3. * rank as f64],
    );
    let inputs = (0..3)
        .map(|r| Tensor::from_slice(&[10. * rank as f64 + r as f64]))
        .collect::<Vec<_>>();
    for (source, t) in group.all_to_all(&inputs)?.iter().enumerate() {
        close(t, &[10. * source as f64 + rank as f64]);
    }
    let bytes = group.all_gather_bytes(format!("rank-{rank}").as_bytes())?;
    assert_eq!(
        bytes,
        [b"rank-0".to_vec(), b"rank-1".to_vec(), b"rank-2".to_vec()]
    );
    if rank == 1 {
        group.send(&Tensor::from(77_f64), 2, 7)?;
    } else if rank == 2 {
        close(&group.recv(1, 7)?, &[77.]);
        group.send(&Tensor::from(88_f64), 0, 8)?;
    } else {
        close(&group.recv(2, 8)?, &[88.]);
    }
    group.barrier()?;
    let empty = Tensor::zeros([0, 3], (Kind::Double, Device::Cpu));
    assert_eq!(group.all_reduce(&empty, ReduceOp::Sum)?.size(), [0, 3]);
    Ok(())
}
fn optimizer(store: &VarStore, name: &str) -> Result<Optimizer> {
    match name {
        "adam" => optim::Adam::builder()
            .learning_rate(0.03)
            .amsgrad(true)
            .build(store),
        "adamw" => optim::AdamW::builder()
            .learning_rate(0.03)
            .weight_decay(0.01)
            .build(store),
        "sgd" => optim::Sgd::builder()
            .learning_rate(0.03)
            .momentum(0.7)
            .nesterov(true)
            .build(store),
        "rmsprop" => optim::RmsProp::builder()
            .learning_rate(0.03)
            .momentum(0.7)
            .centered(true)
            .build(store),
        "adagrad" => optim::Adagrad::builder()
            .learning_rate(0.03)
            .initial_accumulator_value(0.2)
            .build(store),
        "adadelta" => optim::Adadelta::builder().learning_rate(0.03).build(store),
        "adamax" => optim::Adamax::builder().learning_rate(0.03).build(store),
        _ => panic!("bad optimizer"),
    }
}
fn initial() -> Vec<(String, Tensor)> {
    vec![
        ("bias".into(), Tensor::from(0.2_f64)),
        (
            "layer.weight".into(),
            Tensor::from_slice(&[0.4_f64, -0.7, 0.1]),
        ),
    ]
}
fn trainer(group: &mut ProcessGroup, name: &str) -> Result<ShardedTrainer> {
    ShardedTrainer::new(group, initial(), |s| optimizer(s, name))
}
fn local_model(name: &str) -> Result<(VarStore, BTreeMap<String, Tensor>, Optimizer)> {
    let mut store = VarStore::new(Device::Cpu);
    store.set_kind(Kind::Double);
    let mut p = BTreeMap::new();
    for (name, tensor) in initial() {
        let t = if name == "bias" {
            store.root().f_var_copy("bias", &tensor)?
        } else {
            store.root().sub("layer").f_var_copy("weight", &tensor)?
        };
        p.insert(name, t);
    }
    let opt = optimizer(&store, name)?;
    Ok((store, p, opt))
}
fn loss(
    parameters: &BTreeMap<String, Tensor>,
    rank: usize,
    world: usize,
    step: usize,
) -> Result<Tensor> {
    let batch = 6 / world;
    let first = rank * batch;
    let mut data = Vec::new();
    let mut targets = Vec::new();
    for i in first..first + batch {
        let x = i as f64 * 0.2 + step as f64 * 0.01;
        data.extend([x, 1. - x, x * x]);
        targets.push(0.6 * x - 0.4);
    }
    let x = Tensor::from_slice(&data).reshape([batch as i64, 3]);
    let y = Tensor::from_slice(&targets);
    Ok(x.f_matmul(&parameters["layer.weight"])?
        .f_add(&parameters["bias"])?
        .f_sub(&y)?
        .f_square()?
        .f_mean(Kind::Double)?)
}
fn ddp(group: &mut ProcessGroup) -> Result<()> {
    let (store, parameters, mut opt) = local_model("adam")?;
    let (_baseline_store, baseline, mut baseline_opt) = local_model("adam")?;
    let mut buffer = store.root().f_zeros_no_train("running", &[1])?;
    no_grad(|| -> Result<()> {
        let _ = buffer.f_fill_(group.rank() as f64 + 5.)?;
        for p in parameters.values() {
            let _ = p.shallow_clone().f_add_scalar_(group.rank() as f64)?;
        }
        Ok(())
    })?;
    let ddp = DistributedDataParallel::new(group, &store)?;
    close(&buffer, &[5.]);
    for step in 0..4 {
        no_grad(|| -> Result<()> {
            let _ = buffer.f_fill_(group.rank() as f64 + step as f64)?;
            Ok(())
        })?;
        ddp.sync_buffers(group)?;
        close(&buffer, &[step as f64]);
        let objective = loss(&parameters, group.rank(), group.world_size(), step)?;
        ddp.backward_step(group, &mut opt, &objective)?;
        baseline_opt.backward_step(&loss(&baseline, 0, 1, step)?)?;
        for (name, t) in &parameters {
            close_tensor(t, &baseline[name]);
        }
    }
    // Accumulate two local losses, then synchronize once.
    opt.try_zero_grad()?;
    baseline_opt.try_zero_grad()?;
    for step in 4..6 {
        loss(&parameters, group.rank(), group.world_size(), step)?
            .f_div_scalar(2.)?
            .f_backward()?;
        loss(&baseline, 0, 1, step)?
            .f_div_scalar(2.)?
            .f_backward()?;
    }
    ddp.sync_gradients(group)?;
    opt.try_step()?;
    baseline_opt.try_step()?;
    for (name, t) in &parameters {
        close_tensor(t, &baseline[name]);
    }
    group.barrier()
}
fn gradient_mismatch(group: &mut ProcessGroup) -> Result<()> {
    let (store, p, mut opt) = local_model("sgd")?;
    let ddp = DistributedDataParallel::new(group, &store)?;
    opt.try_zero_grad()?;
    if group.rank() == 0 {
        p["bias"].f_square()?.f_backward()?;
    } else {
        p["layer.weight"]
            .f_square()?
            .f_sum(Kind::Double)?
            .f_backward()?;
    }
    assert!(ddp.sync_gradients(group).is_err());
    assert!(group.is_failed());
    Ok(())
}
fn sharded(group: &mut ProcessGroup, name: &str, path: Option<&Path>) -> Result<()> {
    let mut trainer = trainer(group, name)?;
    assert_eq!(trainer.local_parameter_elements(), 3);
    let (_store, reference, mut optimizer) = local_model(name)?;
    for step in 0..2 {
        let rank = group.rank();
        trainer.train_step(group, |p| loss(p, rank, 2, step))?;
        optimizer.backward_step(&loss(&reference, 0, 1, step)?)?;
    }
    let checkpoint = trainer.checkpoint(group)?;
    let consolidated = trainer.consolidate_checkpoint(group)?;
    let mut records = group
        .all_gather_bytes(&serde_json::to_vec(&checkpoint).unwrap())?
        .iter()
        .map(|bytes| serde_json::from_slice::<serde_json::Value>(bytes).unwrap())
        .collect::<Vec<_>>();
    records[1]["checkpoint_id"] = "another-job:42".into();
    let records = records
        .into_iter()
        .map(|record| serde_json::from_value::<ShardedCheckpoint>(record).unwrap())
        .collect::<Vec<_>>();
    assert!(ShardedCheckpoint::consolidate(&records).is_err());
    assert_eq!(checkpoint.rank(), group.rank());
    let encoded = serde_json::to_vec(&checkpoint).unwrap();
    let checkpoint: ShardedCheckpoint = serde_json::from_slice(&encoded).unwrap();
    for step in 2..4 {
        let rank = group.rank();
        trainer.train_step(group, |p| loss(p, rank, 2, step))?;
        optimizer.backward_step(&loss(&reference, 0, 1, step)?)?;
    }
    let expected = trainer.full_parameters(group)?;
    for (key, tensor) in &expected {
        close_tensor(tensor, &reference[key]);
    }
    let expected_state = trainer.consolidate_checkpoint(group)?;
    let mut restored =
        ShardedTrainer::restore(group, &checkpoint, |s| optimizer_for_restore(s, name))?;
    for step in 2..4 {
        let rank = group.rank();
        restored.train_step(group, |p| loss(p, rank, 2, step))?;
    }
    assert_eq!(restored.consolidate_checkpoint(group)?, expected_state);
    if let Some(path) = path
        && group.rank() == 0
    {
        fs::write(
            path,
            serde_json::to_vec(
                &serde_json::json!({"checkpoint":consolidated,"expected":values(&expected)}),
            )
            .unwrap(),
        )
        .unwrap();
    }
    // Check duplicate/missing shards and malformed tensor lengths without a live group.
    assert!(ShardedCheckpoint::consolidate(&[checkpoint.clone(), checkpoint.clone()]).is_err());
    assert!(ShardedCheckpoint::consolidate(&[checkpoint]).is_err());
    group.barrier()
}
fn optimizer_for_restore(store: &VarStore, name: &str) -> Result<Optimizer> {
    let mut opt = optimizer(store, name)?;
    opt.set_learning_rate(9.)?;
    Ok(opt)
}
fn reshard(group: &mut ProcessGroup, path: &Path) -> Result<()> {
    let data: serde_json::Value = serde_json::from_slice(&fs::read(path).unwrap()).unwrap();
    let checkpoint: ConsolidatedCheckpoint =
        serde_json::from_value(data["checkpoint"].clone()).unwrap();
    let expected: BTreeMap<String, Vec<f64>> =
        serde_json::from_value(data["expected"].clone()).unwrap();
    let mut trainer = ShardedTrainer::restore_full(group, &checkpoint, |s| optimizer(s, "adam"))?;
    assert_eq!(trainer.local_parameter_elements(), 2);
    for step in 2..4 {
        let rank = group.rank();
        let world = group.world_size();
        trainer.train_step(group, |p| loss(p, rank, world, step))?;
    }
    for (name, tensor) in trainer.full_parameters(group)? {
        close(&tensor, &expected[&name]);
    }
    group.barrier()
}
#[test]
fn multiprocess_collectives_and_point_to_point_preserve_rank_order() -> Result<()> {
    run(3, "collectives", None)
}
#[test]
fn replicated_training_and_accumulation_match_global_batch() -> Result<()> {
    run(2, "ddp", None)
}
#[test]
fn sharded_training_and_exact_resume_cover_all_optimizer_families() -> Result<()> {
    for name in [
        "adam", "adamw", "sgd", "rmsprop", "adagrad", "adadelta", "adamax",
    ] {
        run(2, &format!("sharded_{name}"), None)?;
    }
    Ok(())
}
#[test]
fn consolidated_checkpoint_reshards_parameters_and_moments_to_new_world_size() -> Result<()> {
    let path = Temp::new();
    run(2, "sharded_adam", Some(&path.0))?;
    run(3, "reshard", Some(&path.0))
}
#[test]
fn mismatched_collectives_and_local_errors_poison_every_rank() -> Result<()> {
    for scenario in [
        "failure_order",
        "failure_shape",
        "failure_root",
        "failure_local",
        "failure_grad",
        "failure_closure",
        "failure_tag",
        "failure_config",
        "failure_layout",
        "failure_shard_layout",
    ] {
        run(2, scenario, None)?;
    }
    Ok(())
}
#[test]
fn disconnected_and_stalled_workers_fail_with_bounded_cleanup() -> Result<()> {
    run(2, "failure_disconnect", None)?;
    run(2, "failure_timeout", None)
}
#[test]
fn configuration_and_untrusted_frames_fail_before_large_allocation() -> Result<()> {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    assert!(ProcessGroup::accept(listener, 0, GroupOptions::new("bad")).is_err());
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let mut opt = GroupOptions::new("bad");
    opt.timeout = Duration::ZERO;
    assert!(ProcessGroup::accept(listener, 1, opt).is_err());
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let peer = thread::spawn(move || {
        let mut stream = TcpStream::connect(address).unwrap();
        stream.write_all(&u64::MAX.to_be_bytes()).unwrap();
    });
    assert!(ProcessGroup::accept(listener, 2, GroupOptions::new("oversize")).is_err());
    peer.join().unwrap();
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let peer = thread::spawn(move || {
        let mut stream = TcpStream::connect(address).unwrap();
        let data = b"{invalid}";
        stream
            .write_all(&(data.len() as u64).to_be_bytes())
            .unwrap();
        stream.write_all(data).unwrap();
    });
    assert!(ProcessGroup::accept(listener, 2, GroupOptions::new("malformed")).is_err());
    peer.join().unwrap();
    Ok(())
}

fn parity_collectives(group: &mut ProcessGroup, path: &Path) -> Result<()> {
    let fixture: serde_json::Value = serde_json::from_slice(&fs::read(path).unwrap()).unwrap();
    let reference = &fixture["collectives"][group.rank()];
    let vector = |key: &str| serde_json::from_value::<Vec<f64>>(reference[key].clone()).unwrap();
    let value = Tensor::from_slice(&[group.rank() as f64 + 1., group.rank() as f64 + 2.]);
    for (key, op) in [
        ("sum", ReduceOp::Sum),
        ("mean", ReduceOp::Mean),
        ("min", ReduceOp::Min),
        ("max", ReduceOp::Max),
        ("product", ReduceOp::Product),
    ] {
        close(&group.all_reduce(&value, op)?, &vector(key));
    }
    close(&group.broadcast(&value, 2)?, &vector("broadcast"));
    let expected: Vec<Vec<f64>> = serde_json::from_value(reference["all_gather"].clone()).unwrap();
    for (tensor, expected) in group.all_gather(&value)?.iter().zip(expected) {
        close(tensor, &expected);
    }
    let chunks = (0..group.world_size())
        .map(|r| Tensor::from_slice(&[group.rank() as f64 + 1. + r as f64]))
        .collect::<Vec<_>>();
    close(
        &group.reduce_scatter(&chunks, ReduceOp::Sum)?,
        &vector("reduce_scatter"),
    );
    let inputs = (0..group.world_size())
        .map(|r| Tensor::from_slice(&[10. * group.rank() as f64 + r as f64]))
        .collect::<Vec<_>>();
    let tensors = group.all_to_all(&inputs)?;
    close(&Tensor::cat(&tensors, 0), &vector("all_to_all"));
    group.barrier()
}
fn parity_training(group: &mut ProcessGroup, path: &Path) -> Result<()> {
    let fixture: serde_json::Value = serde_json::from_slice(&fs::read(path).unwrap()).unwrap();
    let reference = &fixture["training"][group.rank()];
    let (store, parameters, mut optimizer) = local_model("adam")?;
    let ddp = DistributedDataParallel::new(group, &store)?;
    for step in 0..4 {
        let objective = loss(&parameters, group.rank(), group.world_size(), step)?;
        ddp.backward_step(group, &mut optimizer, &objective)?;
    }
    let expected: BTreeMap<String, Vec<f64>> =
        serde_json::from_value(reference["ddp"].clone()).unwrap();
    for (name, tensor) in &parameters {
        close(tensor, &expected[name]);
    }
    let mut sharded = trainer(group, "adam")?;
    for step in 0..4 {
        let rank = group.rank();
        let world = group.world_size();
        sharded.train_step(group, |p| loss(p, rank, world, step))?;
    }
    let expected: BTreeMap<String, Vec<f64>> =
        serde_json::from_value(reference["fsdp"].clone()).unwrap();
    for (name, tensor) in sharded.full_parameters(group)? {
        close(&tensor, &expected[&name]);
    }
    group.barrier()
}

#[test]
#[ignore = "requires generated pinned PyTorch/Gloo fixtures"]
fn cpu_collectives_ddp_and_shards_match_pinned_pytorch_gloo() -> Result<()> {
    let directory = std::env::var_os("RUSTTORCH_PYTHON_REFERENCE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("target/python-reference"));
    let path = directory.join("distributed.json");
    assert!(
        path.is_file(),
        "generate tests/python_reference/distributed.py first"
    );
    run(3, "parity_collectives", Some(&path))?;
    run(2, "parity_training", Some(&path))
}

#[test]
fn corrupted_sharded_state_rejects_weights_optimizer_and_rank_metadata() -> Result<()> {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let mut group = ProcessGroup::accept(listener, 1, GroupOptions::new("corrupt-checkpoint"))?;
    let mut trainer = trainer(&mut group, "adam")?;
    trainer.train_step(&mut group, |p| loss(p, 0, 1, 0))?;
    let saved = trainer.checkpoint(&mut group)?;
    let original = serde_json::to_value(&saved).unwrap();
    for field in [
        "schema",
        "rank",
        "world",
        "weights",
        "shape",
        "dtype",
        "optimizer_shape",
        "optimizer_dtype",
        "optimizer_step",
        "optimizer_bytes",
        "optimizer_slot",
        "duplicate_names",
    ] {
        let mut bad = original.clone();
        match field {
            "schema" => bad["schema_version"] = 99.into(),
            "rank" => bad["rank"] = 2.into(),
            "world" => bad["world_size"] = 0.into(),
            "weights" => bad["weights"]["layer.weight"]["bytes"] = serde_json::json!([0]),
            "shape" => bad["parameters"][1]["shape"] = serde_json::json!([-1]),
            "dtype" => bad["weights"]["layer.weight"]["dtype"] = "Int64".into(),
            "optimizer_shape" => {
                bad["optimizer"]["parameters"]["layer.weight"]["shape"] = serde_json::json!([4])
            }
            "optimizer_dtype" => bad["optimizer"]["parameters"]["bias"]["dtype"] = "Float32".into(),
            "optimizer_step" => bad["optimizer"]["parameters"]["bias"]["step"] = 999.into(),
            "optimizer_bytes" => {
                bad["optimizer"]["parameters"]["bias"]["slots"]["exp_avg"]["bytes"] =
                    serde_json::json!([])
            }
            "optimizer_slot" => {
                bad["optimizer"]["parameters"]["bias"]["slots"]
                    .as_object_mut()
                    .unwrap()
                    .remove("exp_avg");
            }
            "duplicate_names" => bad["parameters"][1]["name"] = "bias".into(),
            _ => unreachable!(),
        }
        let invalid: ShardedCheckpoint = serde_json::from_value(bad).unwrap();
        assert!(
            ShardedCheckpoint::consolidate(&[invalid]).is_err(),
            "accepted corruption: {field}"
        );
    }
    assert_eq!(
        trainer.consolidate_checkpoint(&mut group)?,
        ShardedCheckpoint::consolidate(&[saved])?
    );
    Ok(())
}
