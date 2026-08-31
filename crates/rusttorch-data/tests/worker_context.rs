use std::{
    error::Error,
    fmt,
    panic::{AssertUnwindSafe, catch_unwind},
    sync::{
        Arc, Barrier, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
};

use rusttorch_core::RustTorchError;
use rusttorch_data::{
    FnTransform, FnTransformFactory, FnWorkerInit, TaskContext, TransformFactory, WorkerInfo,
    WorkerInit, get_worker_info, with_worker_info,
};

#[test]
fn worker_info_scopes_are_nested_and_panic_safe() {
    assert_eq!(get_worker_info(), None);
    let outer = WorkerInfo::new(0, 2, 10, 4).expect("valid outer worker");
    let inner = WorkerInfo::new(1, 2, 11, 4).expect("valid inner worker");

    with_worker_info(outer, || {
        assert_eq!(get_worker_info(), Some(outer));
        with_worker_info(inner, || assert_eq!(get_worker_info(), Some(inner)));
        assert_eq!(get_worker_info(), Some(outer));

        let panic = catch_unwind(AssertUnwindSafe(|| {
            with_worker_info(inner, || panic!("simulated worker panic"));
        }));
        assert!(panic.is_err());
        assert_eq!(get_worker_info(), Some(outer));
    });
    assert_eq!(get_worker_info(), None);
}

#[test]
fn worker_info_is_isolated_between_concurrent_threads() {
    let barrier = Arc::new(Barrier::new(2));
    let handles = (0..2)
        .map(|id| {
            let barrier = Arc::clone(&barrier);
            std::thread::spawn(move || {
                let worker = WorkerInfo::new(id, 2, 10 + id as u64, 4).expect("valid worker");
                with_worker_info(worker, || {
                    barrier.wait();
                    assert_eq!(get_worker_info(), Some(worker));
                });
                assert_eq!(get_worker_info(), None);
            })
        })
        .collect::<Vec<_>>();

    for handle in handles {
        handle.join().expect("worker scope must not panic");
    }
    assert_eq!(get_worker_info(), None);
}

#[test]
fn worker_info_validates_identity_and_locks_versioned_seed_derivation() {
    assert_eq!(rusttorch_data::WORKER_SEED_DERIVATION_VERSION, 1);
    assert!(matches!(
        WorkerInfo::new(0, 0, 10, 0),
        Err(RustTorchError::InvalidConfiguration {
            field: "num_workers",
            ..
        })
    ));
    assert!(matches!(
        WorkerInfo::new(2, 2, 10, 0),
        Err(RustTorchError::InvalidConfiguration {
            field: "worker_id",
            ..
        })
    ));

    let workers = (0..4)
        .map(|id| WorkerInfo::from_loader_seed(id, 4, 42, 1, 2).expect("valid worker"))
        .collect::<Vec<_>>();
    assert_eq!(workers[3].seed, 0xdcae_5da8_9952_36e4);
    assert!(
        workers
            .windows(2)
            .all(|pair| pair[1].seed == pair[0].seed + 1)
    );
    assert_eq!(workers[3].id, 3);
    assert_eq!(workers[3].num_workers, 4);
    assert_eq!(workers[3].rank, 1);
    assert_ne!(
        WorkerInfo::from_loader_seed(3, 4, 42, 2, 2)
            .expect("valid rank")
            .seed,
        workers[3].seed
    );
    assert_ne!(
        WorkerInfo::from_loader_seed(3, 4, 42, 1, 3)
            .expect("valid generation")
            .seed,
        workers[3].seed
    );
}

#[test]
fn worker_seed_derivation_does_not_change_libtorch_global_rng() {
    tch::manual_seed(1_234);
    let expected = tch::Tensor::randn([8], (tch::Kind::Float, tch::Device::Cpu));

    tch::manual_seed(1_234);
    let _ = (0..4)
        .map(|id| WorkerInfo::from_loader_seed(id, 4, 42, 1, 2).expect("valid worker"))
        .collect::<Vec<_>>();
    let actual = tch::Tensor::randn([8], (tch::Kind::Float, tch::Device::Cpu));
    assert_eq!(
        Vec::<f32>::try_from(&actual).expect("random tensor must convert"),
        Vec::<f32>::try_from(&expected).expect("random tensor must convert")
    );
}

#[derive(Debug, PartialEq, Eq)]
struct FactoryFailure;

#[derive(Debug, PartialEq, Eq)]
struct InitFailure;

impl fmt::Display for FactoryFailure {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("factory failure")
    }
}

impl Error for FactoryFailure {}

impl fmt::Display for InitFailure {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("init failure")
    }
}

impl Error for InitFailure {}

#[test]
fn factory_and_initializer_adapters_run_once_per_simulated_worker() {
    let factory_calls = Arc::new(Mutex::new(Vec::new()));
    let factory = FnTransformFactory::new({
        let calls = Arc::clone(&factory_calls);
        move |worker: Option<&WorkerInfo>| {
            let worker = *worker.expect("simulated worker context");
            calls.lock().unwrap().push(worker.id);
            Ok::<_, FactoryFailure>(FnTransform::new(|value: usize, _: &TaskContext| {
                Ok::<_, FactoryFailure>(value)
            }))
        }
    });
    let init_calls = Arc::new(AtomicUsize::new(0));
    let initializer = FnWorkerInit::new({
        let calls = Arc::clone(&init_calls);
        move |worker: &WorkerInfo| {
            assert_eq!(get_worker_info(), Some(*worker));
            calls.fetch_add(1, Ordering::SeqCst);
            Ok::<_, InitFailure>(())
        }
    });

    for id in 0..3 {
        let worker = WorkerInfo::from_loader_seed(id, 3, 42, 1, 2).expect("valid worker");
        with_worker_info(worker, || {
            initializer
                .initialize(&worker)
                .expect("initializer succeeds");
            let _transform = factory.create(Some(&worker)).expect("factory succeeds");
        });
    }
    assert_eq!(*factory_calls.lock().unwrap(), [0, 1, 2]);
    assert_eq!(init_calls.load(Ordering::SeqCst), 3);
    assert_eq!(get_worker_info(), None);
}
