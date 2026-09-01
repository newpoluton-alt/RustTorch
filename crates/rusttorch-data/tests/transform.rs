use std::{
    convert::Infallible,
    error::Error,
    fmt,
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
};

use rand::RngCore;
use rusttorch_core::RustTorchError;
use rusttorch_data::{
    CancellationToken, DataLoader, Dataset, Deadline, FnCollate, FnTransform, FnTransformFactory,
    LoaderError, PipelineError, TaskContext, Transform, VecCollate, WorkerContext, WorkerInfo,
    with_worker_info,
};

#[derive(Debug, PartialEq, Eq)]
struct FetchError;

#[derive(Debug, PartialEq, Eq)]
struct TransformError;

#[derive(Debug, PartialEq, Eq)]
struct CollateFailure;

#[derive(Debug, PartialEq, Eq)]
struct FactoryError;

#[derive(Debug, PartialEq, Eq)]
struct InitError;

macro_rules! display_error {
    ($($error:ty),+ $(,)?) => {
        $(
            impl fmt::Display for $error {
                fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                    formatter.write_str(stringify!($error))
                }
            }

            impl Error for $error {}
        )+
    };
}

display_error!(
    FetchError,
    TransformError,
    CollateFailure,
    FactoryError,
    InitError,
);

#[test]
fn task_rng_is_versioned_schedule_independent_and_has_a_locked_seed() {
    let context = TaskContext {
        loader_seed: 42,
        epoch: 3,
        rank: 1,
        logical_sample: 99,
        stage: 7,
        cancellation: CancellationToken::new(),
        deadline: Deadline::none(),
    };
    assert_eq!(rusttorch_data::TASK_RNG_DERIVATION_VERSION, 1);
    assert_eq!(context.deterministic_seed(), 0x1d7d_73dc_f6e9_4f2d);

    let sequence = |context: &TaskContext| {
        let mut rng = context.rng();
        (0..32).map(|_| rng.next_u64()).collect::<Vec<_>>()
    };
    let expected = sequence(&context);
    let worker_zero = WorkerInfo::new(0, 2, 10, 1).expect("valid worker");
    let worker_one = WorkerInfo::new(1, 2, 11, 1).expect("valid worker");
    assert_eq!(
        with_worker_info(worker_zero, || sequence(&context)),
        expected
    );
    assert_eq!(
        with_worker_info(worker_one, || sequence(&context)),
        expected
    );

    let variants = [
        TaskContext {
            loader_seed: 43,
            ..context.clone()
        },
        TaskContext {
            epoch: 4,
            ..context.clone()
        },
        TaskContext {
            rank: 2,
            ..context.clone()
        },
        TaskContext {
            logical_sample: 100,
            ..context.clone()
        },
        TaskContext {
            stage: 8,
            ..context.clone()
        },
    ];
    for variant in variants {
        assert_ne!(sequence(&variant), expected);
    }
}

#[test]
fn task_rng_does_not_change_libtorch_global_rng() {
    tch::manual_seed(1_234);
    let expected = tch::Tensor::randn([8], (tch::Kind::Float, tch::Device::Cpu));

    tch::manual_seed(1_234);
    let context = TaskContext {
        loader_seed: 42,
        epoch: 3,
        rank: 1,
        logical_sample: 99,
        stage: 7,
        cancellation: CancellationToken::new(),
        deadline: Deadline::none(),
    };
    let mut rng = context.rng();
    let _ = (0..32).map(|_| rng.next_u64()).collect::<Vec<_>>();
    let actual = tch::Tensor::randn([8], (tch::Kind::Float, tch::Device::Cpu));

    assert_eq!(
        Vec::<f32>::try_from(&actual).expect("random tensor must convert"),
        Vec::<f32>::try_from(&expected).expect("random tensor must convert")
    );
}

#[test]
fn fn_transform_preserves_output_and_error_types() {
    let mut transform = FnTransform::new(|value: i64, context: &TaskContext| {
        (context.stage == 7)
            .then_some(value * 2)
            .ok_or(TransformError)
    });
    let context = TaskContext {
        loader_seed: 0,
        epoch: 0,
        rank: 0,
        logical_sample: 0,
        stage: 7,
        cancellation: CancellationToken::new(),
        deadline: Deadline::none(),
    };
    assert_eq!(
        rusttorch_data::Transform::transform(&mut transform, 3, &context),
        Ok(6)
    );
}

struct Rows(Vec<i64>);

impl Dataset for Rows {
    type Sample = i64;
    type Error = FetchError;

    fn len(&self) -> usize {
        self.0.len()
    }

    fn get(&self, index: usize) -> Result<Self::Sample, Self::Error> {
        Ok(self.0[index])
    }
}

#[test]
fn serial_factory_creates_one_transform_per_iterator_and_context_ids_restart()
-> Result<(), RustTorchError> {
    let creates = Arc::new(AtomicUsize::new(0));
    let contexts = Arc::new(Mutex::new(Vec::new()));
    let factory = FnTransformFactory::new({
        let creates = Arc::clone(&creates);
        let contexts = Arc::clone(&contexts);
        move |worker: Option<&WorkerContext>| {
            assert!(worker.is_none());
            let transform_id = creates.fetch_add(1, Ordering::SeqCst) + 1;
            let contexts = Arc::clone(&contexts);
            Ok::<_, FactoryError>(FnTransform::new(
                move |value: i64, context: &TaskContext| {
                    contexts
                        .lock()
                        .unwrap()
                        .push((transform_id, context.clone()));
                    Ok::<_, TransformError>(value + transform_id as i64 * 100)
                },
            ))
        }
    });
    let mut loader = DataLoader::builder(Rows(vec![10, 20, 30]))
        .batch_size(2)
        .collate(VecCollate)
        .transform_factory(factory)
        .seed(42)
        .epoch(3)
        .rank(1)
        .build()?;

    let first = loader.iter().collect::<Result<Vec<_>, _>>().unwrap();
    let second = loader.iter().collect::<Result<Vec<_>, _>>().unwrap();
    assert_eq!(first, [vec![110, 120], vec![130]]);
    assert_eq!(second, [vec![210, 220], vec![230]]);
    assert_eq!(creates.load(Ordering::SeqCst), 2);

    let contexts = contexts.lock().unwrap();
    assert_eq!(
        contexts
            .iter()
            .map(|(transform, context)| (*transform, context.logical_sample))
            .collect::<Vec<_>>(),
        [(1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]
    );
    assert!(contexts.iter().all(|(_, context)| {
        context.loader_seed == 42 && context.epoch == 3 && context.rank == 1 && context.stage == 0
    }));
    Ok(())
}

#[derive(Clone)]
struct StatefulTransform {
    calls: i64,
}

impl Transform<i64> for StatefulTransform {
    type Output = i64;
    type Error = Infallible;

    fn transform(
        &mut self,
        input: i64,
        _context: &TaskContext,
    ) -> Result<Self::Output, Self::Error> {
        let output = input + self.calls;
        self.calls += 1;
        Ok(output)
    }
}

#[test]
fn transform_builder_clones_fresh_state_for_each_iterator() -> Result<(), RustTorchError> {
    let mut loader = DataLoader::builder(Rows(vec![10, 20]))
        .transform(StatefulTransform { calls: 0 })
        .collate(VecCollate)
        .build()?;
    assert_eq!(
        loader.iter().collect::<Result<Vec<_>, _>>().unwrap(),
        [vec![10], vec![21]]
    );
    assert_eq!(
        loader.iter().collect::<Result<Vec<_>, _>>().unwrap(),
        [vec![10], vec![21]]
    );
    Ok(())
}

struct FailingRows;

impl Dataset for FailingRows {
    type Sample = i64;
    type Error = FetchError;

    fn len(&self) -> usize {
        1
    }

    fn get(&self, _index: usize) -> Result<Self::Sample, Self::Error> {
        Err(FetchError)
    }
}

#[test]
fn every_serial_stage_keeps_its_concrete_error_and_metadata() -> Result<(), RustTorchError> {
    let mut dataset = DataLoader::builder(FailingRows)
        .transform(FnTransform::new(|value: i64, _: &TaskContext| {
            Ok::<_, TransformError>(value)
        }))
        .collate(VecCollate)
        .build()?;
    let mut iterator = dataset.iter();
    assert!(matches!(
        iterator.next(),
        Some(Err(LoaderError::Pipeline {
            batch: Some(0),
            worker: None,
            source: PipelineError::Dataset(FetchError),
        }))
    ));
    assert!(iterator.next().is_none());

    let mut transform = DataLoader::builder(Rows(vec![1]))
        .transform(FnTransform::new(|_: i64, _: &TaskContext| {
            Err::<i64, _>(TransformError)
        }))
        .collate(VecCollate)
        .build()?;
    let mut iterator = transform.iter();
    assert!(matches!(
        iterator.next(),
        Some(Err(LoaderError::Pipeline {
            batch: Some(0),
            worker: None,
            source: PipelineError::Transform(TransformError),
        }))
    ));
    assert!(iterator.next().is_none());

    let mut collate = DataLoader::builder(Rows(vec![1]))
        .transform(FnTransform::new(|value: i64, _: &TaskContext| {
            Ok::<_, TransformError>(value)
        }))
        .collate(FnCollate::new(|_: Vec<i64>| {
            Err::<Vec<i64>, _>(CollateFailure)
        }))
        .build()?;
    let mut iterator = collate.iter();
    assert!(matches!(
        iterator.next(),
        Some(Err(LoaderError::Pipeline {
            batch: Some(0),
            worker: None,
            source: PipelineError::Collate(CollateFailure),
        }))
    ));
    assert!(iterator.next().is_none());

    let mut factory = DataLoader::builder(Rows(vec![1]))
        .transform_factory(FnTransformFactory::new(|_: Option<&WorkerContext>| {
            Err::<rusttorch_data::IdentityTransform, _>(FactoryError)
        }))
        .collate(VecCollate)
        .build()?;
    let mut iterator = factory.iter();
    assert!(matches!(
        iterator.next(),
        Some(Err(LoaderError::Pipeline {
            batch: None,
            worker: None,
            source: PipelineError::TransformInit(FactoryError),
        }))
    ));
    assert!(iterator.next().is_none());

    let init_error: PipelineError<
        FetchError,
        TransformError,
        CollateFailure,
        FactoryError,
        InitError,
    > = PipelineError::WorkerInit(InitError);
    assert!(matches!(init_error, PipelineError::WorkerInit(InitError)));
    Ok(())
}

#[test]
fn pipeline_error_generic_positions_are_distinct() {
    let errors = [
        PipelineError::<FetchError, TransformError, CollateFailure, FactoryError, InitError>::Source(
            FetchError,
        ),
        PipelineError::<FetchError, TransformError, CollateFailure, FactoryError, InitError>::Dataset(
            FetchError,
        ),
        PipelineError::Transform(TransformError),
        PipelineError::Collate(CollateFailure),
        PipelineError::TransformInit(FactoryError),
        PipelineError::WorkerInit(InitError),
    ];
    assert!(matches!(errors[0], PipelineError::Source(FetchError)));
    assert!(matches!(errors[1], PipelineError::Dataset(FetchError)));
    assert!(matches!(
        errors[2],
        PipelineError::Transform(TransformError)
    ));
    assert!(matches!(errors[3], PipelineError::Collate(CollateFailure)));
    assert!(matches!(
        errors[4],
        PipelineError::TransformInit(FactoryError)
    ));
    assert!(matches!(errors[5], PipelineError::WorkerInit(InitError)));
}

#[test]
fn every_pipeline_variant_preserves_the_standard_error_source_chain() {
    let errors: Vec<
        PipelineError<FetchError, TransformError, CollateFailure, FactoryError, InitError>,
    > = vec![
        PipelineError::Source(FetchError),
        PipelineError::Dataset(FetchError),
        PipelineError::Transform(TransformError),
        PipelineError::Collate(CollateFailure),
        PipelineError::TransformInit(FactoryError),
        PipelineError::WorkerInit(InitError),
    ];

    let stage_names = [
        "stream source failed",
        "dataset fetch failed",
        "transform failed",
        "collation failed",
        "transform initialization failed",
        "worker initialization failed",
    ];
    for (stage, source) in errors.into_iter().enumerate() {
        let loader = LoaderError::Pipeline {
            batch: Some(7),
            worker: None,
            source,
        };
        let display = loader.to_string();
        assert!(display.contains(stage_names[stage]));
        let pipeline = Error::source(&loader).expect("LoaderError exposes PipelineError");
        let leaf = pipeline
            .source()
            .expect("PipelineError exposes its typed stage error");
        let concrete_type_is_preserved = match stage {
            0 => leaf.downcast_ref::<FetchError>().is_some(),
            1 => leaf.downcast_ref::<FetchError>().is_some(),
            2 => leaf.downcast_ref::<TransformError>().is_some(),
            3 => leaf.downcast_ref::<CollateFailure>().is_some(),
            4 => leaf.downcast_ref::<FactoryError>().is_some(),
            5 => leaf.downcast_ref::<InitError>().is_some(),
            _ => unreachable!(),
        };
        assert!(concrete_type_is_preserved);
        assert!(leaf.source().is_none());
    }

    let public_without_error_bounds: PipelineError<u8, u16, u32, u64, u128> =
        PipelineError::Dataset(7);
    assert!(matches!(
        public_without_error_bounds,
        PipelineError::Dataset(7)
    ));
}

fn _assert_default_error_types(
    _error: LoaderError<PipelineError<FetchError, Infallible, Infallible, Infallible, Infallible>>,
) {
}
