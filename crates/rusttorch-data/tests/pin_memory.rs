use std::{collections::BTreeMap, convert::Infallible, sync::Arc};

use rusttorch_core::{Device, RustTorchError, Tensor, available_devices};
use rusttorch_data::{
    Bytes, DataLoader, Dataset, FnCollate, FnTransformFactory, IdentityTransform, LoaderError,
    PinMemory, PinMemoryStatus, StreamDataLoaderBuilder, VecCollate, WorkerContext, WorkerRecord,
    WorkerSourceFactory,
};

#[derive(Debug, Eq, Ord, PartialEq, PartialOrd)]
struct MoveOnlyKey(u8);

#[test]
fn recursive_pin_memory_preserves_identity_leaves_keys_and_shape_without_clone() {
    let device = Device::Cpu;
    assert_eq!(1_u8.pin_memory(device).unwrap(), 1);
    assert_eq!((-2_i8).pin_memory(device).unwrap(), -2);
    assert_eq!((-3_i16).pin_memory(device).unwrap(), -3);
    assert_eq!((-4_i32).pin_memory(device).unwrap(), -4);
    assert_eq!((-5_i64).pin_memory(device).unwrap(), -5);
    assert_eq!(1.5_f32.pin_memory(device).unwrap(), 1.5);
    assert_eq!(2.5_f64.pin_memory(device).unwrap(), 2.5);
    assert!(true.pin_memory(device).unwrap());
    assert_eq!(String::from("move").pin_memory(device).unwrap(), "move");
    assert_eq!(
        Bytes(vec![1, 2]).pin_memory(device).unwrap(),
        Bytes(vec![1, 2])
    );
    assert_eq!(
        Some(vec![(1_i64, false)]).pin_memory(device).unwrap(),
        Some(vec![(1, false)])
    );

    let mut map = BTreeMap::new();
    map.insert(MoveOnlyKey(7), Some(String::from("value")));
    let pinned = map.pin_memory(device).unwrap();
    assert_eq!(pinned.into_iter().next().unwrap().0, MoveOnlyKey(7));
}

#[test]
fn tensor_and_nested_cuda_pinning_is_checked_when_cuda_exists() {
    let capabilities = available_devices();
    if !capabilities.cuda {
        return;
    }
    let device = Device::Cuda(0);
    let tensor = Tensor::from_slice(&[1_i64, 2]).pin_memory(device).unwrap();
    assert!(tensor.f_is_pinned(device).unwrap());
    let nested = Some(vec![(Tensor::from_slice(&[3_i64]), String::from("x"))])
        .pin_memory(device)
        .unwrap();
    assert!(nested.unwrap()[0].0.f_is_pinned(device).unwrap());
}

struct TensorRows;

impl Dataset for TensorRows {
    type Sample = Tensor;
    type Error = Infallible;

    fn len(&self) -> usize {
        1
    }

    fn get(&self, _index: usize) -> Result<Self::Sample, Self::Error> {
        Ok(Tensor::from_slice(&[1_i64, 2]))
    }
}

#[test]
fn automatic_pin_status_is_effective_and_preserves_the_exact_batch_type() {
    let mut loader = DataLoader::builder(TensorRows)
        .pin_memory()
        .build()
        .unwrap();
    assert!(loader.pin_memory_enabled());
    let capabilities = available_devices();
    if capabilities.cuda {
        assert_eq!(
            loader.pin_memory_status(),
            PinMemoryStatus::Enabled(Device::Cuda(0))
        );
    } else {
        assert_eq!(
            loader.pin_memory_status(),
            PinMemoryStatus::DisabledNoAccelerator
        );
    }
    let batch: Tensor = loader.iter().next().unwrap().unwrap();
    if capabilities.cuda {
        assert!(batch.f_is_pinned(Device::Cuda(0)).unwrap());
    } else {
        assert!(!batch.f_is_pinned(Device::Cpu).unwrap());
    }
}

struct CustomBatch;
struct CustomRows;

impl Dataset for CustomRows {
    type Sample = CustomBatch;
    type Error = Infallible;

    fn len(&self) -> usize {
        1
    }

    fn get(&self, _index: usize) -> Result<Self::Sample, Self::Error> {
        Ok(CustomBatch)
    }
}

#[test]
fn disabled_custom_batch_has_no_pin_memory_bound() {
    let mut loader = DataLoader::builder(CustomRows)
        .collate(VecCollate)
        .build()
        .unwrap();
    assert!(!loader.pin_memory_enabled());
    assert_eq!(loader.pin_memory_status(), PinMemoryStatus::Disabled);
    assert_eq!(loader.iter().count(), 1);
}

#[derive(Clone)]
struct EmptyFactory {
    exact_len_calls: Arc<std::sync::atomic::AtomicUsize>,
}

impl WorkerSourceFactory for EmptyFactory {
    type Sample = Tensor;
    type Error = Infallible;
    type Source = std::iter::Empty<Result<WorkerRecord<Tensor>, Infallible>>;

    fn create(&self, _worker: WorkerContext) -> Result<Self::Source, Self::Error> {
        panic!("unsupported explicit devices reject before worker creation")
    }

    fn exact_len(&self) -> Option<usize> {
        self.exact_len_calls
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        Some(0)
    }
}

#[test]
fn explicit_unsupported_devices_reject_before_factory_callbacks() {
    for device in [Device::Cpu, Device::Mps, Device::Vulkan] {
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let result = StreamDataLoaderBuilder::new(EmptyFactory {
            exact_len_calls: Arc::clone(&calls),
        })
        .pin_memory_for(device)
        .build();
        assert!(matches!(
            result,
            Err(RustTorchError::InvalidConfiguration {
                field: "pin_memory",
                ..
            })
        ));
        assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 0);
    }

    let unavailable = available_devices().cuda_device_count;
    let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let result = StreamDataLoaderBuilder::new(EmptyFactory {
        exact_len_calls: Arc::clone(&calls),
    })
    .pin_memory_for(Device::Cuda(unavailable))
    .build();
    assert!(matches!(
        result,
        Err(RustTorchError::InvalidConfiguration {
            field: "pin_memory",
            ..
        })
    ));
    assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 0);

    let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let observed = Arc::clone(&calls);
    let result = DataLoader::builder(TensorRows)
        .transform_factory(FnTransformFactory::new(move |_: Option<&WorkerContext>| {
            observed.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Ok::<_, Infallible>(IdentityTransform)
        }))
        .pin_memory_for(Device::Cpu)
        .build();
    assert!(matches!(
        result,
        Err(RustTorchError::InvalidConfiguration {
            field: "pin_memory",
            ..
        })
    ));
    assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 0);
}

#[derive(Debug)]
struct TraceBatch(Vec<&'static str>);

impl PinMemory for TraceBatch {
    fn pin_memory(mut self, _device: Device) -> rusttorch_core::Result<Self> {
        self.0.push("pin");
        Ok(self)
    }
}

#[test]
fn pinning_runs_after_collation_in_both_setter_orders_when_cuda_exists() {
    if !available_devices().cuda {
        return;
    }
    let make_collator =
        || FnCollate::new(|_: Vec<Tensor>| Ok::<_, Infallible>(TraceBatch(vec!["collate"])));
    let mut first = DataLoader::builder(TensorRows)
        .pin_memory_for(Device::Cuda(0))
        .collate(make_collator())
        .build()
        .unwrap();
    let mut second = DataLoader::builder(TensorRows)
        .collate(make_collator())
        .pin_memory_for(Device::Cuda(0))
        .build()
        .unwrap();
    let mut zero_worker = DataLoader::builder(TensorRows)
        .workers(0)
        .collate(make_collator())
        .pin_memory_for(Device::Cuda(0))
        .build()
        .unwrap();
    let mut positive_worker = DataLoader::builder(TensorRows)
        .workers(1)
        .collate(make_collator())
        .pin_memory_for(Device::Cuda(0))
        .build()
        .unwrap();
    assert_eq!(first.iter().next().unwrap().unwrap().0, ["collate", "pin"]);
    assert_eq!(second.iter().next().unwrap().unwrap().0, ["collate", "pin"]);
    assert_eq!(
        zero_worker.iter().next().unwrap().unwrap().0,
        ["collate", "pin"]
    );
    assert_eq!(
        positive_worker.iter().next().unwrap().unwrap().0,
        ["collate", "pin"]
    );

    struct OneTensor;
    impl WorkerSourceFactory for OneTensor {
        type Sample = Tensor;
        type Error = Infallible;
        type Source = std::vec::IntoIter<Result<WorkerRecord<Tensor>, Infallible>>;

        fn create(&self, _worker: WorkerContext) -> Result<Self::Source, Self::Error> {
            Ok(vec![Ok(WorkerRecord {
                sequence: Some(rusttorch_data::SequenceId::new(0)),
                logical_id: rusttorch_data::LogicalSampleId::new(0),
                sample: Tensor::from_slice(&[1_i64]),
            })]
            .into_iter())
        }
    }
    let mut stream_first = StreamDataLoaderBuilder::new(OneTensor)
        .pin_memory_for(Device::Cuda(0))
        .collate(make_collator())
        .build()
        .unwrap();
    let mut stream_second = StreamDataLoaderBuilder::new(OneTensor)
        .collate(make_collator())
        .pin_memory_for(Device::Cuda(0))
        .build()
        .unwrap();
    assert_eq!(
        stream_first.iter().next().unwrap().unwrap().0,
        ["collate", "pin"]
    );
    assert_eq!(
        stream_second.iter().next().unwrap().unwrap().0,
        ["collate", "pin"]
    );
}

struct FailingPin;

impl PinMemory for FailingPin {
    fn pin_memory(self, _device: Device) -> rusttorch_core::Result<Self> {
        Err(RustTorchError::BackendUnavailable {
            backend: "test pin",
            reason: "rejected".to_owned(),
        })
    }
}

#[test]
fn pin_failure_is_typed_once_when_cuda_exists() {
    if !available_devices().cuda {
        return;
    }
    let mut loader = DataLoader::builder(TensorRows)
        .collate(FnCollate::new(|_: Vec<Tensor>| {
            Ok::<_, Infallible>(FailingPin)
        }))
        .pin_memory_for(Device::Cuda(0))
        .build()
        .unwrap();
    let mut iteration = loader.iter();
    assert!(matches!(
        iteration.next(),
        Some(Err(LoaderError::PinMemory {
            batch: 0,
            source: RustTorchError::BackendUnavailable {
                backend: "test pin",
                ..
            },
        }))
    ));
    assert!(iteration.next().is_none());

    struct OneTensor;
    impl WorkerSourceFactory for OneTensor {
        type Sample = Tensor;
        type Error = Infallible;
        type Source = std::vec::IntoIter<Result<WorkerRecord<Tensor>, Infallible>>;

        fn create(&self, _worker: WorkerContext) -> Result<Self::Source, Self::Error> {
            Ok(vec![Ok(WorkerRecord {
                sequence: Some(rusttorch_data::SequenceId::new(0)),
                logical_id: rusttorch_data::LogicalSampleId::new(0),
                sample: Tensor::from_slice(&[1_i64]),
            })]
            .into_iter())
        }
    }
    let mut stream = StreamDataLoaderBuilder::new(OneTensor)
        .collate(FnCollate::new(|_: Vec<Tensor>| {
            Ok::<_, Infallible>(FailingPin)
        }))
        .pin_memory_for(Device::Cuda(0))
        .build()
        .unwrap();
    let mut iteration = stream.iter();
    assert!(matches!(
        iteration.next(),
        Some(Err(LoaderError::PinMemory {
            batch: 0,
            source: RustTorchError::BackendUnavailable {
                backend: "test pin",
                ..
            },
        }))
    ));
    assert!(iteration.next().is_none());
}

#[test]
fn stream_auto_pin_status_and_batch_type_are_preserved() {
    struct OneTensor;
    impl WorkerSourceFactory for OneTensor {
        type Sample = Tensor;
        type Error = Infallible;
        type Source = std::vec::IntoIter<Result<WorkerRecord<Tensor>, Infallible>>;

        fn create(&self, _worker: WorkerContext) -> Result<Self::Source, Self::Error> {
            Ok(vec![Ok(WorkerRecord {
                sequence: Some(rusttorch_data::SequenceId::new(0)),
                logical_id: rusttorch_data::LogicalSampleId::new(0),
                sample: Tensor::from_slice(&[1_i64]),
            })]
            .into_iter())
        }
    }

    let mut loader = StreamDataLoaderBuilder::new(OneTensor)
        .collate(VecCollate)
        .pin_memory()
        .build()
        .unwrap();
    assert!(loader.pin_memory_enabled());
    let batch: Vec<Tensor> = loader.iter().next().unwrap().unwrap();
    assert_eq!(batch.len(), 1);
}
