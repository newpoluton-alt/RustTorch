use std::{
    collections::BTreeMap,
    convert::Infallible,
    num::NonZeroUsize,
    sync::{
        Arc, Condvar, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    time::Duration,
};

use rusttorch_core::RustTorchError;
use rusttorch_data::{
    DataLoader, Dataset, FnCollate, LoaderError, LogicalSampleId, MemoryFootprint, SequenceId,
    StreamDataLoaderBuilder, TaskContext, VecCollate, WorkerContext, WorkerRecord,
    WorkerSourceFactory,
};

#[derive(Debug, Eq, PartialEq)]
struct SizedRecord {
    value: u64,
    bytes: usize,
}

impl MemoryFootprint for SizedRecord {
    fn resident_bytes(&self) -> usize {
        self.bytes
    }
}

struct Rows(Vec<usize>);

impl Dataset for Rows {
    type Sample = usize;
    type Error = Infallible;

    fn len(&self) -> usize {
        self.0.len()
    }

    fn get(&self, index: usize) -> Result<Self::Sample, Self::Error> {
        Ok(self.0[index])
    }
}

struct EmptyBytes;

impl Dataset for EmptyBytes {
    type Sample = u8;
    type Error = Infallible;

    fn len(&self) -> usize {
        0
    }

    fn get(&self, _index: usize) -> Result<Self::Sample, Self::Error> {
        unreachable!("empty dataset is never fetched")
    }
}

struct EmptyByteShards;

impl WorkerSourceFactory for EmptyByteShards {
    type Sample = u8;
    type Error = Infallible;
    type Source = std::iter::Empty<Result<WorkerRecord<u8>, Infallible>>;

    fn create(&self, _worker: WorkerContext) -> Result<Self::Source, Self::Error> {
        Ok(std::iter::empty())
    }

    fn exact_len(&self) -> Option<usize> {
        Some(0)
    }
}

#[test]
fn disabled_capacity_matches_task9_map_and_stream_boundaries() {
    DataLoader::builder(EmptyBytes)
        .workers(2)
        .prefetch_factor(220_717)
        .build()
        .expect("Task 9 accepted this exact disabled map completion boundary");
    assert!(matches!(
        DataLoader::builder(EmptyBytes)
            .workers(2)
            .prefetch_factor(220_718)
            .build(),
        Err(RustTorchError::InvalidConfiguration {
            field: "prefetch_factor",
            ..
        })
    ));

    StreamDataLoaderBuilder::new(EmptyByteShards)
        .workers(2)
        .batch_size(1)
        .prefetch_factor(220_717)
        .collate(VecCollate)
        .build()
        .expect("Task 9 accepted this exact disabled stream completion boundary");
    assert!(matches!(
        StreamDataLoaderBuilder::new(EmptyByteShards)
            .workers(2)
            .batch_size(1)
            .prefetch_factor(220_718)
            .collate(VecCollate)
            .build(),
        Err(RustTorchError::InvalidConfiguration {
            field: "prefetch_factor",
            ..
        })
    ));
}

#[test]
fn enabled_capacity_charges_permits_and_stream_waiters() {
    let limit = NonZeroUsize::MIN;
    let map_large = DataLoader::builder(EmptyBytes)
        .workers(2)
        .prefetch_factor(200_000)
        .prefetch_bytes(limit)
        .build();
    assert!(matches!(
        map_large,
        Err(RustTorchError::InvalidConfiguration {
            field: "prefetch_factor",
            ..
        })
    ));
    DataLoader::builder(EmptyBytes)
        .workers(2)
        .prefetch_factor(100_000)
        .prefetch_bytes(limit)
        .build()
        .expect("smaller enabled map storage fits");

    StreamDataLoaderBuilder::new(EmptyByteShards)
        .workers(2)
        .batch_size(1)
        .prefetch_factor(182_331)
        .collate(VecCollate)
        .prefetch_bytes(limit)
        .build()
        .expect("exact enabled stream storage boundary fits");
    assert!(matches!(
        StreamDataLoaderBuilder::new(EmptyByteShards)
            .workers(2)
            .batch_size(1)
            .prefetch_factor(182_332)
            .collate(VecCollate)
            .prefetch_bytes(limit)
            .build(),
        Err(RustTorchError::InvalidConfiguration {
            field: "prefetch_factor",
            ..
        })
    ));
}

#[test]
fn footprints_are_recursive_capacity_aware_and_saturating() {
    assert_eq!(String::with_capacity(17).resident_bytes(), 17);
    assert_eq!(
        rusttorch_data::Bytes(Vec::with_capacity(19)).resident_bytes(),
        19
    );
    assert_eq!(Some((1_u8, 2_i64)).resident_bytes(), 9);
    assert_eq!(vec![1_i16, 2_i16].resident_bytes(), 4);

    let mut map = BTreeMap::new();
    map.insert(1_u8, 2_i32);
    assert_eq!(map.resident_bytes(), 5);

    let saturated = vec![
        SizedRecord {
            value: 0,
            bytes: usize::MAX,
        },
        SizedRecord { value: 1, bytes: 1 },
    ];
    assert_eq!(saturated.resident_bytes(), usize::MAX);
}

#[test]
fn byte_accounting_uses_the_final_transform_type_in_either_setter_order() {
    let limit = NonZeroUsize::new(16).unwrap();
    let mut first = DataLoader::builder(Rows(vec![3, 4]))
        .workers(1)
        .prefetch_bytes(limit)
        .transform(rusttorch_data::FnTransform::new(
            |value: usize, _: &TaskContext| {
                Ok::<_, Infallible>(SizedRecord {
                    value: value as u64,
                    bytes: value,
                })
            },
        ))
        .collate(VecCollate)
        .build()
        .unwrap();
    let mut second = DataLoader::builder(Rows(vec![3, 4]))
        .workers(1)
        .transform(rusttorch_data::FnTransform::new(
            |value: usize, _: &TaskContext| {
                Ok::<_, Infallible>(SizedRecord {
                    value: value as u64,
                    bytes: value,
                })
            },
        ))
        .prefetch_bytes(limit)
        .collate(VecCollate)
        .build()
        .unwrap();

    assert_eq!(first.effective_prefetch_bytes(), Some(limit));
    assert_eq!(second.effective_prefetch_bytes(), Some(limit));
    assert_eq!(first.iter().count(), 2);
    assert_eq!(second.iter().count(), 2);
}

struct Unmeasured;

struct UnmeasuredRows;

impl Dataset for UnmeasuredRows {
    type Sample = Unmeasured;
    type Error = Infallible;

    fn len(&self) -> usize {
        1
    }

    fn get(&self, _index: usize) -> Result<Self::Sample, Self::Error> {
        Ok(Unmeasured)
    }
}

#[test]
fn disabled_custom_types_remain_item_bounded_without_new_trait_bounds() {
    let mut loader = DataLoader::builder(UnmeasuredRows)
        .workers(1)
        .collate(VecCollate)
        .build()
        .unwrap();
    assert_eq!(loader.effective_prefetch_bytes(), None);
    assert_eq!(loader.iter().count(), 1);
}

#[test]
fn map_zero_workers_reject_byte_prefetch_before_iteration_callbacks() {
    let created = Arc::new(Mutex::new(0));
    let observed = Arc::clone(&created);
    let result = DataLoader::builder(Rows(vec![1]))
        .transform_factory(rusttorch_data::FnTransformFactory::new(
            move |_: Option<&WorkerContext>| {
                *observed.lock().unwrap() += 1;
                Ok::<_, Infallible>(rusttorch_data::FnTransform::new(
                    |value: usize, _: &TaskContext| {
                        Ok::<_, Infallible>(SizedRecord {
                            value: value as u64,
                            bytes: value,
                        })
                    },
                ))
            },
        ))
        .prefetch_bytes(NonZeroUsize::MIN)
        .collate(VecCollate)
        .build();
    assert!(matches!(
        result,
        Err(RustTorchError::InvalidConfiguration {
            field: "prefetch_bytes",
            ..
        })
    ));
    assert_eq!(*created.lock().unwrap(), 0);
}

#[test]
fn oversized_transformed_map_batch_has_exact_metadata_once() {
    let mut loader = DataLoader::builder(Rows(vec![5, 4]))
        .workers(1)
        .batch_size(2)
        .transform(rusttorch_data::FnTransform::new(
            |value: usize, _: &TaskContext| {
                Ok::<_, Infallible>(SizedRecord {
                    value: value as u64,
                    bytes: value,
                })
            },
        ))
        .collate(VecCollate)
        .prefetch_bytes(NonZeroUsize::new(8).unwrap())
        .build()
        .unwrap();
    let mut iteration = loader.iter();
    assert!(matches!(
        iteration.next(),
        Some(Err(LoaderError::MemoryLimit {
            batch: Some(0),
            worker: Some(0),
            sequence: None,
            logical_id: None,
            limit: 8,
            actual: 9,
        }))
    ));
    assert!(iteration.next().is_none());
}

struct DelayedMapRows {
    high_ready: Arc<(Mutex<bool>, Condvar)>,
}

impl Dataset for DelayedMapRows {
    type Sample = usize;
    type Error = Infallible;

    fn len(&self) -> usize {
        2
    }

    fn get(&self, index: usize) -> Result<Self::Sample, Self::Error> {
        let (lock, wake) = &*self.high_ready;
        if index == 1 {
            *lock.lock().unwrap() = true;
            wake.notify_all();
            return Ok(index);
        }
        let mut ready = lock.lock().unwrap();
        while !*ready {
            ready = wake.wait(ready).unwrap();
        }
        Ok(index)
    }
}

#[test]
fn ordered_map_byte_admission_blocks_high_until_delayed_low_is_consumable() {
    let mut loader = DataLoader::builder(DelayedMapRows {
        high_ready: Arc::new((Mutex::new(false), Condvar::new())),
    })
    .workers(2)
    .transform(rusttorch_data::FnTransform::new(
        |value: usize, _: &TaskContext| {
            Ok::<_, Infallible>(SizedRecord {
                value: value as u64,
                bytes: 1,
            })
        },
    ))
    .collate(FnCollate::new(|records: Vec<SizedRecord>| {
        Ok::<_, Infallible>(records[0].value)
    }))
    .prefetch_bytes(NonZeroUsize::MIN)
    .build()
    .unwrap();

    assert_eq!(
        loader.iter().collect::<Result<Vec<_>, _>>().unwrap(),
        [0, 1]
    );
}

#[derive(Clone)]
struct RecordsFactory {
    records: Arc<Vec<(Option<u64>, u64, usize)>>,
}

impl WorkerSourceFactory for RecordsFactory {
    type Sample = SizedRecord;
    type Error = Infallible;
    type Source = std::vec::IntoIter<Result<WorkerRecord<SizedRecord>, Infallible>>;

    fn create(&self, worker: WorkerContext) -> Result<Self::Source, Self::Error> {
        if worker.info.id != 0 {
            return Ok(Vec::new().into_iter());
        }
        Ok(self
            .records
            .iter()
            .map(|(sequence, logical, bytes)| {
                Ok(WorkerRecord {
                    sequence: sequence.map(SequenceId::new),
                    logical_id: LogicalSampleId::new(*logical),
                    sample: SizedRecord {
                        value: *logical,
                        bytes: *bytes,
                    },
                })
            })
            .collect::<Vec<_>>()
            .into_iter())
    }
}

#[test]
fn unordered_unsequenced_oversize_preserves_exact_stream_metadata_once() {
    let factory = RecordsFactory {
        records: Arc::new(vec![(None, 41, 9)]),
    };
    let mut loader = StreamDataLoaderBuilder::new(factory)
        .ordered(false)
        .collate(VecCollate)
        .prefetch_bytes(NonZeroUsize::new(8).unwrap())
        .build()
        .unwrap();
    let mut iteration = loader.iter();
    assert!(matches!(
        iteration.next(),
        Some(Err(LoaderError::MemoryLimit {
            batch: None,
            worker: Some(0),
            sequence: None,
            logical_id: Some(41),
            limit: 8,
            actual: 9,
        }))
    ));
    assert!(iteration.next().is_none());
}

#[test]
fn ordered_oversize_preserves_stream_identity_without_guessing_a_batch() {
    let factory = RecordsFactory {
        records: Arc::new(vec![(Some(0), 17, 9)]),
    };
    let mut loader = StreamDataLoaderBuilder::new(factory)
        .collate(VecCollate)
        .prefetch_bytes(NonZeroUsize::new(8).unwrap())
        .build()
        .unwrap();
    let mut iteration = loader.iter();
    assert!(matches!(
        iteration.next(),
        Some(Err(LoaderError::MemoryLimit {
            batch: None,
            worker: Some(0),
            sequence: Some(0),
            logical_id: Some(17),
            limit: 8,
            actual: 9,
        }))
    ));
    assert!(iteration.next().is_none());
}

#[test]
fn ordered_byte_stream_rejects_missing_and_nonmonotonic_same_shard_sequences() {
    let missing = RecordsFactory {
        records: Arc::new(vec![(Some(1), 1, 1)]),
    };
    let mut loader = StreamDataLoaderBuilder::new(missing)
        .collate(VecCollate)
        .prefetch_bytes(NonZeroUsize::new(2).unwrap())
        .build()
        .unwrap();
    let mut iteration = loader.iter();
    assert!(matches!(
        iteration.next(),
        Some(Err(LoaderError::StreamProtocol {
            sequence: Some(0),
            ..
        }))
    ));
    assert!(iteration.next().is_none());

    let nonmonotonic = RecordsFactory {
        records: Arc::new(vec![(Some(0), 0, 1), (Some(0), 1, 1)]),
    };
    let mut loader = StreamDataLoaderBuilder::new(nonmonotonic)
        .collate(VecCollate)
        .prefetch_bytes(NonZeroUsize::new(2).unwrap())
        .build()
        .unwrap();
    let mut iteration = loader.iter();
    assert!(iteration.next().unwrap().is_ok());
    assert!(matches!(
        iteration.next(),
        Some(Err(LoaderError::StreamProtocol {
            sequence: Some(0),
            ..
        }))
    ));
    assert!(iteration.next().is_none());
}

#[derive(Clone)]
struct DelayedLowFactory {
    high_ready: Arc<(Mutex<bool>, Condvar)>,
}

impl WorkerSourceFactory for DelayedLowFactory {
    type Sample = SizedRecord;
    type Error = Infallible;
    type Source = std::vec::IntoIter<Result<WorkerRecord<SizedRecord>, Infallible>>;

    fn create(&self, worker: WorkerContext) -> Result<Self::Source, Self::Error> {
        let (lock, wake) = &*self.high_ready;
        if worker.info.id == 1 {
            *lock.lock().unwrap() = true;
            wake.notify_all();
            return Ok(vec![Ok(WorkerRecord {
                sequence: Some(SequenceId::new(1)),
                logical_id: LogicalSampleId::new(1),
                sample: SizedRecord { value: 1, bytes: 1 },
            })]
            .into_iter());
        }
        let mut ready = lock.lock().unwrap();
        while !*ready {
            ready = wake.wait(ready).unwrap();
        }
        Ok(vec![Ok(WorkerRecord {
            sequence: Some(SequenceId::new(0)),
            logical_id: LogicalSampleId::new(0),
            sample: SizedRecord { value: 0, bytes: 1 },
        })]
        .into_iter())
    }
}

#[test]
fn ordered_byte_admission_allows_delayed_low_progress_and_persistent_reuse() {
    let mut delayed = StreamDataLoaderBuilder::new(DelayedLowFactory {
        high_ready: Arc::new((Mutex::new(false), Condvar::new())),
    })
    .workers(2)
    .batch_size(2)
    .collate(FnCollate::new(|records: Vec<SizedRecord>| {
        Ok::<_, Infallible>(
            records
                .into_iter()
                .map(|record| record.value)
                .collect::<Vec<_>>(),
        )
    }))
    .prefetch_bytes(NonZeroUsize::new(2).unwrap())
    .build()
    .unwrap();
    assert_eq!(delayed.iter().next().unwrap().unwrap(), [0, 1]);

    let mut persistent = StreamDataLoaderBuilder::new(RecordsFactory {
        records: Arc::new(vec![(Some(0), 0, 2)]),
    })
    .persistent_workers(true)
    .collate(VecCollate)
    .prefetch_bytes(NonZeroUsize::new(2).unwrap())
    .build()
    .unwrap();
    assert_eq!(persistent.effective_prefetch_bytes().unwrap().get(), 2);
    assert_eq!(persistent.iter().count(), 1);
    assert_eq!(persistent.iter().count(), 1);
}

#[derive(Default)]
struct PrefetchProbeState {
    sources: usize,
    first_generation_requested_second: bool,
}

#[derive(Clone)]
struct PrefetchProbeFactory {
    state: Arc<(Mutex<PrefetchProbeState>, Condvar)>,
}

struct PrefetchProbeSource {
    state: Arc<(Mutex<PrefetchProbeState>, Condvar)>,
    source_number: usize,
    next: u64,
}

impl Iterator for PrefetchProbeSource {
    type Item = Result<WorkerRecord<SizedRecord>, Infallible>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.next == 2 {
            return None;
        }
        if self.source_number == 1 && self.next == 1 {
            let (lock, wake) = &*self.state;
            lock.lock().unwrap().first_generation_requested_second = true;
            wake.notify_all();
        }
        let sequence = self.next;
        self.next += 1;
        Some(Ok(WorkerRecord {
            sequence: Some(SequenceId::new(sequence)),
            logical_id: LogicalSampleId::new(sequence),
            sample: SizedRecord {
                value: sequence,
                bytes: 1,
            },
        }))
    }
}

impl WorkerSourceFactory for PrefetchProbeFactory {
    type Sample = SizedRecord;
    type Error = Infallible;
    type Source = PrefetchProbeSource;

    fn create(&self, _worker: WorkerContext) -> Result<Self::Source, Self::Error> {
        let source_number = {
            let mut state = self.state.0.lock().unwrap();
            state.sources += 1;
            state.sources
        };
        Ok(PrefetchProbeSource {
            state: Arc::clone(&self.state),
            source_number,
            next: 0,
        })
    }

    fn exact_len(&self) -> Option<usize> {
        Some(2)
    }
}

#[test]
fn early_drop_cancels_a_blocked_permit_and_persistent_restart_has_no_leak() {
    let state = Arc::new((Mutex::new(PrefetchProbeState::default()), Condvar::new()));
    let mut loader = StreamDataLoaderBuilder::new(PrefetchProbeFactory {
        state: Arc::clone(&state),
    })
    .persistent_workers(true)
    .prefetch_factor(2)
    .collate(VecCollate)
    .prefetch_bytes(NonZeroUsize::MIN)
    .build()
    .unwrap();

    let iteration = loader.iter();
    let (lock, wake) = &*state;
    let mut observed = lock.lock().unwrap();
    while !observed.first_generation_requested_second {
        observed = wake.wait(observed).unwrap();
    }
    drop(observed);
    drop(iteration);

    assert_eq!(loader.iter().count(), 2);
}

struct TimeoutRows {
    block_low: Arc<AtomicBool>,
}

impl Dataset for TimeoutRows {
    type Sample = usize;
    type Error = &'static str;

    fn len(&self) -> usize {
        2
    }

    fn get(&self, index: usize) -> Result<Self::Sample, Self::Error> {
        Ok(index)
    }

    fn get_batch_with_context(
        &self,
        indices: &[usize],
        context: &WorkerContext,
    ) -> Result<Vec<Self::Sample>, Self::Error> {
        if indices == [0] && self.block_low.load(Ordering::SeqCst) {
            context.cancellation.wait_cancelled();
            return Err("cancelled low record");
        }
        Ok(indices.to_vec())
    }
}

#[test]
fn timeout_cancels_ordered_budget_waiters_before_persistent_restart() {
    let block_low = Arc::new(AtomicBool::new(true));
    let mut loader = DataLoader::builder(TimeoutRows {
        block_low: Arc::clone(&block_low),
    })
    .workers(2)
    .persistent_workers(true)
    .timeout(Duration::from_millis(50))
    .transform(rusttorch_data::FnTransform::new(
        |value: usize, _: &TaskContext| {
            Ok::<_, Infallible>(SizedRecord {
                value: value as u64,
                bytes: 1,
            })
        },
    ))
    .collate(FnCollate::new(|records: Vec<SizedRecord>| {
        Ok::<_, Infallible>(records[0].value)
    }))
    .prefetch_bytes(NonZeroUsize::MIN)
    .build()
    .unwrap();

    let mut first = loader.iter();
    let timed_out = first.next();
    assert!(
        matches!(timed_out, Some(Err(LoaderError::Timeout { batch: 0 }))),
        "unexpected first result: {timed_out:?}"
    );
    assert!(first.next().is_none());
    drop(first);

    block_low.store(false, Ordering::SeqCst);
    assert_eq!(
        loader.iter().collect::<Result<Vec<_>, _>>().unwrap(),
        [0, 1]
    );
}

#[test]
fn stream_byte_setter_order_tracks_final_transform_output() {
    struct UsizeFactory;

    impl WorkerSourceFactory for UsizeFactory {
        type Sample = usize;
        type Error = Infallible;
        type Source = std::vec::IntoIter<Result<WorkerRecord<usize>, Infallible>>;

        fn create(&self, _worker: WorkerContext) -> Result<Self::Source, Self::Error> {
            Ok(vec![Ok(WorkerRecord {
                sequence: Some(SequenceId::new(0)),
                logical_id: LogicalSampleId::new(0),
                sample: 3,
            })]
            .into_iter())
        }
    }

    let limit = NonZeroUsize::new(4).unwrap();
    let mut loader = StreamDataLoaderBuilder::new(UsizeFactory)
        .prefetch_bytes(limit)
        .transform(rusttorch_data::FnTransform::new(
            |value: usize, _: &TaskContext| {
                Ok::<_, Infallible>(SizedRecord {
                    value: 0,
                    bytes: value,
                })
            },
        ))
        .collate(VecCollate)
        .build()
        .unwrap();
    assert_eq!(loader.effective_prefetch_bytes(), Some(limit));
    assert_eq!(loader.iter().count(), 1);
}
