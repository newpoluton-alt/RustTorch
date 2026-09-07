use std::{convert::Infallible, hint::black_box, thread, time::Instant};

use rusttorch_data::{
    DataLoader, Dataset, FnCollate, FnTransform, FnTransformFactory, ReplaySafeDataset,
    ReplaySafeMap, SequentialSampler, TaskContext, VecCollate, WorkerContext,
};

const SAMPLES: usize = 4_096;
const BATCH_SIZE: usize = 64;
const PREFETCH: usize = 2;
const BASE_CHECKSUM: usize = SAMPLES * (SAMPLES - 1) / 2;

struct Rows(usize);

impl Dataset for Rows {
    type Sample = i64;
    type Error = Infallible;

    fn len(&self) -> usize {
        self.0
    }

    fn get(&self, index: usize) -> Result<Self::Sample, Self::Error> {
        Ok(index as i64)
    }
}

impl ReplaySafeDataset for Rows {}

fn measure(name: &str, configuration: &str, expected: usize, mut run: impl FnMut() -> usize) {
    let started = Instant::now();
    let checksum = black_box(run());
    assert_eq!(checksum, expected);
    println!(
        "{name}: {:?}; checksum={checksum}; {configuration}",
        started.elapsed()
    );
}

fn main() {
    let hardware = thread::available_parallelism().map_or(1, usize::from);
    println!(
        "dataset_size={SAMPLES}; hardware_threads={hardware}; batch_size={BATCH_SIZE}; \
         baseline=borrowed sequential DataLoader; timings are observations, not thresholds"
    );

    measure(
        "borrowed-baseline",
        "workers=0; queue_capacity=0; byte_budget=disabled; ordering=ordered; prefetch_factor=n/a; transform=none; collate=Vec; pinning=disabled",
        BASE_CHECKSUM,
        || {
            let rows = Rows(SAMPLES);
            DataLoader::new(&rows, SequentialSampler::new(SAMPLES), BATCH_SIZE, false)
                .unwrap()
                .map(|batch| batch.unwrap().into_iter().sum::<i64>() as usize)
                .sum()
        },
    );

    measure(
        "serial-owned",
        "workers=0; queue_capacity=0; byte_budget=disabled; ordering=ordered; prefetch_factor=n/a; transform=identity; collate=VecCollate; pinning=disabled",
        BASE_CHECKSUM,
        || {
            DataLoader::builder(Rows(SAMPLES))
                .batch_size(BATCH_SIZE)
                .collate(VecCollate)
                .build()
                .unwrap()
                .iter()
                .map(|batch| batch.unwrap().into_iter().sum::<i64>() as usize)
                .sum()
        },
    );

    measure(
        "transform-add-one",
        "workers=0; queue_capacity=0; byte_budget=disabled; ordering=ordered; prefetch_factor=n/a; transform=add-one; collate=VecCollate; pinning=disabled; paired_baseline=serial-owned",
        BASE_CHECKSUM + SAMPLES,
        || {
            DataLoader::builder(Rows(SAMPLES))
                .batch_size(BATCH_SIZE)
                .transform(FnTransform::new(|value: i64, _: &TaskContext| {
                    Ok::<_, Infallible>(value + 1)
                }))
                .collate(VecCollate)
                .build()
                .unwrap()
                .iter()
                .map(|batch| batch.unwrap().into_iter().sum::<i64>() as usize)
                .sum()
        },
    );

    measure(
        "custom-collate-sum",
        "workers=0; queue_capacity=0; byte_budget=disabled; ordering=ordered; prefetch_factor=n/a; transform=identity; collate=sum; pinning=disabled; paired_baseline=serial-owned",
        BASE_CHECKSUM,
        || {
            DataLoader::builder(Rows(SAMPLES))
                .batch_size(BATCH_SIZE)
                .collate(FnCollate::new(|values: Vec<i64>| {
                    Ok::<_, Infallible>(values.into_iter().sum::<i64>())
                }))
                .build()
                .unwrap()
                .iter()
                .map(|batch| batch.unwrap() as usize)
                .sum()
        },
    );

    for workers in [1, 2, 4] {
        for prefetch in [1, PREFETCH, 4] {
            for ordered in [true, false] {
                measure(
                    "worker-scaling",
                    &format!(
                        "workers={workers}; queue_capacity={}; byte_budget=disabled; ordering={}; prefetch_factor={prefetch}; transform=add-one; collate=sum; pinning=disabled",
                        workers * prefetch,
                        if ordered { "ordered" } else { "completion" }
                    ),
                    BASE_CHECKSUM + SAMPLES,
                    || {
                        DataLoader::builder(Rows(SAMPLES))
                            .batch_size(BATCH_SIZE)
                            .workers(workers)
                            .prefetch_factor(prefetch)
                            .ordered(ordered)
                            .transform_factory(FnTransformFactory::new(
                                |_: Option<&WorkerContext>| {
                                    Ok::<_, Infallible>(FnTransform::new(
                                        |value: i64, _: &TaskContext| {
                                            Ok::<_, Infallible>(value + 1)
                                        },
                                    ))
                                },
                            ))
                            .collate(FnCollate::new(|values: Vec<i64>| {
                                Ok::<_, Infallible>(values.into_iter().sum::<i64>())
                            }))
                            .build()
                            .unwrap()
                            .iter()
                            .map(|batch| batch.unwrap() as usize)
                            .sum()
                    },
                );
            }
        }
    }

    let mut unpinned = DataLoader::builder(Rows(SAMPLES))
        .batch_size(BATCH_SIZE)
        .build()
        .unwrap();
    let mut pinned = DataLoader::builder(Rows(SAMPLES))
        .batch_size(BATCH_SIZE)
        .pin_memory()
        .build()
        .unwrap();
    println!(
        "pin_comparison: unpinned_status={:?}; auto_status={:?}; CUDA pinning timing is unavailable when auto_status=DisabledNoAccelerator",
        unpinned.pin_memory_status(),
        pinned.pin_memory_status(),
    );
    measure(
        "tensor-unpinned",
        "workers=0; queue_capacity=0; byte_budget=disabled; ordering=ordered; prefetch_factor=n/a; transform=identity; collate=DefaultCollator; pinning=disabled",
        BASE_CHECKSUM,
        || {
            unpinned
                .iter()
                .map(|batch| {
                    batch
                        .unwrap()
                        .sum(rusttorch_core::Kind::Int64)
                        .int64_value(&[]) as usize
                })
                .sum()
        },
    );
    measure(
        "tensor-auto-pin",
        "workers=0; queue_capacity=0; byte_budget=disabled; ordering=ordered; prefetch_factor=n/a; transform=identity; collate=DefaultCollator; pinning=automatic; paired_baseline=tensor-unpinned",
        BASE_CHECKSUM,
        || {
            pinned
                .iter()
                .map(|batch| {
                    batch
                        .unwrap()
                        .sum(rusttorch_core::Kind::Int64)
                        .int64_value(&[]) as usize
                })
                .sum()
        },
    );

    measure(
        "checkpoint-barrier",
        "workers=2; queue_capacity=4; byte_budget=disabled; ordering=ordered; prefetch_factor=2; transform=stateless identity; collate=VecCollate; pinning=disabled; checkpoint=every batch",
        BASE_CHECKSUM,
        || {
            let mut loader = DataLoader::builder(ReplaySafeMap::new(Rows(SAMPLES)))
                .batch_size(BATCH_SIZE)
                .workers(2)
                .prefetch_factor(PREFETCH)
                .collate(VecCollate)
                .checkpoint_stateless()
                .dataset_identity("benchmark-rows-v1".to_owned())
                .build()
                .unwrap();
            let mut iteration = loader.iter();
            let mut checksum = 0;
            while let Some(batch) = iteration.next() {
                checksum += batch.unwrap().into_iter().sum::<i64>() as usize;
                black_box(iteration.checkpoint().unwrap());
            }
            checksum
        },
    );
}
