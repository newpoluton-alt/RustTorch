use std::{convert::Infallible, hint::black_box, num::NonZeroUsize, thread, time::Instant};

use rusttorch_data::{
    DataLoader, Dataset, FnCollate, FnTransform, ReplaySafeDataset, ReplaySafeMap,
    SequentialSampler, TaskContext, VecCollate,
};

const SAMPLES: usize = 4_096;
const BATCH_SIZE: usize = 64;
const PREFETCH: usize = 2;
const BASE_CHECKSUM: usize = SAMPLES * (SAMPLES - 1) / 2;
const WARMUP_RUNS: usize = 2;
const MEASURED_RUNS: usize = 7;

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
    for _ in 0..WARMUP_RUNS {
        assert_eq!(black_box(run()), expected);
    }
    let nanos = std::array::from_fn::<_, MEASURED_RUNS, _>(|_| {
        let started = Instant::now();
        let checksum = black_box(run());
        let elapsed = started.elapsed().as_nanos();
        assert_eq!(checksum, expected);
        elapsed
    });
    let mean = nanos.iter().map(|&value| value as f64).sum::<f64>() / MEASURED_RUNS as f64;
    let variance = nanos
        .iter()
        .map(|&value| (value as f64 - mean).powi(2))
        .sum::<f64>()
        / (MEASURED_RUNS - 1) as f64;
    println!(
        "{name}: raw_ns={nanos:?}; mean_ns={mean:.0}; sample_stddev_ns={:.0}; checksum={expected}; {configuration}",
        variance.sqrt()
    );
}

fn main() {
    let hardware = thread::available_parallelism().map_or(1, usize::from);
    let cpu = std::env::var("RUSTTORCH_BENCH_HARDWARE")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| "unspecified; set RUSTTORCH_BENCH_HARDWARE to the CPU/GPU model".into());
    let compiler = std::process::Command::new("rustc")
        .arg("--version")
        .output()
        .expect("rustc is available when running cargo bench");
    println!(
        "dataset=generated i64 indices; dataset_size={SAMPLES}; hardware={cpu}; \
         hardware_threads={hardware}; os={}; arch={}; compiler={}; \
         debug_assertions={}; tch=0.26.0; libtorch=2.13.0; batch_size={BATCH_SIZE}; \
         warmup_runs={WARMUP_RUNS}; measured_runs={MEASURED_RUNS}; \
         timing_scope=complete run including build/drop unless noted; \
         baseline=borrowed-baseline; timings are observations, not thresholds",
        std::env::consts::OS,
        std::env::consts::ARCH,
        String::from_utf8_lossy(&compiler.stdout).trim(),
        cfg!(debug_assertions),
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
                        "workers={workers}; queue_capacity={}; queue_unit=batches; byte_budget=disabled; ordering={}; prefetch_factor={prefetch}; transform=identity; collate=VecCollate; pinning=disabled; paired_baseline=serial-owned",
                        workers * prefetch,
                        if ordered { "ordered" } else { "completion" }
                    ),
                    BASE_CHECKSUM,
                    || {
                        DataLoader::builder(Rows(SAMPLES))
                            .batch_size(BATCH_SIZE)
                            .workers(workers)
                            .prefetch_factor(prefetch)
                            .ordered(ordered)
                            .collate(VecCollate)
                            .build()
                            .unwrap()
                            .iter()
                            .map(|batch| batch.unwrap().into_iter().sum::<i64>() as usize)
                            .sum()
                    },
                );
            }
        }
    }

    for bytes in [
        BATCH_SIZE * size_of::<i64>(),
        4 * BATCH_SIZE * size_of::<i64>(),
    ] {
        measure(
            "byte-bounded-workers",
            &format!(
                "workers=2; queue_capacity=4; queue_unit=batches; byte_budget={bytes}; ordering=ordered; prefetch_factor=2; transform=identity; collate=VecCollate; pinning=disabled; paired_baseline=worker-scaling(workers=2,prefetch=2,ordered)"
            ),
            BASE_CHECKSUM,
            || {
                DataLoader::builder(Rows(SAMPLES))
                    .batch_size(BATCH_SIZE)
                    .workers(2)
                    .prefetch_factor(PREFETCH)
                    .prefetch_bytes(NonZeroUsize::new(bytes).unwrap())
                    .collate(VecCollate)
                    .build()
                    .unwrap()
                    .iter()
                    .map(|batch| batch.unwrap().into_iter().sum::<i64>() as usize)
                    .sum()
            },
        );
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
        "workers=0; queue_capacity=0; byte_budget=disabled; ordering=ordered; prefetch_factor=n/a; transform=identity; collate=DefaultCollator; pinning=disabled; timing_scope=iteration only",
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
        "workers=0; queue_capacity=0; byte_budget=disabled; ordering=ordered; prefetch_factor=n/a; transform=identity; collate=DefaultCollator; pinning=automatic; paired_baseline=tensor-unpinned; timing_scope=iteration only",
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

    for checkpoint in [false, true] {
        measure(
            if checkpoint {
                "checkpoint-barrier"
            } else {
                "checkpoint-baseline"
            },
            &format!(
                "workers=2; queue_capacity=4; queue_unit=batches; byte_budget=disabled; ordering=ordered; prefetch_factor=2; transform=stateless identity; collate=VecCollate; pinning=disabled; checkpoint_every_batch={checkpoint}; paired_baseline=checkpoint-baseline"
            ),
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
                    if checkpoint {
                        black_box(iteration.checkpoint().unwrap());
                    }
                }
                checksum
            },
        );
    }
}
