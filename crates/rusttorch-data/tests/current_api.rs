use std::{cell::Cell, convert::Infallible};

use rusttorch_data::{
    AutoBatch, DataLoader, DataLoaderBuilder, Dataset, DefaultCollator, IdentityTransformFactory,
    LoaderIter, MemoryDisabled, NoWorkerInit, OwnedDataLoader, PinDisabled, SequentialSampler,
    SerialExecution, StreamDataLoader, StreamDataLoaderBuilder, StreamLoaderIter, VecCollate,
    WorkerContext, WorkerExecution, WorkerLoaderIter, WorkerRecord, WorkerSourceFactory,
};

struct Rows([i64; 3]);

impl Dataset for Rows {
    type Sample = i64;
    type Error = Infallible;

    fn len(&self) -> usize {
        self.0.len()
    }

    fn get(&self, index: usize) -> Result<i64, Infallible> {
        Ok(self.0[index])
    }
}

#[test]
fn direct_package_matches_the_existing_facade_loader() {
    let rows = Rows([2, 3, 5]);
    let batches = DataLoader::new(&rows, SequentialSampler::new(3), 2, false)
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    assert_eq!(batches, vec![vec![2, 3], vec![5]]);
}

#[test]
fn owned_loader_generic_defaults_remain_source_compatible() {
    let builder: DataLoaderBuilder<Rows, AutoBatch<SequentialSampler>, DefaultCollator> =
        DataLoader::builder(Rows([2, 3, 5]));
    let mut loader: OwnedDataLoader<Rows, AutoBatch<SequentialSampler>, DefaultCollator> =
        builder.build().expect("default configuration is valid");
    assert_eq!(loader.iter().count(), 3);

    let _: DataLoaderBuilder<
        Rows,
        AutoBatch<SequentialSampler>,
        DefaultCollator,
        IdentityTransformFactory,
        NoWorkerInit,
        SerialExecution,
    > = DataLoader::builder(Rows([2, 3, 5]));
    let mut serial: OwnedDataLoader<
        Rows,
        AutoBatch<SequentialSampler>,
        DefaultCollator,
        IdentityTransformFactory,
        NoWorkerInit,
        SerialExecution,
    > = DataLoader::builder(Rows([2, 3, 5])).build().unwrap();
    let _: LoaderIter<'_, Rows, AutoBatch<SequentialSampler>, DefaultCollator> = serial.iter();

    let mut workers: OwnedDataLoader<
        Rows,
        AutoBatch<SequentialSampler>,
        DefaultCollator,
        IdentityTransformFactory,
        NoWorkerInit,
        WorkerExecution,
    > = DataLoader::builder(Rows([2, 3, 5]))
        .workers(1)
        .build()
        .unwrap();
    let _: WorkerLoaderIter<
        '_,
        Rows,
        AutoBatch<SequentialSampler>,
        DefaultCollator,
        IdentityTransformFactory,
        NoWorkerInit,
    > = workers.iter();

    let _: StreamDataLoaderBuilder<
        CellFactory,
        VecCollate,
        IdentityTransformFactory,
        NoWorkerInit,
    > = StreamDataLoaderBuilder::new(CellFactory).collate(VecCollate);
    let mut stream: StreamDataLoader<
        CellFactory,
        VecCollate,
        IdentityTransformFactory,
        NoWorkerInit,
    > = StreamDataLoaderBuilder::new(CellFactory)
        .collate(VecCollate)
        .build()
        .unwrap();
    let _: StreamLoaderIter<'_, CellFactory, VecCollate, IdentityTransformFactory, NoWorkerInit> =
        stream.iter();
}

#[test]
fn trailing_builder_states_are_defaulted_and_can_be_named() {
    let _: DataLoaderBuilder<
        Rows,
        AutoBatch<SequentialSampler>,
        DefaultCollator,
        rusttorch_data::IdentityTransformFactory,
        rusttorch_data::NoWorkerInit,
        rusttorch_data::SerialExecution,
        MemoryDisabled,
        PinDisabled,
    > = DataLoader::builder(Rows([2, 3, 5]));
    let _: StreamDataLoaderBuilder<
        CellFactory,
        VecCollate,
        rusttorch_data::IdentityTransformFactory,
        rusttorch_data::NoWorkerInit,
        MemoryDisabled,
        PinDisabled,
    > = StreamDataLoaderBuilder::new(CellFactory).collate(VecCollate);

    let _: OwnedDataLoader<
        Rows,
        AutoBatch<SequentialSampler>,
        DefaultCollator,
        rusttorch_data::IdentityTransformFactory,
        rusttorch_data::NoWorkerInit,
        rusttorch_data::SerialExecution,
        MemoryDisabled,
        PinDisabled,
    > = DataLoader::builder(Rows([2, 3, 5])).build().unwrap();
    let _: StreamDataLoader<
        CellFactory,
        VecCollate,
        rusttorch_data::IdentityTransformFactory,
        rusttorch_data::NoWorkerInit,
        MemoryDisabled,
        PinDisabled,
    > = StreamDataLoaderBuilder::new(CellFactory)
        .collate(VecCollate)
        .build()
        .unwrap();
}

struct CellFactory;

impl WorkerSourceFactory for CellFactory {
    type Sample = Cell<u8>;
    type Error = Infallible;
    type Source = std::iter::Empty<Result<WorkerRecord<Cell<u8>>, Infallible>>;

    fn create(&self, _worker: WorkerContext) -> Result<Self::Source, Self::Error> {
        Ok(std::iter::empty())
    }
}

fn assert_sync<T: Sync>() {}

#[test]
fn stream_loader_remains_sync_for_send_but_not_sync_output() {
    assert_sync::<StreamDataLoader<CellFactory, VecCollate>>();
}
