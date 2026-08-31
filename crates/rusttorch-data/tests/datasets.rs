use std::{
    convert::Infallible,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

use rusttorch_core::{RustTorchError, Tensor};
use rusttorch_data::{
    ConcatDataset, DataLoader, Dataset, SequentialSampler, SplitLength, StackDataset, Subset,
    TensorDataset, chain_datasets, random_split,
};

#[derive(Clone)]
struct IntDataset(Vec<i64>);

impl IntDataset {
    fn new(values: Vec<i64>) -> Self {
        Self(values)
    }
}

impl Dataset for IntDataset {
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
fn tensor_dataset_returns_one_row_from_each_tensor() {
    let tensors = TensorDataset::new(vec![
        Tensor::from_slice(&[1_i64, 2, 3]),
        Tensor::from_slice(&[10_i64, 20, 30]),
    ])
    .expect("matching first dimensions are valid");

    assert_eq!(tensors.len(), 3);
    let row = tensors.get(1).expect("index one exists");
    assert_eq!(row[0].int64_value(&[]), 2);
    assert_eq!(row[1].int64_value(&[]), 20);
}

#[test]
fn tensor_dataset_rejects_mismatched_first_dimensions() {
    assert!(matches!(
        TensorDataset::new(vec![
            Tensor::from_slice(&[1_i64, 2]),
            Tensor::from_slice(&[10_i64, 20, 30]),
        ]),
        Err(RustTorchError::InvalidDimensions { .. })
    ));
}

#[test]
fn tensor_dataset_get_returns_storage_sharing_views() {
    let dataset =
        TensorDataset::new(vec![Tensor::from_slice(&[1_i64, 2, 3])]).expect("one tensor is valid");
    let mut row = dataset.get(1).expect("index one exists").remove(0);
    let _ = row.fill_(99);

    assert_eq!(
        dataset.get(1).expect("index one exists")[0].int64_value(&[]),
        99
    );
}

#[test]
fn tensor_dataset_get_batch_returns_storage_sharing_views() {
    let dataset =
        TensorDataset::new(vec![Tensor::from_slice(&[1_i64, 2, 3])]).expect("one tensor is valid");
    let mut row = dataset
        .get_batch(&[2])
        .expect("index two exists")
        .remove(0)
        .remove(0);
    let _ = row.fill_(77);

    assert_eq!(
        dataset.get(2).expect("index two exists")[0].int64_value(&[]),
        77
    );
}

#[test]
fn stack_dataset_checks_lengths_and_returns_tuples() {
    let dataset = StackDataset::new((IntDataset::new(vec![1, 2]), IntDataset::new(vec![10, 20])))
        .expect("equal lengths are valid");

    assert_eq!(dataset.len(), 2);
    assert_eq!(dataset.get(1), Ok((2, 20)));
    assert!(matches!(
        StackDataset::new((IntDataset::new(vec![1]), IntDataset::new(vec![10, 20]),)),
        Err(RustTorchError::InvalidDimensions { .. })
    ));
}

#[test]
fn stack_dataset_supports_tuple_arities_two_through_eight() {
    let one = IntDataset::new(vec![1]);
    assert_eq!(
        StackDataset::new((one.clone(), one.clone()))
            .unwrap()
            .get(0),
        Ok((1, 1))
    );
    assert_eq!(
        StackDataset::new((one.clone(), one.clone(), one.clone()))
            .unwrap()
            .len(),
        1
    );
    assert_eq!(
        StackDataset::new((one.clone(), one.clone(), one.clone(), one.clone()))
            .unwrap()
            .len(),
        1
    );
    assert_eq!(
        StackDataset::new((
            one.clone(),
            one.clone(),
            one.clone(),
            one.clone(),
            one.clone()
        ))
        .unwrap()
        .len(),
        1
    );
    assert_eq!(
        StackDataset::new((
            one.clone(),
            one.clone(),
            one.clone(),
            one.clone(),
            one.clone(),
            one.clone(),
        ))
        .unwrap()
        .len(),
        1
    );
    assert_eq!(
        StackDataset::new((
            one.clone(),
            one.clone(),
            one.clone(),
            one.clone(),
            one.clone(),
            one.clone(),
            one.clone(),
        ))
        .unwrap()
        .len(),
        1
    );
    assert_eq!(
        StackDataset::new((
            one.clone(),
            one.clone(),
            one.clone(),
            one.clone(),
            one.clone(),
            one.clone(),
            one.clone(),
            one,
        ))
        .unwrap()
        .len(),
        1
    );
}

#[test]
fn concat_dataset_uses_global_indices_and_rejects_empty_input() {
    let dataset = ConcatDataset::new(vec![IntDataset::new(vec![1, 2]), IntDataset::new(vec![3])])
        .expect("nonempty datasets are valid");

    assert_eq!(dataset.len(), 3);
    assert_eq!(dataset.get(0), Ok(1));
    assert_eq!(dataset.get(2), Ok(3));
    assert!(matches!(
        ConcatDataset::<IntDataset>::new(vec![]),
        Err(RustTorchError::InvalidConfiguration {
            field: "datasets",
            ..
        })
    ));
}

#[test]
fn subset_validates_source_indices_and_arc_delegates_dataset() {
    let rows = Arc::new(IntDataset::new(vec![10, 20, 30, 40]));
    let subset = Subset::new(Arc::clone(&rows), vec![3, 1]).expect("indices are in range");

    assert_eq!(subset.len(), 2);
    assert_eq!(subset.get(0), Ok(40));
    assert!(matches!(
        Subset::new(Arc::clone(&rows), vec![4]),
        Err(RustTorchError::InvalidConfiguration {
            field: "indices",
            ..
        })
    ));
}

fn split_values(parts: &[Subset<Arc<IntDataset>>]) -> Vec<Vec<i64>> {
    parts
        .iter()
        .map(|part| {
            part.samples()
                .collect::<Result<Vec<_>, _>>()
                .expect("the source is infallible")
        })
        .collect()
}

#[test]
fn random_split_requires_integer_lengths_to_sum_to_dataset_length() {
    let rows = Arc::new(IntDataset::new((0..5).collect()));

    assert!(
        random_split(
            Arc::clone(&rows),
            &[SplitLength::Count(2), SplitLength::Count(3)],
            7
        )
        .is_ok()
    );
    assert!(matches!(
        random_split(rows, &[SplitLength::Count(2), SplitLength::Count(2)], 7),
        Err(RustTorchError::InvalidConfiguration {
            field: "lengths",
            ..
        })
    ));
}

#[test]
fn random_split_floors_fractions_and_distributes_remainder_round_robin() {
    let rows = Arc::new(IntDataset::new((0..11).collect()));
    let parts = random_split(
        rows,
        &[
            SplitLength::Fraction(0.3),
            SplitLength::Fraction(0.3),
            SplitLength::Fraction(0.4),
        ],
        7,
    )
    .expect("fractions sum to one");

    assert_eq!(
        parts.iter().map(Dataset::len).collect::<Vec<_>>(),
        vec![4, 3, 4]
    );
}

#[test]
fn random_split_rejects_fraction_totals_above_one_within_tolerance() {
    let rows = Arc::new(IntDataset::new((0..5).collect()));

    assert!(matches!(
        random_split(
            rows,
            &[
                SplitLength::Fraction(0.500_000_000_4),
                SplitLength::Fraction(0.500_000_000_4),
            ],
            7,
        ),
        Err(RustTorchError::InvalidConfiguration {
            field: "lengths",
            ..
        })
    ));
}

struct HugeDataset;

impl Dataset for HugeDataset {
    type Sample = ();
    type Error = Infallible;

    fn len(&self) -> usize {
        1_000_000_000_000
    }

    fn get(&self, _index: usize) -> Result<Self::Sample, Self::Error> {
        unreachable!("invalid split lengths must fail before sampling")
    }
}

#[test]
fn random_split_rejects_fraction_floors_larger_than_the_dataset() {
    assert!(matches!(
        random_split(
            Arc::new(HugeDataset),
            &[
                SplitLength::Fraction(0.500_000_000_4),
                SplitLength::Fraction(0.500_000_000_4),
            ],
            7,
        ),
        Err(RustTorchError::InvalidConfiguration {
            field: "lengths",
            ..
        })
    ));
}

#[test]
fn random_split_is_deterministic_and_allows_zero_length_parts() {
    let rows = Arc::new(IntDataset::new((0..6).collect()));
    let lengths = [
        SplitLength::Count(0),
        SplitLength::Count(2),
        SplitLength::Count(4),
    ];
    let first = random_split(Arc::clone(&rows), &lengths, 42).expect("lengths sum to six");
    let second = random_split(rows, &lengths, 42).expect("lengths sum to six");

    assert_eq!(first[0].len(), 0);
    assert_eq!(split_values(&first), split_values(&second));
    let mut flattened = split_values(&first)
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
    flattened.sort_unstable();
    assert_eq!(flattened, vec![0, 1, 2, 3, 4, 5]);
}

#[test]
fn chain_datasets_is_the_standard_flattened_iterator() {
    assert_eq!(
        chain_datasets([vec![1, 2], vec![], vec![3]]).collect::<Vec<_>>(),
        vec![1, 2, 3]
    );
}

struct BatchedDataset {
    calls: AtomicUsize,
    wrong_cardinality: bool,
}

impl Dataset for BatchedDataset {
    type Sample = usize;
    type Error = Infallible;

    fn len(&self) -> usize {
        4
    }

    fn get(&self, _index: usize) -> Result<Self::Sample, Self::Error> {
        panic!("the loader must use get_batch")
    }

    fn get_batch(&self, indices: &[usize]) -> Result<Vec<Self::Sample>, Self::Error> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        if self.wrong_cardinality {
            Ok(vec![])
        } else {
            Ok(indices.to_vec())
        }
    }
}

#[test]
fn loader_uses_one_batched_fetch_per_index_batch() {
    let dataset = BatchedDataset {
        calls: AtomicUsize::new(0),
        wrong_cardinality: false,
    };
    let batches = DataLoader::new(&dataset, SequentialSampler::new(4), 4, false)
        .expect("batch size is nonzero")
        .collect::<Result<Vec<_>, _>>()
        .expect("the dataset is infallible");

    assert_eq!(batches, vec![vec![0, 1, 2, 3]]);
    assert_eq!(dataset.calls.load(Ordering::SeqCst), 1);
}

#[derive(Debug, PartialEq, Eq)]
struct TailError;

struct FailingTailDataset;

impl Dataset for FailingTailDataset {
    type Sample = usize;
    type Error = TailError;

    fn len(&self) -> usize {
        3
    }

    fn get(&self, index: usize) -> Result<Self::Sample, Self::Error> {
        if index == 2 {
            Err(TailError)
        } else {
            Ok(index)
        }
    }
}

#[test]
fn loader_preserves_errors_from_a_short_dropped_tail() {
    let mut loader = DataLoader::new(&FailingTailDataset, SequentialSampler::new(3), 2, true)
        .expect("batch size is nonzero");

    assert_eq!(loader.next(), Some(Ok(vec![0, 1])));
    assert_eq!(loader.next(), Some(Err(TailError)));
    assert_eq!(loader.next(), None);
}

#[test]
fn loader_rejects_wrong_batched_fetch_cardinality_before_collation() {
    let dataset = BatchedDataset {
        calls: AtomicUsize::new(0),
        wrong_cardinality: true,
    };
    let collations = AtomicUsize::new(0);
    let mut loader =
        DataLoader::with_collate(&dataset, SequentialSampler::new(4), 4, false, |samples| {
            collations.fetch_add(1, Ordering::SeqCst);
            Ok::<_, Infallible>(samples)
        })
        .expect("batch size is nonzero");

    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| loader.next()));

    assert!(panic.is_err());
    assert_eq!(dataset.calls.load(Ordering::SeqCst), 1);
    assert_eq!(collations.load(Ordering::SeqCst), 0);
}
