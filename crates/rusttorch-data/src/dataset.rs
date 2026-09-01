use std::sync::Arc;

use rand::{SeedableRng, seq::SliceRandom};
use rand_chacha::ChaCha12Rng;
use rusttorch_core::{Result, RustTorchError, Tensor};

use crate::{Checkpointable, Dataset, DatasetCheckpoint, ReplaySafeDataset, WorkerContext};

/// A map-style dataset backed by tensors with a shared first dimension.
///
/// Indexed samples are ordinary LibTorch row views and therefore share
/// storage with the tensors supplied to [`TensorDataset::new`].
pub struct TensorDataset {
    tensors: Vec<Tensor>,
    len: usize,
}

impl TensorDataset {
    /// Creates a tensor dataset whose tensors have the same non-scalar first dimension.
    ///
    /// # Errors
    ///
    /// Returns [`RustTorchError::InvalidConfiguration`] for no tensors and
    /// [`RustTorchError::InvalidDimensions`] for a scalar tensor or mismatched
    /// first dimensions.
    pub fn new(tensors: Vec<Tensor>) -> Result<Self> {
        let first = tensors
            .first()
            .ok_or_else(|| RustTorchError::InvalidConfiguration {
                field: "tensors",
                reason: "must not be empty".to_owned(),
            })?;
        let first_size = first.size();
        let len = first_size
            .first()
            .copied()
            .ok_or_else(|| RustTorchError::InvalidDimensions {
                context: "TensorDataset tensor 0".to_owned(),
                expected: "at least one dimension".to_owned(),
                actual: format!("{first_size:?}"),
            })?;

        for (index, tensor) in tensors.iter().enumerate().skip(1) {
            let size = tensor.size();
            if size.first().copied() != Some(len) {
                return Err(RustTorchError::InvalidDimensions {
                    context: format!("TensorDataset tensor {index}"),
                    expected: format!("first dimension {len}"),
                    actual: format!("{size:?}"),
                });
            }
        }

        Ok(Self {
            tensors,
            len: usize::try_from(len).map_err(|_| RustTorchError::InvalidDimensions {
                context: "TensorDataset tensor 0".to_owned(),
                expected: "a nonnegative first dimension representable as usize".to_owned(),
                actual: len.to_string(),
            })?,
        })
    }

    /// Deep-copies the backing tensors for exact replay-safe loading.
    ///
    /// The returned dataset also deep-copies every fetched row, so callers
    /// cannot mutate its private backing storage through a sample view.
    ///
    /// # Errors
    ///
    /// Returns the preserved LibTorch backend error when allocation or copy
    /// fails.
    pub fn into_replay_safe(self) -> Result<ReplaySafeTensorDataset> {
        let tensors = self
            .tensors
            .iter()
            .map(deep_copy_tensor)
            .collect::<Result<Vec<_>>>()?;
        Ok(ReplaySafeTensorDataset {
            tensors,
            len: self.len,
        })
    }
}

impl Dataset for TensorDataset {
    type Sample = Vec<Tensor>;
    type Error = RustTorchError;

    fn len(&self) -> usize {
        self.len
    }

    fn get(&self, index: usize) -> std::result::Result<Self::Sample, Self::Error> {
        let index = i64::try_from(index).map_err(|_| RustTorchError::InvalidConfiguration {
            field: "index",
            reason: "must fit in LibTorch's signed index range".to_owned(),
        })?;
        self.tensors
            .iter()
            .map(|tensor| tensor.f_get(index).map_err(RustTorchError::from))
            .collect()
    }
}

/// Tensor dataset with private backing storage and independently owned rows.
pub struct ReplaySafeTensorDataset {
    tensors: Vec<Tensor>,
    len: usize,
}

impl Dataset for ReplaySafeTensorDataset {
    type Sample = Vec<Tensor>;
    type Error = RustTorchError;

    fn len(&self) -> usize {
        self.len
    }

    fn get(&self, index: usize) -> std::result::Result<Self::Sample, Self::Error> {
        let index = i64::try_from(index).map_err(|_| RustTorchError::InvalidConfiguration {
            field: "index",
            reason: "must fit in LibTorch's signed index range".to_owned(),
        })?;
        self.tensors
            .iter()
            .map(|tensor| {
                let row = tensor.f_get(index).map_err(RustTorchError::from)?;
                deep_copy_tensor(&row)
            })
            .collect()
    }
}

impl ReplaySafeDataset for ReplaySafeTensorDataset {}

impl DatasetCheckpoint for ReplaySafeTensorDataset {
    type State = ();

    fn snapshot_dataset(&self) -> Self::State {}

    fn validate_dataset_state(&self, _state: &Self::State) -> Result<()> {
        Ok(())
    }

    fn restore_dataset_validated(&mut self, _state: &Self::State) {}
}

fn deep_copy_tensor(tensor: &Tensor) -> Result<Tensor> {
    let mut copied = tensor.f_empty_like().map_err(RustTorchError::from)?;
    copied.f_copy_(tensor).map_err(RustTorchError::from)?;
    Ok(copied)
}

/// Explicit exact-checkpoint adapter for a replay-safe map dataset.
#[derive(Clone, Copy, Debug, Default)]
pub struct ReplaySafeMap<D> {
    dataset: D,
}

impl<D> ReplaySafeMap<D> {
    /// Wraps a dataset that explicitly implements [`ReplaySafeDataset`].
    pub fn new(dataset: D) -> Self {
        Self { dataset }
    }

    /// Returns the wrapped dataset.
    pub fn into_inner(self) -> D {
        self.dataset
    }
}

impl<D> Dataset for ReplaySafeMap<D>
where
    D: ReplaySafeDataset,
{
    type Sample = D::Sample;
    type Error = D::Error;

    fn len(&self) -> usize {
        self.dataset.len()
    }

    fn get(&self, index: usize) -> std::result::Result<Self::Sample, Self::Error> {
        self.dataset.get(index)
    }

    fn get_batch(&self, indices: &[usize]) -> std::result::Result<Vec<Self::Sample>, Self::Error> {
        self.dataset.get_batch(indices)
    }

    fn get_batch_with_context(
        &self,
        indices: &[usize],
        context: &WorkerContext,
    ) -> std::result::Result<Vec<Self::Sample>, Self::Error> {
        self.dataset.get_batch_with_context(indices, context)
    }
}

impl<D> ReplaySafeDataset for ReplaySafeMap<D> where D: ReplaySafeDataset {}

impl<D> DatasetCheckpoint for ReplaySafeMap<D>
where
    D: ReplaySafeDataset,
{
    type State = ();

    fn snapshot_dataset(&self) -> Self::State {}

    fn validate_dataset_state(&self, _state: &Self::State) -> Result<()> {
        Ok(())
    }

    fn restore_dataset_validated(&mut self, _state: &Self::State) {}
}

/// Explicit serial exact-checkpoint adapter for a transactional dataset.
#[derive(Clone, Copy, Debug, Default)]
pub struct TransactionalMap<D> {
    dataset: D,
}

impl<D> TransactionalMap<D> {
    /// Wraps a dataset that implements [`Checkpointable`].
    pub fn new(dataset: D) -> Self {
        Self { dataset }
    }

    /// Returns the wrapped dataset.
    pub fn into_inner(self) -> D {
        self.dataset
    }
}

impl<D> Dataset for TransactionalMap<D>
where
    D: Dataset + Checkpointable,
{
    type Sample = D::Sample;
    type Error = D::Error;

    fn len(&self) -> usize {
        self.dataset.len()
    }

    fn get(&self, index: usize) -> std::result::Result<Self::Sample, Self::Error> {
        self.dataset.get(index)
    }

    fn get_batch(&self, indices: &[usize]) -> std::result::Result<Vec<Self::Sample>, Self::Error> {
        self.dataset.get_batch(indices)
    }

    fn get_batch_with_context(
        &self,
        indices: &[usize],
        context: &WorkerContext,
    ) -> std::result::Result<Vec<Self::Sample>, Self::Error> {
        self.dataset.get_batch_with_context(indices, context)
    }
}

impl<D> DatasetCheckpoint for TransactionalMap<D>
where
    D: Dataset + Checkpointable,
{
    type State = D::State;

    fn snapshot_dataset(&self) -> Self::State {
        self.dataset.save_state()
    }

    fn validate_dataset_state(&self, state: &Self::State) -> Result<()> {
        self.dataset.validate_state(state)
    }

    fn restore_dataset_validated(&mut self, state: &Self::State) {
        self.dataset.load_validated(state);
    }
}

/// A dataset that returns aligned samples from a tuple of child datasets.
pub struct StackDataset<T> {
    datasets: T,
    len: usize,
}

mod sealed {
    pub trait Sealed {}
}

/// A sealed tuple shape accepted by [`StackDataset`].
///
/// RustTorch implements this trait for dataset tuples with arities two
/// through eight whose children share one error type.
pub trait StackTuple: sealed::Sealed {
    /// The tuple of child samples.
    type Sample;
    /// The error shared by every child dataset.
    type Error;
    /// Returns the child lengths for constructor validation.
    #[doc(hidden)]
    fn lengths(&self) -> Vec<usize>;
    /// Loads the aligned child samples at `index`.
    #[doc(hidden)]
    fn get(&self, index: usize) -> std::result::Result<Self::Sample, Self::Error>;
}

impl<T> StackDataset<T>
where
    T: StackTuple,
{
    /// Creates a stack from child datasets with equal lengths.
    ///
    /// # Errors
    ///
    /// Returns [`RustTorchError::InvalidDimensions`] when a child length
    /// differs from the first child's length.
    pub fn new(datasets: T) -> Result<Self> {
        let lengths = datasets.lengths();
        let len = lengths[0];
        if lengths.iter().any(|&child_len| child_len != len) {
            return Err(RustTorchError::InvalidDimensions {
                context: "StackDataset child lengths".to_owned(),
                expected: format!("all lengths equal to {len}"),
                actual: format!("{lengths:?}"),
            });
        }
        Ok(Self { datasets, len })
    }
}

impl<T> Dataset for StackDataset<T>
where
    T: StackTuple,
{
    type Sample = T::Sample;
    type Error = T::Error;

    fn len(&self) -> usize {
        self.len
    }

    fn get(&self, index: usize) -> std::result::Result<Self::Sample, Self::Error> {
        StackTuple::get(&self.datasets, index)
    }
}

macro_rules! impl_stack_dataset {
    ($First:ident:$first:tt, $($Dataset:ident:$field:tt),+) => {
        impl<$First, $($Dataset),+> sealed::Sealed for ($First, $($Dataset,)+)
        where
            $First: Dataset,
            $($Dataset: Dataset<Error = $First::Error>),+
        {}

        impl<$First, $($Dataset),+> StackTuple for ($First, $($Dataset,)+)
        where
            $First: Dataset,
            $($Dataset: Dataset<Error = $First::Error>),+
        {
            type Sample = ($First::Sample, $($Dataset::Sample,)+);
            type Error = $First::Error;

            fn lengths(&self) -> Vec<usize> {
                vec![self.$first.len(), $(self.$field.len()),+]
            }

            fn get(&self, index: usize) -> std::result::Result<Self::Sample, Self::Error> {
                Ok((self.$first.get(index)?, $(self.$field.get(index)?,)+))
            }
        }
    };
}

impl_stack_dataset!(A:0, B:1);
impl_stack_dataset!(A:0, B:1, C:2);
impl_stack_dataset!(A:0, B:1, C:2, D:3);
impl_stack_dataset!(A:0, B:1, C:2, D:3, E:4);
impl_stack_dataset!(A:0, B:1, C:2, D:3, E:4, F:5);
impl_stack_dataset!(A:0, B:1, C:2, D:3, E:4, F:5, G:6);
impl_stack_dataset!(A:0, B:1, C:2, D:3, E:4, F:5, G:6, H:7);

trait ReplaySafeStackTuple {}

macro_rules! impl_replay_safe_stack_tuple {
    ($($type:ident),+ $(,)?) => {
        impl<$($type),+> ReplaySafeStackTuple for ($($type,)+)
        where
            $($type: ReplaySafeDataset),+
        {}
    };
}

impl_replay_safe_stack_tuple!(A, B);
impl_replay_safe_stack_tuple!(A, B, C);
impl_replay_safe_stack_tuple!(A, B, C, D);
impl_replay_safe_stack_tuple!(A, B, C, D, E);
impl_replay_safe_stack_tuple!(A, B, C, D, E, F);
impl_replay_safe_stack_tuple!(A, B, C, D, E, F, G);
impl_replay_safe_stack_tuple!(A, B, C, D, E, F, G, H);

impl<T> ReplaySafeDataset for StackDataset<T> where T: StackTuple + ReplaySafeStackTuple {}

/// A dataset that maps one global index across consecutive child datasets.
pub struct ConcatDataset<D> {
    datasets: Vec<D>,
    cumulative_sizes: Vec<usize>,
}

impl<D> ConcatDataset<D>
where
    D: Dataset,
{
    /// Creates a concatenation of one or more child datasets.
    ///
    /// # Errors
    ///
    /// Returns [`RustTorchError::InvalidConfiguration`] when `datasets` is
    /// empty or their combined length overflows [`usize`].
    pub fn new(datasets: Vec<D>) -> Result<Self> {
        if datasets.is_empty() {
            return Err(RustTorchError::InvalidConfiguration {
                field: "datasets",
                reason: "must not be empty".to_owned(),
            });
        }
        let mut total = 0usize;
        let cumulative_sizes = datasets
            .iter()
            .map(|dataset| {
                total = total.checked_add(dataset.len()).ok_or_else(|| {
                    RustTorchError::InvalidConfiguration {
                        field: "datasets",
                        reason: "combined length overflows usize".to_owned(),
                    }
                })?;
                Ok(total)
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            datasets,
            cumulative_sizes,
        })
    }
}

impl<D> Dataset for ConcatDataset<D>
where
    D: Dataset,
{
    type Sample = D::Sample;
    type Error = D::Error;

    fn len(&self) -> usize {
        self.cumulative_sizes.last().copied().unwrap_or(0)
    }

    fn get(&self, index: usize) -> std::result::Result<Self::Sample, Self::Error> {
        let dataset_index = self
            .cumulative_sizes
            .partition_point(|&cumulative_size| cumulative_size <= index);
        let previous_size = dataset_index
            .checked_sub(1)
            .map_or(0, |previous| self.cumulative_sizes[previous]);
        self.datasets[dataset_index].get(index - previous_size)
    }
}

impl<D> ReplaySafeDataset for ConcatDataset<D> where D: ReplaySafeDataset {}

/// A dataset containing selected indices from another dataset.
pub struct Subset<D> {
    dataset: D,
    indices: Vec<usize>,
}

impl<D> Subset<D>
where
    D: Dataset,
{
    /// Creates a subset after validating every source index.
    ///
    /// # Errors
    ///
    /// Returns [`RustTorchError::InvalidConfiguration`] when an index is not
    /// less than the source dataset length.
    pub fn new(dataset: D, indices: Vec<usize>) -> Result<Self> {
        if let Some(&index) = indices.iter().find(|&&index| index >= dataset.len()) {
            return Err(RustTorchError::InvalidConfiguration {
                field: "indices",
                reason: format!("index {index} is out of range for length {}", dataset.len()),
            });
        }
        Ok(Self { dataset, indices })
    }
}

impl<D> Dataset for Subset<D>
where
    D: Dataset,
{
    type Sample = D::Sample;
    type Error = D::Error;

    fn len(&self) -> usize {
        self.indices.len()
    }

    fn get(&self, index: usize) -> std::result::Result<Self::Sample, Self::Error> {
        self.dataset.get(self.indices[index])
    }
}

impl<D> ReplaySafeDataset for Subset<D> where D: ReplaySafeDataset {}

/// One requested output length for [`random_split`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SplitLength {
    /// An exact sample count.
    Count(usize),
    /// A fraction of the source length; all fractions must sum to one.
    Fraction(f64),
}

/// Randomly partitions a shared dataset into non-overlapping subsets.
///
/// Count lengths must sum exactly to the dataset length. Fractional lengths
/// are floored, then leftover samples are assigned round-robin from the first
/// split. The seeded permutation is reproducible within this implementation.
///
/// # Errors
///
/// Returns [`RustTorchError::InvalidConfiguration`] for mixed length kinds,
/// invalid fractions, incorrect totals, or count overflow.
pub fn random_split<D>(
    dataset: Arc<D>,
    lengths: &[SplitLength],
    seed: u64,
) -> Result<Vec<Subset<Arc<D>>>>
where
    D: Dataset,
{
    let counts = match lengths.first() {
        None | Some(SplitLength::Count(_)) => {
            if lengths
                .iter()
                .any(|length| matches!(length, SplitLength::Fraction(_)))
            {
                return Err(invalid_split("count and fraction lengths cannot be mixed"));
            }
            let counts = lengths
                .iter()
                .map(|length| match length {
                    SplitLength::Count(count) => Ok(*count),
                    SplitLength::Fraction(_) => unreachable!(),
                })
                .collect::<Result<Vec<_>>>()?;
            let total = counts.iter().try_fold(0usize, |total, &count| {
                total
                    .checked_add(count)
                    .ok_or_else(|| invalid_split("count total overflows usize"))
            })?;
            if total != dataset.len() {
                return Err(invalid_split(&format!(
                    "counts sum to {total}, expected {}",
                    dataset.len()
                )));
            }
            counts
        }
        Some(SplitLength::Fraction(_)) => {
            let fractions = lengths
                .iter()
                .map(|length| match length {
                    SplitLength::Fraction(fraction)
                        if fraction.is_finite() && (0.0..=1.0).contains(fraction) =>
                    {
                        Ok(*fraction)
                    }
                    SplitLength::Fraction(_) => Err(invalid_split(
                        "fractions must be finite and between zero and one",
                    )),
                    SplitLength::Count(_) => {
                        Err(invalid_split("count and fraction lengths cannot be mixed"))
                    }
                })
                .collect::<Result<Vec<_>>>()?;
            let total = fractions.iter().sum::<f64>();
            if total > 1.0 || (total - 1.0).abs() > 1e-9 {
                return Err(invalid_split(&format!(
                    "fractions sum to {total}, expected 1"
                )));
            }
            let mut counts = fractions
                .iter()
                .map(|fraction| (fraction * dataset.len() as f64).floor() as usize)
                .collect::<Vec<_>>();
            let assigned = counts.iter().try_fold(0usize, |total, &count| {
                total
                    .checked_add(count)
                    .ok_or_else(|| invalid_split("fractional count total overflows usize"))
            })?;
            let remainder = dataset.len().checked_sub(assigned).ok_or_else(|| {
                invalid_split("fractional floors exceed the source dataset length")
            })?;
            let split_count = counts.len();
            for index in 0..remainder {
                counts[index % split_count] += 1;
            }
            counts
        }
    };

    let mut indices = (0..dataset.len()).collect::<Vec<_>>();
    indices.shuffle(&mut ChaCha12Rng::seed_from_u64(seed));
    let mut offset = 0;
    counts
        .into_iter()
        .map(|count| {
            let next = offset + count;
            let subset = Subset::new(Arc::clone(&dataset), indices[offset..next].to_vec());
            offset = next;
            subset
        })
        .collect()
}

fn invalid_split(reason: &str) -> RustTorchError {
    RustTorchError::InvalidConfiguration {
        field: "lengths",
        reason: reason.to_owned(),
    }
}

/// Flattens datasets or other iterables using the standard iterator adapter.
pub fn chain_datasets<I>(datasets: I) -> impl Iterator<Item = <I::Item as IntoIterator>::Item>
where
    I: IntoIterator,
    I::Item: IntoIterator,
{
    datasets.into_iter().flatten()
}

impl<D> Dataset for Arc<D>
where
    D: Dataset + ?Sized,
{
    type Sample = D::Sample;
    type Error = D::Error;

    fn len(&self) -> usize {
        self.as_ref().len()
    }

    fn get(&self, index: usize) -> std::result::Result<Self::Sample, Self::Error> {
        self.as_ref().get(index)
    }

    fn get_batch(&self, indices: &[usize]) -> std::result::Result<Vec<Self::Sample>, Self::Error> {
        self.as_ref().get_batch(indices)
    }

    fn get_batch_with_context(
        &self,
        indices: &[usize],
        context: &WorkerContext,
    ) -> std::result::Result<Vec<Self::Sample>, Self::Error> {
        self.as_ref().get_batch_with_context(indices, context)
    }
}

impl<D> ReplaySafeDataset for Arc<D> where D: ReplaySafeDataset + ?Sized {}
