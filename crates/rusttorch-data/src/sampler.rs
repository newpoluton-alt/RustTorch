use std::{num::NonZeroUsize, ops::Range, vec::IntoIter};

use rand::{
    Rng, SeedableRng,
    distributions::{Distribution, Open01, WeightedIndex},
    seq::SliceRandom,
};
use rand_chacha::ChaCha12Rng;
use rusttorch_core::{Result, RustTorchError};
use serde::{Deserialize, Serialize, de::DeserializeOwned};

/// Serialized cursor for [`SequentialSampler`].
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct SequentialSamplerState {
    /// Configured index count.
    pub length: u64,
    /// Active epoch.
    pub epoch: u64,
    /// Next index position.
    pub position: u64,
}

/// Replacement mode stored by [`RandomSamplerState`].
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum RandomReplacement {
    /// Draw each generated index independently.
    With,
    /// Generate successive shuffled permutations.
    Without,
}

/// Serialized configuration and cursor for [`RandomSampler`].
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct RandomSamplerState {
    /// Configured index range length.
    pub length: u64,
    /// Number of generated occurrences.
    pub num_samples: u64,
    /// Replacement policy.
    pub replacement: RandomReplacement,
    /// Base seed.
    pub seed: u64,
    /// Active epoch.
    pub epoch: u64,
    /// Next occurrence position.
    pub position: u64,
}

/// Serialized configuration and cursor for [`SubsetRandomSampler`].
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct SubsetRandomSamplerState {
    /// Exact caller-supplied subset indices.
    pub indices: Vec<u64>,
    /// Base seed.
    pub seed: u64,
    /// Active epoch.
    pub epoch: u64,
    /// Next occurrence position.
    pub position: u64,
}

/// Serialized configuration and cursor for [`WeightedRandomSampler`].
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct WeightedRandomSamplerState {
    /// Exact IEEE-754 bit patterns of the configured weights.
    pub weight_bits: Vec<u64>,
    /// Number of generated occurrences.
    pub num_samples: u64,
    /// Replacement policy.
    pub replacement: bool,
    /// Base seed.
    pub seed: u64,
    /// Active epoch.
    pub epoch: u64,
    /// Next occurrence position.
    pub position: u64,
}

/// Serialized configuration and cursor for [`DistributedSampler`].
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct DistributedSamplerState {
    /// Global dataset length.
    pub length: u64,
    /// Replica count.
    pub replicas: u64,
    /// This sampler's rank.
    pub rank: u64,
    /// Whether the global indices are shuffled.
    pub shuffle: bool,
    /// Base seed.
    pub seed: u64,
    /// Whether the global tail is truncated.
    pub drop_last: bool,
    /// Active epoch.
    pub epoch: u64,
    /// Next rank-local occurrence position.
    pub position: u64,
}

/// Serialized cursor for a compositional [`BatchSampler`].
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct BatchSamplerState<S> {
    /// Inner sampler configuration and cursor.
    pub sampler: S,
    /// Configured batch size.
    pub batch_size: u64,
    /// Whether an incomplete final batch is omitted.
    pub drop_last: bool,
    /// Next visible batch number.
    pub next_batch: u64,
    /// Next logical sample occurrence.
    pub next_logical_sample: u64,
}

#[doc(hidden)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DistributedConfiguration {
    pub(crate) replicas: usize,
    pub(crate) rank: usize,
    pub(crate) shuffle: bool,
    pub(crate) seed: u64,
    pub(crate) drop_last: bool,
}

#[doc(hidden)]
pub trait SamplerCheckpoint: Sampler {
    type State: Clone + Serialize + DeserializeOwned;

    fn kind(&self) -> &'static str;
    fn checkpoint_state(&self, position: u64) -> Result<Self::State>;
    fn validate_checkpoint_state(
        &self,
        state: &Self::State,
        epoch: u64,
        position: u64,
    ) -> Result<()>;
    fn restore_checkpoint_iter_validated(&mut self, state: &Self::State) -> Self::Iter;

    fn distributed_configuration(&self) -> Option<DistributedConfiguration> {
        None
    }
}

#[doc(hidden)]
pub trait BatchSourceCheckpoint: BatchSource {
    type State: Clone + Serialize + DeserializeOwned;

    fn kind(&self) -> String;
    fn checkpoint_state(&self, next_batch: u64, next_logical_sample: u64) -> Result<Self::State>;
    fn validate_checkpoint_state(
        &self,
        state: &Self::State,
        epoch: u64,
        next_batch: u64,
        next_logical_sample: u64,
    ) -> Result<()>;
    fn restore_checkpoint_iter_validated(&mut self, state: &Self::State) -> Self::Iter;
    fn distributed_configuration(&self) -> Option<DistributedConfiguration>;
}

/// A reusable, epoch-aware source of finite sample indices.
///
/// Every call to [`Sampler::iter`] creates a fresh iterator. An exact length
/// is optional and is never required for iteration.
pub trait Sampler {
    /// A fresh finite iterator over sample indices.
    type Iter: Iterator<Item = usize>;

    /// Creates an iterator for the current epoch.
    fn iter(&self) -> Self::Iter;

    /// Returns the exact number of indices when it is known.
    fn exact_len(&self) -> Option<usize> {
        None
    }

    /// Returns the current epoch.
    fn epoch(&self) -> u64;

    /// Selects the epoch used by future iterators.
    fn set_epoch(&mut self, epoch: u64);
}

/// A reusable, epoch-aware source of finite index batches.
///
/// Every call to [`BatchSource::iter`] creates a fresh iterator. An exact
/// number of batches is optional and is never required for iteration.
pub trait BatchSource {
    /// A fresh finite iterator over index batches.
    type Iter: Iterator<Item = Vec<usize>>;

    /// Creates an iterator for the current epoch.
    fn iter(&self) -> Self::Iter;

    /// Returns the exact number of batches when it is known.
    fn exact_len(&self) -> Option<usize> {
        None
    }

    /// Returns the current epoch.
    fn epoch(&self) -> u64;

    /// Selects the epoch used by future iterators.
    fn set_epoch(&mut self, epoch: u64);
}

/// Adapts an epoch-aware iterator factory into a reusable [`Sampler`].
pub struct FnSampler<F> {
    exact_len: Option<usize>,
    epoch: u64,
    make_iter: F,
}

impl<F> FnSampler<F> {
    /// Creates a sampler with an optional exact index count.
    pub fn new(exact_len: Option<usize>, make_iter: F) -> Self {
        Self {
            exact_len,
            epoch: 0,
            make_iter,
        }
    }
}

impl<F, I> Sampler for FnSampler<F>
where
    F: Fn(u64) -> I,
    I: Iterator<Item = usize>,
{
    type Iter = I;

    fn iter(&self) -> Self::Iter {
        (self.make_iter)(self.epoch)
    }

    fn exact_len(&self) -> Option<usize> {
        self.exact_len
    }

    fn epoch(&self) -> u64 {
        self.epoch
    }

    fn set_epoch(&mut self, epoch: u64) {
        self.epoch = epoch;
    }
}

/// Adapts an epoch-aware batch-iterator factory into a reusable [`BatchSource`].
pub struct FnBatchSource<F> {
    exact_len: Option<usize>,
    epoch: u64,
    make_iter: F,
}

impl<F> FnBatchSource<F> {
    /// Creates a batch source with an optional exact batch count.
    pub fn new(exact_len: Option<usize>, make_iter: F) -> Self {
        Self {
            exact_len,
            epoch: 0,
            make_iter,
        }
    }
}

impl<F, I> BatchSource for FnBatchSource<F>
where
    F: Fn(u64) -> I,
    I: Iterator<Item = Vec<usize>>,
{
    type Iter = I;

    fn iter(&self) -> Self::Iter {
        (self.make_iter)(self.epoch)
    }

    fn exact_len(&self) -> Option<usize> {
        self.exact_len
    }

    fn epoch(&self) -> u64 {
        self.epoch
    }

    fn set_epoch(&mut self, epoch: u64) {
        self.epoch = epoch;
    }
}

/// Groups sampler indices into fixed-size batches.
///
/// The sampler remains directly iterable when `S` is an iterator. When `S`
/// implements [`Sampler`], this type also implements [`BatchSource`] and
/// creates fresh batches for every iteration.
///
/// ```
/// use rusttorch_data::BatchSampler;
///
/// let batches = BatchSampler::new(0..5, 2, false)
///     .expect("batch size is nonzero")
///     .collect::<Vec<_>>();
/// assert_eq!(batches, vec![vec![0, 1], vec![2, 3], vec![4]]);
/// ```
pub struct BatchSampler<S> {
    sampler: S,
    batch_size: usize,
    drop_last: bool,
}

impl<S> BatchSampler<S> {
    /// Creates a batch sampler.
    ///
    /// A short final batch is omitted when `drop_last` is `true`.
    ///
    /// # Errors
    ///
    /// Returns [`RustTorchError::InvalidConfiguration`] when `batch_size` is
    /// zero or cannot describe an allocatable index batch.
    pub fn new(sampler: S, batch_size: usize, drop_last: bool) -> Result<Self> {
        let batch_size = validate_batch_size(batch_size)?;
        Ok(Self::from_validated(sampler, batch_size, drop_last))
    }

    pub(crate) fn from_validated(sampler: S, batch_size: NonZeroUsize, drop_last: bool) -> Self {
        Self {
            sampler,
            batch_size: batch_size.get(),
            drop_last,
        }
    }
}

impl<S> BatchSampler<S>
where
    S: Iterator<Item = usize>,
{
    /// Returns the exact number of remaining batches when the sampler reports
    /// an exact remaining length.
    pub fn exact_len(&self) -> Option<usize> {
        let (lower, upper) = self.sampler.size_hint();
        (upper == Some(lower)).then(|| batch_count(lower, self.batch_size, self.drop_last))
    }
}

impl<S> Iterator for BatchSampler<S>
where
    S: Iterator<Item = usize>,
{
    type Item = Vec<usize>;

    fn next(&mut self) -> Option<Self::Item> {
        let first = self.sampler.next()?;
        let mut batch = try_vec(self.batch_size, "batch_size")
            .expect("batch storage dimensions were validated during construction");
        batch.push(first);
        batch.extend(self.sampler.by_ref().take(self.batch_size - 1));
        (!self.drop_last || batch.len() == self.batch_size).then_some(batch)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let (lower, upper) = self.sampler.size_hint();
        (
            batch_count(lower, self.batch_size, self.drop_last),
            upper.map(|length| batch_count(length, self.batch_size, self.drop_last)),
        )
    }
}

impl<S> BatchSource for BatchSampler<S>
where
    S: Sampler,
{
    type Iter = BatchSampler<S::Iter>;

    fn iter(&self) -> Self::Iter {
        BatchSampler::new(self.sampler.iter(), self.batch_size, self.drop_last)
            .expect("batch storage dimensions were validated during construction")
    }

    fn exact_len(&self) -> Option<usize> {
        self.sampler
            .exact_len()
            .map(|length| batch_count(length, self.batch_size, self.drop_last))
    }

    fn epoch(&self) -> u64 {
        self.sampler.epoch()
    }

    fn set_epoch(&mut self, epoch: u64) {
        self.sampler.set_epoch(epoch);
    }
}

/// An allocation-free sampler that yields indices in ascending order.
pub struct SequentialSampler {
    length: usize,
    epoch: u64,
    indices: Range<usize>,
}

impl SequentialSampler {
    /// Creates a sampler for indices `0..length`.
    pub fn new(length: usize) -> Self {
        Self {
            length,
            epoch: 0,
            indices: 0..length,
        }
    }
}

impl Sampler for SequentialSampler {
    type Iter = Range<usize>;

    fn iter(&self) -> Self::Iter {
        0..self.length
    }

    fn exact_len(&self) -> Option<usize> {
        Some(self.length)
    }

    fn epoch(&self) -> u64 {
        self.epoch
    }

    fn set_epoch(&mut self, epoch: u64) {
        self.epoch = epoch;
        self.indices = 0..self.length;
    }
}

impl Iterator for SequentialSampler {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        self.indices.next()
    }
}

#[derive(Clone, Copy)]
enum Replacement {
    With,
    Without,
}

/// A seeded sampler over a finite index range.
///
/// Shuffling uses sampler-local ChaCha12 state, so it does not alter
/// LibTorch's global random state. Its sequence intentionally does not claim
/// exact PyTorch Philox parity.
pub struct RandomSampler {
    length: usize,
    num_samples: usize,
    replacement: Replacement,
    seed: u64,
    epoch: u64,
    indices: IntoIter<usize>,
}

impl RandomSampler {
    /// Creates a reproducible shuffled permutation of `0..length`.
    ///
    /// # Errors
    ///
    /// Returns [`RustTorchError::InvalidConfiguration`] when `length` is zero
    /// or storage for the permutation cannot be reserved.
    pub fn new(length: usize, seed: u64) -> Result<Self> {
        Self::without_replacement(length, length, seed)
    }

    /// Creates a sampler that draws `num_samples` indices with replacement.
    ///
    /// # Errors
    ///
    /// Returns [`RustTorchError::InvalidConfiguration`] when `length` or
    /// `num_samples` is zero or requested sampler storage cannot be reserved.
    pub fn with_replacement(length: usize, num_samples: usize, seed: u64) -> Result<Self> {
        Self::build(length, num_samples, Replacement::With, seed)
    }

    /// Creates a sampler that draws shuffled indices without replacement.
    ///
    /// Counts larger than `length` concatenate independent complete
    /// permutations and one final permutation prefix.
    ///
    /// # Errors
    ///
    /// Returns [`RustTorchError::InvalidConfiguration`] when `length` or
    /// `num_samples` is zero or requested sampler storage cannot be reserved.
    pub fn without_replacement(length: usize, num_samples: usize, seed: u64) -> Result<Self> {
        Self::build(length, num_samples, Replacement::Without, seed)
    }

    fn build(
        length: usize,
        num_samples: usize,
        replacement: Replacement,
        seed: u64,
    ) -> Result<Self> {
        validate_positive(length, "length")?;
        validate_positive(num_samples, "num_samples")?;
        let indices = random_indices(length, num_samples, replacement, seed)?.into_iter();
        Ok(Self {
            length,
            num_samples,
            replacement,
            seed,
            epoch: 0,
            indices,
        })
    }

    fn indices_for_epoch(&self) -> Vec<usize> {
        random_indices(
            self.length,
            self.num_samples,
            self.replacement,
            epoch_seed(self.seed, self.epoch),
        )
        .expect("sampler storage dimensions were validated during construction")
    }
}

impl Sampler for RandomSampler {
    type Iter = IntoIter<usize>;

    fn iter(&self) -> Self::Iter {
        self.indices_for_epoch().into_iter()
    }

    fn exact_len(&self) -> Option<usize> {
        Some(self.num_samples)
    }

    fn epoch(&self) -> u64 {
        self.epoch
    }

    fn set_epoch(&mut self, epoch: u64) {
        self.epoch = epoch;
        self.indices = self.indices_for_epoch().into_iter();
    }
}

impl Iterator for RandomSampler {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        self.indices.next()
    }
}

/// A seeded sampler that shuffles a caller-provided subset of indices.
pub struct SubsetRandomSampler {
    source: Vec<usize>,
    seed: u64,
    epoch: u64,
    indices: IntoIter<usize>,
}

impl SubsetRandomSampler {
    /// Creates a sampler that yields each supplied index once.
    ///
    /// # Errors
    ///
    /// Returns [`RustTorchError::InvalidConfiguration`] when storage for a
    /// shuffled copy of the supplied indices cannot be reserved.
    pub fn new(indices: Vec<usize>, seed: u64) -> Result<Self> {
        let shuffled = shuffled_copy(&indices, seed)?.into_iter();
        Ok(Self {
            source: indices,
            seed,
            epoch: 0,
            indices: shuffled,
        })
    }

    fn indices_for_epoch(&self) -> Vec<usize> {
        shuffled_copy(&self.source, epoch_seed(self.seed, self.epoch))
            .expect("sampler storage dimensions were validated during construction")
    }
}

impl Sampler for SubsetRandomSampler {
    type Iter = IntoIter<usize>;

    fn iter(&self) -> Self::Iter {
        self.indices_for_epoch().into_iter()
    }

    fn exact_len(&self) -> Option<usize> {
        Some(self.source.len())
    }

    fn epoch(&self) -> u64 {
        self.epoch
    }

    fn set_epoch(&mut self, epoch: u64) {
        self.epoch = epoch;
        self.indices = self.indices_for_epoch().into_iter();
    }
}

impl Iterator for SubsetRandomSampler {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        self.indices.next()
    }
}

/// A seeded sampler that draws indices according to nonnegative weights.
pub struct WeightedRandomSampler {
    weights: Vec<f64>,
    num_samples: usize,
    replacement: bool,
    seed: u64,
    epoch: u64,
    distribution: WeightedIndex<f64>,
    indices: IntoIter<usize>,
}

impl WeightedRandomSampler {
    /// Creates a weighted sampler.
    ///
    /// Without replacement, `num_samples` may be at most `weights.len()`.
    /// Positive-weight entries are selected first; deterministically shuffled
    /// zero-weight entries fill any remaining positions.
    ///
    /// # Errors
    ///
    /// Returns [`RustTorchError::InvalidConfiguration`] when weights are
    /// empty, non-finite, negative, or have no positive finite total; when
    /// `num_samples` is zero; when a without-replacement count exceeds the
    /// number of weights; or when requested sampler storage cannot be
    /// reserved.
    pub fn new(
        weights: Vec<f64>,
        num_samples: usize,
        replacement: bool,
        seed: u64,
    ) -> Result<Self> {
        validate_weights(&weights)?;
        validate_positive(num_samples, "num_samples")?;
        if !replacement && num_samples > weights.len() {
            return Err(invalid_configuration(
                "num_samples",
                "must not exceed the number of weights without replacement",
            ));
        }
        let distribution = WeightedIndex::new(&weights).map_err(|error| {
            invalid_configuration("weights", format!("cannot form a distribution: {error}"))
        })?;
        let indices =
            weighted_indices(&weights, num_samples, replacement, seed, &distribution)?.into_iter();
        Ok(Self {
            weights,
            num_samples,
            replacement,
            seed,
            epoch: 0,
            distribution,
            indices,
        })
    }

    fn indices_for_epoch(&self) -> Vec<usize> {
        weighted_indices(
            &self.weights,
            self.num_samples,
            self.replacement,
            epoch_seed(self.seed, self.epoch),
            &self.distribution,
        )
        .expect("sampler storage dimensions were validated during construction")
    }
}

impl Sampler for WeightedRandomSampler {
    type Iter = IntoIter<usize>;

    fn iter(&self) -> Self::Iter {
        self.indices_for_epoch().into_iter()
    }

    fn exact_len(&self) -> Option<usize> {
        Some(self.num_samples)
    }

    fn epoch(&self) -> u64 {
        self.epoch
    }

    fn set_epoch(&mut self, epoch: u64) {
        self.epoch = epoch;
        self.indices = self.indices_for_epoch().into_iter();
    }
}

impl Iterator for WeightedRandomSampler {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        self.indices.next()
    }
}

/// Deterministically shards a finite index range across distributed ranks.
///
/// When `drop_last` is false, indices are cyclically padded so every rank has
/// the same length. When it is true, the global sequence is truncated to a
/// multiple of the replica count. Optional shuffling uses sampler-local
/// ChaCha12 state and does not alter LibTorch's global random state.
///
/// ```
/// use rusttorch_data::DistributedSampler;
///
/// let rank_one = DistributedSampler::new(5, 2, 1, false, 0, false)
///     .expect("replica and rank are valid")
///     .collect::<Vec<_>>();
/// assert_eq!(rank_one, vec![1, 3, 0]);
/// ```
pub struct DistributedSampler {
    length: usize,
    replicas: usize,
    rank: usize,
    shuffle: bool,
    seed: u64,
    drop_last: bool,
    epoch: u64,
    num_samples: usize,
    indices: IntoIter<usize>,
}

impl DistributedSampler {
    /// Creates a rank-aware sampler over `0..length`.
    ///
    /// Empty datasets are accepted and yield no indices on every valid rank.
    ///
    /// # Errors
    ///
    /// Returns [`RustTorchError::InvalidConfiguration`] when `replicas` is
    /// zero, `rank` is outside `0..replicas`, padding arithmetic overflows, or
    /// requested sampler storage cannot be reserved.
    pub fn new(
        length: usize,
        replicas: usize,
        rank: usize,
        shuffle: bool,
        seed: u64,
        drop_last: bool,
    ) -> Result<Self> {
        validate_positive(replicas, "replicas")?;
        if rank >= replicas {
            return Err(invalid_configuration(
                "rank",
                "must be less than the replica count",
            ));
        }
        let num_samples = distributed_sample_count(length, replicas, drop_last)?;
        let indices = distributed_indices(
            length,
            replicas,
            rank,
            shuffle,
            seed,
            drop_last,
            num_samples,
        )?
        .into_iter();
        Ok(Self {
            length,
            replicas,
            rank,
            shuffle,
            seed,
            drop_last,
            epoch: 0,
            num_samples,
            indices,
        })
    }

    /// Returns the number of indices assigned to this rank.
    pub fn len(&self) -> usize {
        self.num_samples
    }

    /// Returns `true` when this rank has no indices.
    pub fn is_empty(&self) -> bool {
        self.num_samples == 0
    }

    /// Returns the current epoch.
    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    /// Returns the number of indices consumed by direct iteration.
    pub fn position(&self) -> usize {
        self.num_samples - self.indices.len()
    }

    /// Selects the epoch and resets direct iteration to its beginning.
    pub fn set_epoch(&mut self, epoch: u64) {
        self.epoch = epoch;
        self.indices = self.indices_for_epoch().into_iter();
    }

    fn indices_for_epoch(&self) -> Vec<usize> {
        distributed_indices(
            self.length,
            self.replicas,
            self.rank,
            self.shuffle,
            epoch_seed(self.seed, self.epoch),
            self.drop_last,
            self.num_samples,
        )
        .expect("distributed sampler dimensions were validated during construction")
    }
}

impl Sampler for DistributedSampler {
    type Iter = IntoIter<usize>;

    fn iter(&self) -> Self::Iter {
        self.indices_for_epoch().into_iter()
    }

    fn exact_len(&self) -> Option<usize> {
        Some(self.num_samples)
    }

    fn epoch(&self) -> u64 {
        self.epoch()
    }

    fn set_epoch(&mut self, epoch: u64) {
        self.set_epoch(epoch);
    }
}

impl Iterator for DistributedSampler {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        self.indices.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.indices.size_hint()
    }
}

impl ExactSizeIterator for DistributedSampler {}

impl SamplerCheckpoint for SequentialSampler {
    type State = SequentialSamplerState;

    fn kind(&self) -> &'static str {
        "sequential"
    }

    fn checkpoint_state(&self, position: u64) -> Result<Self::State> {
        let length = usize_to_u64(self.length, "sampler")?;
        validate_position(position, length)?;
        Ok(SequentialSamplerState {
            length,
            epoch: self.epoch,
            position,
        })
    }

    fn validate_checkpoint_state(
        &self,
        state: &Self::State,
        epoch: u64,
        position: u64,
    ) -> Result<()> {
        require_equal(
            "sampler",
            state.length,
            usize_to_u64(self.length, "sampler")?,
        )?;
        validate_epoch_position(state.epoch, state.position, epoch, position)?;
        validate_position(state.position, state.length)
    }

    fn restore_checkpoint_iter_validated(&mut self, state: &Self::State) -> Self::Iter {
        self.set_epoch(state.epoch);
        let position = usize::try_from(state.position)
            .expect("validated sequential sampler position fits usize");
        position..self.length
    }
}

impl SamplerCheckpoint for RandomSampler {
    type State = RandomSamplerState;

    fn kind(&self) -> &'static str {
        match self.replacement {
            Replacement::With => "random_replacement",
            Replacement::Without => "random",
        }
    }

    fn checkpoint_state(&self, position: u64) -> Result<Self::State> {
        let num_samples = usize_to_u64(self.num_samples, "sampler")?;
        validate_position(position, num_samples)?;
        Ok(RandomSamplerState {
            length: usize_to_u64(self.length, "sampler")?,
            num_samples,
            replacement: match self.replacement {
                Replacement::With => RandomReplacement::With,
                Replacement::Without => RandomReplacement::Without,
            },
            seed: self.seed,
            epoch: self.epoch,
            position,
        })
    }

    fn validate_checkpoint_state(
        &self,
        state: &Self::State,
        epoch: u64,
        position: u64,
    ) -> Result<()> {
        require_equal(
            "sampler",
            state.length,
            usize_to_u64(self.length, "sampler")?,
        )?;
        require_equal(
            "sampler",
            state.num_samples,
            usize_to_u64(self.num_samples, "sampler")?,
        )?;
        let replacement = match self.replacement {
            Replacement::With => RandomReplacement::With,
            Replacement::Without => RandomReplacement::Without,
        };
        require_equal("sampler", state.replacement, replacement)?;
        require_equal("sampler", state.seed, self.seed)?;
        validate_epoch_position(state.epoch, state.position, epoch, position)?;
        validate_position(state.position, state.num_samples)
    }

    fn restore_checkpoint_iter_validated(&mut self, state: &Self::State) -> Self::Iter {
        self.set_epoch(state.epoch);
        let mut iterator = self.iter();
        advance(&mut iterator, state.position)
            .expect("validated random sampler cursor is regenerable");
        iterator
    }
}

impl SamplerCheckpoint for SubsetRandomSampler {
    type State = SubsetRandomSamplerState;

    fn kind(&self) -> &'static str {
        "subset_random"
    }

    fn checkpoint_state(&self, position: u64) -> Result<Self::State> {
        let length = usize_to_u64(self.source.len(), "sampler")?;
        validate_position(position, length)?;
        Ok(SubsetRandomSamplerState {
            indices: self
                .source
                .iter()
                .copied()
                .map(|index| usize_to_u64(index, "sampler"))
                .collect::<Result<Vec<_>>>()?,
            seed: self.seed,
            epoch: self.epoch,
            position,
        })
    }

    fn validate_checkpoint_state(
        &self,
        state: &Self::State,
        epoch: u64,
        position: u64,
    ) -> Result<()> {
        let expected = self
            .source
            .iter()
            .copied()
            .map(|index| usize_to_u64(index, "sampler"))
            .collect::<Result<Vec<_>>>()?;
        require_equal("sampler", &state.indices, &expected)?;
        require_equal("sampler", state.seed, self.seed)?;
        validate_epoch_position(state.epoch, state.position, epoch, position)?;
        validate_position(
            state.position,
            usize_to_u64(state.indices.len(), "sampler")?,
        )
    }

    fn restore_checkpoint_iter_validated(&mut self, state: &Self::State) -> Self::Iter {
        self.set_epoch(state.epoch);
        let mut iterator = self.iter();
        advance(&mut iterator, state.position)
            .expect("validated subset sampler cursor is regenerable");
        iterator
    }
}

impl SamplerCheckpoint for WeightedRandomSampler {
    type State = WeightedRandomSamplerState;

    fn kind(&self) -> &'static str {
        if self.replacement {
            "weighted_random_replacement"
        } else {
            "weighted_random"
        }
    }

    fn checkpoint_state(&self, position: u64) -> Result<Self::State> {
        let num_samples = usize_to_u64(self.num_samples, "sampler")?;
        validate_position(position, num_samples)?;
        Ok(WeightedRandomSamplerState {
            weight_bits: self.weights.iter().map(|weight| weight.to_bits()).collect(),
            num_samples,
            replacement: self.replacement,
            seed: self.seed,
            epoch: self.epoch,
            position,
        })
    }

    fn validate_checkpoint_state(
        &self,
        state: &Self::State,
        epoch: u64,
        position: u64,
    ) -> Result<()> {
        let expected = self
            .weights
            .iter()
            .map(|weight| weight.to_bits())
            .collect::<Vec<_>>();
        require_equal("sampler", &state.weight_bits, &expected)?;
        require_equal(
            "sampler",
            state.num_samples,
            usize_to_u64(self.num_samples, "sampler")?,
        )?;
        require_equal("sampler", state.replacement, self.replacement)?;
        require_equal("sampler", state.seed, self.seed)?;
        validate_epoch_position(state.epoch, state.position, epoch, position)?;
        validate_position(state.position, state.num_samples)
    }

    fn restore_checkpoint_iter_validated(&mut self, state: &Self::State) -> Self::Iter {
        self.set_epoch(state.epoch);
        let mut iterator = self.iter();
        advance(&mut iterator, state.position)
            .expect("validated weighted sampler cursor is regenerable");
        iterator
    }
}

impl SamplerCheckpoint for DistributedSampler {
    type State = DistributedSamplerState;

    fn kind(&self) -> &'static str {
        "distributed"
    }

    fn checkpoint_state(&self, position: u64) -> Result<Self::State> {
        let num_samples = usize_to_u64(self.num_samples, "sampler")?;
        validate_position(position, num_samples)?;
        Ok(DistributedSamplerState {
            length: usize_to_u64(self.length, "sampler")?,
            replicas: usize_to_u64(self.replicas, "sampler")?,
            rank: usize_to_u64(self.rank, "sampler")?,
            shuffle: self.shuffle,
            seed: self.seed,
            drop_last: self.drop_last,
            epoch: self.epoch,
            position,
        })
    }

    fn validate_checkpoint_state(
        &self,
        state: &Self::State,
        epoch: u64,
        position: u64,
    ) -> Result<()> {
        require_equal(
            "sampler",
            state.length,
            usize_to_u64(self.length, "sampler")?,
        )?;
        require_equal(
            "sampler",
            state.replicas,
            usize_to_u64(self.replicas, "sampler")?,
        )?;
        require_equal("sampler", state.rank, usize_to_u64(self.rank, "sampler")?)?;
        require_equal("sampler", state.shuffle, self.shuffle)?;
        require_equal("sampler", state.seed, self.seed)?;
        require_equal("sampler", state.drop_last, self.drop_last)?;
        validate_epoch_position(state.epoch, state.position, epoch, position)?;
        validate_position(state.position, usize_to_u64(self.num_samples, "sampler")?)
    }

    fn restore_checkpoint_iter_validated(&mut self, state: &Self::State) -> Self::Iter {
        self.set_epoch(state.epoch);
        let mut iterator = self.iter();
        advance(&mut iterator, state.position)
            .expect("validated distributed sampler cursor is regenerable");
        iterator
    }

    fn distributed_configuration(&self) -> Option<DistributedConfiguration> {
        Some(DistributedConfiguration {
            replicas: self.replicas,
            rank: self.rank,
            shuffle: self.shuffle,
            seed: self.seed,
            drop_last: self.drop_last,
        })
    }
}

impl<S> BatchSourceCheckpoint for BatchSampler<S>
where
    S: SamplerCheckpoint,
{
    type State = BatchSamplerState<S::State>;

    fn kind(&self) -> String {
        format!("batch_sampler/{}", self.sampler.kind())
    }

    fn checkpoint_state(&self, next_batch: u64, next_logical_sample: u64) -> Result<Self::State> {
        validate_batch_boundary(
            self.sampler.exact_len(),
            self.batch_size,
            self.drop_last,
            next_batch,
            next_logical_sample,
        )?;
        Ok(BatchSamplerState {
            sampler: self.sampler.checkpoint_state(next_logical_sample)?,
            batch_size: usize_to_u64(self.batch_size, "batch_sampler")?,
            drop_last: self.drop_last,
            next_batch,
            next_logical_sample,
        })
    }

    fn validate_checkpoint_state(
        &self,
        state: &Self::State,
        epoch: u64,
        next_batch: u64,
        next_logical_sample: u64,
    ) -> Result<()> {
        require_equal(
            "batch_sampler",
            state.batch_size,
            usize_to_u64(self.batch_size, "batch_sampler")?,
        )?;
        require_equal("batch_sampler", state.drop_last, self.drop_last)?;
        require_equal("checkpoint", state.next_batch, next_batch)?;
        require_equal("checkpoint", state.next_logical_sample, next_logical_sample)?;
        validate_batch_boundary(
            self.sampler.exact_len(),
            self.batch_size,
            self.drop_last,
            state.next_batch,
            state.next_logical_sample,
        )?;
        self.sampler
            .validate_checkpoint_state(&state.sampler, epoch, state.next_logical_sample)
    }

    fn restore_checkpoint_iter_validated(&mut self, state: &Self::State) -> Self::Iter {
        let iterator = self
            .sampler
            .restore_checkpoint_iter_validated(&state.sampler);
        BatchSampler::from_validated(
            iterator,
            NonZeroUsize::new(self.batch_size)
                .expect("batch sampler construction validated a nonzero size"),
            self.drop_last,
        )
    }

    fn distributed_configuration(&self) -> Option<DistributedConfiguration> {
        self.sampler.distributed_configuration()
    }
}

fn validate_epoch_position(
    state_epoch: u64,
    state_position: u64,
    epoch: u64,
    position: u64,
) -> Result<()> {
    require_equal("checkpoint", state_epoch, epoch)?;
    require_equal("checkpoint", state_position, position)
}

fn validate_position(position: u64, length: u64) -> Result<()> {
    if position > length {
        return Err(invalid_configuration(
            "checkpoint",
            format!("sampler position {position} exceeds length {length}"),
        ));
    }
    let _ = u64_to_usize(position, "checkpoint")?;
    Ok(())
}

fn validate_batch_boundary(
    exact_len: Option<usize>,
    batch_size: usize,
    drop_last: bool,
    next_batch: u64,
    next_logical_sample: u64,
) -> Result<()> {
    let batch_size = usize_to_u64(batch_size, "batch_sampler")?;
    let complete_boundary = next_logical_sample
        .checked_div(batch_size)
        .ok_or_else(|| invalid_configuration("batch_sampler", "batch size is zero"))?;
    let is_aligned = next_logical_sample.is_multiple_of(batch_size);
    let final_short = exact_len
        .map(|length| usize_to_u64(length, "batch_sampler"))
        .transpose()?
        .is_some_and(|length| {
            !drop_last
                && next_logical_sample == length
                && next_batch == complete_boundary + u64::from(!is_aligned)
        });
    if (!is_aligned || next_batch != complete_boundary) && !final_short {
        return Err(invalid_configuration(
            "checkpoint",
            "batch and logical sample cursors are not on a visible batch boundary",
        ));
    }
    Ok(())
}

fn advance<I>(iterator: &mut I, position: u64) -> Result<()>
where
    I: Iterator,
{
    let position = u64_to_usize(position, "checkpoint")?;
    for _ in 0..position {
        if iterator.next().is_none() {
            return Err(invalid_configuration(
                "checkpoint",
                "sampler cursor exceeds the regenerated sequence",
            ));
        }
    }
    Ok(())
}

fn usize_to_u64(value: usize, field: &'static str) -> Result<u64> {
    u64::try_from(value)
        .map_err(|_| invalid_configuration(field, "value does not fit the checkpoint schema"))
}

fn u64_to_usize(value: u64, field: &'static str) -> Result<usize> {
    usize::try_from(value)
        .map_err(|_| invalid_configuration(field, "checkpoint value does not fit this platform"))
}

fn require_equal<T>(field: &'static str, actual: T, expected: T) -> Result<()>
where
    T: PartialEq,
{
    if actual != expected {
        return Err(invalid_configuration(
            field,
            "checkpoint configuration does not match the loader",
        ));
    }
    Ok(())
}

fn validate_positive(value: usize, field: &'static str) -> Result<()> {
    if value == 0 {
        return Err(invalid_configuration(field, "must be greater than zero"));
    }
    Ok(())
}

pub(crate) fn validate_batch_size(batch_size: usize) -> Result<NonZeroUsize> {
    validate_positive(batch_size, "batch_size")?;
    let _ = try_vec::<usize>(batch_size, "batch_size")?;
    NonZeroUsize::new(batch_size)
        .ok_or_else(|| invalid_configuration("batch_size", "must be greater than zero"))
}

fn validate_weights(weights: &[f64]) -> Result<()> {
    if weights.is_empty() {
        return Err(invalid_configuration("weights", "must not be empty"));
    }
    if weights.iter().any(|weight| !weight.is_finite()) {
        return Err(invalid_configuration("weights", "must be finite"));
    }
    if weights.iter().any(|&weight| weight < 0.0) {
        return Err(invalid_configuration("weights", "must be nonnegative"));
    }
    let total = weights.iter().sum::<f64>();
    if !total.is_finite() || total <= 0.0 {
        return Err(invalid_configuration(
            "weights",
            "must have a positive finite total",
        ));
    }
    Ok(())
}

fn invalid_configuration(field: &'static str, reason: impl Into<String>) -> RustTorchError {
    RustTorchError::InvalidConfiguration {
        field,
        reason: reason.into(),
    }
}

fn epoch_seed(seed: u64, epoch: u64) -> u64 {
    seed.wrapping_add(epoch)
}

fn batch_count(length: usize, batch_size: usize, drop_last: bool) -> usize {
    let complete = length / batch_size;
    complete + usize::from(!drop_last && !length.is_multiple_of(batch_size))
}

fn distributed_sample_count(length: usize, replicas: usize, drop_last: bool) -> Result<usize> {
    if drop_last {
        return Ok(length / replicas);
    }
    let complete = length / replicas;
    complete
        .checked_add(usize::from(!length.is_multiple_of(replicas)))
        .ok_or_else(|| invalid_configuration("length", "padded rank length overflows usize"))
}

#[allow(clippy::too_many_arguments)]
fn distributed_indices(
    length: usize,
    replicas: usize,
    rank: usize,
    shuffle: bool,
    seed: u64,
    drop_last: bool,
    num_samples: usize,
) -> Result<Vec<usize>> {
    let total_size = num_samples
        .checked_mul(replicas)
        .ok_or_else(|| invalid_configuration("length", "padded global length overflows usize"))?;
    let materialized_length = if shuffle || !drop_last {
        length
    } else {
        total_size
    };
    let mut indices = try_vec(total_size.max(materialized_length), "length")?;
    indices.extend(0..materialized_length);
    if shuffle {
        indices.shuffle(&mut ChaCha12Rng::seed_from_u64(seed));
    }
    if drop_last {
        indices.truncate(total_size);
    } else if length != 0 {
        for position in length..total_size {
            indices.push(indices[(position - length) % length]);
        }
    }

    let mut rank_indices = try_vec(num_samples, "length")?;
    rank_indices.extend(indices.into_iter().skip(rank).step_by(replicas));
    Ok(rank_indices)
}

fn random_indices(
    length: usize,
    num_samples: usize,
    replacement: Replacement,
    seed: u64,
) -> Result<Vec<usize>> {
    let mut rng = ChaCha12Rng::seed_from_u64(seed);
    match replacement {
        Replacement::With => {
            let mut output = try_vec(num_samples, "num_samples")?;
            output.extend((0..num_samples).map(|_| rng.gen_range(0..length)));
            Ok(output)
        }
        Replacement::Without => {
            let mut permutation = try_vec(length, "length")?;
            permutation.extend(0..length);
            let mut output = try_vec(num_samples, "num_samples")?;
            while output.len() < num_samples {
                permutation.shuffle(&mut rng);
                let remaining = num_samples - output.len();
                output.extend(permutation.iter().copied().take(remaining));
            }
            Ok(output)
        }
    }
}

fn try_vec<T>(capacity: usize, field: &'static str) -> Result<Vec<T>> {
    let mut values = Vec::new();
    values.try_reserve_exact(capacity).map_err(|error| {
        invalid_configuration(field, format!("cannot reserve requested storage: {error}"))
    })?;
    Ok(values)
}

fn shuffled_copy(indices: &[usize], seed: u64) -> Result<Vec<usize>> {
    let mut shuffled = try_vec(indices.len(), "indices")?;
    shuffled.extend_from_slice(indices);
    shuffled.shuffle(&mut ChaCha12Rng::seed_from_u64(seed));
    Ok(shuffled)
}

fn weighted_indices(
    weights: &[f64],
    num_samples: usize,
    replacement: bool,
    seed: u64,
    distribution: &WeightedIndex<f64>,
) -> Result<Vec<usize>> {
    let mut rng = ChaCha12Rng::seed_from_u64(seed);
    if replacement {
        let mut output = try_vec(num_samples, "num_samples")?;
        output.extend((0..num_samples).map(|_| distribution.sample(&mut rng)));
        return Ok(output);
    }

    let mut positive = try_vec(weights.len(), "weights")?;
    positive.extend(
        weights
            .iter()
            .enumerate()
            .filter(|(_, weight)| **weight > 0.0)
            .map(|(index, &weight)| {
                let draw: f64 = rng.sample(Open01);
                (index, weight.ln() - (-draw.ln()).ln())
            }),
    );
    positive.sort_unstable_by(|(left_index, left_key), (right_index, right_key)| {
        right_key
            .total_cmp(left_key)
            .then_with(|| left_index.cmp(right_index))
    });

    let mut output = try_vec(num_samples, "num_samples")?;
    output.extend(
        positive
            .into_iter()
            .take(num_samples)
            .map(|(index, _)| index),
    );
    if output.len() < num_samples {
        let mut zero_weight_indices = try_vec(weights.len(), "weights")?;
        zero_weight_indices.extend(
            weights
                .iter()
                .enumerate()
                .filter_map(|(index, &weight)| (weight == 0.0).then_some(index)),
        );
        zero_weight_indices.shuffle(&mut rng);
        output.extend(
            zero_weight_indices
                .into_iter()
                .take(num_samples - output.len()),
        );
    }
    Ok(output)
}
