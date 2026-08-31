use std::{ops::Range, vec::IntoIter};

use rand::{
    Rng, SeedableRng,
    distributions::{Distribution, Open01, WeightedIndex},
    seq::SliceRandom,
};
use rand_chacha::ChaCha12Rng;
use rusttorch_core::{Result, RustTorchError};

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
    /// Returns [`RustTorchError::InvalidConfiguration`] when `length` is zero.
    pub fn new(length: usize, seed: u64) -> Result<Self> {
        Self::without_replacement(length, length, seed)
    }

    /// Creates a sampler that draws `num_samples` indices with replacement.
    ///
    /// # Errors
    ///
    /// Returns [`RustTorchError::InvalidConfiguration`] when `length` or
    /// `num_samples` is zero.
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
    /// `num_samples` is zero.
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
        let indices = random_indices(length, num_samples, replacement, seed).into_iter();
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
    pub fn new(indices: Vec<usize>, seed: u64) -> Self {
        let shuffled = shuffled_copy(&indices, seed).into_iter();
        Self {
            source: indices,
            seed,
            epoch: 0,
            indices: shuffled,
        }
    }

    fn indices_for_epoch(&self) -> Vec<usize> {
        shuffled_copy(&self.source, epoch_seed(self.seed, self.epoch))
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
    /// `num_samples` is zero; or when a without-replacement count exceeds the
    /// number of weights.
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
            weighted_indices(&weights, num_samples, replacement, seed, &distribution).into_iter();
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

fn validate_positive(value: usize, field: &'static str) -> Result<()> {
    if value == 0 {
        return Err(invalid_configuration(field, "must be greater than zero"));
    }
    Ok(())
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

fn random_indices(
    length: usize,
    num_samples: usize,
    replacement: Replacement,
    seed: u64,
) -> Vec<usize> {
    let mut rng = ChaCha12Rng::seed_from_u64(seed);
    match replacement {
        Replacement::With => (0..num_samples).map(|_| rng.gen_range(0..length)).collect(),
        Replacement::Without => {
            let mut output = Vec::with_capacity(num_samples);
            while output.len() < num_samples {
                let mut permutation = (0..length).collect::<Vec<_>>();
                permutation.shuffle(&mut rng);
                let remaining = num_samples - output.len();
                output.extend(permutation.into_iter().take(remaining));
            }
            output
        }
    }
}

fn shuffled_copy(indices: &[usize], seed: u64) -> Vec<usize> {
    let mut shuffled = indices.to_vec();
    shuffled.shuffle(&mut ChaCha12Rng::seed_from_u64(seed));
    shuffled
}

fn weighted_indices(
    weights: &[f64],
    num_samples: usize,
    replacement: bool,
    seed: u64,
    distribution: &WeightedIndex<f64>,
) -> Vec<usize> {
    let mut rng = ChaCha12Rng::seed_from_u64(seed);
    if replacement {
        return (0..num_samples)
            .map(|_| distribution.sample(&mut rng))
            .collect();
    }

    let mut positive = weights
        .iter()
        .enumerate()
        .filter(|(_, weight)| **weight > 0.0)
        .map(|(index, &weight)| {
            let draw: f64 = rng.sample(Open01);
            (index, draw.ln() / weight)
        })
        .collect::<Vec<_>>();
    positive.sort_unstable_by(|(left_index, left_key), (right_index, right_key)| {
        right_key
            .total_cmp(left_key)
            .then_with(|| left_index.cmp(right_index))
    });

    let mut output = positive
        .into_iter()
        .take(num_samples)
        .map(|(index, _)| index)
        .collect::<Vec<_>>();
    if output.len() < num_samples {
        let mut zero_weight_indices = weights
            .iter()
            .enumerate()
            .filter_map(|(index, &weight)| (weight == 0.0).then_some(index))
            .collect::<Vec<_>>();
        zero_weight_indices.shuffle(&mut rng);
        output.extend(
            zero_weight_indices
                .into_iter()
                .take(num_samples - output.len()),
        );
    }
    output
}
