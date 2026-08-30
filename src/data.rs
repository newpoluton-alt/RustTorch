//! Fallible datasets and deterministic sampling utilities.

use std::{marker::PhantomData, ops::Range, vec::IntoIter};

use rand::{SeedableRng, seq::SliceRandom};
use rand_chacha::ChaCha12Rng;

use crate::{Result, RustTorchError};

/// A finite, indexable collection of samples.
///
/// Implementations return owned samples so loaders can move them into batches
/// without cloning. Dataset-specific failures remain in [`Dataset::Error`].
pub trait Dataset {
    /// One owned item produced by the dataset.
    type Sample;

    /// An error returned while loading a sample.
    type Error;

    /// Returns the number of addressable samples.
    fn len(&self) -> usize;

    /// Loads the sample at `index`.
    fn get(&self, index: usize) -> std::result::Result<Self::Sample, Self::Error>;

    /// Returns `true` when the dataset has no samples.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Iterates over every sample in index order.
    ///
    /// The iterator borrows the dataset and calls [`Dataset::get`] lazily.
    fn samples(&self) -> DatasetSamples<'_, Self>
    where
        Self: Sized,
    {
        DatasetSamples {
            dataset: self,
            next_index: 0,
        }
    }
}

/// A borrowing, sequential iterator over a [`Dataset`].
pub struct DatasetSamples<'a, D> {
    dataset: &'a D,
    next_index: usize,
}

impl<D> Iterator for DatasetSamples<'_, D>
where
    D: Dataset,
{
    type Item = std::result::Result<D::Sample, D::Error>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.next_index >= self.dataset.len() {
            return None;
        }

        let index = self.next_index;
        self.next_index += 1;
        Some(self.dataset.get(index))
    }
}

/// An allocation-free sampler that yields indices in ascending order.
pub struct SequentialSampler {
    indices: Range<usize>,
}

impl SequentialSampler {
    /// Creates a sampler for indices `0..length`.
    pub fn new(length: usize) -> Self {
        Self { indices: 0..length }
    }
}

impl Iterator for SequentialSampler {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        self.indices.next()
    }
}

/// A seeded sampler that yields each index exactly once in shuffled order.
///
/// Shuffling uses a sampler-local ChaCha12 generator, so it does not alter
/// LibTorch's global random state. The exact ordering is not a PyTorch RNG
/// compatibility guarantee.
pub struct RandomSampler {
    indices: IntoIter<usize>,
}

impl RandomSampler {
    /// Creates a reproducible shuffled sampler for `0..length`.
    ///
    /// Returns an error when `length` is zero.
    pub fn new(length: usize, seed: u64) -> Result<Self> {
        if length == 0 {
            return Err(RustTorchError::InvalidConfiguration {
                field: "length",
                reason: "must be greater than zero".to_owned(),
            });
        }

        let mut indices = (0..length).collect::<Vec<_>>();
        indices.shuffle(&mut ChaCha12Rng::seed_from_u64(seed));
        Ok(Self {
            indices: indices.into_iter(),
        })
    }
}

impl Iterator for RandomSampler {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        self.indices.next()
    }
}

type IdentityCollate<T, E> = fn(Vec<T>) -> std::result::Result<Vec<T>, E>;

fn identity_batch<T, E>(samples: Vec<T>) -> std::result::Result<Vec<T>, E> {
    Ok(samples)
}

/// A single-threaded iterator over batches from a map-style [`Dataset`].
///
/// The loader borrows its dataset and owns both its sampler and collation
/// closure. Samples are moved into one pre-sized vector per batch without a
/// `Clone` requirement. Dataset and collation errors are yielded once and
/// then terminate the iterator.
pub struct DataLoader<'a, D, S, C, B, E>
where
    D: Dataset,
{
    dataset: &'a D,
    sampler: S,
    batch_size: usize,
    drop_last: bool,
    collate: C,
    exhausted: bool,
    output: PhantomData<fn() -> (B, E)>,
}

impl<'a, D, S> DataLoader<'a, D, S, IdentityCollate<D::Sample, D::Error>, Vec<D::Sample>, D::Error>
where
    D: Dataset,
    S: Iterator<Item = usize>,
{
    /// Creates a loader whose batches are vectors of owned samples.
    ///
    /// `dataset` is borrowed, while `sampler` is consumed by the loader.
    /// A short final batch is omitted when `drop_last` is `true`.
    ///
    /// # Errors
    ///
    /// Returns [`RustTorchError::InvalidConfiguration`] when `batch_size` is
    /// zero.
    pub fn new(dataset: &'a D, sampler: S, batch_size: usize, drop_last: bool) -> Result<Self> {
        Self::with_collate(
            dataset,
            sampler,
            batch_size,
            drop_last,
            identity_batch::<D::Sample, D::Error>,
        )
    }
}

impl<'a, D, S, C, B, E> DataLoader<'a, D, S, C, B, E>
where
    D: Dataset,
    S: Iterator<Item = usize>,
    C: FnMut(Vec<D::Sample>) -> std::result::Result<B, E>,
    E: From<D::Error>,
{
    /// Creates a loader with fallible custom collation.
    ///
    /// The closure receives ownership of exactly one vector of samples and
    /// may return any batch type. Dataset failures are converted into the
    /// closure's error type through [`From`]. A short final batch is omitted
    /// when `drop_last` is `true`.
    ///
    /// # Errors
    ///
    /// Returns [`RustTorchError::InvalidConfiguration`] when `batch_size` is
    /// zero. Dataset and collation failures are returned by iteration.
    pub fn with_collate(
        dataset: &'a D,
        sampler: S,
        batch_size: usize,
        drop_last: bool,
        collate: C,
    ) -> Result<Self> {
        if batch_size == 0 {
            return Err(RustTorchError::InvalidConfiguration {
                field: "batch_size",
                reason: "must be greater than zero".to_owned(),
            });
        }

        Ok(Self {
            dataset,
            sampler,
            batch_size,
            drop_last,
            collate,
            exhausted: false,
            output: PhantomData,
        })
    }
}

impl<D, S, C, B, E> Iterator for DataLoader<'_, D, S, C, B, E>
where
    D: Dataset,
    S: Iterator<Item = usize>,
    C: FnMut(Vec<D::Sample>) -> std::result::Result<B, E>,
    E: From<D::Error>,
{
    type Item = std::result::Result<B, E>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.exhausted {
            return None;
        }

        let first_index = match self.sampler.next() {
            Some(index) => index,
            None => {
                self.exhausted = true;
                return None;
            }
        };
        let first = match self.dataset.get(first_index) {
            Ok(sample) => sample,
            Err(error) => {
                self.exhausted = true;
                return Some(Err(E::from(error)));
            }
        };

        let mut samples = Vec::with_capacity(self.batch_size);
        samples.push(first);
        while samples.len() < self.batch_size {
            let index = match self.sampler.next() {
                Some(index) => index,
                None => {
                    self.exhausted = true;
                    if self.drop_last {
                        return None;
                    }
                    break;
                }
            };

            match self.dataset.get(index) {
                Ok(sample) => samples.push(sample),
                Err(error) => {
                    self.exhausted = true;
                    return Some(Err(E::from(error)));
                }
            }
        }

        let batch = (self.collate)(samples);
        if batch.is_err() {
            self.exhausted = true;
        }
        Some(batch)
    }
}
