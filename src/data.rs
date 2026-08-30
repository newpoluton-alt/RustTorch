//! Fallible datasets and deterministic sampling utilities.

use std::{ops::Range, vec::IntoIter};

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
