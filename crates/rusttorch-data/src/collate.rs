use std::{
    collections::{BTreeMap, TryReserveError},
    convert::Infallible,
    error::Error,
    fmt,
};

use rusttorch_core::{RustTorchError, Tensor};

/// Converts an owned group of samples into one batch.
///
/// See [`DefaultCollator`], [`DefaultConverter`], [`FnCollate`], and
/// [`VecCollate`] for compiling examples of the built-in implementations.
pub trait Collate<Sample> {
    /// Batch produced from the samples.
    type Batch;

    /// Error returned when collation fails.
    type Error;

    /// Collates `samples` into one batch.
    fn collate(&mut self, samples: Vec<Sample>) -> std::result::Result<Self::Batch, Self::Error>;
}

/// Moves samples unchanged into their existing vector.
///
/// # Examples
///
/// ```
/// use rusttorch_data::{Collate, VecCollate};
///
/// let mut collator = VecCollate;
/// let batch = collator
///     .collate(vec![String::from("first"), String::from("second")])
///     .expect("VecCollate is infallible");
/// assert_eq!(batch, ["first", "second"]);
/// ```
#[derive(Clone, Copy, Debug, Default)]
pub struct VecCollate;

impl<Sample> Collate<Sample> for VecCollate {
    type Batch = Vec<Sample>;
    type Error = Infallible;

    fn collate(&mut self, samples: Vec<Sample>) -> std::result::Result<Self::Batch, Self::Error> {
        Ok(samples)
    }
}

/// Adapts an ordinary fallible closure to [`Collate`].
///
/// # Examples
///
/// ```
/// use std::convert::Infallible;
/// use rusttorch_data::{Collate, FnCollate};
///
/// # fn main() -> Result<(), Infallible> {
/// let mut collator = FnCollate::new(|samples: Vec<Vec<i64>>| {
///     Ok::<_, Infallible>(samples.into_iter().flatten().collect::<Vec<_>>())
/// });
/// assert_eq!(collator.collate(vec![vec![1, 2], vec![3]])?, [1, 2, 3]);
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Copy, Debug)]
pub struct FnCollate<F> {
    collate: F,
}

impl<F> FnCollate<F> {
    /// Creates a collator backed by `collate`.
    pub fn new(collate: F) -> Self {
        Self { collate }
    }
}

impl<Sample, Batch, E, F> Collate<Sample> for FnCollate<F>
where
    F: FnMut(Vec<Sample>) -> std::result::Result<Batch, E>,
{
    type Batch = Batch;
    type Error = E;

    fn collate(&mut self, samples: Vec<Sample>) -> std::result::Result<Self::Batch, Self::Error> {
        (self.collate)(samples)
    }
}

/// A byte-string record that is not treated as a recursive numeric sequence.
///
/// `Vec<u8>` remains available for equal-length sequence collation. Wrap a
/// byte string in `Bytes` when each sample must remain one byte record.
///
/// # Examples
///
/// ```
/// use rusttorch_data::{Bytes, Collate, DefaultCollator};
///
/// let mut collator = DefaultCollator;
/// let records = collator
///     .collate(vec![Bytes(vec![1, 2]), Bytes(vec![3])])
///     .expect("byte records collate");
/// assert_eq!(records, [Bytes(vec![1, 2]), Bytes(vec![3])]);
/// ```
#[derive(Clone, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct Bytes(pub Vec<u8>);

/// An error produced by built-in collation or conversion.
///
/// The [`DefaultCollator`] and [`DefaultConverter`] examples show the public
/// operations that return this error.
#[derive(Debug)]
#[non_exhaustive]
pub enum CollateError {
    /// A collator received no top-level samples.
    EmptyBatch,
    /// A no-batching converter received other than one sample.
    ConversionBatchSize {
        /// Number of samples received.
        actual: usize,
    },
    /// One sequence had a different length from the first sample.
    SequenceLengthMismatch {
        /// Zero-based sample position containing the mismatch.
        sample: usize,
        /// Length of the first sequence.
        expected: usize,
        /// Length of the mismatched sequence.
        actual: usize,
    },
    /// One option had different presence from the first sample.
    OptionPresenceMismatch {
        /// Zero-based sample position containing the mismatch.
        sample: usize,
        /// Whether the first option was present.
        expected_some: bool,
        /// Whether the mismatched option was present.
        actual_some: bool,
    },
    /// One map had a different ordered key set from the first sample.
    MapKeysMismatch {
        /// Zero-based sample position containing the mismatch.
        sample: usize,
    },
    /// A caller-controlled size cannot be represented by the native API.
    SizeOverflow {
        /// Operation whose size was not representable.
        context: &'static str,
        /// Size supplied by the caller.
        value: usize,
    },
    /// A required intermediate allocation could not be reserved.
    Allocation {
        /// Structure whose allocation failed.
        context: &'static str,
        /// Number of additional elements requested.
        requested: usize,
        /// Standard-library allocation error.
        source: TryReserveError,
    },
    /// LibTorch rejected tensor construction or stacking.
    LibTorch {
        /// Preserved backend error and source chain.
        source: RustTorchError,
    },
}

impl fmt::Display for CollateError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyBatch => formatter.write_str("cannot collate an empty batch"),
            Self::ConversionBatchSize { actual } => write!(
                formatter,
                "default conversion requires exactly one sample, got {actual}"
            ),
            Self::SequenceLengthMismatch {
                sample,
                expected,
                actual,
            } => write!(
                formatter,
                "sequence sample {sample} has length {actual}, expected {expected}"
            ),
            Self::OptionPresenceMismatch {
                sample,
                expected_some,
                actual_some,
            } => write!(
                formatter,
                "option sample {sample} presence is {actual_some}, expected {expected_some}"
            ),
            Self::MapKeysMismatch { sample } => {
                write!(formatter, "map sample {sample} has a different key set")
            }
            Self::SizeOverflow { context, value } => {
                write!(formatter, "{context} size {value} is not representable")
            }
            Self::Allocation {
                context,
                requested,
                source,
            } => write!(
                formatter,
                "failed to reserve {requested} elements for {context}: {source}"
            ),
            Self::LibTorch { source } => write!(formatter, "LibTorch collation failed: {source}"),
        }
    }
}

impl Error for CollateError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Allocation { source, .. } => Some(source),
            Self::LibTorch { source } => Some(source),
            _ => None,
        }
    }
}

fn reserve_vec<T>(
    capacity: usize,
    context: &'static str,
) -> std::result::Result<Vec<T>, CollateError> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(capacity)
        .map_err(|source| CollateError::Allocation {
            context,
            requested: capacity,
            source,
        })?;
    Ok(values)
}

fn require_nonempty<T>(samples: &[T]) -> std::result::Result<(), CollateError> {
    if samples.is_empty() {
        Err(CollateError::EmptyBatch)
    } else {
        Ok(())
    }
}

fn backend_error(source: impl Into<RustTorchError>) -> CollateError {
    CollateError::LibTorch {
        source: source.into(),
    }
}

/// Recursively converts one Rust sample type into its default batch type.
///
/// Tensor samples stack on dimension zero. Scalar kinds are `u8` → `Uint8`,
/// `i8` → `Int8`, `i16` → `Int16`, `i32` → `Int`, `i64` → `Int64`, `f32` →
/// `Float`, `f64` → `Double`, and `bool` → `Bool`. Strings and [`Bytes`]
/// remain records. Options, equal-length vectors, tuples of arity two through
/// eight, and `BTreeMap` values recurse through this trait.
///
/// Use this trait through [`DefaultCollator`], whose example demonstrates
/// numeric, Tensor, and typed recursive output.
pub trait DefaultCollate: Sized {
    /// Batch produced from a vector of this sample type.
    type Batch;

    /// Applies built-in typed collation.
    fn default_collate(samples: Vec<Self>) -> std::result::Result<Self::Batch, CollateError>;
}

/// Applies [`DefaultCollate`] to an owned group of samples.
///
/// # Examples
///
/// ```
/// use rusttorch_core::{Kind, Tensor};
/// use rusttorch_data::{Bytes, Collate, CollateError, DefaultCollator};
///
/// # fn main() -> Result<(), CollateError> {
/// let mut collator = DefaultCollator;
/// let numbers: Tensor = collator.collate(vec![1_i64, 2])?;
/// assert_eq!(numbers.kind(), Kind::Int64);
/// assert_eq!(numbers.size(), [2]);
///
/// let tensors = collator.collate(vec![
///     Tensor::from_slice(&[1_i64, 2]),
///     Tensor::from_slice(&[3_i64, 4]),
/// ])?;
/// assert_eq!(tensors.size(), [2, 2]);
///
/// let nested: (Tensor, Vec<Bytes>) = collator.collate(vec![
///     (1_i64, Bytes(vec![1, 2])),
///     (2_i64, Bytes(vec![3])),
/// ])?;
/// assert_eq!(nested.0.size(), [2]);
/// assert_eq!(nested.1, [Bytes(vec![1, 2]), Bytes(vec![3])]);
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Copy, Debug, Default)]
pub struct DefaultCollator;

impl<Sample> Collate<Sample> for DefaultCollator
where
    Sample: DefaultCollate,
{
    type Batch = Sample::Batch;
    type Error = CollateError;

    fn collate(&mut self, samples: Vec<Sample>) -> std::result::Result<Self::Batch, Self::Error> {
        Sample::default_collate(samples)
    }
}

impl DefaultCollate for Tensor {
    type Batch = Tensor;

    fn default_collate(samples: Vec<Self>) -> std::result::Result<Self::Batch, CollateError> {
        require_nonempty(&samples)?;
        i32::try_from(samples.len()).map_err(|_| CollateError::SizeOverflow {
            context: "tensor stack",
            value: samples.len(),
        })?;
        Tensor::f_stack(&samples, 0).map_err(backend_error)
    }
}

macro_rules! impl_numeric_collate {
    ($($type:ty),+ $(,)?) => {
        $(
            impl DefaultCollate for $type {
                type Batch = Tensor;

                fn default_collate(
                    samples: Vec<Self>,
                ) -> std::result::Result<Self::Batch, CollateError> {
                    require_nonempty(&samples)?;
                    i64::try_from(samples.len()).map_err(|_| CollateError::SizeOverflow {
                        context: "numeric tensor",
                        value: samples.len(),
                    })?;
                    Tensor::f_from_slice(&samples).map_err(backend_error)
                }
            }
        )+
    };
}

impl_numeric_collate!(u8, i8, i16, i32, i64, f32, f64, bool);

impl DefaultCollate for String {
    type Batch = Vec<String>;

    fn default_collate(samples: Vec<Self>) -> std::result::Result<Self::Batch, CollateError> {
        require_nonempty(&samples)?;
        Ok(samples)
    }
}

impl DefaultCollate for Bytes {
    type Batch = Vec<Bytes>;

    fn default_collate(samples: Vec<Self>) -> std::result::Result<Self::Batch, CollateError> {
        require_nonempty(&samples)?;
        Ok(samples)
    }
}

impl<T> DefaultCollate for Option<T>
where
    T: DefaultCollate,
{
    type Batch = Option<T::Batch>;

    fn default_collate(samples: Vec<Self>) -> std::result::Result<Self::Batch, CollateError> {
        require_nonempty(&samples)?;
        let expected_some = samples[0].is_some();
        for (sample, value) in samples.iter().enumerate().skip(1) {
            let actual_some = value.is_some();
            if actual_some != expected_some {
                return Err(CollateError::OptionPresenceMismatch {
                    sample,
                    expected_some,
                    actual_some,
                });
            }
        }
        if !expected_some {
            return Ok(None);
        }

        let sample_count = samples.len();
        let mut present = reserve_vec(sample_count, "option values")?;
        for value in samples {
            match value {
                Some(value) => present.push(value),
                None => {
                    return Err(CollateError::OptionPresenceMismatch {
                        sample: present.len(),
                        expected_some: true,
                        actual_some: false,
                    });
                }
            }
        }
        T::default_collate(present).map(Some)
    }
}

impl<T> DefaultCollate for Vec<T>
where
    T: DefaultCollate,
{
    type Batch = Vec<T::Batch>;

    fn default_collate(samples: Vec<Self>) -> std::result::Result<Self::Batch, CollateError> {
        require_nonempty(&samples)?;
        let sequence_len = samples[0].len();
        for (sample, sequence) in samples.iter().enumerate().skip(1) {
            if sequence.len() != sequence_len {
                return Err(CollateError::SequenceLengthMismatch {
                    sample,
                    expected: sequence_len,
                    actual: sequence.len(),
                });
            }
        }

        let sample_count = samples.len();
        let mut columns = reserve_vec(sequence_len, "sequence columns")?;
        for _ in 0..sequence_len {
            columns.push(reserve_vec(sample_count, "sequence column values")?);
        }
        for sequence in samples {
            for (column, value) in columns.iter_mut().zip(sequence) {
                column.push(value);
            }
        }

        let mut batch = reserve_vec(sequence_len, "collated sequence")?;
        for column in columns {
            batch.push(T::default_collate(column)?);
        }
        Ok(batch)
    }
}

impl<K, V> DefaultCollate for BTreeMap<K, V>
where
    K: Ord,
    V: DefaultCollate,
{
    type Batch = BTreeMap<K, V::Batch>;

    fn default_collate(samples: Vec<Self>) -> std::result::Result<Self::Batch, CollateError> {
        require_nonempty(&samples)?;
        for (sample, map) in samples.iter().enumerate().skip(1) {
            if !samples[0].keys().eq(map.keys()) {
                return Err(CollateError::MapKeysMismatch { sample });
            }
        }

        let sample_count = samples.len();
        let mut maps = samples.into_iter();
        let first = maps.next().ok_or(CollateError::EmptyBatch)?;
        let mut columns = BTreeMap::new();
        for (key, value) in first {
            let mut values = reserve_vec(sample_count, "map values")?;
            values.push(value);
            columns.insert(key, values);
        }
        for map in maps {
            for (values, value) in columns.values_mut().zip(map.into_values()) {
                values.push(value);
            }
        }

        let mut batch = BTreeMap::new();
        for (key, values) in columns {
            batch.insert(key, V::default_collate(values)?);
        }
        Ok(batch)
    }
}

/// Recursively converts one sample without adding a batch dimension.
///
/// Tensors, supported scalar primitives, strings, and [`Bytes`] are preserved.
/// Options, vectors, tuples of arity two through eight, and `BTreeMap` values
/// are converted recursively. Unlike [`DefaultCollate`], vectors are not
/// transposed.
///
/// Use this trait through [`DefaultConverter`], whose example demonstrates
/// recursive conversion without automatic batching.
pub trait DefaultConvert: Sized {
    /// Output produced from this sample type.
    type Output;

    /// Applies built-in conversion without automatic batching.
    fn default_convert(self) -> std::result::Result<Self::Output, CollateError>;
}

/// Applies [`DefaultConvert`] in no-auto-batching mode.
///
/// # Examples
///
/// ```
/// use rusttorch_data::{Bytes, Collate, CollateError, DefaultConverter};
///
/// # fn main() -> Result<(), CollateError> {
/// let mut converter = DefaultConverter;
/// assert_eq!(converter.convert(vec![1_i64, 2])?, [1, 2]);
/// assert_eq!(
///     converter.convert(Some(Bytes(vec![3, 4])))?,
///     Some(Bytes(vec![3, 4])),
/// );
///
/// // The `Collate` adapter accepts the loader's one-sample no-batch group.
/// assert_eq!(converter.collate(vec![vec![5_i64, 6]])?, [5, 6]);
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Copy, Debug, Default)]
pub struct DefaultConverter;

impl DefaultConverter {
    /// Converts one sample without adding a batch dimension.
    pub fn convert<T>(&mut self, sample: T) -> std::result::Result<T::Output, CollateError>
    where
        T: DefaultConvert,
    {
        sample.default_convert()
    }
}

impl<Sample> Collate<Sample> for DefaultConverter
where
    Sample: DefaultConvert,
{
    type Batch = Sample::Output;
    type Error = CollateError;

    fn collate(&mut self, samples: Vec<Sample>) -> std::result::Result<Self::Batch, Self::Error> {
        let actual = samples.len();
        if actual != 1 {
            return Err(CollateError::ConversionBatchSize { actual });
        }
        match samples.into_iter().next() {
            Some(sample) => sample.default_convert(),
            None => Err(CollateError::ConversionBatchSize { actual: 0 }),
        }
    }
}

macro_rules! impl_leaf_convert {
    ($($type:ty),+ $(,)?) => {
        $(
            impl DefaultConvert for $type {
                type Output = Self;

                fn default_convert(self) -> std::result::Result<Self::Output, CollateError> {
                    Ok(self)
                }
            }
        )+
    };
}

impl_leaf_convert!(Tensor, u8, i8, i16, i32, i64, f32, f64, bool, String, Bytes);

impl<T> DefaultConvert for Option<T>
where
    T: DefaultConvert,
{
    type Output = Option<T::Output>;

    fn default_convert(self) -> std::result::Result<Self::Output, CollateError> {
        self.map(DefaultConvert::default_convert).transpose()
    }
}

impl<T> DefaultConvert for Vec<T>
where
    T: DefaultConvert,
{
    type Output = Vec<T::Output>;

    fn default_convert(self) -> std::result::Result<Self::Output, CollateError> {
        let mut output = reserve_vec(self.len(), "converted sequence")?;
        for value in self {
            output.push(value.default_convert()?);
        }
        Ok(output)
    }
}

impl<K, V> DefaultConvert for BTreeMap<K, V>
where
    K: Ord,
    V: DefaultConvert,
{
    type Output = BTreeMap<K, V::Output>;

    fn default_convert(self) -> std::result::Result<Self::Output, CollateError> {
        let mut output = BTreeMap::new();
        for (key, value) in self {
            output.insert(key, value.default_convert()?);
        }
        Ok(output)
    }
}

macro_rules! impl_tuple {
    ($(($type:ident, $index:tt, $value:ident)),+ $(,)?) => {
        impl<$($type),+> DefaultCollate for ($($type,)+)
        where
            $($type: DefaultCollate),+
        {
            type Batch = ($($type::Batch,)+);

            fn default_collate(
                samples: Vec<Self>,
            ) -> std::result::Result<Self::Batch, CollateError> {
                require_nonempty(&samples)?;
                let sample_count = samples.len();
                let mut fields = ($(reserve_vec::<$type>(sample_count, "tuple field")?,)+);
                for ($($value,)+) in samples {
                    $(fields.$index.push($value);)+
                }
                Ok(($($type::default_collate(fields.$index)?,)+))
            }
        }

        impl<$($type),+> DefaultConvert for ($($type,)+)
        where
            $($type: DefaultConvert),+
        {
            type Output = ($($type::Output,)+);

            fn default_convert(self) -> std::result::Result<Self::Output, CollateError> {
                let ($($value,)+) = self;
                Ok(($($value.default_convert()?,)+))
            }
        }
    };
}

impl_tuple!((A, 0, a), (B, 1, b));
impl_tuple!((A, 0, a), (B, 1, b), (C, 2, c));
impl_tuple!((A, 0, a), (B, 1, b), (C, 2, c), (D, 3, d));
impl_tuple!((A, 0, a), (B, 1, b), (C, 2, c), (D, 3, d), (E, 4, e));
impl_tuple!(
    (A, 0, a),
    (B, 1, b),
    (C, 2, c),
    (D, 3, d),
    (E, 4, e),
    (F, 5, f)
);
impl_tuple!(
    (A, 0, a),
    (B, 1, b),
    (C, 2, c),
    (D, 3, d),
    (E, 4, e),
    (F, 5, f),
    (G, 6, g)
);
impl_tuple!(
    (A, 0, a),
    (B, 1, b),
    (C, 2, c),
    (D, 3, d),
    (E, 4, e),
    (F, 5, f),
    (G, 6, g),
    (H, 7, h)
);
