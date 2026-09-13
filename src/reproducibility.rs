//! Repeat stochastic application steps without sharing LibTorch's global RNG.
//!
//! [`TensorRng`] assigns one ChaCha12 stream to each draw call. Its small state
//! can be saved next to optimizer and loader state to reproduce the next draw.
//! This is a RustTorch sequence, not a claim of PyTorch RNG stream identity.
//!
//! ```
//! use rusttorch::{Device, Kind, reproducibility::TensorRng};
//! let mut noise = TensorRng::new(42);
//! let saved = noise.state_dict();
//! let first = noise.normal(&[4, 2], (Kind::Float, Device::Cpu))?;
//! let mut restored = TensorRng::from_state(saved)?;
//! assert!(first.f_equal(&restored.normal(&[4, 2], (Kind::Float, Device::Cpu))?)?);
//! # Ok::<(), rusttorch::RustTorchError>(())
//! ```

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha12Rng;
use serde::{Deserialize, Serialize};

use crate::{Device, Kind, Result, RustTorchError, Tensor, data::ResourceLimits};

/// Startup settings for LibTorch's process-wide seed and CPU thread count.
///
/// Apply before concurrent model work. Other threads share these settings;
/// changing the seed does not restore an advanced RNG stream or guarantee
/// deterministic kernels across platforms. Use [`TensorRng`] for isolated,
/// checkpointable application noise and loader seeds for data ordering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeConfig {
    /// Seed passed to LibTorch's native generator.
    pub seed: i64,
    /// Positive CPU intra-operation thread count, or retain the current count.
    pub threads: Option<i32>,
}

impl RuntimeConfig {
    /// Seeds LibTorch and optionally changes its CPU thread count after validation.
    ///
    /// ```no_run
    /// rusttorch::reproducibility::RuntimeConfig { seed: 42, threads: Some(1) }.apply()?;
    /// # Ok::<(), rusttorch::RustTorchError>(())
    /// ```
    pub fn apply(self) -> Result<()> {
        if self.threads.is_some_and(|threads| threads <= 0) {
            return Err(invalid("runtime thread count must be positive"));
        }
        if let Some(threads) = self.threads {
            tch::set_num_threads(threads);
        }
        crate::manual_seed(self.seed);
        Ok(())
    }
}

/// Versioned state representing the next stochastic tensor draw.
///
/// Serialize with `serde_json` and restore with [`TensorRng::from_state`].
/// The schema identifies ChaCha12, one stream per draw, and the documented
/// uniform/Box-Muller transformations; unknown versions are rejected.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TensorRngState {
    version: u32,
    seed: u64,
    next_stream: u64,
}

/// Deterministic tensor noise with independent state and bounded allocations.
///
/// Float32 and Float64 samples are generated on CPU, then copied to the
/// requested device. Float32 uniforms use the native `rand` Float32 sampler
/// so rounding cannot introduce the excluded endpoint 1. Normal samples use
/// the Box-Muller transform in Float64 before casting. Cross-platform libm
/// differences can affect normal samples' last bits; exact resume targets the
/// same build/platform. These routines do not change the global torch seed.
#[derive(Debug, Clone)]
pub struct TensorRng {
    state: TensorRngState,
    limits: ResourceLimits,
}

impl TensorRng {
    /// Starts at stream zero using finite default resource limits.
    pub fn new(seed: u64) -> Self {
        Self {
            state: TensorRngState {
                version: 1,
                seed,
                next_stream: 0,
            },
            limits: ResourceLimits::default(),
        }
    }

    /// Sets explicit generation limits; these are policy and are not checkpoint state.
    #[must_use]
    pub const fn with_limits(mut self, limits: ResourceLimits) -> Self {
        self.limits = limits;
        self
    }

    /// Captures the next draw position without retaining tensor values.
    pub fn state_dict(&self) -> TensorRngState {
        self.state.clone()
    }

    /// Validates a saved sequence version/counter and restores with default limits.
    pub fn from_state(state: TensorRngState) -> Result<Self> {
        if state.version != 1 || state.next_stream == u64::MAX {
            return Err(invalid(
                "unsupported RNG state version or exhausted stream counter",
            ));
        }
        Ok(Self {
            state,
            limits: ResourceLimits::default(),
        })
    }

    /// Draws uniform values in `[0, 1)`; a successful call consumes one stream.
    ///
    /// Shape must have at most 64 nonnegative dimensions. Invalid shape, resource,
    /// dtype or backend requests return an error without advancing the stream.
    pub fn uniform(&mut self, shape: &[i64], options: (Kind, Device)) -> Result<Tensor> {
        self.draw(shape, options, false)
    }

    /// Draws zero-mean, unit-variance normal noise; see the module example.
    ///
    /// A successful call consumes one stream even for an empty tensor. Each
    /// draw starts independently, so changing its shape does not change the
    /// next draw's stream. Failed calls preserve the previous state.
    pub fn normal(&mut self, shape: &[i64], options: (Kind, Device)) -> Result<Tensor> {
        self.draw(shape, options, true)
    }

    fn draw(
        &mut self,
        shape: &[i64],
        (kind, device): (Kind, Device),
        normal: bool,
    ) -> Result<Tensor> {
        if !matches!(kind, Kind::Float | Kind::Double) || shape.len() > 64 {
            return Err(invalid(
                "noise requires Float32/Float64 and at most 64 dimensions",
            ));
        }
        let next = self
            .state
            .next_stream
            .checked_add(1)
            .filter(|&value| value < u64::MAX)
            .ok_or_else(|| invalid("RNG stream counter exhausted"))?;
        let dimensions = shape
            .iter()
            .map(|&d| {
                usize::try_from(d).map_err(|_| invalid("noise dimensions must be nonnegative"))
            })
            .collect::<Result<Vec<_>>>()?;
        let count = self.limits.tensor_elements(&dimensions)?;
        let bytes = count
            .checked_mul(kind.elt_size_in_bytes())
            .ok_or_else(|| invalid("noise byte size overflow"))?;
        self.limits
            .check("noise bytes", bytes, self.limits.max_decoded_bytes)?;
        let mut rng = ChaCha12Rng::seed_from_u64(self.state.seed);
        rng.set_stream(self.state.next_stream);
        let mut sample = || {
            let u = 1. - rng.r#gen::<f64>();
            let angle = std::f64::consts::TAU * rng.r#gen::<f64>();
            (-2. * u.ln()).sqrt() * angle.cos()
        };
        let tensor = if kind == Kind::Float {
            let mut values = Vec::new();
            values
                .try_reserve_exact(count)
                .map_err(|_| invalid("cannot reserve noise payload"))?;
            if normal {
                values.extend((0..count).map(|_| sample() as f32));
            } else {
                values.extend((0..count).map(|_| rng.r#gen::<f32>()));
            }
            Tensor::f_from_slice(&values)?
        } else {
            let mut values = Vec::new();
            values
                .try_reserve_exact(count)
                .map_err(|_| invalid("cannot reserve noise payload"))?;
            if normal {
                values.extend((0..count).map(|_| sample()));
            } else {
                values.extend((0..count).map(|_| rng.r#gen::<f64>()));
            }
            Tensor::f_from_slice(&values)?
        };
        let result = tensor.f_reshape(shape)?.f_to_device(device)?;
        self.state.next_stream = next;
        Ok(result)
    }
}

fn invalid(reason: &str) -> RustTorchError {
    RustTorchError::InvalidConfiguration {
        field: "reproducibility",
        reason: reason.into(),
    }
}
