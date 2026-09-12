//! Train neural networks, run inference, and build tensor pipelines in Rust.
//!
//! RustTorch combines [`Tensor`] operations and automatic differentiation with
//! fallible [neural-network layers](nn), [optimizers](optim), [data loaders](data),
//! and [model-weight storage](interop). Computation runs through LibTorch;
//! tensors are the same Rust types exposed by [`tch`], so existing tensor code
//! can be used directly. Operations with an `f_` prefix return errors that can
//! be propagated with `?`.
//!
//! # Choose an API for your task
//!
//! | Task | Start here |
//! | --- | --- |
//! | Transform numerical data or compute gradients | [`Tensor`], [`Kind`], [`no_grad`] |
//! | Build a model from layers | [`nn::Sequential`], [`nn::Module`] |
//! | Extract image features | [`nn::Conv2d`], [`nn::BatchNorm2d`], [`nn::AdaptiveAvgPool2d`] |
//! | Model sequences or token relationships | [`nn::Lstm`], [`nn::MultiheadAttention`], [`nn::TransformerConfig`] |
//! | Train a regressor or classifier | [`nn::functional`], [`optim::Adam`], [`optim::Sgd`] |
//! | Batch samples or load data in workers | [`data::DataLoader`], [`data::TensorDataset`] |
//! | Resume training or schedule updates | [`optim::OptimizerState`], [`optim::StepLr`], [`amp::GradScaler`] |
//! | Save or restore model parameters | [`nn::Sequential::save_weights`], [`interop`] |
//! | Inspect an explicit computation graph | [`graph`] |
//!
//! # Work with tensors
//!
//! Tensor shapes describe the axes of your data. For example, a matrix with
//! shape `[2, 3]` contains two samples with three features each. Multiplying
//! it by a `[3, 1]` weight matrix produces one value per sample.
//!
//! ```
//! use rusttorch::{Device, Kind, Result, Tensor};
//!
//! # fn main() -> Result<()> {
//! let samples = Tensor::f_from_slice(&[1_f32, 2., 3., 4., 5., 6.])?
//!     .f_reshape([2, 3])?;
//! let weights = Tensor::f_ones([3, 1], (Kind::Float, Device::Cpu))?;
//! let totals = samples.f_matmul(&weights)?;
//!
//! assert_eq!(totals.size(), [2, 1]);
//! assert_eq!(totals.double_value(&[0, 0]), 6.0);
//! # Ok(())
//! # }
//! ```
//!
//! # Train a model and run inference
//!
//! This regressor learns one output from one input feature. A model owns its
//! parameter store; the optimizer tracks those parameters and updates them
//! from a scalar loss. Mean squared error is suitable for continuous targets.
//! For class labels, use [`nn::functional::cross_entropy`] with raw class
//! scores and integer targets.
//!
//! ```
//! use rusttorch::nn::{Sequential, functional};
//! use rusttorch::optim::Adam;
//! use rusttorch::{DeviceSpec, Result, Tensor, no_grad};
//!
//! # fn main() -> Result<()> {
//! let mut model = Sequential::builder()
//!     .linear(1, 1)
//!     .build(DeviceSpec::Cpu)?;
//! let input = Tensor::f_from_slice(&[-1_f32, 0., 1., 2.])?.f_reshape([4, 1])?;
//! let target = Tensor::f_from_slice(&[-2_f32, 1., 4., 7.])?.f_reshape([4, 1])?;
//! let mut optimizer = Adam::builder()
//!     .learning_rate(0.05)
//!     .build(model.var_store())?;
//!
//! model.train();
//! for _ in 0..100 {
//!     let prediction = model.forward(&input)?;
//!     let loss = functional::mse_loss(&prediction, &target)?;
//!     optimizer.backward_step(&loss)?;
//! }
//!
//! model.eval();
//! let prediction = no_grad(|| model.forward(&input))?;
//! assert_eq!(prediction.size(), [4, 1]);
//! # Ok(())
//! # }
//! ```
//!
//! [`nn::Sequential::eval`] changes layer behavior, such as disabling dropout.
//! [`no_grad`] separately disables gradient recording during inference. New
//! sequential models start in training mode.
//!
//! # Save and restore weights
//!
//! SafeTensors files store named parameters and buffers. Rebuild the same model
//! architecture before loading; strict loading checks parameter names, shapes,
//! and dtypes. The file does not contain the model architecture or optimizer
//! state. Use this pattern after training to save weights for inference:
//!
//! ```no_run
//! use std::path::Path;
//! use rusttorch::nn::Sequential;
//! use rusttorch::{DeviceSpec, Result};
//!
//! fn save_regressor(model: &Sequential, path: &Path) -> Result<()> {
//!     model.save_weights(path)
//! }
//!
//! fn load_regressor(path: &Path) -> Result<Sequential> {
//!     let mut model = Sequential::builder()
//!         .linear(1, 1)
//!         .build(DeviceSpec::Cpu)?;
//!     model.load_weights(path)?;
//!     model.eval();
//!     Ok(model)
//! }
//! ```
//!
//! Use a path ending in `.safetensors`, such as `regressor.safetensors`.
//! [`interop::StateDictMapping`] handles files whose parameter names differ.
//!
//! # Devices and installation
//!
//! The examples use CPU for portability. [`DeviceSpec::Auto`] selects an
//! available accelerator before falling back to CPU. Explicit CUDA or MPS
//! requests return an error if unavailable. Create or move inputs onto
//! [`nn::Sequential::device`] before passing them to the model.
//!
//! RustTorch requires a compatible LibTorch runtime. See the
//! [installation guide](https://github.com/newpoluton-alt/RustTorch#installation)
//! for setup, and the [repository documentation](https://github.com/newpoluton-alt/RustTorch/tree/main/docs)
//! for additional guides and the feature roadmap.

#![deny(missing_docs)]

pub mod amp;
pub mod data;
pub mod device;
pub mod error;
pub mod graph;
pub mod interop;
pub mod nn;
pub mod optim;

pub use device::{DeviceCapabilities, DeviceSpec, available_devices, resolve_device};
pub use error::{Result, RustTorchError};
pub use rusttorch_core::{Device, Kind, Reduction, Tensor, manual_seed, no_grad, no_grad_guard};

// Keep the GitHub training recipes executable without adding a public API.
#[cfg(doctest)]
#[doc = include_str!("../docs/training.md")]
mod training_guide {}

#[cfg(doctest)]
#[doc = include_str!("../docs/sequence-models.md")]
mod sequence_guide {}
