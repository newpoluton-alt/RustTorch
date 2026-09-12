//! Recurrent sequence models and masked attention for time series and tokens.

use tch::{Kind, Tensor, nn::Init, no_grad};

use super::{LayerNorm, LayerNormConfig, Linear, LinearConfig, Module, ParameterPath, functional};
use crate::{Result, RustTorchError, device::ensure_device};

fn invalid(field: &'static str, reason: &str) -> RustTorchError {
    RustTorchError::InvalidConfiguration {
        field,
        reason: reason.to_owned(),
    }
}

fn real_parameters(path: &ParameterPath<'_>) -> Result<()> {
    if !matches!(
        path.kind(),
        Kind::Float | Kind::Double | Kind::Half | Kind::BFloat16
    ) {
        return Err(invalid(
            "parameter dtype",
            "sequence models require float, double, half, or bfloat16 parameters",
        ));
    }
    Ok(())
}

fn same_tensor(input: &Tensor, reference: &Tensor, context: &'static str) -> Result<()> {
    ensure_device(context, input, reference.device())?;
    if input.kind() != reference.kind() {
        return Err(invalid(
            context,
            "tensor dtype must match the model parameters",
        ));
    }
    Ok(())
}

fn product(field: &'static str, a: i64, b: i64) -> Result<i64> {
    a.checked_mul(b)
        .ok_or_else(|| invalid(field, "dimension product overflows i64"))
}

/// Nonlinearity for a vanilla recurrent network. See [`RnnConfig::build_rnn`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum RnnActivation {
    /// Bounded hyperbolic tangent activation, the default.
    #[default]
    Tanh,
    /// Rectified linear activation.
    Relu,
}

/// Configure a recurrent model for time series, speech features, or token embeddings.
///
/// Defaults are one layer, sequence-first input, biased parameters, no dropout,
/// one direction, and no LSTM projection. Hidden states always put layers first.
/// Packed sequences and recurrent cells are separate, currently unsupported APIs.
///
/// ```
/// use rusttorch::{Device, Kind, Tensor, nn::{RnnConfig, VarStore}};
/// let store = VarStore::new(Device::Cpu);
/// let model = RnnConfig::new(3, 8).num_layers(2).batch_first(true)
///     .bidirectional(true).dropout(0.1).build_lstm(&store.root())?;
/// let series = Tensor::zeros([4, 10, 3], (Kind::Float, Device::Cpu));
/// let (features, (hidden, cell)) = model.forward_t(&series, None, true)?;
/// assert_eq!(features.size(), [4, 10, 16]);
/// assert_eq!(hidden.size(), [4, 4, 8]);
/// assert_eq!(cell.size(), hidden.size());
/// # Ok::<(), rusttorch::RustTorchError>(())
/// ```
#[derive(Debug, Clone, Copy)]
pub struct RnnConfig {
    input_size: i64,
    hidden_size: i64,
    num_layers: i64,
    bias: bool,
    batch_first: bool,
    dropout: f64,
    bidirectional: bool,
    projection: i64,
}

impl RnnConfig {
    /// Creates a model configuration; input and hidden widths must be positive.
    pub const fn new(input_size: i64, hidden_size: i64) -> Self {
        Self {
            input_size,
            hidden_size,
            num_layers: 1,
            bias: true,
            batch_first: false,
            dropout: 0.,
            bidirectional: false,
            projection: 0,
        }
    }
    /// Sets the number of stacked recurrent layers; see the type example.
    #[must_use]
    pub const fn num_layers(mut self, value: i64) -> Self {
        self.num_layers = value;
        self
    }
    /// Enables input and recurrent bias vectors. Defaults to true.
    #[must_use]
    pub const fn bias(mut self, value: bool) -> Self {
        self.bias = value;
        self
    }
    /// Uses `[batch, time, features]` instead of `[time, batch, features]`.
    #[must_use]
    pub const fn batch_first(mut self, value: bool) -> Self {
        self.batch_first = value;
        self
    }
    /// Sets dropout between recurrent layers during training, in `[0, 1]`.
    #[must_use]
    pub const fn dropout(mut self, value: f64) -> Self {
        self.dropout = value;
        self
    }
    /// Concatenates forward and reverse features and returns both final states.
    #[must_use]
    pub const fn bidirectional(mut self, value: bool) -> Self {
        self.bidirectional = value;
        self
    }
    /// Projects LSTM hidden states to this width; zero disables projection.
    ///
    /// A positive width must be less than `hidden_size`. Cell states retain the
    /// original hidden width. RNN and GRU reject nonzero projections.
    #[must_use]
    pub const fn projection(mut self, value: i64) -> Self {
        self.projection = value;
        self
    }
    /// Registers a tanh or ReLU recurrent network with stable `weight_ih_l0`
    /// and `weight_hh_l0` parameter names (and `_reverse` for reverse layers).
    ///
    /// ```
    /// # use rusttorch::{Device, Kind, Tensor, nn::{VarStore, RnnConfig, RnnActivation}};
    /// # let store = VarStore::new(Device::Cpu);
    /// let model = RnnConfig::new(2, 4).build_rnn(&store.root(), RnnActivation::Tanh)?;
    /// let (output, state) = model.forward_t(&Tensor::zeros([3, 2], (Kind::Float, Device::Cpu)), None, false)?;
    /// assert_eq!(state.size(), [1, 4]);
    /// # Ok::<(), rusttorch::RustTorchError>(())
    /// ```
    pub fn build_rnn(self, path: &ParameterPath<'_>, activation: RnnActivation) -> Result<Rnn> {
        Ok(Rnn {
            core: Recurrent::build(self, path, 1)?,
            activation,
        })
    }
    /// Registers a long short-term memory network; see the type example.
    pub fn build_lstm(self, path: &ParameterPath<'_>) -> Result<Lstm> {
        Ok(Lstm {
            core: Recurrent::build(self, path, 4)?,
        })
    }
    /// Registers a gated recurrent network for compact sequence encoders.
    ///
    /// ```
    /// # use rusttorch::{Device, Kind, Tensor, nn::{VarStore, RnnConfig}};
    /// # let store = VarStore::new(Device::Cpu);
    /// let model = RnnConfig::new(2, 4).build_gru(&store.root())?;
    /// let input = Tensor::zeros([3, 1, 2], (Kind::Float, Device::Cpu));
    /// let (output, state) = model.forward_t(&input, None, true)?;
    /// assert_eq!(output.size(), [3, 1, 4]);
    /// # Ok::<(), rusttorch::RustTorchError>(())
    /// ```
    pub fn build_gru(self, path: &ParameterPath<'_>) -> Result<Gru> {
        Ok(Gru {
            core: Recurrent::build(self, path, 3)?,
        })
    }
}

#[derive(Debug)]
struct Recurrent {
    config: RnnConfig,
    weights: Vec<Tensor>,
}

impl Recurrent {
    // Parameter order and uniform bounds follow torch/nn/modules/rnn.py,
    // PyTorch v2.13.0 (cf30153). See THIRD_PARTY_NOTICES.md.
    fn build(config: RnnConfig, path: &ParameterPath<'_>, gates: i64) -> Result<Self> {
        real_parameters(path)?;
        functional::validate_dropout(config.dropout)?;
        if config.input_size <= 0 || config.hidden_size <= 0 || config.num_layers <= 0 {
            return Err(invalid(
                "recurrent dimensions",
                "input_size, hidden_size, and num_layers must be positive",
            ));
        }
        if config.projection < 0
            || config.projection >= config.hidden_size
            || (gates != 4 && config.projection != 0)
        {
            return Err(invalid(
                "projection",
                "only LSTM accepts projections; width must be nonnegative and less than hidden_size",
            ));
        }
        let directions = if config.bidirectional { 2 } else { 1 };
        let width = if config.projection == 0 {
            config.hidden_size
        } else {
            config.projection
        };
        let gate_size = product("hidden_size", config.hidden_size, gates)?;
        let layer_width = product("hidden_size", width, directions)?;
        let layer_count = product("num_layers", config.num_layers, directions)?;
        let parameters_per_layer =
            2 + i64::from(config.bias) * 2 + i64::from(config.projection > 0);
        if product("recurrent parameters", layer_count, parameters_per_layer)? > i64::from(i32::MAX)
        {
            return Err(invalid(
                "num_layers",
                "parameter count exceeds the native recurrent interface limit",
            ));
        }
        for size in [config.input_size, layer_width, width] {
            product("recurrent weights", gate_size, size)?;
        }
        let bound = 1. / (config.hidden_size as f64).sqrt();
        let init = Init::Uniform {
            lo: -bound,
            up: bound,
        };
        let mut weights = Vec::new();
        for layer in 0..config.num_layers {
            for direction in 0..directions {
                let suffix = if direction == 0 { "" } else { "_reverse" };
                let input_width = if layer == 0 {
                    config.input_size
                } else {
                    layer_width
                };
                for (name, shape) in [
                    ("weight_ih", vec![gate_size, input_width]),
                    ("weight_hh", vec![gate_size, width]),
                ] {
                    weights.push(path.f_var(&format!("{name}_l{layer}{suffix}"), &shape, init)?);
                }
                if config.bias {
                    for name in ["bias_ih", "bias_hh"] {
                        weights.push(path.f_var(
                            &format!("{name}_l{layer}{suffix}"),
                            &[gate_size],
                            init,
                        )?);
                    }
                }
                if config.projection > 0 {
                    weights.push(path.f_var(
                        &format!("weight_hr_l{layer}{suffix}"),
                        &[width, config.hidden_size],
                        init,
                    )?);
                }
            }
        }
        Ok(Self { config, weights })
    }
    fn input(&self, input: &Tensor) -> Result<(Tensor, bool, i64)> {
        same_tensor(input, &self.weights[0], "recurrent input")?;
        let size = input.size();
        if !(2..=3).contains(&size.len())
            || size.last() != Some(&self.config.input_size)
            || size[..size.len() - 1].contains(&0)
        {
            return Err(invalid(
                "recurrent input",
                "expected nonempty [time, features] or rank-three sequences with the configured feature width",
            ));
        }
        let unbatched = size.len() == 2;
        let input = if unbatched {
            input.f_unsqueeze(1)?
        } else {
            input.shallow_clone()
        };
        let batch_dim = usize::from(!self.config.batch_first || unbatched);
        let batch = input.size()[batch_dim];
        Ok((input, unbatched, batch))
    }
    fn state(
        &self,
        state: Option<&Tensor>,
        batch: i64,
        width: i64,
        unbatched: bool,
    ) -> Result<Tensor> {
        let layers = self.config.num_layers * if self.config.bidirectional { 2 } else { 1 };
        let expected = if unbatched {
            vec![layers, width]
        } else {
            vec![layers, batch, width]
        };
        if let Some(state) = state {
            same_tensor(state, &self.weights[0], "recurrent state")?;
            if state.size() != expected {
                return Err(RustTorchError::ShapeMismatch {
                    name: "recurrent state".to_owned(),
                    expected,
                    actual: state.size(),
                });
            }
            Ok(if unbatched {
                state.f_unsqueeze(1)?
            } else {
                state.shallow_clone()
            })
        } else {
            Ok(Tensor::f_zeros(
                [layers, batch, width],
                (self.weights[0].kind(), self.weights[0].device()),
            )?)
        }
    }
    fn width(&self) -> i64 {
        if self.config.projection == 0 {
            self.config.hidden_size
        } else {
            self.config.projection
        }
    }
}

/// A vanilla recurrent network; construct it with [`RnnConfig::build_rnn`].
/// [`Module`] execution starts at zero, returns only outputs, and defaults to evaluation.
#[derive(Debug)]
pub struct Rnn {
    core: Recurrent,
    activation: RnnActivation,
}
/// A gated recurrent network; construct it with [`RnnConfig::build_gru`].
/// [`Module`] execution starts at zero, returns only outputs, and defaults to evaluation.
#[derive(Debug)]
pub struct Gru {
    core: Recurrent,
}
/// A long short-term memory network; construct it with [`RnnConfig::build_lstm`].
/// [`Module`] execution starts at zero, returns only outputs, and defaults to evaluation.
#[derive(Debug)]
pub struct Lstm {
    core: Recurrent,
}

impl Rnn {
    /// Returns every time-step output and the final hidden state. Pass a previous
    /// state to continue a stream, or `None` to start at zero. See [`RnnConfig`].
    pub fn forward_t(
        &self,
        input: &Tensor,
        state: Option<&Tensor>,
        training: bool,
    ) -> Result<(Tensor, Tensor)> {
        let (input, unbatched, batch) = self.core.input(input)?;
        let state = self
            .core
            .state(state, batch, self.core.width(), unbatched)?;
        let c = self.core.config;
        let (output, hidden) = match self.activation {
            RnnActivation::Tanh => input.f_rnn_tanh(
                &state,
                &self.core.weights,
                c.bias,
                c.num_layers,
                c.dropout,
                training,
                c.bidirectional,
                c.batch_first && !unbatched,
            )?,
            RnnActivation::Relu => input.f_rnn_relu(
                &state,
                &self.core.weights,
                c.bias,
                c.num_layers,
                c.dropout,
                training,
                c.bidirectional,
                c.batch_first && !unbatched,
            )?,
        };
        Ok(if unbatched {
            (output.f_squeeze_dim(1)?, hidden.f_squeeze_dim(1)?)
        } else {
            (output, hidden)
        })
    }
}

impl Gru {
    /// Returns all outputs and the final hidden state, with optional initial
    /// state and explicit dropout mode. See [`RnnConfig::build_gru`].
    pub fn forward_t(
        &self,
        input: &Tensor,
        state: Option<&Tensor>,
        training: bool,
    ) -> Result<(Tensor, Tensor)> {
        let (input, unbatched, batch) = self.core.input(input)?;
        let state = self
            .core
            .state(state, batch, self.core.width(), unbatched)?;
        let c = self.core.config;
        let (output, hidden) = input.f_gru(
            &state,
            &self.core.weights,
            c.bias,
            c.num_layers,
            c.dropout,
            training,
            c.bidirectional,
            c.batch_first && !unbatched,
        )?;
        Ok(if unbatched {
            (output.f_squeeze_dim(1)?, hidden.f_squeeze_dim(1)?)
        } else {
            (output, hidden)
        })
    }
}

impl Lstm {
    /// Returns `(outputs, (hidden, cell))`. An initial state continues a stream;
    /// `None` creates zero states. Projected models retain full-width cells.
    /// See [`RnnConfig`] for a training example.
    pub fn forward_t(
        &self,
        input: &Tensor,
        state: Option<(&Tensor, &Tensor)>,
        training: bool,
    ) -> Result<(Tensor, (Tensor, Tensor))> {
        let (input, unbatched, batch) = self.core.input(input)?;
        let hidden = self
            .core
            .state(state.map(|s| s.0), batch, self.core.width(), unbatched)?;
        let cell = self.core.state(
            state.map(|s| s.1),
            batch,
            self.core.config.hidden_size,
            unbatched,
        )?;
        let c = self.core.config;
        let (output, hidden, cell) = input.f_lstm(
            &[hidden, cell],
            &self.core.weights,
            c.bias,
            c.num_layers,
            c.dropout,
            training,
            c.bidirectional,
            c.batch_first && !unbatched,
        )?;
        Ok(if unbatched {
            (
                output.f_squeeze_dim(1)?,
                (hidden.f_squeeze_dim(1)?, cell.f_squeeze_dim(1)?),
            )
        } else {
            (output, (hidden, cell))
        })
    }
}

impl Module for Rnn {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        Ok(self.forward_t(input, None, false)?.0)
    }
    fn forward_t(&self, input: &Tensor, training: bool) -> Result<Tensor> {
        Ok(self.forward_t(input, None, training)?.0)
    }
}

impl Module for Gru {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        Ok(self.forward_t(input, None, false)?.0)
    }
    fn forward_t(&self, input: &Tensor, training: bool) -> Result<Tensor> {
        Ok(self.forward_t(input, None, training)?.0)
    }
}

impl Module for Lstm {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        Ok(self.forward_t(input, None, false)?.0)
    }
    fn forward_t(&self, input: &Tensor, training: bool) -> Result<Tensor> {
        Ok(self.forward_t(input, None, training)?.0)
    }
}

/// Borrowed restrictions for attention. Boolean `true` blocks a position;
/// floating masks add to attention scores. Both can be combined with causality.
///
/// ```
/// use rusttorch::nn::AttentionMask;
/// let causal = AttentionMask { causal: true, ..Default::default() };
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct AttentionMask<'a> {
    /// `[target_time, source_time]` or `[batch * heads, target_time, source_time]`.
    pub attention: Option<&'a Tensor>,
    /// `[batch, source_time]`, or `[source_time]` for unbatched input.
    pub key_padding: Option<&'a Tensor>,
    /// Blocks positions above the diagonal, including for cross attention.
    pub causal: bool,
}

/// Configure learned multihead attention over equal-width query, key, and value
/// vectors. Defaults: sequence-first input, bias enabled, and zero dropout.
///
/// ```
/// use rusttorch::{Device, Kind, Tensor, nn::{VarStore, MultiheadAttentionConfig, AttentionMask}};
/// let store = VarStore::new(Device::Cpu);
/// let attention = MultiheadAttentionConfig::new(8, 2).batch_first(true).build(&store.root())?;
/// let tokens = Tensor::zeros([3, 5, 8], (Kind::Float, Device::Cpu));
/// let mask = AttentionMask { causal: true, ..Default::default() };
/// let (features, weights) = attention.forward_t(&tokens, &tokens, &tokens, mask, false)?;
/// assert_eq!(features.size(), [3, 5, 8]);
/// assert_eq!(weights.size(), [3, 5, 5]);
/// # Ok::<(), rusttorch::RustTorchError>(())
/// ```
#[derive(Debug, Clone, Copy)]
pub struct MultiheadAttentionConfig {
    embed_dim: i64,
    num_heads: i64,
    dropout: f64,
    bias: bool,
    batch_first: bool,
}

impl MultiheadAttentionConfig {
    /// Creates a configuration; the feature width must be divisible by heads.
    pub const fn new(embed_dim: i64, num_heads: i64) -> Self {
        Self {
            embed_dim,
            num_heads,
            dropout: 0.,
            bias: true,
            batch_first: false,
        }
    }
    /// Sets dropout on attention weights during training. See the type example.
    #[must_use]
    pub const fn dropout(mut self, value: f64) -> Self {
        self.dropout = value;
        self
    }
    /// Enables projection biases, initialized to zero.
    #[must_use]
    pub const fn bias(mut self, value: bool) -> Self {
        self.bias = value;
        self
    }
    /// Uses `[batch, time, features]` for batched inputs.
    #[must_use]
    pub const fn batch_first(mut self, value: bool) -> Self {
        self.batch_first = value;
        self
    }
    fn validate(self, path: &ParameterPath<'_>) -> Result<()> {
        real_parameters(path)?;
        functional::validate_dropout(self.dropout)?;
        if self.embed_dim <= 0 || self.num_heads <= 0 || self.embed_dim % self.num_heads != 0 {
            return Err(invalid(
                "attention dimensions",
                "positive embed_dim must be divisible by positive num_heads",
            ));
        }
        product(
            "attention weights",
            product("embed_dim", self.embed_dim, 3)?,
            self.embed_dim,
        )?;
        Ok(())
    }
    /// Registers `in_proj_weight`, optional `in_proj_bias`, and `out_proj`
    /// parameters. See the type example for causal token attention.
    pub fn build(self, path: &ParameterPath<'_>) -> Result<MultiheadAttention> {
        self.validate(path)?;
        let bound = (6. / (4. * self.embed_dim as f64)).sqrt();
        let mut in_proj_weight =
            path.f_zeros("in_proj_weight", &[3 * self.embed_dim, self.embed_dim])?;
        let in_proj_bias = self
            .bias
            .then(|| path.f_zeros("in_proj_bias", &[3 * self.embed_dim]))
            .transpose()?;
        let out_proj = LinearConfig::new(self.embed_dim, self.embed_dim)
            .bias(self.bias)
            .build(&(path / "out_proj"))?;
        let _ = no_grad(|| in_proj_weight.f_uniform_(-bound, bound))?;
        if let Some(bias) = out_proj.bias() {
            let _ = no_grad(|| bias.shallow_clone().f_zero_())?;
        }
        Ok(MultiheadAttention {
            config: self,
            in_proj_weight,
            in_proj_bias,
            out_proj,
        })
    }
}

/// Trainable self or cross attention. See [`MultiheadAttentionConfig`].
///
/// Equal query/key/value feature widths are supported. Separate `kdim`/`vdim`,
/// appended bias/zero tokens, nested tensors, and cached keys are not exposed.
/// [`Module`] execution performs unmasked self attention and defaults to evaluation.
#[derive(Debug)]
pub struct MultiheadAttention {
    config: MultiheadAttentionConfig,
    in_proj_weight: Tensor,
    in_proj_bias: Option<Tensor>,
    out_proj: Linear,
}

impl MultiheadAttention {
    /// Computes attention and head-averaged weights. Output follows the query
    /// layout; weights are `[batch, target_time, source_time]` or rank two for
    /// unbatched input. Use [`Self::forward_per_head_t`] to inspect each head.
    pub fn forward_t(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        mask: AttentionMask<'_>,
        training: bool,
    ) -> Result<(Tensor, Tensor)> {
        let (output, weights) = self.forward_per_head_t(query, key, value, mask, training)?;
        let dim = if weights.size().len() == 4 { 1 } else { 0 };
        Ok((output, weights.f_mean_dim(&[dim][..], false, None)?))
    }
    /// Returns separate attention maps `[batch, heads, target_time, source_time]`
    /// (without batch for rank-two inputs). Masks and dropout follow [`Self::forward_t`].
    /// Every tensor is checked before accessing metadata; invalid shapes, dtypes,
    /// and devices return errors. Fully blocked rows produce NaN weights, as a
    /// softmax over only negative infinity is undefined.
    pub fn forward_per_head_t(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        mask: AttentionMask<'_>,
        training: bool,
    ) -> Result<(Tensor, Tensor)> {
        let (output, weights) = self.compute(query, key, value, mask, training, true)?;
        Ok((output, weights.expect("attention weights requested")))
    }
    /// Computes only output features with native scaled dot-product attention.
    /// Fully blocked rows contribute zero features before output projection.
    /// Use this when attention maps are unnecessary; [`MultiheadAttentionConfig`]
    /// demonstrates the same input and mask conventions.
    pub fn forward_without_weights_t(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        mask: AttentionMask<'_>,
        training: bool,
    ) -> Result<Tensor> {
        Ok(self.compute(query, key, value, mask, training, false)?.0)
    }
    #[allow(clippy::too_many_arguments)]
    fn compute(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        mask: AttentionMask<'_>,
        training: bool,
        need_weights: bool,
    ) -> Result<(Tensor, Option<Tensor>)> {
        for (input, name) in [
            (query, "attention query"),
            (key, "attention key"),
            (value, "attention value"),
        ] {
            same_tensor(input, &self.in_proj_weight, name)?;
            if !(2..=3).contains(&input.size().len())
                || input.size().last() != Some(&self.config.embed_dim)
            {
                return Err(invalid(
                    name,
                    "expected rank-two or rank-three input ending in embed_dim",
                ));
            }
        }
        if query.size().len() != key.size().len() || key.size() != value.size() {
            return Err(invalid(
                "attention inputs",
                "query, key and value must have equal rank; key and value shapes must match",
            ));
        }
        let unbatched = query.size().len() == 2;
        let canonical = |x: &Tensor| -> Result<Tensor> {
            Ok(if unbatched {
                x.f_unsqueeze(0)?
            } else if self.config.batch_first {
                x.shallow_clone()
            } else {
                x.f_transpose(0, 1)?
            })
        };
        let q = canonical(query)?;
        let k = canonical(key)?;
        let v = canonical(value)?;
        let [batch, target, width] =
            <[i64; 3]>::try_from(q.size()).expect("validated attention rank");
        let source = k.size()[1];
        if k.size()[0] != batch {
            return Err(invalid(
                "attention batch",
                "query and key batch sizes must match",
            ));
        }
        let heads = self.config.num_heads;
        let head_width = width / heads;
        let project = |x: &Tensor, index: i64, time: i64| -> Result<Tensor> {
            let weight = self.in_proj_weight.f_narrow(0, index * width, width)?;
            let bias = self
                .in_proj_bias
                .as_ref()
                .map(|b| b.f_narrow(0, index * width, width))
                .transpose()?;
            Ok(x.f_linear(&weight, bias.as_ref())?
                .f_reshape([batch, time, heads, head_width])?
                .f_transpose(1, 2)?)
        };
        let q = project(&q, 0, target)?;
        let k = project(&k, 1, source)?;
        let v = project(&v, 2, source)?;
        let mut bias = None;
        if let Some(attention) = mask.attention {
            check_mask(attention, query)?;
            let shape = attention.size();
            let attention = if shape == [target, source] {
                attention.f_unsqueeze(0)?.f_unsqueeze(0)?
            } else if shape == [product("attention batch", batch, heads)?, target, source] {
                attention.f_reshape([batch, heads, target, source])?
            } else {
                return Err(invalid(
                    "attention mask",
                    "expected [target, source] or [batch * heads, target, source]",
                ));
            };
            bias = merge_mask(bias, &attention, query.kind())?;
        }
        if let Some(padding) = mask.key_padding {
            check_mask(padding, query)?;
            let expected = if unbatched {
                vec![source]
            } else {
                vec![batch, source]
            };
            if padding.size() != expected {
                return Err(RustTorchError::ShapeMismatch {
                    name: "key padding mask".to_owned(),
                    expected,
                    actual: padding.size(),
                });
            }
            bias = merge_mask(
                bias,
                &padding.f_reshape([batch, 1, 1, source])?,
                query.kind(),
            )?;
        }
        if mask.causal {
            bias = merge_mask(
                bias,
                &Tensor::f_ones([target, source], (Kind::Bool, query.device()))?.f_triu(1)?,
                query.kind(),
            )?;
        }
        let (attended, weights) = if need_weights {
            // Scale before the product so low-precision dot products do not
            // overflow when their scaled attention scores are representable.
            let scores = q
                .f_mul_scalar((1. / head_width as f64).sqrt())?
                .f_matmul(&k.f_transpose(-2, -1)?)?;
            let scores = if let Some(bias) = &bias {
                scores.f_add(bias)?
            } else {
                scores
            };
            let weights = scores
                .f_softmax(-1, None)?
                .f_dropout(self.config.dropout, training)?;
            (weights.f_matmul(&v)?, Some(weights))
        } else {
            (
                Tensor::f_scaled_dot_product_attention(
                    &q,
                    &k,
                    &v,
                    bias.as_ref(),
                    if training { self.config.dropout } else { 0. },
                    false,
                    None,
                    false,
                )?,
                None,
            )
        };
        // Keep sequence-first projection order so subsequent dropout sees the
        // same element layout as the standard dense attention implementation.
        let output = self
            .out_proj
            .forward(
                &attended
                    .f_permute([2, 0, 1, 3])?
                    .f_contiguous()?
                    .f_reshape([product("attention output", target, batch)?, width])?,
            )?
            .f_reshape([target, batch, width])?;
        Ok(if unbatched {
            (
                output.f_squeeze_dim(1)?,
                weights.map(|w| w.f_squeeze_dim(0)).transpose()?,
            )
        } else if self.config.batch_first {
            (output.f_transpose(0, 1)?, weights)
        } else {
            (output, weights)
        })
    }
    fn parameters(&self) -> Vec<&Tensor> {
        let mut parameters = vec![&self.in_proj_weight];
        parameters.extend(self.in_proj_bias.iter());
        parameters.push(self.out_proj.weight());
        parameters.extend(self.out_proj.bias());
        parameters
    }
}

impl Module for MultiheadAttention {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.forward_without_weights_t(input, input, input, AttentionMask::default(), false)
    }
    fn forward_t(&self, input: &Tensor, training: bool) -> Result<Tensor> {
        self.forward_without_weights_t(input, input, input, AttentionMask::default(), training)
    }
}

fn check_mask(mask: &Tensor, input: &Tensor) -> Result<()> {
    ensure_device("attention mask", mask, input.device())?;
    if mask.kind() != Kind::Bool && mask.kind() != input.kind() {
        return Err(invalid(
            "attention mask",
            "expected bool or the query floating-point dtype",
        ));
    }
    Ok(())
}

fn merge_mask(current: Option<Tensor>, mask: &Tensor, kind: Kind) -> Result<Option<Tensor>> {
    let mask = if mask.kind() == Kind::Bool {
        Tensor::f_zeros(mask.size(), (kind, mask.device()))?
            .f_masked_fill(mask, f64::NEG_INFINITY)?
    } else {
        mask.shallow_clone()
    };
    Ok(Some(if let Some(current) = current {
        current.f_add(&mask)?
    } else {
        mask
    }))
}

/// Feed-forward activation for transformer layers; see [`TransformerConfig`].
#[derive(Debug, Clone, Copy, Default)]
pub enum TransformerActivation {
    /// Rectified linear units, the default.
    #[default]
    Relu,
    /// Exact Gaussian error linear units.
    Gelu,
}

/// Configure sequence encoders, autoregressive decoders, and encoder-decoder models.
///
/// Defaults are feed-forward width 2048, dropout 0.1, ReLU, normalization epsilon
/// 1e-5, normalization after residual addition, biased projections, sequence-first
/// input, and six layers in each half of a complete model. Token embeddings and
/// positional information belong to the calling model.
///
/// ```
/// use rusttorch::{Device, Kind, Tensor, nn::{VarStore, TransformerConfig, TransformerMasks, AttentionMask}};
/// let store = VarStore::new(Device::Cpu);
/// let model = TransformerConfig::new(8, 2).dim_feedforward(16).batch_first(true)
///     .num_encoder_layers(2).num_decoder_layers(2).build(&store.root())?;
/// let source = Tensor::zeros([3, 5, 8], (Kind::Float, Device::Cpu));
/// let target = Tensor::zeros([3, 4, 8], (Kind::Float, Device::Cpu));
/// let masks = TransformerMasks { target: AttentionMask { causal: true, ..Default::default() }, ..Default::default() };
/// let decoded = model.forward_t(&source, &target, masks, false)?;
/// assert_eq!(decoded.size(), [3, 4, 8]);
/// # Ok::<(), rusttorch::RustTorchError>(())
/// ```
#[derive(Debug, Clone, Copy)]
pub struct TransformerConfig {
    d_model: i64,
    num_heads: i64,
    dim_feedforward: i64,
    dropout: f64,
    activation: TransformerActivation,
    layer_norm_eps: f64,
    batch_first: bool,
    norm_first: bool,
    bias: bool,
    num_encoder_layers: i64,
    num_decoder_layers: i64,
}

impl TransformerConfig {
    /// Sets the model width and attention heads; see the type example.
    pub const fn new(d_model: i64, num_heads: i64) -> Self {
        Self {
            d_model,
            num_heads,
            dim_feedforward: 2048,
            dropout: 0.1,
            activation: TransformerActivation::Relu,
            layer_norm_eps: 1e-5,
            batch_first: false,
            norm_first: false,
            bias: true,
            num_encoder_layers: 6,
            num_decoder_layers: 6,
        }
    }
    /// Sets the hidden width of the two feed-forward projections.
    #[must_use]
    pub const fn dim_feedforward(mut self, value: i64) -> Self {
        self.dim_feedforward = value;
        self
    }
    /// Sets dropout on attention probabilities and residual/feed-forward branches.
    #[must_use]
    pub const fn dropout(mut self, value: f64) -> Self {
        self.dropout = value;
        self
    }
    /// Selects ReLU or exact GELU between feed-forward projections.
    #[must_use]
    pub const fn activation(mut self, value: TransformerActivation) -> Self {
        self.activation = value;
        self
    }
    /// Sets the nonnegative, finite layer normalization epsilon.
    #[must_use]
    pub const fn layer_norm_eps(mut self, value: f64) -> Self {
        self.layer_norm_eps = value;
        self
    }
    /// Uses `[batch, time, features]` for batched input.
    #[must_use]
    pub const fn batch_first(mut self, value: bool) -> Self {
        self.batch_first = value;
        self
    }
    /// Applies layer normalization before each branch when true.
    #[must_use]
    pub const fn norm_first(mut self, value: bool) -> Self {
        self.norm_first = value;
        self
    }
    /// Enables additive biases in projections and layer normalization.
    #[must_use]
    pub const fn bias(mut self, value: bool) -> Self {
        self.bias = value;
        self
    }
    /// Sets the positive encoder depth for [`Self::build`].
    #[must_use]
    pub const fn num_encoder_layers(mut self, value: i64) -> Self {
        self.num_encoder_layers = value;
        self
    }
    /// Sets the positive decoder depth for [`Self::build`].
    #[must_use]
    pub const fn num_decoder_layers(mut self, value: i64) -> Self {
        self.num_decoder_layers = value;
        self
    }
    fn validate(self, path: &ParameterPath<'_>) -> Result<()> {
        self.attention().validate(path)?;
        if self.dim_feedforward <= 0 {
            return Err(invalid("dim_feedforward", "must be positive"));
        }
        product("feed-forward weights", self.d_model, self.dim_feedforward)?;
        if !self.layer_norm_eps.is_finite() || self.layer_norm_eps < 0. {
            return Err(invalid("layer_norm_eps", "must be finite and nonnegative"));
        }
        Ok(())
    }
    fn attention(self) -> MultiheadAttentionConfig {
        MultiheadAttentionConfig::new(self.d_model, self.num_heads)
            .dropout(self.dropout)
            .bias(self.bias)
            .batch_first(self.batch_first)
    }
    fn validate_depth(self, num_layers: i64, field: &'static str) -> Result<()> {
        if num_layers <= 0 {
            return Err(invalid(field, "must be positive"));
        }
        product(
            field,
            product(field, self.d_model, self.dim_feedforward)?,
            num_layers,
        )?;
        Ok(())
    }
    fn norm(self, path: &ParameterPath<'_>) -> Result<LayerNorm> {
        LayerNormConfig::new([self.d_model])
            .eps(self.layer_norm_eps)
            .bias(self.bias)
            .build(path)
    }
    /// Registers one self-attention encoder block.
    ///
    /// ```
    /// # use rusttorch::{Device, Kind, Tensor, nn::{VarStore, TransformerConfig, AttentionMask}};
    /// # let store = VarStore::new(Device::Cpu);
    /// let encoder = TransformerConfig::new(4, 2).dim_feedforward(8).build_encoder_layer(&store.root())?;
    /// let input = Tensor::zeros([3, 1, 4], (Kind::Float, Device::Cpu));
    /// assert_eq!(encoder.forward_t(&input, AttentionMask::default(), false)?.size(), [3, 1, 4]);
    /// # Ok::<(), rusttorch::RustTorchError>(())
    /// ```
    pub fn build_encoder_layer(self, path: &ParameterPath<'_>) -> Result<TransformerEncoderLayer> {
        self.validate(path)?;
        Ok(TransformerEncoderLayer {
            block: TransformerBlock::build(self, path, false)?,
        })
    }
    /// Registers one decoder block with self attention and attention over memory.
    ///
    /// ```
    /// # use rusttorch::{Device, Kind, Tensor, nn::{VarStore, TransformerConfig, AttentionMask}};
    /// # let store = VarStore::new(Device::Cpu);
    /// let decoder = TransformerConfig::new(4, 2).dim_feedforward(8).build_decoder_layer(&store.root())?;
    /// let target = Tensor::zeros([3, 1, 4], (Kind::Float, Device::Cpu));
    /// let memory = Tensor::zeros([5, 1, 4], (Kind::Float, Device::Cpu));
    /// let causal = AttentionMask { causal: true, ..Default::default() };
    /// assert_eq!(decoder.forward_t(&target, &memory, causal, AttentionMask::default(), false)?.size(), [3, 1, 4]);
    /// # Ok::<(), rusttorch::RustTorchError>(())
    /// ```
    pub fn build_decoder_layer(self, path: &ParameterPath<'_>) -> Result<TransformerDecoderLayer> {
        self.validate(path)?;
        Ok(TransformerDecoderLayer {
            block: TransformerBlock::build(self, path, true)?,
        })
    }
    /// Registers an encoder stack under `layers.0`, `layers.1`, etc., optionally
    /// followed by `norm`. Layers begin with equal values in independent parameters.
    ///
    /// ```
    /// # use rusttorch::{Device, Kind, Tensor, nn::{VarStore, TransformerConfig, AttentionMask}};
    /// # let store = VarStore::new(Device::Cpu);
    /// let encoder = TransformerConfig::new(4, 2).dim_feedforward(8).build_encoder(&store.root(), 2, true)?;
    /// let input = Tensor::zeros([3, 1, 4], (Kind::Float, Device::Cpu));
    /// let output = encoder.forward_t(&input, AttentionMask::default(), false)?;
    /// # Ok::<(), rusttorch::RustTorchError>(())
    /// ```
    pub fn build_encoder(
        self,
        path: &ParameterPath<'_>,
        num_layers: i64,
        final_norm: bool,
    ) -> Result<TransformerEncoder> {
        self.validate(path)?;
        self.validate_depth(num_layers, "encoder layers")?;
        let mut layers: Vec<TransformerEncoderLayer> = Vec::new();
        for index in 0..num_layers {
            let layer = self.build_encoder_layer(&(path / "layers" / index))?;
            if let Some(first) = layers.first() {
                copy_parameters(&layer.block.parameters(), &first.block.parameters())?;
            }
            layers.push(layer);
        }
        let norm = final_norm
            .then(|| self.norm(&(path / "norm")))
            .transpose()?;
        Ok(TransformerEncoder { layers, norm })
    }
    /// Registers a decoder stack and optional final normalization. Layers have
    /// independent parameters initialized equally, as in [`Self::build_encoder`].
    pub fn build_decoder(
        self,
        path: &ParameterPath<'_>,
        num_layers: i64,
        final_norm: bool,
    ) -> Result<TransformerDecoder> {
        self.validate(path)?;
        self.validate_depth(num_layers, "decoder layers")?;
        let mut layers: Vec<TransformerDecoderLayer> = Vec::new();
        for index in 0..num_layers {
            let layer = self.build_decoder_layer(&(path / "layers" / index))?;
            if let Some(first) = layers.first() {
                copy_parameters(&layer.block.parameters(), &first.block.parameters())?;
            }
            layers.push(layer);
        }
        let norm = final_norm
            .then(|| self.norm(&(path / "norm")))
            .transpose()?;
        Ok(TransformerDecoder { layers, norm })
    }
    /// Registers a complete encoder-decoder model, including final normalization
    /// in both halves and Xavier-uniform matrix initialization. See the type example.
    pub fn build(self, path: &ParameterPath<'_>) -> Result<Transformer> {
        self.validate(path)?;
        self.validate_depth(self.num_encoder_layers, "encoder layers")?;
        self.validate_depth(self.num_decoder_layers, "decoder layers")?;
        let encoder = self.build_encoder(&(path / "encoder"), self.num_encoder_layers, true)?;
        let decoder = self.build_decoder(&(path / "decoder"), self.num_decoder_layers, true)?;
        for layer in &encoder.layers {
            xavier_matrices(&layer.block.parameters())?;
        }
        for layer in &decoder.layers {
            xavier_matrices(&layer.block.parameters())?;
        }
        Ok(Transformer { encoder, decoder })
    }
}

#[derive(Debug)]
struct TransformerBlock {
    config: TransformerConfig,
    self_attn: MultiheadAttention,
    cross_attn: Option<MultiheadAttention>,
    linear1: Linear,
    linear2: Linear,
    norms: Vec<LayerNorm>,
}

impl TransformerBlock {
    // Residual ordering and initialization follow torch/nn/modules/transformer.py,
    // PyTorch v2.13.0 (cf30153). See THIRD_PARTY_NOTICES.md.
    fn build(config: TransformerConfig, path: &ParameterPath<'_>, decoder: bool) -> Result<Self> {
        let self_attn = config.attention().build(&(path / "self_attn"))?;
        let cross_attn = decoder
            .then(|| config.attention().build(&(path / "multihead_attn")))
            .transpose()?;
        let linear1 = LinearConfig::new(config.d_model, config.dim_feedforward)
            .bias(config.bias)
            .build(&(path / "linear1"))?;
        let linear2 = LinearConfig::new(config.dim_feedforward, config.d_model)
            .bias(config.bias)
            .build(&(path / "linear2"))?;
        let norms = (1..=if decoder { 3 } else { 2 })
            .map(|i| config.norm(&(path / format!("norm{i}"))))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            config,
            self_attn,
            cross_attn,
            linear1,
            linear2,
            norms,
        })
    }
    fn feedforward(&self, input: &Tensor, training: bool) -> Result<Tensor> {
        let projected = self.linear1.forward(input)?;
        let activated = match self.config.activation {
            TransformerActivation::Relu => projected.f_relu()?,
            TransformerActivation::Gelu => projected.f_gelu("none")?,
        };
        Ok(self
            .linear2
            .forward(&activated.f_dropout(self.config.dropout, training)?)?
            .f_dropout(self.config.dropout, training)?)
    }
    fn forward(
        &self,
        input: &Tensor,
        memory: Option<&Tensor>,
        target_mask: AttentionMask<'_>,
        memory_mask: AttentionMask<'_>,
        training: bool,
    ) -> Result<Tensor> {
        let mut output = input.shallow_clone();
        let normalized = if self.config.norm_first {
            self.norms[0].forward(&output)?
        } else {
            output.shallow_clone()
        };
        let branch = self.self_attn.forward_without_weights_t(
            &normalized,
            &normalized,
            &normalized,
            target_mask,
            training,
        )?;
        output = output.f_add(&branch.f_dropout(self.config.dropout, training)?)?;
        if !self.config.norm_first {
            output = self.norms[0].forward(&output)?;
        }
        if let (Some(attention), Some(memory)) = (&self.cross_attn, memory) {
            let normalized = if self.config.norm_first {
                self.norms[1].forward(&output)?
            } else {
                output.shallow_clone()
            };
            let branch = attention.forward_without_weights_t(
                &normalized,
                memory,
                memory,
                memory_mask,
                training,
            )?;
            output = output.f_add(&branch.f_dropout(self.config.dropout, training)?)?;
            if !self.config.norm_first {
                output = self.norms[1].forward(&output)?;
            }
        }
        let norm = self.norms.last().expect("transformer has normalization");
        let normalized = if self.config.norm_first {
            norm.forward(&output)?
        } else {
            output.shallow_clone()
        };
        output = output.f_add(&self.feedforward(&normalized, training)?)?;
        if !self.config.norm_first {
            output = norm.forward(&output)?;
        }
        Ok(output)
    }
    fn parameters(&self) -> Vec<&Tensor> {
        let mut parameters = self.self_attn.parameters();
        if let Some(attention) = &self.cross_attn {
            parameters.extend(attention.parameters());
        }
        for linear in [&self.linear1, &self.linear2] {
            parameters.push(linear.weight());
            parameters.extend(linear.bias());
        }
        for norm in &self.norms {
            parameters.extend(norm.weight());
            parameters.extend(norm.bias());
        }
        parameters
    }
}

fn copy_parameters(target: &[&Tensor], source: &[&Tensor]) -> Result<()> {
    no_grad(|| -> Result<()> {
        for (target, source) in target.iter().zip(source) {
            target.shallow_clone().f_copy_(source)?;
        }
        Ok(())
    })
}

fn xavier_matrices(parameters: &[&Tensor]) -> Result<()> {
    no_grad(|| -> Result<()> {
        for parameter in parameters {
            if parameter.size().len() == 2 {
                let size = parameter.size();
                let bound = (6. / (size[0] as f64 + size[1] as f64)).sqrt();
                let _ = parameter.shallow_clone().f_uniform_(-bound, bound)?;
            }
        }
        Ok(())
    })
}

/// A self-attention and feed-forward residual block. See [`TransformerConfig::build_encoder_layer`].
/// [`Module`] execution uses no mask and defaults to evaluation.
#[derive(Debug)]
pub struct TransformerEncoderLayer {
    block: TransformerBlock,
}
/// A decoder block attending to both target tokens and encoder memory.
/// See [`TransformerConfig::build_decoder_layer`].
#[derive(Debug)]
pub struct TransformerDecoderLayer {
    block: TransformerBlock,
}
/// A stack of encoder layers with optional final normalization.
/// See [`TransformerConfig::build_encoder`].
/// [`Module`] execution uses no mask and defaults to evaluation.
#[derive(Debug)]
pub struct TransformerEncoder {
    layers: Vec<TransformerEncoderLayer>,
    norm: Option<LayerNorm>,
}
/// A stack of decoder layers with optional final normalization.
/// See [`TransformerConfig::build_decoder`].
#[derive(Debug)]
pub struct TransformerDecoder {
    layers: Vec<TransformerDecoderLayer>,
    norm: Option<LayerNorm>,
}
/// A complete encoder-decoder transformer. See [`TransformerConfig`] for a
/// causal decoding example. Embeddings and output heads can share its parameter store.
#[derive(Debug)]
pub struct Transformer {
    encoder: TransformerEncoder,
    decoder: TransformerDecoder,
}

/// Separate source, target, and memory attention restrictions for [`Transformer`].
///
/// Source padding normally also belongs in `memory.key_padding` to prevent the
/// decoder attending to padded encoder positions. See [`TransformerConfig`].
#[derive(Debug, Clone, Copy, Default)]
pub struct TransformerMasks<'a> {
    /// Encoder self-attention restrictions.
    pub source: AttentionMask<'a>,
    /// Decoder self-attention restrictions; enable `causal` for next-token training.
    pub target: AttentionMask<'a>,
    /// Decoder attention restrictions over encoder memory.
    pub memory: AttentionMask<'a>,
}

impl TransformerEncoderLayer {
    /// Transforms source features with explicit attention restrictions and dropout
    /// mode. See [`TransformerConfig::build_encoder_layer`].
    pub fn forward_t(
        &self,
        input: &Tensor,
        mask: AttentionMask<'_>,
        training: bool,
    ) -> Result<Tensor> {
        self.block
            .forward(input, None, mask, AttentionMask::default(), training)
    }
}

impl TransformerDecoderLayer {
    /// Transforms target features while attending to `memory`. All tensors share
    /// feature width, batch layout, dtype, and device. See [`TransformerConfig::build_decoder_layer`].
    pub fn forward_t(
        &self,
        input: &Tensor,
        memory: &Tensor,
        target_mask: AttentionMask<'_>,
        memory_mask: AttentionMask<'_>,
        training: bool,
    ) -> Result<Tensor> {
        self.block
            .forward(input, Some(memory), target_mask, memory_mask, training)
    }
}

impl TransformerEncoder {
    /// Applies every encoder layer and optional final normalization. See
    /// [`TransformerConfig::build_encoder`].
    pub fn forward_t(
        &self,
        input: &Tensor,
        mask: AttentionMask<'_>,
        training: bool,
    ) -> Result<Tensor> {
        let mut output = input.shallow_clone();
        for layer in &self.layers {
            output = layer.forward_t(&output, mask, training)?;
        }
        if let Some(norm) = &self.norm {
            output = norm.forward(&output)?;
        }
        Ok(output)
    }
}

impl TransformerDecoder {
    /// Applies every decoder layer and optional final normalization with separate
    /// target and memory masks. See [`TransformerConfig::build_decoder_layer`].
    pub fn forward_t(
        &self,
        input: &Tensor,
        memory: &Tensor,
        target_mask: AttentionMask<'_>,
        memory_mask: AttentionMask<'_>,
        training: bool,
    ) -> Result<Tensor> {
        let mut output = input.shallow_clone();
        for layer in &self.layers {
            output = layer.forward_t(&output, memory, target_mask, memory_mask, training)?;
        }
        if let Some(norm) = &self.norm {
            output = norm.forward(&output)?;
        }
        Ok(output)
    }
}

impl Transformer {
    /// Encodes `source`, then decodes `target` against the resulting memory.
    /// Both inputs must share batch size and model width. See [`TransformerConfig`].
    pub fn forward_t(
        &self,
        source: &Tensor,
        target: &Tensor,
        masks: TransformerMasks<'_>,
        training: bool,
    ) -> Result<Tensor> {
        let memory = self.encoder.forward_t(source, masks.source, training)?;
        self.decoder
            .forward_t(target, &memory, masks.target, masks.memory, training)
    }
}

impl Module for TransformerEncoderLayer {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.forward_t(input, AttentionMask::default(), false)
    }
    fn forward_t(&self, input: &Tensor, training: bool) -> Result<Tensor> {
        self.forward_t(input, AttentionMask::default(), training)
    }
}

impl Module for TransformerEncoder {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.forward_t(input, AttentionMask::default(), false)
    }
    fn forward_t(&self, input: &Tensor, training: bool) -> Result<Tensor> {
        self.forward_t(input, AttentionMask::default(), training)
    }
}
