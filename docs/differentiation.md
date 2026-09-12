# Gradients, sensitivity and probability models

Use gradients to train parameters, measure how predictions react to inputs, or
optimize a continuous latent variable. These examples run on CPU and use
fallible RustTorch APIs so native operation errors reach the caller.

## Get a gradient without changing training buffers

A scalar gradient answers how much a loss changes when each input changes.
`autograd::grad` returns the derivatives without accumulating into `.grad()`.
Use `create_graph` when another derivative is needed, such as a curvature penalty.

```rust
use rusttorch::{Kind, Result, Tensor, autograd::{self, GradOptions}};

fn main() -> Result<()> {
    let weights = Tensor::from_slice(&[2_f64, 3.]).set_requires_grad(true);
    let objective = weights.f_pow_tensor_scalar(3)?.f_sum(Kind::Double)?;
    let first = autograd::grad(&objective, &[&weights], GradOptions {
        create_graph: true, ..Default::default()
    })?.remove(0);
    let second = autograd::grad(
        &first.f_sum(Kind::Double)?, &[&weights], GradOptions::default()
    )?.remove(0);
    assert_eq!(Vec::<f64>::try_from(first)?, [12., 27.]);
    assert_eq!(Vec::<f64>::try_from(second)?, [12., 18.]);
    assert!(!weights.grad().defined());
    Ok(())
}
```

The loss must be a scalar and every requested input must participate in its
graph. This direct API reports disconnected inputs as errors. In a normal
training loop, an optimizer's `backward_step` already handles differentiation;
use that when you just need a parameter update.

## Measure prediction sensitivity

A Jacobian records each output's derivative with respect to every input. Its
shape is `output_shape + input_shape`. For large outputs, compute a vector
product instead: a JVP measures a directional input perturbation; a VJP weights
output sensitivities and propagates them back to an input.

```rust
use rusttorch::{Result, Tensor, autograd::{self, GradOptions}};

fn main() -> Result<()> {
    let input = Tensor::from_slice(&[2_f64, 3.]).set_requires_grad(true);
    let predict = |x: &Tensor| -> Result<Tensor> { Ok(x.f_square()?) };
    let jacobian = autograd::jacobian(predict, &input, false)?;
    assert_eq!(jacobian.size(), [2, 2]);

    let direction = Tensor::from_slice(&[0.1_f64, -0.2]);
    let (prediction, sensitivity) = autograd::jvp(predict, &input, &direction, false)?;
    assert_eq!(prediction.size(), [2]);
    assert_eq!(sensitivity.size(), [2]);
    let weighted = autograd::vjp(
        &predict(&input)?, &input, &Tensor::from_slice(&[1_f64, 0.]),
        GradOptions::default(),
    )?;
    assert_eq!(Vec::<f64>::try_from(weighted)?, [4., 0.]);
    Ok(())
}
```

Functional Jacobians and Hessians return zeros for constant or unused inputs.
A full Jacobian runs one backward pass per output element. The JVP uses two
reverse passes and therefore requires operators with double-backward support.
It does not expose native forward-mode dual tensors. These helpers accept one
dense real tensor input; tangent and cotangent seeds are treated as constants.

## Inspect curvature of a scalar objective

The Hessian has `input_shape + input_shape` and can help diagnose ill-conditioned
small objectives. Its memory grows quadratically with the number of inputs.

```rust
use rusttorch::{Kind, Result, Tensor, autograd};

fn main() -> Result<()> {
    let input = Tensor::from_slice(&[2_f64, 3.]);
    let curvature = autograd::hessian(
        |x| Ok(x.f_pow_tensor_scalar(3)?.f_sum(Kind::Double)?), &input, false,
    )?;
    assert_eq!(Vec::<f64>::try_from(curvature.f_reshape([-1])?)?, [12., 0., 0., 18.]);
    Ok(())
}
```

## Train through a chosen surrogate derivative

Some models need a discrete forward value and a smooth training approximation.
`with_surrogate_gradient` returns the first tensor's values while using only the
second tensor's derivatives. This straight-through rounding example uses the
identity derivative. The surrogate is an explicit modeling choice; it is not
the mathematical derivative of rounding.

```rust
use rusttorch::{Kind, Result, Tensor, autograd::{self, GradOptions}};

fn main() -> Result<()> {
    let latent = Tensor::from_slice(&[0.2_f64, 1.7]).set_requires_grad(true);
    let codes = autograd::with_surrogate_gradient(&latent.f_round()?, &latent)?;
    assert_eq!(Vec::<f64>::try_from(&codes)?, [0., 2.]);
    let gradient = autograd::grad(
        &codes.f_sum(Kind::Double)?, &[&latent], GradOptions::default()
    )?.remove(0);
    assert_eq!(Vec::<f64>::try_from(gradient)?, [1., 1.]);
    Ok(())
}
```

Both tensors must have the same shape, dtype and device, and the surrogate must
be finite. This is ordinary tensor composition; registering custom backward
callbacks or Python-style custom function objects is outside this API.

## Sample continuous latent variables with gradients

Use `Normal::sample` for detached observations and `Normal::rsample` to optimize
location and scale through sampled values. Leading dimensions specify how many
samples to draw; the broadcast parameter shape comes afterward.

```rust
use rusttorch::{Kind, Result, Tensor, autograd::{self, GradOptions}, distributions::Normal};

fn main() -> Result<()> {
    let location = Tensor::from_slice(&[0_f64, 1.]).set_requires_grad(true);
    let scale = Tensor::from(0.5_f64).set_requires_grad(true);
    let latent = Normal::new(&location, &scale)?;
    let samples = latent.rsample(&[8])?;
    assert_eq!(samples.size(), [8, 2]);
    let objective = samples.f_square()?.f_mean(Kind::Double)?;
    let gradients = autograd::grad(&objective, &[&location, &scale], GradOptions::default())?;
    assert_eq!(gradients[0].size(), [2]);
    assert!(gradients[1].size().is_empty());
    assert!(!latent.sample(&[8])?.requires_grad());
    Ok(())
}
```

Scale must be positive and parameters must be finite real tensors on one dtype
and device. Rebuild a distribution after changing its parameters in place.
`manual_seed` controls the shared native random generator; concurrent random
operations affect its draw order. Equal seeds do not guarantee equal streams
on different devices.

## Score binary events and choose classes

Use Bernoulli for independent yes/no events and Categorical for one choice
among classes. A categorical parameter's last axis contains the class scores;
its samples are integer labels. Log probabilities retain parameter gradients,
while discrete samples do not.

```rust
use rusttorch::{Result, Tensor, distributions::{Bernoulli, Categorical}};

fn main() -> Result<()> {
    let events = Bernoulli::from_probs(&Tensor::from_slice(&[0.2_f32, 0.8]))?;
    let event_scores = events.log_prob(&Tensor::from_slice(&[0_f32, 1.]))?;
    assert_eq!(event_scores.size(), [2]);

    let choices = Categorical::from_logits(
        &Tensor::from_slice(&[1_f32, 2., 3., 3., 2., 1.]).f_reshape([2, 3])?
    )?;
    let labels = choices.sample(&[4])?;
    assert_eq!(labels.size(), [4, 2]);
    assert_eq!(choices.log_prob(&labels)?.size(), [4, 2]);
    assert_eq!(choices.entropy()?.size(), [2]);
    Ok(())
}
```

For a supervised classifier, prefer the regular cross-entropy training loss.
Distribution objects are useful when the model explicitly samples or evaluates
probabilities. The current families are univariate Normal, Bernoulli and
Categorical; transformed, multivariate and other distribution families remain
outside this interface.
