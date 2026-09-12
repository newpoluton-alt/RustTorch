use std::{fs, path::PathBuf};

use rusttorch::{
    Device, Kind, Result, Tensor,
    interop::{load_state_dict, save_state_dict},
    nn::{
        AttentionMask, MultiheadAttentionConfig, RnnActivation, RnnConfig, TransformerActivation,
        TransformerConfig, TransformerMasks, VarStore,
    },
    no_grad,
};
use serde_json::Value;

fn close(left: &Tensor, right: &Tensor, name: &str) {
    assert_eq!(left.size(), right.size(), "{name} shape");
    assert!(
        left.allclose(right, 2e-4, 2e-6, false),
        "{name}: max difference {}",
        (left - right).abs().max().double_value(&[])
    );
}

#[test]
fn recurrent_models_preserve_states_layouts_and_parameter_names() -> Result<()> {
    let store = VarStore::new(Device::Cpu);
    let config = RnnConfig::new(2, 3)
        .num_layers(2)
        .bidirectional(true)
        .batch_first(true);
    let lstm = config.build_lstm(&(store.root() / "lstm"))?;
    let input = Tensor::ones([2, 4, 2], (Kind::Float, Device::Cpu)).set_requires_grad(true);
    let hidden = Tensor::ones([4, 2, 3], (Kind::Float, Device::Cpu)).set_requires_grad(true);
    let cell = Tensor::zeros_like(&hidden).set_requires_grad(true);
    let (output, (h, c)) = lstm.forward_t(&input, Some((&hidden, &cell)), true)?;
    assert_eq!(output.size(), [2, 4, 6]);
    assert_eq!(h.size(), [4, 2, 3]);
    assert_eq!(c.size(), h.size());
    (output.square().mean(Kind::Float)
        + h.square().mean(Kind::Float)
        + c.square().mean(Kind::Float))
    .f_backward()?;
    for tensor in [&input, &hidden, &cell] {
        assert!(tensor.grad().defined());
    }
    assert_eq!(store.variables().len(), 16);
    for name in [
        "lstm.weight_ih_l0",
        "lstm.weight_hh_l1_reverse",
        "lstm.bias_ih_l1",
        "lstm.bias_hh_l0_reverse",
    ] {
        assert!(store.variables()[name].grad().defined(), "{name}");
    }
    for tensor in store.variables().values() {
        assert!(tensor.abs().max().double_value(&[]) <= 1. / 3_f64.sqrt());
    }

    let projected = RnnConfig::new(2, 4)
        .projection(2)
        .bias(false)
        .batch_first(true)
        .build_lstm(&(store.root() / "projected"))?;
    let (output, (h, c)) = projected.forward_t(&input.get(0), None, false)?;
    assert_eq!(output.size(), [4, 2]);
    assert_eq!(h.size(), [1, 2]);
    assert_eq!(c.size(), [1, 4]);
    assert!(store.variables().contains_key("projected.weight_hr_l0"));
    assert!(!store.variables().contains_key("projected.bias_ih_l0"));
    Ok(())
}

#[test]
fn recurrent_streaming_matches_complete_sequences_and_dropout_obeys_mode() -> Result<()> {
    let store = VarStore::new(Device::Cpu);
    let input = Tensor::ones([6, 2, 3], (Kind::Float, Device::Cpu));
    let gru = RnnConfig::new(3, 4).build_gru(&(store.root() / "gru"))?;
    let (all, final_state) = gru.forward_t(&input, None, false)?;
    let (first, state) = gru.forward_t(&input.narrow(0, 0, 2), None, false)?;
    let (second, state) = gru.forward_t(&input.narrow(0, 2, 4), Some(&state), false)?;
    close(&Tensor::cat(&[first, second], 0), &all, "stream output");
    close(&state, &final_state, "stream state");
    let rnn = RnnConfig::new(3, 8)
        .num_layers(2)
        .dropout(0.8)
        .build_rnn(&(store.root() / "rnn"), RnnActivation::Tanh)?;
    let (eval, _) = rnn.forward_t(&input, None, false)?;
    close(&rnn.forward_t(&input, None, false)?.0, &eval, "eval repeat");
    assert!(
        !rnn.forward_t(&input, None, true)?
            .0
            .allclose(&eval, 1e-6, 1e-6, false)
    );
    Ok(())
}

#[test]
fn recurrent_validation_rejects_undefined_shapes_dtypes_and_bad_configuration() -> Result<()> {
    let store = VarStore::new(Device::Cpu);
    for config in [
        RnnConfig::new(0, 2),
        RnnConfig::new(2, 0),
        RnnConfig::new(2, 3).num_layers(0),
        RnnConfig::new(2, 3).dropout(f64::NAN),
        RnnConfig::new(2, 3).projection(3),
        RnnConfig::new(2, i64::MAX),
    ] {
        assert!(config.build_lstm(&store.root()).is_err());
        assert!(store.variables().is_empty());
    }
    assert!(
        RnnConfig::new(2, 3)
            .projection(1)
            .build_gru(&store.root())
            .is_err()
    );
    let model = RnnConfig::new(2, 3).build_lstm(&store.root())?;
    for input in [
        Tensor::new(),
        Tensor::zeros([2], (Kind::Float, Device::Cpu)),
        Tensor::zeros([2, 1, 3], (Kind::Float, Device::Cpu)),
        Tensor::zeros([0, 1, 2], (Kind::Float, Device::Cpu)),
        Tensor::zeros([2, 1, 2], (Kind::Int64, Device::Cpu)),
    ] {
        assert!(model.forward_t(&input, None, false).is_err());
    }
    let input = Tensor::zeros([2, 1, 2], (Kind::Float, Device::Cpu));
    let valid = Tensor::zeros([1, 1, 3], (Kind::Float, Device::Cpu));
    for state in [
        Tensor::new(),
        Tensor::zeros([1, 3], (Kind::Float, Device::Cpu)),
        Tensor::zeros([1, 1, 3], (Kind::Double, Device::Cpu)),
    ] {
        assert!(
            model
                .forward_t(&input, Some((&state, &valid)), false)
                .is_err()
        );
        assert!(
            model
                .forward_t(&input, Some((&valid, &state)), false)
                .is_err()
        );
    }
    Ok(())
}

#[test]
fn attention_masks_block_future_and_padding_tokens_with_gradients() -> Result<()> {
    let store = VarStore::new(Device::Cpu);
    let model = MultiheadAttentionConfig::new(4, 2)
        .batch_first(true)
        .build(&store.root())?;
    let input = Tensor::randn([2, 3, 4], (Kind::Float, Device::Cpu)).set_requires_grad(true);
    let padding = Tensor::from_slice(&[false, false, true, false, true, false]).reshape([2, 3]);
    let masks = AttentionMask {
        key_padding: Some(&padding),
        causal: true,
        ..Default::default()
    };
    let (output, weights) = model.forward_per_head_t(&input, &input, &input, masks, true)?;
    assert_eq!(output.size(), [2, 3, 4]);
    assert_eq!(weights.size(), [2, 2, 3, 3]);
    assert_eq!(weights.triu(1).abs().sum(Kind::Float).double_value(&[]), 0.);
    assert_eq!(
        weights
            .get(0)
            .select(-1, 2)
            .abs()
            .sum(Kind::Float)
            .double_value(&[]),
        0.
    );
    assert_eq!(
        weights
            .get(1)
            .select(-1, 1)
            .abs()
            .sum(Kind::Float)
            .double_value(&[]),
        0.
    );
    output.square().mean(Kind::Float).f_backward()?;
    assert!(input.grad().defined());
    assert!(store.variables()["in_proj_weight"].grad().defined());
    assert!(store.variables()["out_proj.weight"].grad().defined());
    assert_eq!(
        store.variables()["in_proj_bias"]
            .abs()
            .sum(Kind::Float)
            .double_value(&[]),
        0.
    );
    assert_eq!(
        store.variables()["out_proj.bias"]
            .abs()
            .sum(Kind::Float)
            .double_value(&[]),
        0.
    );
    let floating = Tensor::zeros([3, 3], (Kind::Float, Device::Cpu));
    let ordinary = model.forward_t(&input, &input, &input, AttentionMask::default(), false)?;
    let masked = model.forward_t(
        &input,
        &input,
        &input,
        AttentionMask {
            attention: Some(&floating),
            ..Default::default()
        },
        false,
    )?;
    close(&ordinary.0, &masked.0, "zero mask output");
    assert_eq!(ordinary.1.size(), [2, 3, 3]);
    Ok(())
}

#[test]
fn attention_scales_queries_before_half_precision_dot_products() -> Result<()> {
    let mut store = VarStore::new(Device::Cpu);
    store.set_kind(Kind::Half);
    let model = MultiheadAttentionConfig::new(4, 1)
        .bias(false)
        .batch_first(true)
        .build(&store.root())?;
    let identity = Tensor::eye(4, (Kind::Half, Device::Cpu));
    no_grad(|| -> Result<()> {
        let variables = store.variables();
        variables["in_proj_weight"]
            .shallow_clone()
            .f_copy_(&Tensor::cat(&[&identity, &identity, &identity], 0))?;
        variables["out_proj.weight"]
            .shallow_clone()
            .f_copy_(&identity)?;
        Ok(())
    })?;
    let input = Tensor::full([1, 2, 4], 150., (Kind::Half, Device::Cpu));
    let (output, weights) =
        model.forward_t(&input, &input, &input, AttentionMask::default(), false)?;
    close(&output, &input, "finite half attention output");
    close(
        &weights,
        &Tensor::full([1, 2, 2], 0.5, (Kind::Half, Device::Cpu)),
        "finite half attention weights",
    );
    Ok(())
}

#[test]
fn attention_validation_and_dropout_cover_cross_attention_and_unbatched_inputs() -> Result<()> {
    let store = VarStore::new(Device::Cpu);
    for config in [
        MultiheadAttentionConfig::new(0, 2),
        MultiheadAttentionConfig::new(4, 0),
        MultiheadAttentionConfig::new(5, 2),
        MultiheadAttentionConfig::new(4, 2).dropout(-0.1),
        MultiheadAttentionConfig::new(i64::MAX, 1),
    ] {
        assert!(config.build(&store.root()).is_err());
        assert!(store.variables().is_empty());
    }
    let model = MultiheadAttentionConfig::new(4, 2)
        .bias(false)
        .dropout(1.)
        .build(&store.root())?;
    let query = Tensor::ones([3, 4], (Kind::Float, Device::Cpu));
    let key = Tensor::ones([5, 4], (Kind::Float, Device::Cpu));
    let trained = model.forward_t(&query, &key, &key, AttentionMask::default(), true)?;
    assert_eq!(trained.0.abs().sum(Kind::Float).double_value(&[]), 0.);
    assert_eq!(trained.1.size(), [3, 5]);
    assert!(
        model
            .forward_t(&query, &key, &key, AttentionMask::default(), false)?
            .1
            .sum(Kind::Float)
            .double_value(&[])
            > 0.
    );
    let undefined = Tensor::new();
    for (q, k, v) in [
        (&undefined, &key, &key),
        (&query, &undefined, &key),
        (&query, &key, &undefined),
    ] {
        assert!(
            model
                .forward_t(q, k, v, AttentionMask::default(), false)
                .is_err()
        );
    }
    for mask in [
        Tensor::new(),
        Tensor::zeros([3, 5], (Kind::Int64, Device::Cpu)),
        Tensor::zeros([3, 4], (Kind::Float, Device::Cpu)),
        Tensor::zeros([1, 3, 5], (Kind::Bool, Device::Cpu)),
    ] {
        assert!(
            model
                .forward_t(
                    &query,
                    &key,
                    &key,
                    AttentionMask {
                        attention: Some(&mask),
                        ..Default::default()
                    },
                    false
                )
                .is_err()
        );
    }
    assert!(
        model
            .forward_t(
                &query,
                &key,
                &key,
                AttentionMask {
                    key_padding: Some(&undefined),
                    ..Default::default()
                },
                false
            )
            .is_err()
    );
    Ok(())
}

#[test]
fn transformer_stacks_clone_values_without_sharing_parameters_and_train_end_to_end() -> Result<()> {
    let store = VarStore::new(Device::Cpu);
    let config = TransformerConfig::new(4, 2)
        .dim_feedforward(7)
        .dropout(0.)
        .batch_first(true);
    let encoder = config.build_encoder(&(store.root() / "standalone"), 2, true)?;
    let variables = store.variables();
    let first = &variables["standalone.layers.0.self_attn.in_proj_weight"];
    let second = &variables["standalone.layers.1.self_attn.in_proj_weight"];
    close(first, second, "cloned initialization");
    no_grad(|| second.shallow_clone().f_copy_(&Tensor::zeros_like(second)))?;
    assert!(first.abs().sum(Kind::Float).double_value(&[]) > 0.);
    let input = Tensor::randn([2, 3, 4], (Kind::Float, Device::Cpu));
    assert_eq!(
        encoder
            .forward_t(&input, AttentionMask::default(), false)?
            .size(),
        [2, 3, 4]
    );

    let model = config
        .num_encoder_layers(2)
        .num_decoder_layers(2)
        .build(&(store.root() / "model"))?;
    let source = Tensor::randn([2, 5, 4], (Kind::Float, Device::Cpu)).set_requires_grad(true);
    let target = Tensor::randn([2, 3, 4], (Kind::Float, Device::Cpu)).set_requires_grad(true);
    let output = model.forward_t(
        &source,
        &target,
        TransformerMasks {
            target: AttentionMask {
                causal: true,
                ..Default::default()
            },
            ..Default::default()
        },
        true,
    )?;
    assert_eq!(output.size(), target.size());
    output.select(-1, 0).sum(Kind::Float).f_backward()?;
    assert!(source.grad().defined());
    assert!(target.grad().defined());
    for (name, parameter) in store.variables() {
        if name.starts_with("model.") {
            assert!(parameter.grad().defined(), "{name}");
        }
    }
    Ok(())
}

#[test]
fn transformer_modes_masked_rows_and_invalid_inputs_are_fallible() -> Result<()> {
    let store = VarStore::new(Device::Cpu);
    let config = TransformerConfig::new(4, 2).dim_feedforward(7);
    for bad in [
        config.dim_feedforward(0),
        config.layer_norm_eps(-1.),
        config.dropout(f64::NAN),
        TransformerConfig::new(3, 2),
    ] {
        assert!(bad.build_encoder_layer(&store.root()).is_err());
        assert!(store.variables().is_empty());
    }
    assert!(config.num_decoder_layers(0).build(&store.root()).is_err());
    assert!(
        config
            .num_encoder_layers(1)
            .num_decoder_layers(i64::MAX)
            .build(&store.root())
            .is_err()
    );
    assert!(store.variables().is_empty());
    assert!(config.build_encoder(&store.root(), 0, false).is_err());
    assert!(store.variables().is_empty());
    let encoder = config
        .dropout(0.9)
        .norm_first(true)
        .activation(TransformerActivation::Gelu)
        .build_encoder_layer(&store.root())?;
    let input = Tensor::randn([3, 2, 4], (Kind::Float, Device::Cpu)).set_requires_grad(true);
    let eval = encoder.forward_t(&input, AttentionMask::default(), false)?;
    close(
        &eval,
        &encoder.forward_t(&input, AttentionMask::default(), false)?,
        "transformer eval",
    );
    assert!(!eval.allclose(
        &encoder.forward_t(&input, AttentionMask::default(), true)?,
        1e-6,
        1e-6,
        false
    ));
    let blocked = Tensor::ones([3, 3], (Kind::Bool, Device::Cpu));
    let output = encoder.forward_t(
        &input,
        AttentionMask {
            attention: Some(&blocked),
            ..Default::default()
        },
        false,
    )?;
    assert_eq!(output.isfinite().all().int64_value(&[]), 1);
    output.sum(Kind::Float).f_backward()?;
    assert_eq!(input.grad().isfinite().all().int64_value(&[]), 1);
    assert!(
        encoder
            .forward_t(&Tensor::new(), AttentionMask::default(), false)
            .is_err()
    );
    assert!(
        encoder
            .forward_t(
                &Tensor::zeros([3, 2, 5], (Kind::Float, Device::Cpu)),
                AttentionMask::default(),
                false
            )
            .is_err()
    );
    for kind in [Kind::Int64, Kind::Bool, Kind::ComplexFloat] {
        let mut store = VarStore::new(Device::Cpu);
        store.set_kind(kind);
        assert!(RnnConfig::new(2, 3).build_gru(&store.root()).is_err());
        assert!(
            MultiheadAttentionConfig::new(4, 2)
                .build(&store.root())
                .is_err()
        );
        assert!(config.build(&store.root()).is_err());
        assert!(store.variables().is_empty());
    }
    Ok(())
}

#[test]
fn sequence_modules_compose_in_sequential_with_registered_parameters() -> Result<()> {
    let model = rusttorch::nn::Sequential::builder()
        .layer(|path| RnnConfig::new(3, 4).build_rnn(path, RnnActivation::Tanh))
        .layer(|path| RnnConfig::new(4, 4).build_lstm(path))
        .layer(|path| RnnConfig::new(4, 4).build_gru(path))
        .layer(|path| MultiheadAttentionConfig::new(4, 2).build(path))
        .layer(|path| {
            TransformerConfig::new(4, 2)
                .dim_feedforward(7)
                .build_encoder_layer(path)
        })
        .layer(|path| {
            TransformerConfig::new(4, 2)
                .dim_feedforward(7)
                .build_encoder(path, 1, false)
        })
        .linear(4, 2)
        .build(rusttorch::DeviceSpec::Cpu)?;
    let input = Tensor::ones([3, 2, 3], (Kind::Float, Device::Cpu));
    let output = model.forward_t(&input, true)?;
    assert_eq!(output.size(), [3, 2, 2]);
    output.square().mean(Kind::Float).f_backward()?;
    for (name, parameter) in model.var_store().variables() {
        assert!(parameter.grad().defined(), "{name}");
    }
    Ok(())
}

#[test]
fn sequence_parameters_restore_exact_recurrent_states_and_masked_transformer_outputs() -> Result<()>
{
    let checkpoint = std::env::temp_dir().join(format!(
        "rusttorch-sequence-replay-{}.safetensors",
        std::process::id()
    ));
    let original_store = VarStore::new(Device::Cpu);
    let restored_store = VarStore::new(Device::Cpu);
    let config = RnnConfig::new(2, 4)
        .num_layers(2)
        .projection(2)
        .bidirectional(true)
        .batch_first(true);
    let original = config.build_lstm(&original_store.root())?;
    let restored = config.build_lstm(&restored_store.root())?;
    save_state_dict(&checkpoint, &original_store)?;
    load_state_dict(&checkpoint, &restored_store)?;

    let input = Tensor::arange(12, (Kind::Float, Device::Cpu)).reshape([2, 3, 2]) / 12.;
    let (output, (hidden, cell)) = original.forward_t(&input, None, false)?;
    let (restored_output, (restored_hidden, restored_cell)) =
        restored.forward_t(&input, None, false)?;
    for (actual, expected, name) in [
        (&restored_output, &output, "recurrent output"),
        (&restored_hidden, &hidden, "recurrent hidden state"),
        (&restored_cell, &cell, "recurrent cell state"),
    ] {
        assert!(actual.equal(expected), "{name}");
    }

    let original_store = VarStore::new(Device::Cpu);
    let restored_store = VarStore::new(Device::Cpu);
    let config = TransformerConfig::new(4, 2)
        .dim_feedforward(7)
        .num_encoder_layers(2)
        .num_decoder_layers(2)
        .batch_first(true)
        .dropout(0.);
    let original = config.build(&original_store.root())?;
    let restored = config.build(&restored_store.root())?;
    save_state_dict(&checkpoint, &original_store)?;
    load_state_dict(&checkpoint, &restored_store)?;

    let source = Tensor::arange(32, (Kind::Float, Device::Cpu)).reshape([2, 4, 4]) / 32.;
    let target = Tensor::arange(24, (Kind::Float, Device::Cpu)).reshape([2, 3, 4]) / 24.;
    let masks = TransformerMasks {
        target: AttentionMask {
            causal: true,
            ..Default::default()
        },
        ..Default::default()
    };
    assert!(
        restored
            .forward_t(&source, &target, masks, false)?
            .equal(&original.forward_t(&source, &target, masks, false)?),
        "masked transformer output"
    );
    fs::remove_file(checkpoint).expect("remove sequence checkpoint");
    Ok(())
}

fn tensor(value: &Value, requires_grad: bool) -> Tensor {
    fn flatten(value: &Value, data: &mut Vec<f32>) {
        match value {
            Value::Array(values) => {
                for item in values {
                    flatten(item, data);
                }
            }
            Value::Bool(value) => data.push(u8::from(*value) as f32),
            Value::String(value) if value == "-inf" => data.push(f32::NEG_INFINITY),
            Value::Number(value) => data.push(value.as_f64().unwrap() as f32),
            _ => panic!("unexpected tensor data: {value}"),
        }
    }
    let mut shape = Vec::new();
    let mut current = value;
    while let Some(values) = current.as_array() {
        shape.push(values.len() as i64);
        if values.is_empty() {
            break;
        }
        current = &values[0];
    }
    let mut data = Vec::new();
    flatten(value, &mut data);
    Tensor::from_slice(&data)
        .reshape(&shape)
        .set_requires_grad(requires_grad)
}

fn load_parameters(store: &VarStore, reference: &Value) -> Result<()> {
    assert_eq!(
        store.variables().len(),
        reference.as_object().unwrap().len()
    );
    no_grad(|| -> Result<()> {
        for (name, value) in store.variables() {
            value
                .shallow_clone()
                .f_copy_(&tensor(&reference[&name], false))?;
        }
        Ok(())
    })
}

fn check_parameters(store: &VarStore, reference: &Value, gradients: bool, prefix: &str) {
    for (name, value) in store.variables() {
        let actual = if gradients { value.grad() } else { value };
        close(
            &actual,
            &tensor(&reference[&name], false),
            &format!("{prefix}.{name}"),
        );
    }
}

fn verify(
    reference: &Value,
    output: &Tensor,
    extra: &[&Tensor],
    inputs: &[&Tensor],
    store: &VarStore,
    name: &str,
) -> Result<()> {
    close(
        output,
        &tensor(&reference["output"], false),
        &format!("{name}.output"),
    );
    let mut loss = output.square().mean(Kind::Float);
    for (index, value) in extra.iter().enumerate() {
        close(
            value,
            &tensor(&reference["extra"][index], false),
            &format!("{name}.extra.{index}"),
        );
        loss += value.square().mean(Kind::Float);
    }
    loss.f_backward()?;
    for (index, input) in inputs.iter().enumerate() {
        close(
            &input.grad(),
            &tensor(&reference["input_grads"][index], false),
            &format!("{name}.input_grad.{index}"),
        );
    }
    check_parameters(store, &reference["parameter_grads"], true, name);
    Ok(())
}

#[test]
#[ignore = "run through scripts/run-python-parity.sh"]
fn sequence_python_parity() -> Result<()> {
    let directory = PathBuf::from(
        std::env::var_os("RUSTTORCH_PYTHON_REFERENCE_DIR")
            .expect("run through scripts/run-python-parity.sh"),
    );
    let reference: Value = serde_json::from_slice(
        &fs::read(directory.join("sequence.json")).expect("read sequence fixture"),
    )
    .expect("parse sequence fixture");
    for (name, case) in reference["recurrent"].as_object().unwrap() {
        rusttorch::manual_seed(127);
        let store = VarStore::new(Device::Cpu);
        let kwargs = &case["kwargs"];
        let config = RnnConfig::new(2, 3)
            .num_layers(kwargs["num_layers"].as_i64().unwrap_or(1))
            .bias(kwargs["bias"].as_bool().unwrap_or(true))
            .batch_first(kwargs["batch_first"].as_bool().unwrap_or(false))
            .bidirectional(kwargs["bidirectional"].as_bool().unwrap_or(false))
            .projection(kwargs["proj_size"].as_i64().unwrap_or(0))
            .dropout(kwargs["dropout"].as_f64().unwrap_or(0.));
        let input = tensor(&case["input"], true);
        let states = case["states"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| tensor(v, true))
            .collect::<Vec<_>>();
        let mut inputs = vec![&input];
        inputs.extend(states.iter());
        let training = case["training"].as_bool().unwrap();
        let run = |forward: &dyn Fn() -> Result<(Tensor, Vec<Tensor>)>| -> Result<()> {
            check_parameters(&store, &case["initial"], false, &format!("{name}.initial"));
            load_parameters(&store, &case["state"])?;
            rusttorch::manual_seed(541);
            let (output, extra) = forward()?;
            verify(
                case,
                &output,
                &extra.iter().collect::<Vec<_>>(),
                &inputs,
                &store,
                name,
            )
        };
        match case["family"].as_str().unwrap() {
            "rnn" => {
                let activation = if kwargs["nonlinearity"] == "relu" {
                    RnnActivation::Relu
                } else {
                    RnnActivation::Tanh
                };
                let model = config.build_rnn(&store.root(), activation)?;
                run(&|| {
                    let (out, h) = model.forward_t(&input, states.first(), training)?;
                    Ok((out, vec![h]))
                })?;
            }
            "gru" => {
                let model = config.build_gru(&store.root())?;
                run(&|| {
                    let (out, h) = model.forward_t(&input, states.first(), training)?;
                    Ok((out, vec![h]))
                })?;
            }
            "lstm" => {
                let model = config.build_lstm(&store.root())?;
                run(&|| {
                    let (out, (h, c)) = model.forward_t(
                        &input,
                        if states.is_empty() {
                            None
                        } else {
                            Some((&states[0], &states[1]))
                        },
                        training,
                    )?;
                    Ok((out, vec![h, c]))
                })?;
            }
            _ => unreachable!(),
        }
    }
    for (name, case) in reference["attention"].as_object().unwrap() {
        rusttorch::manual_seed(127);
        let store = VarStore::new(Device::Cpu);
        let model = MultiheadAttentionConfig::new(4, 2)
            .batch_first(case["batch_first"].as_bool().unwrap())
            .bias(case["bias"].as_bool().unwrap())
            .dropout(case["dropout"].as_f64().unwrap())
            .build(&store.root())?;
        check_parameters(&store, &case["initial"], false, &format!("{name}.initial"));
        load_parameters(&store, &case["state"])?;
        let query = tensor(&case["query"], true);
        let key = tensor(&case["key"], true);
        let value = tensor(&case["value"], true);
        let padding =
            tensor(&case["padding"], false).to_kind(if case["padding_bool"].as_bool().unwrap() {
                Kind::Bool
            } else {
                Kind::Float
            });
        let attention = tensor(&case["attention"], false).to_kind(
            if case["attention_bool"].as_bool().unwrap() {
                Kind::Bool
            } else {
                Kind::Float
            },
        );
        let mask = AttentionMask {
            attention: Some(&attention),
            key_padding: Some(&padding),
            causal: case["causal"].as_bool().unwrap(),
        };
        rusttorch::manual_seed(541);
        let (output, weights) = if case["per_head"].as_bool().unwrap() {
            model.forward_per_head_t(
                &query,
                &key,
                &value,
                mask,
                case["training"].as_bool().unwrap(),
            )?
        } else {
            model.forward_t(
                &query,
                &key,
                &value,
                mask,
                case["training"].as_bool().unwrap(),
            )?
        };
        verify(
            case,
            &output,
            &[&weights],
            &[&query, &key, &value],
            &store,
            name,
        )?;
    }
    for (name, case) in reference["transformer"].as_object().unwrap() {
        rusttorch::manual_seed(127);
        let store = VarStore::new(Device::Cpu);
        let kwargs = &case["kwargs"];
        let config = TransformerConfig::new(4, 2)
            .dim_feedforward(7)
            .dropout(kwargs["dropout"].as_f64().unwrap_or(0.))
            .batch_first(kwargs["batch_first"].as_bool().unwrap_or(false))
            .norm_first(kwargs["norm_first"].as_bool().unwrap_or(false))
            .bias(kwargs["bias"].as_bool().unwrap_or(true))
            .activation(if kwargs["activation"] == "gelu" {
                TransformerActivation::Gelu
            } else {
                TransformerActivation::Relu
            });
        let input = tensor(&case["input"], true);
        let memory = tensor(&case["memory"], true);
        let padding = tensor(&case["target_padding"], false).to_kind(Kind::Bool);
        let source_padding = tensor(&case["source_padding"], false).to_kind(Kind::Bool);
        let mask = AttentionMask {
            causal: true,
            key_padding: Some(&padding),
            attention: None,
        };
        let memory_mask = AttentionMask {
            key_padding: Some(&source_padding),
            ..Default::default()
        };
        let training = case["training"].as_bool().unwrap();
        let family = case["family"].as_str().unwrap();
        let prepare = || -> Result<()> {
            if !case["initial"].is_null() {
                check_parameters(&store, &case["initial"], false, &format!("{name}.initial"));
            }
            load_parameters(&store, &case["state"])?;
            rusttorch::manual_seed(541);
            Ok(())
        };
        let output = match family {
            "encoder_layer" => {
                let model = config.build_encoder_layer(&store.root())?;
                prepare()?;
                model.forward_t(&input, mask, training)?
            }
            "encoder" => {
                let model = config.build_encoder(&store.root(), 2, true)?;
                prepare()?;
                model.forward_t(&input, mask, training)?
            }
            "decoder_layer" => {
                let model = config.build_decoder_layer(&store.root())?;
                prepare()?;
                model.forward_t(&input, &memory, mask, memory_mask, training)?
            }
            "decoder" => {
                let model = config.build_decoder(&store.root(), 2, true)?;
                prepare()?;
                model.forward_t(&input, &memory, mask, memory_mask, training)?
            }
            "transformer" => {
                let model = config
                    .num_encoder_layers(2)
                    .num_decoder_layers(2)
                    .build(&store.root())?;
                prepare()?;
                model.forward_t(
                    &memory,
                    &input,
                    TransformerMasks {
                        source: memory_mask,
                        target: mask,
                        memory: memory_mask,
                    },
                    training,
                )?
            }
            _ => unreachable!(),
        };
        let inputs = if family.starts_with("encoder") {
            vec![&input]
        } else {
            vec![&input, &memory]
        };
        verify(case, &output, &[], &inputs, &store, name)?;
    }
    Ok(())
}
