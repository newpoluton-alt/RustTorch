fn conditional() -> Result<Model> {
    let input = ValueSpec::new("operand", DType::Float, vec![Dimension::Known(2)]);
    let a = Program::new(
        vec![input.clone()],
        vec![Operation::new(
            "a",
            Operator::Multiply,
            ["operand", "operand"],
        )],
        Tree::Tensor("a".into()),
    );
    let b = Program::new(
        vec![input.clone()],
        vec![Operation::new("b", Operator::Add, ["operand", "operand"])],
        Tree::Tensor("b".into()),
    );
    let p = Program::new(
        vec![
            ValueSpec::new("choose", DType::Bool, vec![]),
            ValueSpec::new("x", DType::Float, vec![Dimension::Known(2)]),
        ],
        vec![Operation::new(
            "result",
            Operator::If {
                then_branch: Box::new(a),
                else_branch: Box::new(b),
                result: input,
            },
            ["choose", "x"],
        )],
        Tree::Tensor("result".into()),
    );
    Model::new(p, [])
}

use rusttorch::deployment::{
    DType, Dimension, Model, Operation, Operator, Program, StateRole, TracedModel, Tree, ValueSpec,
};
use rusttorch::{Device, Kind, Result, Tensor};
use std::{
    collections::BTreeMap,
    path::PathBuf,
    sync::atomic::{AtomicUsize, Ordering},
};

fn file(suffix: &str) -> PathBuf {
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    std::env::temp_dir().join(format!(
        "rusttorch-deployment-{}-{}.{suffix}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ))
}
fn batch() -> Dimension {
    Dimension::Symbol {
        name: "batch".into(),
        min: 1,
        max: Some(8),
    }
}
fn dense() -> Result<Model> {
    let input = ValueSpec::new("x", DType::Float, vec![batch(), Dimension::Known(3)]);
    let p = Program::new(
        vec![input],
        vec![
            Operation::new(
                "linear",
                Operator::Linear,
                ["x", "head.weight", "head.bias"],
            ),
            Operation::new("relu", Operator::Relu, ["linear"]),
            Operation::new("score", Operator::Multiply, ["relu", "scale"]),
        ],
        Tree::Dict(vec![
            ("score".into(), Tree::Tensor("score".into())),
            ("input".into(), Tree::Tensor("x".into())),
        ]),
    );
    Model::new(
        p,
        [
            (
                "head.weight".into(),
                StateRole::Parameter,
                Tensor::from_slice(&[1_f32, 2., 3., -1., 2., -3.])
                    .reshape([2, 3])
                    .set_requires_grad(true),
            ),
            (
                "head.bias".into(),
                StateRole::Parameter,
                Tensor::from_slice(&[0.5_f32, 1.]).set_requires_grad(true),
            ),
            (
                "scale".into(),
                StateRole::Constant,
                Tensor::from_slice(&[2_f32, 3.]),
            ),
        ],
    )
}
fn x() -> Tensor {
    Tensor::from_slice(&[1_f32, 2., 3., 4., 5., 6.]).reshape([2, 3])
}
fn close(a: &Tensor, b: &Tensor) {
    assert!(a.allclose(b, 1e-5, 1e-6, false), "{a:?} != {b:?}");
}

#[test]
fn portable_graph_guards_state_and_gradients_round_trip() -> Result<()> {
    let model = dense()?;
    let input = x().set_requires_grad(true);
    let output = model.run(&[input.shallow_clone()])?;
    assert_eq!(output[0].size(), [2, 2]);
    output[0].sum(Kind::Float).backward();
    assert!(model.state()["head.weight"].grad().defined());
    assert!(input.grad().defined());
    let restored = Model::from_json(&model.to_json()?)?;
    close(&restored.run(&[x()])?[0], &output[0]);
    assert_eq!(restored.program(), model.program());
    assert!(
        model
            .run(&[Tensor::zeros([9, 3], (Kind::Float, Device::Cpu))])
            .is_err()
    );
    assert!(
        model
            .run(&[Tensor::zeros([2, 3], (Kind::Double, Device::Cpu))])
            .is_err()
    );
    let mut json: serde_json::Value = serde_json::from_str(&model.to_json()?).unwrap();
    json["program"]["version"] = 99.into();
    assert!(Model::from_json(&json.to_string()).is_err());
    json["program"]["version"] = 1.into();
    json["state"][0]["offset"] = i64::MAX.into();
    assert!(Model::from_json(&json.to_string()).is_err());
    Ok(())
}

#[test]
fn conditional_executes_one_branch_and_preserves_selected_gradient() -> Result<()> {
    let model = conditional()?;
    let x = Tensor::from_slice(&[2_f32, 3.]).set_requires_grad(true);
    let a = model
        .run(&[Tensor::from(true), x.shallow_clone()])?
        .remove(0);
    a.sum(Kind::Float).backward();
    close(&x.grad(), &Tensor::from_slice(&[4_f32, 6.]));
    close(
        &model.run(&[Tensor::from(false), x])?[0],
        &Tensor::from_slice(&[4_f32, 6.]),
    );
    assert!(
        model
            .trace(&[
                Tensor::from(true),
                Tensor::zeros([2], (Kind::Float, Device::Cpu))
            ])
            .is_err()
    );
    let mut json: serde_json::Value = serde_json::from_str(&model.to_json()?).unwrap();
    json["program"]["operations"][0]["operator"]["If"]["else_branch"]["operations"][0]["operator"] =
        serde_json::json!({"Reshape":[3]});
    json["program"]["operations"][0]["operator"]["If"]["else_branch"]["operations"][0]["inputs"] =
        serde_json::json!(["operand"]);
    let model = Model::from_json(&json.to_string())?;
    assert!(
        model
            .run(&[
                Tensor::from(true),
                Tensor::ones([2], (Kind::Float, Device::Cpu))
            ])
            .is_ok()
    );
    assert!(
        model
            .run(&[
                Tensor::from(false),
                Tensor::ones([2], (Kind::Float, Device::Cpu))
            ])
            .is_err()
    );
    Ok(())
}

#[test]
fn pt2_and_onnx_native_round_trips_preserve_values_and_guards() -> Result<()> {
    let model = dense()?;
    let pt2 = file("pt2");
    model.save_pt2(&pt2)?;
    let restored = Model::load_pt2(&pt2)?;
    close(&restored.run(&[x()])?[0], &model.run(&[x()])?[0]);
    assert_eq!(
        model.state().keys().collect::<Vec<_>>(),
        restored.state().keys().collect::<Vec<_>>()
    );
    assert!(
        restored
            .run(&[Tensor::zeros([10, 3], (Kind::Float, Device::Cpu))])
            .is_err()
    );
    let onnx = file("onnx");
    model.save_onnx(&onnx)?;
    let restored = Model::load_onnx(&onnx)?;
    close(&restored.run(&[x()])?[0], &model.run(&[x()])?[0]);
    assert!(
        restored
            .run(&[Tensor::zeros([10, 3], (Kind::Float, Device::Cpu))])
            .is_err()
    );
    std::fs::remove_file(pt2).unwrap();
    std::fs::remove_file(onnx).unwrap();
    Ok(())
}

#[test]
fn actual_torchscript_trace_executes_and_reloads_with_guards() -> Result<()> {
    let model = dense()?;
    let mut json: serde_json::Value = serde_json::from_str(&model.to_json()?).unwrap();
    json["program"]["output_tree"] = serde_json::json!({"Tensor":"score"});
    let model = Model::from_json(&json.to_string())?;
    let traced = model.trace(&[x()])?;
    close(&traced.run(&[x()])?[0], &model.run(&[x()])?[0]);
    let path = file("pt");
    traced.save(&path)?;
    let restored = TracedModel::load(&path, Device::Cpu)?;
    close(&restored.run(&[x()])?[0], &model.run(&[x()])?[0]);
    assert!(
        restored
            .run(&[Tensor::zeros([9, 3], (Kind::Float, Device::Cpu))])
            .is_err()
    );
    std::fs::remove_file(&path).unwrap();
    std::fs::remove_file(format!("{}.guards.json", path.display())).unwrap();
    Ok(())
}

#[test]
fn state_aliases_survive_json_and_pt2_with_bounded_views() -> Result<()> {
    let tensor = Tensor::from_slice(&[1_f32, 2.]);
    let p = Program::new(
        vec![],
        vec![Operation::new("sum", Operator::Add, ["a", "b"])],
        Tree::Tensor("sum".into()),
    );
    let model = Model::new(
        p,
        [
            ("a".into(), StateRole::Buffer, tensor.shallow_clone()),
            ("b".into(), StateRole::Buffer, tensor),
        ],
    )?;
    assert_eq!(model.state()["a"].data_ptr(), model.state()["b"].data_ptr());
    let restored = Model::from_json(&model.to_json()?)?;
    assert_eq!(
        restored.state()["a"].data_ptr(),
        restored.state()["b"].data_ptr()
    );
    let path = file("pt2");
    model.save_pt2(&path)?;
    let restored = Model::load_pt2(&path)?;
    assert_eq!(
        restored.state()["a"].data_ptr(),
        restored.state()["b"].data_ptr()
    );
    assert!(model.save_onnx(file("onnx")).is_err());
    std::fs::remove_file(path).unwrap();
    Ok(())
}

#[test]
fn graph_module_lowering_reuses_named_state_and_evaluation_behavior() -> Result<()> {
    use rusttorch::{
        DeviceSpec,
        graph::{GraphBuilder, GraphInputs, TensorSpec},
    };
    let mut builder = GraphBuilder::new();
    let input = builder.input(
        "x",
        TensorSpec::new().known_dimensions([2, 3]).kind(Kind::Float),
    )?;
    let hidden = builder.linear("head", input, 3, 2)?;
    let out = builder.relu("relu", hidden)?;
    let mut graph = builder.output("scores", out)?.build(DeviceSpec::Cpu)?;
    assert!(Model::from_graph(&graph).is_err());
    graph.eval();
    let portable = Model::from_graph(&graph)?;
    let expected = graph.forward(GraphInputs::new().with("x", x())?)?;
    close(&portable.run(&[x()])?[0], expected.get("scores")?);
    assert!(portable.state().contains_key("head.weight"));
    Ok(())
}

#[test]
fn pt2_rejects_versions_pickle_mutation_truncated_storage_and_unsafe_paths() -> Result<()> {
    use std::io::{Cursor, Read, Write};
    use zip::{ZipArchive, ZipWriter, write::FileOptions};
    let path = file("pt2");
    dense()?.save_pt2(&path)?;
    let bytes = std::fs::read(&path).unwrap();
    let mut archive = ZipArchive::new(Cursor::new(bytes)).unwrap();
    let mut entries = Vec::new();
    for i in 0..archive.len() {
        let mut entry = archive.by_index(i).unwrap();
        let mut bytes = Vec::new();
        entry.read_to_end(&mut bytes).unwrap();
        entries.push((entry.name().to_owned(), bytes));
    }
    for case in 0..7 {
        let mut bad = entries.clone();
        match case {
            0 => {
                bad.iter_mut()
                    .find(|(n, _)| n.ends_with("archive_version"))
                    .unwrap()
                    .1 = b"100".to_vec()
            }
            1..=3 => {
                let (_, bytes) = bad
                    .iter_mut()
                    .find(|(n, _)| n.ends_with("models/model.json"))
                    .unwrap();
                let mut json: serde_json::Value = serde_json::from_slice(bytes).unwrap();
                match case {
                    1 => {
                        json["guards_code"] =
                            serde_json::json!(["__import__('os').system('exit 1')"])
                    }
                    2 => {
                        json["graph_module"]["signature"]["output_specs"][0] = serde_json::json!({"buffer_mutation":{"arg":{"name":"score"},"buffer_name":"scale"}})
                    }
                    _ => json["schema_version"]["minor"] = 21.into(),
                };
                *bytes = serde_json::to_vec(&json).unwrap();
            }
            4 => {
                let (_, bytes) = bad
                    .iter_mut()
                    .find(|(n, _)| n.ends_with("model_weights_config.json"))
                    .unwrap();
                let mut json: serde_json::Value = serde_json::from_slice(bytes).unwrap();
                json["config"]["head.weight"]["use_pickle"] = true.into();
                *bytes = serde_json::to_vec(&json).unwrap();
            }
            5 => bad
                .iter_mut()
                .find(|(n, _)| n.ends_with("data/weights/tensor_0"))
                .unwrap()
                .1
                .truncate(1),
            _ => bad.push(("package/../outside".into(), vec![])),
        }
        let mut writer = ZipWriter::new(Cursor::new(Vec::new()));
        for (name, bytes) in bad {
            writer.start_file(name, FileOptions::default()).unwrap();
            writer.write_all(&bytes).unwrap();
        }
        std::fs::write(&path, writer.finish().unwrap().into_inner()).unwrap();
        assert!(
            Model::load_pt2(&path).is_err(),
            "malformed archive case {case} was accepted"
        );
    }
    std::fs::remove_file(path).unwrap();
    Ok(())
}

#[test]
fn tied_parameters_keep_one_gradient_leaf_and_symbol_guards_agree() -> Result<()> {
    let tensor = Tensor::from_slice(&[1_f32, 2.]).set_requires_grad(true);
    let p = Program::new(
        vec![],
        vec![Operation::new("sum", Operator::Add, ["a", "b"])],
        Tree::Tensor("sum".into()),
    );
    let model = Model::new(
        p,
        [
            ("a".into(), StateRole::Parameter, tensor.shallow_clone()),
            ("b".into(), StateRole::Parameter, tensor),
        ],
    )?;
    model.run(&[])?.remove(0).sum(Kind::Float).backward();
    close(
        &model.state()["a"].grad(),
        &Tensor::from_slice(&[2_f32, 2.]),
    );
    assert_eq!(
        model.state()["a"].grad().data_ptr(),
        model.state()["b"].grad().data_ptr()
    );
    let p = Program::new(
        vec![
            ValueSpec::new("a", DType::Float, vec![batch()]),
            ValueSpec::new("b", DType::Float, vec![batch()]),
        ],
        vec![Operation::new("out", Operator::Add, ["a", "b"])],
        Tree::Tensor("out".into()),
    );
    let model = Model::new(p, [])?;
    assert!(
        model
            .run(&[
                Tensor::ones([1], (Kind::Float, Device::Cpu)),
                Tensor::ones([2], (Kind::Float, Device::Cpu))
            ])
            .is_err()
    );
    Ok(())
}

#[test]
fn onnx_rejects_malformed_shapes_unknown_fields_and_versions() -> Result<()> {
    let path = file("onnx");
    dense()?.save_onnx(&path)?;
    let original = std::fs::read(&path).unwrap();
    let mut wrong_version = original.clone();
    wrong_version.extend([0x08, 99]);
    std::fs::write(&path, wrong_version).unwrap();
    assert!(Model::load_onnx(&path).is_err());
    let mut unknown = original.clone();
    unknown.extend([0xa0, 0x06, 0x00]);
    std::fs::write(&path, unknown).unwrap();
    assert!(Model::load_onnx(&path).is_err());
    let mut wrong_shape = original.clone();
    let index = wrong_shape
        .windows(4)
        .position(|x| x == [0x0a, 0x02, 0x02, 0x03])
        .expect("packed weight dimensions [2,3]");
    wrong_shape[index + 3] = 127;
    std::fs::write(&path, wrong_shape).unwrap();
    let error = Model::load_onnx(&path).unwrap_err().to_string();
    assert!(error.contains("byte length"), "{error}");
    std::fs::write(&path, &original[..original.len() / 2]).unwrap();
    assert!(Model::load_onnx(&path).is_err());
    std::fs::remove_file(path).unwrap();
    Ok(())
}

#[test]
fn portable_execution_matches_cpu_on_available_accelerators() -> Result<()> {
    let reference = dense()?;
    let input = x().set_requires_grad(true);
    let expected = reference.run(&[input.shallow_clone()])?.remove(0);
    let exact = Tensor::from_slice(&[29_f32, 0., 65., 0.]).reshape([2, 2]);
    assert!(expected.equal(&exact), "CPU fixture changed: {expected:?}");
    expected.sum(Kind::Float).backward();
    let caps = rusttorch::available_devices();
    let values = |tensor: &Tensor| -> Result<Vec<f32>> {
        Ok(Vec::<f32>::try_from(
            &tensor.f_to_device(Device::Cpu)?.f_reshape([-1])?,
        )?)
    };
    for (name, device, available) in [
        ("CUDA", Device::Cuda(0), caps.cuda),
        ("MPS", Device::Mps, caps.mps),
    ] {
        if !available {
            eprintln!("skipped {name} deployment numerical parity: unavailable hardware");
            continue;
        }
        let mut model = dense()?;
        model.to_device(device)?;
        let input = x().to_device(device).set_requires_grad(true);
        let output = model.run(&[input.shallow_clone()])?.remove(0);
        let before_backward = output.f_to_device(Device::Cpu)?;
        if !before_backward.allclose(&expected, 1e-4, 1e-4, false) {
            let direct = input
                .f_linear(
                    &model.state()["head.weight"],
                    Some(&model.state()["head.bias"]),
                )?
                .f_relu()?
                .f_mul(&model.state()["scale"])?;
            eprintln!(
                "{name} before backward: model={:?}, direct native={:?}",
                values(&before_backward)?,
                values(&direct)?
            );
        }
        output.sum(Kind::Float).backward();
        let actual = output.f_to_device(Device::Cpu)?;
        if !actual.allclose(&expected, 1e-4, 1e-4, false)
            || !before_backward.allclose(&expected, 1e-4, 1e-4, false)
        {
            // Capture direct backend and decomposed results only on failure;
            // these extra synchronizations must not mask the original result.
            let state = model.state();
            let weight = &state["head.weight"];
            let bias = &state["head.bias"];
            let scale = &state["scale"];
            let direct_linear = input.f_linear(weight, Some(bias))?;
            let decomposed = input.f_matmul(&weight.f_transpose(0, 1)?)?.f_add(bias)?;
            eprintln!(
                "{name} input={:?}, weight={:?}, bias={:?}, scale={:?}",
                values(&input)?,
                values(weight)?,
                values(bias)?,
                values(scale)?
            );
            eprintln!(
                "{name} direct linear={:?}, matmul+bias={:?}",
                values(&direct_linear)?,
                values(&decomposed)?
            );
            eprintln!(
                "{name} direct score={:?}, decomposed score={:?}",
                values(&direct_linear.f_relu()?.f_mul(scale)?)?,
                values(&decomposed.f_relu()?.f_mul(scale)?)?
            );
            #[cfg(target_os = "macos")]
            for property in ["machdep.cpu.brand_string", "hw.model"] {
                if let Ok(info) = std::process::Command::new("sysctl")
                    .args(["-n", property])
                    .output()
                {
                    eprintln!(
                        "{property}: {}",
                        String::from_utf8_lossy(&info.stdout).trim()
                    );
                }
            }
        }
        assert!(
            actual.allclose(&expected, 1e-4, 1e-4, false)
                && before_backward.allclose(&expected, 1e-4, 1e-4, false),
            "{name} score mismatch: before backward={:?}, after backward={:?}, expected={:?}",
            values(&before_backward)?,
            values(&actual)?,
            values(&expected)?
        );
        let input_gradient = input.grad().f_to_device(Device::Cpu)?;
        let expected_input_gradient =
            Tensor::from_slice(&[2_f32, 4., 6., 2., 4., 6.]).reshape([2, 3]);
        assert!(
            input_gradient.allclose(&expected_input_gradient, 1e-4, 1e-4, false),
            "{name} input gradient mismatch: actual={:?}, expected={:?}",
            values(&input_gradient)?,
            values(&expected_input_gradient)?
        );
        let weight_gradient = model.state()["head.weight"]
            .grad()
            .f_to_device(Device::Cpu)?;
        let expected_weight_gradient = reference.state()["head.weight"].grad();
        assert!(
            weight_gradient.allclose(&expected_weight_gradient, 1e-4, 1e-4, false),
            "{name} weight gradient mismatch: actual={:?}, expected={:?}",
            values(&weight_gradient)?,
            values(&expected_weight_gradient)?
        );
    }
    Ok(())
}

#[test]
fn all_portable_dtypes_and_operator_shapes_round_trip() -> Result<()> {
    for (dtype, tensor) in [
        (DType::Float, Tensor::from_slice(&[1_f32, 2.])),
        (DType::Double, Tensor::from_slice(&[1_f64, 2.])),
        (DType::Int64, Tensor::from_slice(&[1_i64, 2])),
        (DType::Bool, Tensor::from_slice(&[true, false])),
    ] {
        let p = Program::new(
            vec![ValueSpec::new("x", dtype, vec![Dimension::Known(2)])],
            vec![Operation::new("out", Operator::Identity, ["x"])],
            Tree::Tensor("out".into()),
        );
        let model = Model::new(p, [])?;
        for suffix in ["pt2", "onnx"] {
            let path = file(suffix);
            let restored = if suffix == "pt2" {
                model.save_pt2(&path)?;
                Model::load_pt2(&path)?
            } else {
                model.save_onnx(&path)?;
                Model::load_onnx(&path)?
            };
            assert!(restored.run(&[tensor.shallow_clone()])?[0].equal(&tensor));
            std::fs::remove_file(path).unwrap();
        }
    }
    let p = Program::new(
        vec![ValueSpec::new(
            "x",
            DType::Float,
            vec![Dimension::Known(2), Dimension::Known(3)],
        )],
        vec![
            Operation::new("a", Operator::Sigmoid, ["x"]),
            Operation::new("b", Operator::Tanh, ["x"]),
            Operation::new("c", Operator::Subtract, ["a", "b"]),
            Operation::new("d", Operator::Transpose { dim0: 0, dim1: 1 }, ["c"]),
            Operation::new("e", Operator::Matmul, ["c", "d"]),
            Operation::new("f", Operator::Permute(vec![1, 0]), ["e"]),
            Operation::new("g", Operator::Cat(1), ["e", "f"]),
            Operation::new("h", Operator::Reshape(vec![2, 2, 2]), ["g"]),
            Operation::new("out", Operator::Flatten { start: 1, end: -1 }, ["h"]),
        ],
        Tree::Tensor("out".into()),
    );
    let model = Model::new(p, [])?;
    let c = x().sigmoid() - x().tanh();
    let e = c.matmul(&c.transpose(0, 1));
    let expected = Tensor::cat(&[e.shallow_clone(), e.transpose(0, 1)], 1);
    for suffix in ["pt2", "onnx"] {
        let path = file(suffix);
        let restored = if suffix == "pt2" {
            model.save_pt2(&path)?;
            Model::load_pt2(&path)?
        } else {
            model.save_onnx(&path)?;
            Model::load_onnx(&path)?
        };
        close(&restored.run(&[x()])?[0], &expected);
        std::fs::remove_file(path).unwrap();
    }
    Ok(())
}

#[test]
#[ignore = "requires the locked Python/PyTorch/ONNX reference environment"]
fn python_pt2_onnx_and_torchscript_numerical_round_trips() -> Result<()> {
    let directory = file("reference");
    std::fs::create_dir(&directory).unwrap();
    let python = std::env::var("RUSTTORCH_PYTHON").unwrap_or_else(|_| "python3".into());
    assert!(
        std::process::Command::new(&python)
            .args(["tests/python_reference/deployment.py", "generate"])
            .arg(&directory)
            .status()
            .unwrap()
            .success()
    );
    let expected = Tensor::read_safetensors(directory.join("conditional.safetensors"))?
        .into_iter()
        .collect::<BTreeMap<_, _>>();
    for pred in [false, true] {
        let x = Tensor::from_slice(&[2_f32, 3.]).set_requires_grad(true);
        let y = conditional()?
            .run(&[Tensor::from(pred), x.shallow_clone()])?
            .remove(0);
        y.sum(Kind::Float).backward();
        close(&y, &expected[&format!("output_{}", u8::from(pred))]);
        close(
            &x.grad(),
            &expected[&format!("gradient_{}", u8::from(pred))],
        );
    }
    for name in ["reference.pt2", "reference.onnx"] {
        let model = if name.ends_with("pt2") {
            Model::load_pt2(directory.join(name))?
        } else {
            Model::load_onnx(directory.join(name))?
        };
        let output = model.run(&[x()])?;
        let expected = Tensor::read_safetensors(directory.join("expected.safetensors"))?
            .into_iter()
            .collect::<BTreeMap<_, _>>();
        close(&output[0], &expected["output"]);
        if name.ends_with("pt2") {
            let input = x().set_requires_grad(true);
            model.run(&[input.shallow_clone()])?[0]
                .sum(Kind::Float)
                .backward();
            close(&input.grad(), &expected["input_gradient"]);
            close(
                &model.state()["head.weight"].grad(),
                &expected["weight_gradient"],
            );
        }
        model.save_pt2(directory.join(format!("{name}.rust.pt2")))?;
        model.save_onnx(directory.join(format!("{name}.rust.onnx")))?;
    }
    let model = dense()?;
    model.save_pt2(directory.join("rust.pt2"))?;
    model.save_onnx(directory.join("rust.onnx"))?;
    let native = TracedModel::load(directory.join("reference.pt"), Device::Cpu)?;
    close(&native.run(&[x()])?[0], &model.run(&[x()])?[0]);
    let mut json: serde_json::Value = serde_json::from_str(&model.to_json()?).unwrap();
    json["program"]["output_tree"] = serde_json::json!({"Tensor":"score"});
    Model::from_json(&json.to_string())?
        .trace(&[x()])?
        .save(directory.join("rust.pt"))?;
    assert!(
        std::process::Command::new(&python)
            .args(["tests/python_reference/deployment.py", "verify"])
            .arg(&directory)
            .status()
            .unwrap()
            .success()
    );
    std::fs::remove_dir_all(directory).unwrap();
    Ok(())
}
