// PT2 wire fields adapted from PyTorch 2.13.0, cf30153c4c131c8164ee7798e5022d810682e2cb:
// torch/_export/serde/schema.py, torch/export/pt2_archive/_package.py and
// torch/csrc/export/pt2_archive_constants.h (BSD-3-Clause; see third-party notices).
use super::*;
use serde_json::{Value, json};
use std::io::Cursor;
use zip::{CompressionMethod, ZipArchive, ZipWriter, write::FileOptions};

fn object(v: &Value) -> Result<&serde_json::Map<String, Value>> {
    v.as_object()
        .ok_or_else(|| error("expected PT2 JSON object"))
}
fn array(v: &Value) -> Result<&Vec<Value>> {
    v.as_array().ok_or_else(|| error("expected PT2 JSON array"))
}
fn string(v: &Value) -> Result<&str> {
    v.as_str().ok_or_else(|| error("expected PT2 JSON string"))
}
fn integer(v: &Value) -> Result<i64> {
    v.as_i64().ok_or_else(|| error("expected PT2 JSON integer"))
}
fn boolean(v: &Value) -> Result<bool> {
    v.as_bool()
        .ok_or_else(|| error("expected PT2 JSON Boolean"))
}
fn tensor_name(v: &Value) -> Result<&str> {
    if object(v)?.len() != 1 {
        return Err(error("unsupported PT2 argument union"));
    }
    string(&v["as_tensor"]["name"])
}
fn dtype(v: &Value) -> Result<DType> {
    match integer(v)? {
        7 => Ok(DType::Float),
        8 => Ok(DType::Double),
        5 => Ok(DType::Int64),
        12 => Ok(DType::Bool),
        n => Err(error(format!("unsupported PT2 dtype {n}"))),
    }
}
fn scalar_code(d: DType) -> i64 {
    match d {
        DType::Float => 7,
        DType::Double => 8,
        DType::Int64 => 5,
        DType::Bool => 12,
    }
}
fn fixed(v: &Value) -> Result<i64> {
    if object(v)?.len() != 1 {
        return Err(error("invalid PT2 dimension union"));
    }
    integer(&v["as_int"])
}
fn fixed_list(v: &Value) -> Result<Vec<i64>> {
    array(v)?.iter().map(fixed).collect()
}

fn entry_path(name: &str) -> Result<()> {
    if name.is_empty()
        || name.len() > 2048
        || name.contains('\\')
        || name.starts_with('/')
        || name
            .split('/')
            .any(|p| p.is_empty() || p == "." || p == ".." || p.contains(':'))
    {
        return Err(error("invalid PT2 ZIP path"));
    }
    Ok(())
}
fn read_archive(path: &Path) -> Result<BTreeMap<String, Vec<u8>>> {
    let bytes = read_bounded(path)?;
    validate_zip_envelope(&bytes, MAX_NODES, MAX_BYTES)?;
    let mut zip = ZipArchive::new(Cursor::new(bytes)).map_err(mapped)?;
    if zip.len() > MAX_NODES {
        return Err(error("too many PT2 ZIP entries"));
    }
    let mut files = BTreeMap::new();
    let mut total = 0_u64;
    let mut root = None;
    for i in 0..zip.len() {
        let mut file = zip.by_index(i).map_err(mapped)?;
        let name = file.name().to_owned();
        entry_path(&name)?;
        if !matches!(
            file.compression(),
            CompressionMethod::Stored | CompressionMethod::Deflated
        ) || file.is_dir()
        {
            return Err(error("unsupported PT2 ZIP entry"));
        }
        total = total
            .checked_add(file.size())
            .ok_or_else(|| error("archive length overflow"))?;
        if total > MAX_BYTES as u64 {
            return Err(error("uncompressed PT2 exceeds limit"));
        }
        let (prefix, relative) = name
            .split_once('/')
            .ok_or_else(|| error("PT2 ZIP requires one root directory"))?;
        if root.as_deref().is_some_and(|r| r != prefix) {
            return Err(error("PT2 ZIP has multiple roots"));
        }
        root = Some(prefix.to_owned());
        let mut data = Vec::new();
        (&mut file)
            .take(MAX_BYTES as u64 + 1)
            .read_to_end(&mut data)
            .map_err(mapped)?;
        if data.len() as u64 != file.size() || files.insert(relative.to_owned(), data).is_some() {
            return Err(error("duplicate or truncated PT2 ZIP entry"));
        }
    }
    for (name, expected) in [
        ("archive_format", b"pt2".as_slice()),
        ("archive_version", b"0".as_slice()),
        ("byteorder", b"little".as_slice()),
    ] {
        if files.get(name).map(Vec::as_slice) != Some(expected) {
            return Err(error(format!("unsupported or missing PT2 {name}")));
        }
    }
    if files
        .get(".data/version")
        .map(|x| String::from_utf8_lossy(x).trim().to_owned())
        .as_deref()
        != Some("6")
        || !files.contains_key(".data/serialization_id")
    {
        return Err(error("unsupported PT2 serialization headers"));
    }
    Ok(files)
}

fn json_file(files: &BTreeMap<String, Vec<u8>>, name: &str) -> Result<Value> {
    let bytes = files
        .get(name)
        .ok_or_else(|| error(format!("missing PT2 member {name}")))?;
    if bytes.len() > 16 * 1024 * 1024 {
        return Err(error("PT2 JSON member exceeds 16 MiB"));
    }
    strict_json(bytes)
}

fn strict_json(bytes: &[u8]) -> Result<Value> {
    struct Checked(Value);
    impl<'de> Deserialize<'de> for Checked {
        fn deserialize<D: serde::Deserializer<'de>>(
            deserializer: D,
        ) -> std::result::Result<Self, D::Error> {
            struct Visitor;
            impl<'de> serde::de::Visitor<'de> for Visitor {
                type Value = Checked;
                fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                    f.write_str("JSON with unique object keys")
                }
                fn visit_bool<E: serde::de::Error>(
                    self,
                    v: bool,
                ) -> std::result::Result<Checked, E> {
                    Ok(Checked(v.into()))
                }
                fn visit_i64<E: serde::de::Error>(self, v: i64) -> std::result::Result<Checked, E> {
                    Ok(Checked(v.into()))
                }
                fn visit_u64<E: serde::de::Error>(self, v: u64) -> std::result::Result<Checked, E> {
                    Ok(Checked(v.into()))
                }
                fn visit_f64<E: serde::de::Error>(self, v: f64) -> std::result::Result<Checked, E> {
                    Ok(Checked(v.into()))
                }
                fn visit_str<E: serde::de::Error>(
                    self,
                    v: &str,
                ) -> std::result::Result<Checked, E> {
                    Ok(Checked(v.into()))
                }
                fn visit_unit<E: serde::de::Error>(self) -> std::result::Result<Checked, E> {
                    Ok(Checked(Value::Null))
                }
                fn visit_seq<A: serde::de::SeqAccess<'de>>(
                    self,
                    mut seq: A,
                ) -> std::result::Result<Checked, A::Error> {
                    let mut out = Vec::new();
                    while let Some(Checked(v)) = seq.next_element()? {
                        if out.len() > MAX_NODES * 32 {
                            return Err(serde::de::Error::custom("PT2 array exceeds limit"));
                        }
                        out.push(v);
                    }
                    Ok(Checked(Value::Array(out)))
                }
                fn visit_map<A: serde::de::MapAccess<'de>>(
                    self,
                    mut map: A,
                ) -> std::result::Result<Checked, A::Error> {
                    let mut out = serde_json::Map::new();
                    while let Some((key, Checked(v))) = map.next_entry::<String, Checked>()? {
                        if out.len() > MAX_NODES * 32 || out.insert(key, v).is_some() {
                            return Err(serde::de::Error::custom(
                                "duplicate PT2 JSON key or excessive object",
                            ));
                        }
                    }
                    Ok(Checked(Value::Object(out)))
                }
            }
            deserializer.deserialize_any(Visitor)
        }
    }
    serde_json::from_slice::<Checked>(bytes)
        .map(|v| v.0)
        .map_err(mapped)
}
fn symbol(v: &Value) -> Result<String> {
    let expression = string(&v["as_expr"]["expr_str"])?;
    // Only one serialized SymPy Symbol; never evaluate Python/SymPy expressions.
    let inner = expression
        .strip_prefix("Symbol('")
        .and_then(|s| s.strip_suffix("', positive=True, integer=True)"))
        .or_else(|| {
            expression
                .strip_prefix("Symbol('")
                .and_then(|s| s.strip_suffix("', integer=True)"))
        })
        .ok_or_else(|| error(format!("unsupported symbolic expression {expression}")))?;
    if inner.is_empty()
        || !inner
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_')
    {
        return Err(error("invalid symbolic dimension name"));
    }
    Ok(inner.to_owned())
}
fn input_spec(name: &str, meta: &Value, ranges: &Value) -> Result<ValueSpec> {
    meta_layout(meta)?;
    let dimensions = array(&meta["sizes"])?
        .iter()
        .map(|d| {
            if d.get("as_int").is_some() {
                Ok(Dimension::Known(fixed(d)?))
            } else {
                let name = symbol(d)?;
                let range = &ranges[&name];
                let min = if range["min_val"].is_null() {
                    0
                } else {
                    integer(&range["min_val"])?
                };
                let max = if range["max_val"].is_null() {
                    None
                } else {
                    Some(integer(&range["max_val"])?)
                };
                Ok(Dimension::Symbol { name, min, max })
            }
        })
        .collect::<Result<_>>()?;
    Ok(ValueSpec::new(name, dtype(&meta["dtype"])?, dimensions))
}
fn meta_layout(meta: &Value) -> Result<()> {
    if integer(&meta["layout"])? != 7 || string(&meta["device"]["type"])? != "cpu" {
        return Err(error(
            "PT2 supports strided CPU metadata only; move a loaded model explicitly",
        ));
    }
    Ok(())
}
fn parse_tree(serialized: &str, names: &[String]) -> Result<Tree> {
    let root: Value = strict_json(serialized.as_bytes())?;
    let mut leaves = names.iter();
    fn walk(v: &Value, names: &mut std::slice::Iter<'_, String>, depth: usize) -> Result<Tree> {
        if depth > MAX_DEPTH {
            return Err(error("PT2 calling tree exceeds depth limit"));
        }
        let children = array(&v["children_spec"])?;
        if v["type"].is_null() {
            if !children.is_empty() {
                return Err(error("invalid PT2 tensor leaf"));
            }
            return Ok(Tree::Tensor(
                names
                    .next()
                    .ok_or_else(|| error("too many calling tree leaves"))?
                    .clone(),
            ));
        }
        let children = children
            .iter()
            .map(|v| walk(v, names, depth + 1))
            .collect::<Result<Vec<_>>>()?;
        match string(&v["type"])? {
            "builtins.tuple" => Ok(Tree::Tuple(children)),
            "builtins.list" => Ok(Tree::List(children)),
            "builtins.dict" => {
                let keys: Vec<String> =
                    serde_json::from_str(string(&v["context"])?).map_err(mapped)?;
                if keys.len() != children.len() {
                    return Err(error("PT2 dictionary key count mismatch"));
                }
                Ok(Tree::Dict(keys.into_iter().zip(children).collect()))
            }
            other => Err(error(format!("unsupported PT2 calling tree {other}"))),
        }
    }
    if root.get(0) != Some(&json!(1)) {
        return Err(error("unsupported PT2 tree version"));
    }
    let tree = walk(&root[1], &mut leaves, 0)?;
    if leaves.next().is_some() {
        return Err(error("too few calling tree leaves"));
    }
    tree.leaves()?;
    Ok(tree)
}
fn tree_json(tree: &Tree) -> Result<String> {
    fn walk(t: &Tree) -> Value {
        match t {
            Tree::Tensor(_) => json!({"type":null,"context":null,"children_spec":[]}),
            Tree::Tuple(v) | Tree::List(v) => {
                json!({"type":if matches!(t,Tree::Tuple(_)){"builtins.tuple"}else{"builtins.list"},"context":"null","children_spec":v.iter().map(walk).collect::<Vec<_>>()})
            }
            Tree::Dict(v) => {
                json!({"type":"builtins.dict","context":serde_json::to_string(&v.iter().map(|(n,_)|n).collect::<Vec<_>>()).expect("string serialization"),"children_spec":v.iter().map(|(_,t)|walk(t)).collect::<Vec<_>>()})
            }
        }
    }
    serde_json::to_string(&json!([1, walk(tree)])).map_err(mapped)
}

pub(super) fn load(path: &Path) -> Result<Model> {
    let files = read_archive(path)?;
    let models = files
        .keys()
        .filter(|n| n.starts_with("models/"))
        .collect::<Vec<_>>();
    if models.len() != 1 {
        return Err(error("PT2 import requires exactly one model"));
    }
    let name = models[0]
        .strip_prefix("models/")
        .and_then(|n| n.strip_suffix(".json"))
        .ok_or_else(|| error("invalid PT2 model filename"))?;
    if name.contains('/') {
        return Err(error("invalid PT2 model name"));
    }
    let ep = json_file(&files, models[0])?;
    if ep["schema_version"] != json!({"major":8,"minor":20})
        || ep["opset_version"] != json!({"aten":10})
        || !string(&ep["torch_version"])?.starts_with("2.13.0")
    {
        return Err(error("PT2 requires PyTorch2.13.0/schema8.20/ATen10"));
    }
    if !array(&ep["guards_code"])?.is_empty() {
        return Err(error("PT2 executable guard code is unsupported"));
    }
    let gm = &ep["graph_module"];
    let graph = &gm["graph"];
    for key in [
        "custom_obj_values",
        "sym_bool_values",
        "sym_float_values",
        "sym_int_values",
    ] {
        if !object(&graph[key])?.is_empty() {
            return Err(error(format!("unsupported PT2 {key}")));
        }
    }
    let mut storages = Vec::new();
    let mut states = Vec::new();
    let mut storage_paths = BTreeMap::new();
    let mut loaded = BTreeMap::new();
    for (directory, suffix, constant) in [
        ("data/weights/", "weights", false),
        ("data/constants/", "constants", true),
    ] {
        let config = json_file(&files, &format!("{directory}{name}_{suffix}_config.json"))?;
        for (state_name, payload) in object(&config["config"])? {
            if boolean(&payload["use_pickle"])? {
                return Err(error(format!("pickled state {state_name} is unsupported")));
            }
            let meta = &payload["tensor_meta"];
            meta_layout(meta)?;
            let dtype = dtype(&meta["dtype"])?;
            let leaf = string(&payload["path_name"])?;
            if leaf.contains('/') || leaf.contains('\\') {
                return Err(error("invalid PT2 state path"));
            }
            let path = format!("{directory}{leaf}");
            entry_path(&path)?;
            let storage = if let Some(&index) = storage_paths.get(&path) {
                let old: &Storage = &storages[index];
                if old.dtype != dtype {
                    return Err(error("shared PT2 storage has conflicting dtypes"));
                }
                index
            } else {
                let bytes = files
                    .get(&path)
                    .ok_or_else(|| error("missing raw PT2 state"))?
                    .clone();
                let index = storages.len();
                storages.push(Storage { dtype, bytes });
                storage_paths.insert(path, index);
                index
            };
            let role = if constant {
                StateRole::Constant
            } else if boolean(&payload["is_param"])? {
                StateRole::Parameter
            } else {
                StateRole::Buffer
            };
            let descriptor = State {
                name: state_name.clone(),
                role,
                storage,
                shape: fixed_list(&meta["sizes"])?,
                strides: fixed_list(&meta["strides"])?,
                offset: fixed(&meta["storage_offset"])?,
                requires_grad: boolean(&meta["requires_grad"])?,
            };
            if loaded.insert(state_name.clone(), states.len()).is_some() {
                return Err(error("duplicate PT2 state name"));
            }
            states.push(descriptor);
        }
    }
    let mut rename = BTreeMap::new();
    let mut inputs = Vec::new();
    let mut declared = Vec::new();
    let mut used_state = BTreeSet::new();
    for spec in array(&gm["signature"]["input_specs"])? {
        if object(spec)?.len() != 1 {
            return Err(error("invalid PT2 input signature"));
        }
        if let Some(user) = spec.get("user_input") {
            let name = tensor_name(&user["arg"])?;
            inputs.push(input_spec(
                name,
                &graph["tensor_values"][name],
                &ep["range_constraints"],
            )?);
            declared.push(name.to_owned());
        } else {
            let (variant, name_key, expected) = if spec.get("parameter").is_some() {
                ("parameter", "parameter_name", StateRole::Parameter)
            } else if spec.get("buffer").is_some() {
                let role = if boolean(&spec["buffer"]["persistent"])? {
                    StateRole::Buffer
                } else {
                    StateRole::Constant
                };
                ("buffer", "buffer_name", role)
            } else if spec.get("tensor_constant").is_some() {
                (
                    "tensor_constant",
                    "tensor_constant_name",
                    StateRole::Constant,
                )
            } else {
                return Err(error("unsupported PT2 input kind"));
            };
            let arg = string(&spec[variant]["arg"]["name"])?;
            let name = string(&spec[variant][name_key])?;
            let state = &states[*loaded
                .get(name)
                .ok_or_else(|| error("PT2 signature state is missing"))?];
            if state.role != expected {
                return Err(error("PT2 state role disagrees with signature"));
            }
            let meta = &graph["tensor_values"][arg];
            meta_layout(meta)?;
            if fixed_list(&meta["sizes"])? != state.shape
                || dtype(&meta["dtype"])? != storages[state.storage].dtype
                || boolean(&meta["requires_grad"])? != state.requires_grad
            {
                return Err(error("PT2 state metadata disagrees with raw payload"));
            }
            if rename.insert(arg.to_owned(), name.to_owned()).is_some() {
                return Err(error("duplicate PT2 placeholder"));
            }
            declared.push(arg.to_owned());
            used_state.insert(name.to_owned());
        }
    }
    if used_state.len() != states.len() {
        return Err(error("PT2 contains undeclared state"));
    }
    let graph_inputs = array(&graph["inputs"])?
        .iter()
        .map(|x| tensor_name(x).map(str::to_owned))
        .collect::<Result<Vec<_>>>()?;
    if declared != graph_inputs {
        return Err(error("PT2 graph inputs disagree with signature"));
    }
    let map_name = |n: &str| rename.get(n).cloned().unwrap_or_else(|| n.to_owned());
    let mut operations = Vec::new();
    if array(&graph["nodes"])?.len() > MAX_NODES {
        return Err(error("PT2 node count exceeds limit"));
    }
    for node in array(&graph["nodes"])? {
        let output = array(&node["outputs"])?;
        if output.len() != 1 {
            return Err(error("PT2 operator must return one tensor"));
        }
        let name = map_name(tensor_name(&output[0])?);
        let mut args = BTreeMap::new();
        for arg in array(&node["inputs"])? {
            let n = string(&arg["name"])?;
            if args.insert(n, &arg["arg"]).is_some() {
                return Err(error("duplicate PT2 operator argument"));
            }
        }
        let target = string(&node["target"])?;
        let (operator, operands) = parse_operator(target, &mut args, &map_name)?;
        if !args.is_empty() {
            return Err(error(format!(
                "unsupported arguments for {target}: {:?}",
                args.keys()
            )));
        }
        operations.push(Operation {
            name,
            operator,
            inputs: operands,
        });
    }
    let outputs = array(&graph["outputs"])?
        .iter()
        .map(|v| tensor_name(v).map(map_name))
        .collect::<Result<Vec<_>>>()?;
    let sig_outputs = array(&gm["signature"]["output_specs"])?;
    if sig_outputs.len() != outputs.len() {
        return Err(error("PT2 output count mismatch"));
    }
    for (spec, name) in sig_outputs.iter().zip(&outputs) {
        if object(spec)?.len() != 1
            || tensor_name(&spec["user_output"]["arg"]).map(map_name)? != *name
        {
            return Err(error(
                "PT2 mutations, gradients or non-user outputs are unsupported",
            ));
        }
    }
    let call = array(&gm["module_call_graph"])?
        .iter()
        .find(|c| c["fqn"] == "")
        .ok_or_else(|| error("missing PT2 root call signature"))?;
    let input_names = inputs.iter().map(|s| s.name.clone()).collect::<Vec<_>>();
    let program = Program {
        version: 1,
        operator_version: 1,
        inputs,
        input_tree: parse_tree(string(&call["signature"]["in_spec"])?, &input_names)?,
        operations,
        output_tree: parse_tree(string(&call["signature"]["out_spec"])?, &outputs)?,
    };
    Model::from_artifact(
        Artifact {
            program,
            storages,
            state: states,
        },
        Device::Cpu,
    )
}

fn parse_operator(
    target: &str,
    args: &mut BTreeMap<&str, &Value>,
    map: &impl Fn(&str) -> String,
) -> Result<(Operator, Vec<String>)> {
    fn tensor(
        args: &mut BTreeMap<&str, &Value>,
        key: &str,
        map: &impl Fn(&str) -> String,
    ) -> Result<String> {
        tensor_name(
            args.remove(key)
                .ok_or_else(|| error(format!("missing PT2 argument {key}")))?,
        )
        .map(map)
    }
    fn int(args: &mut BTreeMap<&str, &Value>, key: &str, default: Option<i64>) -> Result<i64> {
        match args.remove(key) {
            Some(v) => integer(&v["as_int"]),
            None => default.ok_or_else(|| error(format!("missing PT2 argument {key}"))),
        }
    }
    let unary = match target {
        "torch.ops.aten.relu.default" => Some(Operator::Relu),
        "torch.ops.aten.sigmoid.default" => Some(Operator::Sigmoid),
        "torch.ops.aten.tanh.default" => Some(Operator::Tanh),
        "torch.ops.aten.alias.default" => Some(Operator::Identity),
        _ => None,
    };
    if let Some(op) = unary {
        return Ok((op, vec![tensor(args, "self", map)?]));
    }
    match target {
        "torch.ops.aten.add.Tensor"
        | "torch.ops.aten.sub.Tensor"
        | "torch.ops.aten.mul.Tensor"
        | "torch.ops.aten.matmul.default"
        | "torch.ops.aten.mm.default" => {
            let a = tensor(args, "self", map)?;
            let b = tensor(
                args,
                if target.ends_with("mm.default") {
                    "mat2"
                } else {
                    "other"
                },
                map,
            )?;
            if let Some(alpha) = args.remove("alpha")
                && alpha != &json!({"as_int":1})
                && alpha != &json!({"as_float":1.0})
            {
                return Err(error("PT2 add/sub alpha other than one is unsupported"));
            }
            Ok((
                if target.contains(".add.") {
                    Operator::Add
                } else if target.contains(".sub.") {
                    Operator::Subtract
                } else if target.contains(".mul.") {
                    Operator::Multiply
                } else {
                    Operator::Matmul
                },
                vec![a, b],
            ))
        }
        "torch.ops.aten.linear.default" => {
            let mut x = vec![tensor(args, "input", map)?, tensor(args, "weight", map)?];
            if let Some(bias) = args.remove("bias")
                && bias != &json!({"as_none":true})
            {
                x.push(map(tensor_name(bias)?));
            }
            Ok((Operator::Linear, x))
        }
        "torch.ops.aten.reshape.default" | "torch.ops.aten.view.default" => {
            let x = tensor(args, "self", map)?;
            let v = args
                .remove(if target.contains(".view.") {
                    "size"
                } else {
                    "shape"
                })
                .ok_or_else(|| error("missing reshape shape"))?;
            let shape = if let Some(v) = v.get("as_ints") {
                array(v)?.iter().map(integer).collect::<Result<Vec<_>>>()?
            } else {
                array(&v["as_sym_ints"])?
                    .iter()
                    .map(|x| integer(&x["as_int"]))
                    .collect::<Result<Vec<_>>>()?
            };
            Ok((Operator::Reshape(shape), vec![x]))
        }
        "torch.ops.aten.flatten.using_ints" => {
            let x = tensor(args, "self", map)?;
            Ok((
                Operator::Flatten {
                    start: int(args, "start_dim", Some(0))?,
                    end: int(args, "end_dim", Some(-1))?,
                },
                vec![x],
            ))
        }
        "torch.ops.aten.transpose.int" => {
            let x = tensor(args, "self", map)?;
            Ok((
                Operator::Transpose {
                    dim0: int(args, "dim0", None)?,
                    dim1: int(args, "dim1", None)?,
                },
                vec![x],
            ))
        }
        "torch.ops.aten.permute.default" => {
            let x = tensor(args, "self", map)?;
            let dims = args
                .remove("dims")
                .ok_or_else(|| error("missing permutation"))?;
            Ok((
                Operator::Permute(
                    array(&dims["as_ints"])?
                        .iter()
                        .map(integer)
                        .collect::<Result<_>>()?,
                ),
                vec![x],
            ))
        }
        "torch.ops.aten.cat.default" => {
            let x = args
                .remove("tensors")
                .ok_or_else(|| error("missing cat tensors"))?;
            let names = array(&x["as_tensors"])?
                .iter()
                .map(|v| string(&v["name"]).map(map))
                .collect::<Result<Vec<_>>>()?;
            Ok((Operator::Cat(int(args, "dim", Some(0))?), names))
        }
        _ => Err(error(format!("unsupported PT2 operator {target}"))),
    }
}

fn tensor_arg(name: &str) -> Value {
    json!({"as_tensor":{"name":name}})
}
fn named_arg(name: &str, value: Value) -> Value {
    json!({"name":name,"arg":value,"kind":1})
}
fn meta(
    tensor: &Tensor,
    dimensions: Option<&[Dimension]>,
    symbol_names: &BTreeMap<String, String>,
) -> Value {
    let sizes = if let Some(d) = dimensions {
        d.iter().map(|d|match d{Dimension::Known(n)=>json!({"as_int":n}),Dimension::Symbol{name,min,..}=>json!({"as_expr":{"expr_str":format!("Symbol('{}', positive=True, integer=True)",symbol_names[name]),"hint":{"as_int":(*min).max(2)}}})}).collect::<Vec<_>>()
    } else {
        tensor.size().iter().map(|n| json!({"as_int":n})).collect()
    };
    json!({"dtype":scalar_code(DType::from_kind(tensor.kind()).expect("validated dtype")),"sizes":sizes,"requires_grad":tensor.requires_grad(),"device":{"type":"cpu","index":null},"strides":tensor.stride().iter().map(|s|json!({"as_int":s})).collect::<Vec<_>>(),"storage_offset":{"as_int":0},"layout":7})
}
pub(super) fn save(model: &Model, path: &Path) -> Result<()> {
    if model.device != Device::Cpu {
        return Err(error("PT2 export requires CPU model state"));
    }
    if model
        .program()
        .operations
        .iter()
        .any(|n| matches!(n.operator, Operator::If { .. }))
    {
        return Err(error("PT2 conditional export is unsupported"));
    }
    let artifact = model.snapshot()?;
    let mut storage_roles = BTreeMap::new();
    for state in &artifact.state {
        let constants = state.role == StateRole::Constant;
        if storage_roles
            .insert(state.storage, constants)
            .is_some_and(|old| old != constants)
        {
            return Err(error(
                "PT2 cannot preserve storage shared across weights and constants",
            ));
        }
    }
    let program = &artifact.program;
    let mut symbols = BTreeMap::new();
    let mut ranges = serde_json::Map::new();
    for input in &program.inputs {
        for d in &input.dimensions {
            if let Dimension::Symbol { name, min, max } = d {
                let next = format!("s{}", symbols.len());
                let s = symbols.entry(name.clone()).or_insert(next);
                let range = json!({"min_val":min,"max_val":max});
                if ranges.get(s).is_some_and(|old| old != &range) {
                    return Err(error("symbol bounds must agree for PT2 export"));
                }
                ranges.insert(s.clone(), range);
            }
        }
    }
    let examples = program
        .inputs
        .iter()
        .map(|s| {
            let shape = s
                .dimensions
                .iter()
                .map(|d| match d {
                    Dimension::Known(n) => *n,
                    Dimension::Symbol { min, max, .. } => {
                        max.unwrap_or(i64::MAX).min((*min).max(2))
                    }
                })
                .collect::<Vec<_>>();
            byte_count(&shape, s.dtype)?;
            Tensor::f_ones(&shape, (s.dtype.kind(), Device::Cpu)).map_err(Into::into)
        })
        .collect::<Result<Vec<_>>>()?;
    let values = evaluate(program, &examples, &model.state, Device::Cpu)?;
    let mut tensor_values = serde_json::Map::new();
    let mut signature_inputs = Vec::new();
    let mut graph_inputs = Vec::new();
    let mut rename = BTreeMap::new();
    for (i, state) in artifact.state.iter().enumerate() {
        let placeholder = format!("state_{i}");
        rename.insert(state.name.clone(), placeholder.clone());
        graph_inputs.push(tensor_arg(&placeholder));
        let spec = match state.role {
            StateRole::Parameter => {
                json!({"parameter":{"arg":{"name":placeholder},"parameter_name":state.name}})
            }
            StateRole::Buffer => {
                json!({"buffer":{"arg":{"name":placeholder},"buffer_name":state.name,"persistent":true}})
            }
            StateRole::Constant => {
                json!({"tensor_constant":{"arg":{"name":placeholder},"tensor_constant_name":state.name}})
            }
        };
        signature_inputs.push(spec);
        tensor_values.insert(placeholder, meta(&model.state[&state.name], None, &symbols));
    }
    for input in &program.inputs {
        if rename.values().any(|n| n == &input.name) {
            return Err(error(
                "input name conflicts with exported state placeholder",
            ));
        }
        graph_inputs.push(tensor_arg(&input.name));
        signature_inputs.push(json!({"user_input":{"arg":tensor_arg(&input.name)}}));
        tensor_values.insert(
            input.name.clone(),
            meta(&values[&input.name], Some(&input.dimensions), &symbols),
        );
    }
    let mapped_name = |n: &str| rename.get(n).cloned().unwrap_or_else(|| n.to_owned());
    let inferred = infer_specs(model)?;
    let mut nodes = Vec::new();
    for node in &program.operations {
        let n = node
            .inputs
            .iter()
            .map(|n| mapped_name(n))
            .collect::<Vec<_>>();
        let mut args = Vec::new();
        let target = match &node.operator {
            Operator::Identity | Operator::Relu | Operator::Sigmoid | Operator::Tanh => {
                args.push(named_arg("self", tensor_arg(&n[0])));
                match node.operator {
                    Operator::Identity => "alias.default",
                    Operator::Relu => "relu.default",
                    Operator::Sigmoid => "sigmoid.default",
                    _ => "tanh.default",
                }
            }
            Operator::Add | Operator::Subtract | Operator::Multiply | Operator::Matmul => {
                args.push(named_arg("self", tensor_arg(&n[0])));
                args.push(named_arg("other", tensor_arg(&n[1])));
                match node.operator {
                    Operator::Add => "add.Tensor",
                    Operator::Subtract => "sub.Tensor",
                    Operator::Multiply => "mul.Tensor",
                    _ => "matmul.default",
                }
            }
            Operator::Linear => {
                for (key, name) in ["input", "weight", "bias"].iter().zip(&n) {
                    args.push(named_arg(key, tensor_arg(name)));
                }
                "linear.default"
            }
            Operator::Reshape(shape) => {
                args.push(named_arg("self", tensor_arg(&n[0])));
                args.push(named_arg("shape",json!({"as_sym_ints":shape.iter().map(|v|json!({"as_int":v})).collect::<Vec<_>>()})));
                "reshape.default"
            }
            Operator::Flatten { start, end } => {
                args.push(named_arg("self", tensor_arg(&n[0])));
                args.push(named_arg("start_dim", json!({"as_int":start})));
                args.push(named_arg("end_dim", json!({"as_int":end})));
                "flatten.using_ints"
            }
            Operator::Transpose { dim0, dim1 } => {
                args.push(named_arg("self", tensor_arg(&n[0])));
                args.push(named_arg("dim0", json!({"as_int":dim0})));
                args.push(named_arg("dim1", json!({"as_int":dim1})));
                "transpose.int"
            }
            Operator::Permute(axes) => {
                args.push(named_arg("self", tensor_arg(&n[0])));
                args.push(named_arg("dims", json!({"as_ints":axes})));
                "permute.default"
            }
            Operator::Cat(dim) => {
                args.push(named_arg(
                    "tensors",
                    json!({"as_tensors":n.iter().map(|n|json!({"name":n})).collect::<Vec<_>>()}),
                ));
                args.push(named_arg("dim", json!({"as_int":dim})));
                "cat.default"
            }
            Operator::If { .. } => unreachable!(),
        };
        nodes.push(json!({"target":format!("torch.ops.aten.{target}"),"inputs":args,"outputs":[tensor_arg(&node.name)],"metadata":{},"name":node.name,"is_hop_single_tensor_return":null}));
        tensor_values.insert(
            node.name.clone(),
            meta(
                &values[&node.name],
                Some(&inferred[&node.name].dimensions),
                &symbols,
            ),
        );
    }
    let outputs = program
        .output_tree
        .leaves()?
        .iter()
        .map(|n| tensor_arg(&mapped_name(n)))
        .collect::<Vec<_>>();
    let input_tree = match &program.input_tree {
        Tree::Tuple(v)
            if v.len() == 2
                && matches!(&v[0], Tree::Tuple(_))
                && matches!(&v[1], Tree::Dict(_)) =>
        {
            program.input_tree.clone()
        }
        _ => Tree::Tuple(vec![program.input_tree.clone(), Tree::Dict(vec![])]),
    };
    let ep = json!({"graph_module":{"graph":{"inputs":graph_inputs,"outputs":outputs,"nodes":nodes,"tensor_values":tensor_values,"sym_int_values":{},"sym_bool_values":{},"sym_float_values":{},"custom_obj_values":{},"is_single_tensor_return":false},"signature":{"input_specs":signature_inputs,"output_specs":outputs.iter().map(|a|json!({"user_output":{"arg":a}})).collect::<Vec<_>>()},"module_call_graph":[{"fqn":"","signature":{"inputs":[],"outputs":[],"in_spec":tree_json(&input_tree)?,"out_spec":tree_json(&program.output_tree)?,"forward_arg_names":null}}],"metadata":{},"treespec_namedtuple_fields":{}},"opset_version":{"aten":10},"range_constraints":ranges,"schema_version":{"major":8,"minor":20},"verifiers":["TRAINING"],"torch_version":"2.13.0","guards_code":[]});
    let mut entries = BTreeMap::<String, Vec<u8>>::new();
    entries.insert(
        "models/model.json".into(),
        serde_json::to_vec(&ep).map_err(mapped)?,
    );
    for (directory, suffix, constant) in [
        ("data/weights/", "weights", false),
        ("data/constants/", "constants", true),
    ] {
        let mut config = serde_json::Map::new();
        for state in artifact
            .state
            .iter()
            .filter(|s| (s.role == StateRole::Constant) == constant)
        {
            let storage = &artifact.storages[state.storage];
            let path_name = format!("tensor_{}", state.storage);
            entries.insert(format!("{directory}{path_name}"), storage.bytes.clone());
            let tensor_meta = json!({"dtype":scalar_code(storage.dtype),"sizes":state.shape.iter().map(|n|json!({"as_int":n})).collect::<Vec<_>>(),"strides":state.strides.iter().map(|n|json!({"as_int":n})).collect::<Vec<_>>(),"storage_offset":{"as_int":state.offset},"requires_grad":state.requires_grad,"device":{"type":"cpu","index":null},"layout":7});
            config.insert(state.name.clone(),json!({"path_name":path_name,"is_param":state.role==StateRole::Parameter,"use_pickle":false,"tensor_meta":tensor_meta}));
        }
        entries.insert(
            format!("{directory}model_{suffix}_config.json"),
            serde_json::to_vec(&json!({"config":config})).map_err(mapped)?,
        );
    }
    for (name, data) in [
        ("archive_format", b"pt2".as_slice()),
        ("archive_version", b"0".as_slice()),
        ("byteorder", b"little".as_slice()),
        (".data/version", b"6\n".as_slice()),
        (
            ".data/serialization_id",
            b"0000000000000000000000000000000000000000".as_slice(),
        ),
        ("data/sample_inputs/model.pt", b"".as_slice()),
    ] {
        entries.insert(name.into(), data.to_vec());
    }
    let mut writer = ZipWriter::new(Cursor::new(Vec::new()));
    for (name, data) in entries {
        writer
            .start_file(
                format!("package/{name}"),
                FileOptions::default().compression_method(CompressionMethod::Stored),
            )
            .map_err(mapped)?;
        writer.write_all(&data).map_err(mapped)?;
    }
    let bytes = writer.finish().map_err(mapped)?.into_inner();
    if bytes.len() > MAX_BYTES {
        return Err(error("PT2 output exceeds limit"));
    }
    std::fs::write(path, bytes).map_err(mapped)
}
