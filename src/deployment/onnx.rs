// Explicit wire-field subset from ONNX 1.22.0 onnx/onnx.proto (Apache-2.0),
// commit 2bb50465112feca9003e1ed654d77f01ff1415ca.
// prost supplies the bounded protobuf parser; no hand-written wire decoder.
use super::*;
use prost::Message;

#[derive(Clone, PartialEq, Message)]
struct ModelProto {
    #[prost(int64, tag = "1")]
    ir_version: i64,
    #[prost(string, tag = "2")]
    producer_name: String,
    #[prost(string, tag = "3")]
    producer_version: String,
    #[prost(message, optional, tag = "7")]
    graph: Option<GraphProto>,
    #[prost(message, repeated, tag = "8")]
    opset: Vec<Opset>,
    #[prost(message, repeated, tag = "14")]
    metadata: Vec<Entry>,
    #[prost(bytes = "vec", repeated, tag = "20")]
    training: Vec<Vec<u8>>,
    #[prost(bytes = "vec", repeated, tag = "25")]
    functions: Vec<Vec<u8>>,
    #[prost(bytes = "vec", repeated, tag = "26")]
    configuration: Vec<Vec<u8>>,
}
#[derive(Clone, PartialEq, Message)]
struct Opset {
    #[prost(string, tag = "1")]
    domain: String,
    #[prost(int64, tag = "2")]
    version: i64,
}
#[derive(Clone, PartialEq, Message)]
struct Entry {
    #[prost(string, tag = "1")]
    key: String,
    #[prost(string, tag = "2")]
    value: String,
}
#[derive(Clone, PartialEq, Message)]
struct GraphProto {
    #[prost(message, repeated, tag = "1")]
    nodes: Vec<NodeProto>,
    #[prost(string, tag = "2")]
    name: String,
    #[prost(message, repeated, tag = "5")]
    initializers: Vec<TensorProto>,
    #[prost(message, repeated, tag = "11")]
    inputs: Vec<ValueProto>,
    #[prost(message, repeated, tag = "12")]
    outputs: Vec<ValueProto>,
    #[prost(message, repeated, tag = "13")]
    values: Vec<ValueProto>,
    #[prost(bytes = "vec", repeated, tag = "14")]
    quantization: Vec<Vec<u8>>,
    #[prost(bytes = "vec", repeated, tag = "15")]
    sparse: Vec<Vec<u8>>,
}
#[derive(Clone, PartialEq, Message)]
struct NodeProto {
    #[prost(string, repeated, tag = "1")]
    inputs: Vec<String>,
    #[prost(string, repeated, tag = "2")]
    outputs: Vec<String>,
    #[prost(string, tag = "3")]
    name: String,
    #[prost(string, tag = "4")]
    op: String,
    #[prost(message, repeated, tag = "5")]
    attrs: Vec<Attribute>,
    #[prost(string, tag = "7")]
    domain: String,
    #[prost(string, tag = "8")]
    overload: String,
    #[prost(bytes = "vec", repeated, tag = "10")]
    devices: Vec<Vec<u8>>,
}
#[derive(Clone, PartialEq, Message)]
struct Attribute {
    #[prost(string, tag = "1")]
    name: String,
    #[prost(float, optional, tag = "2")]
    float: Option<f32>,
    #[prost(int64, optional, tag = "3")]
    int: Option<i64>,
    #[prost(bytes = "vec", optional, tag = "4")]
    string: Option<Vec<u8>>,
    #[prost(message, optional, tag = "5")]
    tensor: Option<TensorProto>,
    #[prost(bytes = "vec", optional, tag = "6")]
    graph: Option<Vec<u8>>,
    #[prost(float, repeated, tag = "7")]
    floats: Vec<f32>,
    #[prost(int64, repeated, tag = "8")]
    ints: Vec<i64>,
    #[prost(bytes = "vec", repeated, tag = "9")]
    strings: Vec<Vec<u8>>,
    #[prost(bytes = "vec", repeated, tag = "10")]
    tensors: Vec<Vec<u8>>,
    #[prost(bytes = "vec", repeated, tag = "11")]
    graphs: Vec<Vec<u8>>,
    #[prost(bytes = "vec", optional, tag = "14")]
    type_proto: Option<Vec<u8>>,
    #[prost(bytes = "vec", repeated, tag = "15")]
    type_protos: Vec<Vec<u8>>,
    #[prost(int32, tag = "20")]
    kind: i32,
    #[prost(string, tag = "21")]
    reference: String,
    #[prost(bytes = "vec", optional, tag = "22")]
    sparse: Option<Vec<u8>>,
    #[prost(bytes = "vec", repeated, tag = "23")]
    sparse_tensors: Vec<Vec<u8>>,
}
#[derive(Clone, PartialEq, Message)]
struct TensorProto {
    #[prost(int64, repeated, tag = "1")]
    dims: Vec<i64>,
    #[prost(int32, tag = "2")]
    dtype: i32,
    #[prost(bytes = "vec", optional, tag = "3")]
    segment: Option<Vec<u8>>,
    #[prost(float, repeated, tag = "4")]
    floats: Vec<f32>,
    #[prost(int32, repeated, tag = "5")]
    ints32: Vec<i32>,
    #[prost(bytes = "vec", repeated, tag = "6")]
    strings: Vec<Vec<u8>>,
    #[prost(int64, repeated, tag = "7")]
    ints64: Vec<i64>,
    #[prost(string, tag = "8")]
    name: String,
    #[prost(bytes = "vec", tag = "9")]
    raw: Vec<u8>,
    #[prost(double, repeated, tag = "10")]
    doubles: Vec<f64>,
    #[prost(uint64, repeated, tag = "11")]
    uints64: Vec<u64>,
    #[prost(bytes = "vec", repeated, tag = "13")]
    external: Vec<Vec<u8>>,
    #[prost(int32, tag = "14")]
    location: i32,
}
#[derive(Clone, PartialEq, Message)]
struct ValueProto {
    #[prost(string, tag = "1")]
    name: String,
    #[prost(message, optional, tag = "2")]
    r#type: Option<TypeProto>,
}
#[derive(Clone, PartialEq, Message)]
struct TypeProto {
    #[prost(message, optional, tag = "1")]
    tensor: Option<TensorType>,
    #[prost(bytes = "vec", optional, tag = "4")]
    sequence: Option<Vec<u8>>,
    #[prost(bytes = "vec", optional, tag = "5")]
    map: Option<Vec<u8>>,
    #[prost(bytes = "vec", optional, tag = "7")]
    opaque: Option<Vec<u8>>,
    #[prost(bytes = "vec", optional, tag = "8")]
    sparse: Option<Vec<u8>>,
    #[prost(bytes = "vec", optional, tag = "9")]
    optional: Option<Vec<u8>>,
}
#[derive(Clone, PartialEq, Message)]
struct TensorType {
    #[prost(int32, tag = "1")]
    dtype: i32,
    #[prost(message, optional, tag = "2")]
    shape: Option<Shape>,
}
#[derive(Clone, PartialEq, Message)]
struct Shape {
    #[prost(message, repeated, tag = "1")]
    dims: Vec<Dim>,
}
#[derive(Clone, PartialEq, Message)]
struct Dim {
    #[prost(int64, optional, tag = "1")]
    value: Option<i64>,
    #[prost(string, optional, tag = "2")]
    symbol: Option<String>,
}
fn dtype(n: i32) -> Result<DType> {
    match n {
        1 => Ok(DType::Float),
        11 => Ok(DType::Double),
        7 => Ok(DType::Int64),
        9 => Ok(DType::Bool),
        _ => Err(error(format!("unsupported ONNX dtype {n}"))),
    }
}
fn code(n: DType) -> i32 {
    match n {
        DType::Float => 1,
        DType::Double => 11,
        DType::Int64 => 7,
        DType::Bool => 9,
    }
}
fn value_spec(v: &ValueProto) -> Result<ValueSpec> {
    let ty = v
        .r#type
        .as_ref()
        .ok_or_else(|| error("missing ONNX value type"))?;
    if ty.sequence.is_some()
        || ty.map.is_some()
        || ty.opaque.is_some()
        || ty.sparse.is_some()
        || ty.optional.is_some()
    {
        return Err(error("ONNX requires dense tensor values"));
    }
    let tensor = ty
        .tensor
        .as_ref()
        .ok_or_else(|| error("missing ONNX tensor type"))?;
    let dims = &tensor
        .shape
        .as_ref()
        .ok_or_else(|| error("unknown ONNX rank is unsupported"))?
        .dims;
    let dimensions = dims
        .iter()
        .map(|d| match (&d.value, &d.symbol) {
            (Some(n), None) => Ok(Dimension::Known(*n)),
            (None, Some(name)) if !name.is_empty() => Ok(Dimension::Symbol {
                name: name.clone(),
                min: 0,
                max: None,
            }),
            _ => Err(error("invalid or unnamed ONNX dimension")),
        })
        .collect::<Result<_>>()?;
    Ok(ValueSpec::new(&v.name, dtype(tensor.dtype)?, dimensions))
}
fn value_proto(spec: &ValueSpec) -> ValueProto {
    ValueProto {
        name: spec.name.clone(),
        r#type: Some(TypeProto {
            tensor: Some(TensorType {
                dtype: code(spec.dtype),
                shape: Some(Shape {
                    dims: spec
                        .dimensions
                        .iter()
                        .map(|d| match d {
                            Dimension::Known(n) => Dim {
                                value: Some(*n),
                                symbol: None,
                            },
                            Dimension::Symbol { name, .. } => Dim {
                                value: None,
                                symbol: Some(name.clone()),
                            },
                        })
                        .collect(),
                }),
            }),
            ..Default::default()
        }),
    }
}
fn raw_tensor(t: &TensorProto) -> Result<(DType, Vec<u8>)> {
    if t.segment.is_some()
        || !t.external.is_empty()
        || t.location != 0
        || !t.strings.is_empty()
        || !t.uints64.is_empty()
    {
        return Err(error("ONNX external, segmented or unsupported tensor data"));
    }
    let dtype = dtype(t.dtype)?;
    let expected = byte_count(&t.dims, dtype)?;
    let alternatives = [
        !t.floats.is_empty(),
        !t.doubles.is_empty(),
        !t.ints64.is_empty(),
        !t.ints32.is_empty(),
        !t.raw.is_empty(),
    ]
    .iter()
    .filter(|v| **v)
    .count();
    if alternatives > 1 {
        return Err(error("ONNX tensor has conflicting data fields"));
    }
    let bytes = if !t.raw.is_empty() {
        t.raw.clone()
    } else {
        match dtype {
            DType::Float if t.doubles.is_empty() && t.ints64.is_empty() && t.ints32.is_empty() => {
                t.floats.iter().flat_map(|x| x.to_le_bytes()).collect()
            }
            DType::Double if t.floats.is_empty() && t.ints64.is_empty() && t.ints32.is_empty() => {
                t.doubles.iter().flat_map(|x| x.to_le_bytes()).collect()
            }
            DType::Int64 if t.floats.is_empty() && t.doubles.is_empty() && t.ints32.is_empty() => {
                t.ints64.iter().flat_map(|x| x.to_le_bytes()).collect()
            }
            DType::Bool if t.floats.is_empty() && t.doubles.is_empty() && t.ints64.is_empty() => t
                .ints32
                .iter()
                .map(|x| {
                    if *x == 0 || *x == 1 {
                        Ok(*x as u8)
                    } else {
                        Err(error("invalid ONNX Boolean"))
                    }
                })
                .collect::<Result<_>>()?,
            _ => return Err(error("ONNX data field disagrees with dtype")),
        }
    };
    if bytes.len() != expected || (dtype == DType::Bool && bytes.iter().any(|x| *x > 1)) {
        return Err(error("ONNX tensor byte length or Boolean data is invalid"));
    }
    Ok((dtype, bytes))
}
fn int_attr(
    attrs: &mut BTreeMap<String, Attribute>,
    name: &str,
    default: Option<i64>,
) -> Result<i64> {
    match attrs.remove(name) {
        Some(a) if a.kind == 2 => a
            .int
            .ok_or_else(|| error("missing ONNX integer attribute value")),
        Some(_) => Err(error("ONNX attribute type mismatch")),
        None => default.ok_or_else(|| error(format!("missing ONNX attribute {name}"))),
    }
}
fn attribute_check(a: &Attribute) -> Result<()> {
    if !a.reference.is_empty()
        || a.graph.is_some()
        || a.type_proto.is_some()
        || a.sparse.is_some()
        || !a.graphs.is_empty()
        || !a.type_protos.is_empty()
        || !a.sparse_tensors.is_empty()
        || !a.tensors.is_empty()
        || !a.strings.is_empty()
        || a.string.is_some()
        || !a.floats.is_empty()
    {
        return Err(error("unsupported ONNX attribute payload"));
    }
    let count = [
        a.int.is_some(),
        a.float.is_some(),
        a.tensor.is_some(),
        !a.ints.is_empty(),
    ]
    .iter()
    .filter(|x| **x)
    .count();
    if count > 1 {
        return Err(error("conflicting ONNX attribute fields"));
    }
    Ok(())
}
// Preflight message counts before prost allocates repeated message vectors.
// Scalar wire decoding/skipping remains entirely in prost.
fn preflight(mut bytes: &[u8], schema: u8, depth: usize, budget: &mut usize) -> Result<()> {
    use prost::encoding::{DecodeContext, WireType, decode_key, decode_varint, skip_field};
    if depth > MAX_DEPTH {
        return Err(error("ONNX protobuf nesting exceeds limit"));
    }
    while !bytes.is_empty() {
        *budget += 1;
        if *budget > MAX_NODES * 32 {
            return Err(error("ONNX protobuf field count exceeds limit"));
        }
        let (tag, wire) = decode_key(&mut bytes).map_err(mapped)?;
        let child = match (schema, tag) {
            (0, 7) => Some(1),
            (0, 8) => Some(9),
            (0, 14) => Some(10),
            (1, 1) => Some(2),
            (1, 5) => Some(4),
            (1, 11..=13) => Some(5),
            (2, 5) => Some(3),
            (3, 5) => Some(4),
            (5, 2) => Some(6),
            (6, 1) => Some(7),
            (7, 2) => Some(8),
            (8, 1) => Some(11),
            _ => None,
        };
        let known = match schema {
            0 => matches!(tag, 1..=8 | 14 | 20 | 25 | 26),
            1 => matches!(tag, 1 | 2 | 5 | 10..=16),
            2 => matches!(tag, 1..=10),
            3 => matches!(tag,1..=11|13..=15|20..=23),
            4 => matches!(tag, 1..=16),
            5 => matches!(tag, 1..=4),
            6 => matches!(tag, 1 | 4..=9),
            7 => matches!(tag, 1 | 2),
            8 => tag == 1,
            9 => matches!(tag, 1 | 2),
            10 => matches!(tag, 1 | 2),
            11 => matches!(tag, 1..=3),
            _ => false,
        };
        if !known {
            return Err(error(format!("unknown ONNX schema field {schema}:{tag}")));
        }
        if let Some(child) = child {
            if wire != WireType::LengthDelimited {
                return Err(error("invalid ONNX message wire type"));
            }
            let size =
                usize::try_from(decode_varint(&mut bytes).map_err(mapped)?).map_err(mapped)?;
            if size > bytes.len() {
                return Err(error("truncated ONNX message"));
            }
            let (part, rest) = bytes.split_at(size);
            preflight(part, child, depth + 1, budget)?;
            bytes = rest;
        } else {
            skip_field(wire, tag, &mut bytes, DecodeContext::default()).map_err(mapped)?;
        }
    }
    Ok(())
}
pub(super) fn load(path: &Path) -> Result<Model> {
    let bytes = read_bounded(path)?;
    preflight(&bytes, 0, 0, &mut 0)?;
    let model = ModelProto::decode(bytes.as_slice()).map_err(mapped)?;
    if model.ir_version != 10
        || model.opset.len() != 1
        || !model.opset[0].domain.is_empty()
        || model.opset[0].version != 18
    {
        return Err(error("ONNX requires IR10 and default-domain opset18"));
    }
    if !model.training.is_empty() || !model.functions.is_empty() || !model.configuration.is_empty()
    {
        return Err(error(
            "ONNX training, functions and device configuration are unsupported",
        ));
    }
    let graph = model.graph.ok_or_else(|| error("missing ONNX graph"))?;
    if graph.nodes.len() > MAX_NODES
        || graph.initializers.len() > MAX_NODES
        || !graph.sparse.is_empty()
        || !graph.quantization.is_empty()
    {
        return Err(error("unsupported or excessive ONNX graph"));
    }
    let mut storages = Vec::new();
    let mut state = Vec::new();
    let mut constants = BTreeMap::new();
    for tensor in &graph.initializers {
        let (dtype, bytes) = raw_tensor(tensor)?;
        let storage = storages.len();
        storages.push(Storage { dtype, bytes });
        state.push(State {
            name: tensor.name.clone(),
            role: StateRole::Constant,
            storage,
            shape: tensor.dims.clone(),
            strides: contiguous_strides(&tensor.dims)?,
            offset: 0,
            requires_grad: false,
        });
        if constants
            .insert(tensor.name.clone(), tensor.clone())
            .is_some()
        {
            return Err(error("duplicate ONNX initializer"));
        }
    }
    let mut inputs = graph
        .inputs
        .iter()
        .map(value_spec)
        .collect::<Result<Vec<_>>>()?;
    if inputs.iter().any(|s| constants.contains_key(&s.name)) {
        return Err(error("overridable ONNX initializers are unsupported"));
    }
    let mut ranks = inputs
        .iter()
        .map(|s| (s.name.clone(), s.dimensions.len()))
        .collect::<BTreeMap<_, _>>();
    for s in &state {
        ranks.insert(s.name.clone(), s.shape.len());
    }
    for v in &graph.values {
        let s = value_spec(v)?;
        ranks.insert(s.name, s.dimensions.len());
    }
    let mut operations = Vec::new();
    let mut reserved = inputs
        .iter()
        .map(|s| s.name.clone())
        .chain(state.iter().map(|s| s.name.clone()))
        .collect::<BTreeSet<_>>();
    for node in graph.nodes {
        if !node.domain.is_empty()
            || !node.overload.is_empty()
            || !node.devices.is_empty()
            || node.outputs.len() != 1
        {
            return Err(error(
                "ONNX custom operators, overloads, devices or multiple outputs are unsupported",
            ));
        }
        let name = &node.outputs[0];
        if reserved.contains(name) || node.inputs.iter().any(|name| !reserved.contains(name)) {
            return Err(error("ONNX duplicate result or non-topological operand"));
        }
        reserved.insert(name.clone());
        let mut attrs = BTreeMap::new();
        for a in node.attrs {
            attribute_check(&a)?;
            if attrs.insert(a.name.clone(), a).is_some() {
                return Err(error("duplicate ONNX attribute"));
            }
        }
        let mut operands = node.inputs;
        let operator = match node.op.as_str() {
            "Identity" => Operator::Identity,
            "Relu" => Operator::Relu,
            "Sigmoid" => Operator::Sigmoid,
            "Tanh" => Operator::Tanh,
            "Add" => Operator::Add,
            "Sub" => Operator::Subtract,
            "Mul" => Operator::Multiply,
            "MatMul" => Operator::Matmul,
            "Transpose" => {
                let axes = match attrs.remove("perm") {
                    Some(a) if a.kind == 7 => a.ints,
                    Some(_) => return Err(error("ONNX permutation attribute type mismatch")),
                    None => (0..*ranks
                        .get(
                            operands
                                .first()
                                .ok_or_else(|| error("missing transpose operand"))?,
                        )
                        .ok_or_else(|| error("ONNX default transpose requires known rank"))?
                        as i64)
                        .rev()
                        .collect(),
                };
                Operator::Permute(axes)
            }
            "Concat" => Operator::Cat(int_attr(&mut attrs, "axis", None)?),
            "Flatten" => {
                let rank = *ranks
                    .get(
                        operands
                            .first()
                            .ok_or_else(|| error("missing flatten operand"))?,
                    )
                    .ok_or_else(|| error("ONNX Flatten requires known rank"))?;
                let axis = int_attr(&mut attrs, "axis", Some(1))?;
                let axis = if axis < 0 { rank as i64 + axis } else { axis };
                if axis != 1 {
                    return Err(error("ONNX Flatten currently supports axis1 only"));
                }
                Operator::Flatten { start: 1, end: -1 }
            }
            "Reshape" => {
                if operands.len() != 2 {
                    return Err(error("ONNX Reshape requires data and constant shape"));
                }
                let shape = constants
                    .get(&operands[1])
                    .ok_or_else(|| error("ONNX Reshape requires initializer shape"))?;
                let (dt, bytes) = raw_tensor(shape)?;
                if dt != DType::Int64 || shape.dims.len() != 1 {
                    return Err(error("ONNX reshape shape requires Int64 vector"));
                }
                let dims = bytes
                    .as_chunks::<8>()
                    .0
                    .iter()
                    .map(|x| i64::from_le_bytes(*x))
                    .collect::<Vec<_>>();
                let allowzero = int_attr(&mut attrs, "allowzero", Some(0))?;
                if allowzero != 1 && dims.contains(&0) {
                    return Err(error("ONNX reshape copy-zero dimensions are unsupported"));
                }
                if allowzero != 0 && allowzero != 1 {
                    return Err(error("invalid ONNX reshape allowzero"));
                }
                operands.pop();
                Operator::Reshape(dims)
            }
            "Gemm" => {
                if !(operands.len() == 2 || operands.len() == 3) {
                    return Err(error("invalid ONNX Gemm arity"));
                }
                for key in ["alpha", "beta"] {
                    if let Some(a) = attrs.remove(key)
                        && (a.kind != 1 || a.float != Some(1.))
                    {
                        return Err(error("ONNX Gemm alpha/beta must equal one"));
                    }
                }
                let trans_a = int_attr(&mut attrs, "transA", Some(0))?;
                let trans_b = int_attr(&mut attrs, "transB", Some(0))?;
                if trans_a != 0 || trans_b != 1 {
                    return Err(error("ONNX Gemm requires transA0/transB1"));
                }
                if operands[..2].iter().any(|n| ranks.get(n) != Some(&2)) {
                    return Err(error("ONNX Gemm requires rank2 operands"));
                }
                Operator::Linear
            }
            "Constant" => {
                if !operands.is_empty() {
                    return Err(error("ONNX Constant has operands"));
                }
                let a = attrs
                    .remove("value")
                    .ok_or_else(|| error("ONNX Constant requires tensor value"))?;
                if a.kind != 4 {
                    return Err(error("ONNX Constant attribute must be a tensor"));
                }
                let mut tensor = a
                    .tensor
                    .ok_or_else(|| error("missing ONNX Constant tensor"))?;
                tensor.name = name.clone();
                let (dtype, bytes) = raw_tensor(&tensor)?;
                let storage = storages.len();
                storages.push(Storage { dtype, bytes });
                state.push(State {
                    name: name.clone(),
                    role: StateRole::Constant,
                    storage,
                    shape: tensor.dims.clone(),
                    strides: contiguous_strides(&tensor.dims)?,
                    offset: 0,
                    requires_grad: false,
                });
                ranks.insert(name.clone(), tensor.dims.len());
                constants.insert(name.clone(), tensor);
                if !attrs.is_empty() {
                    return Err(error("unsupported ONNX Constant attributes"));
                }
                continue;
            }
            other => return Err(error(format!("unsupported ONNX operator {other}"))),
        };
        if !attrs.is_empty() {
            return Err(error(format!(
                "unsupported ONNX {} attributes {:?}",
                node.op,
                attrs.keys()
            )));
        }
        if matches!(operator, Operator::Linear | Operator::Flatten { .. }) {
            ranks.insert(name.clone(), 2);
        } else if let Operator::Reshape(shape) = &operator {
            ranks.insert(name.clone(), shape.len());
        } else if let Operator::Permute(axes) = &operator {
            ranks.insert(name.clone(), axes.len());
        } else if matches!(operator, Operator::Matmul) && operands.len() == 2 {
            if let (Some(a), Some(b)) = (ranks.get(&operands[0]), ranks.get(&operands[1])) {
                if *a == 0 || *b == 0 {
                    return Err(error("ONNX MatMul requires nonscalar operands"));
                }
                let rank = if *a == 1 && *b == 1 {
                    0
                } else if *a == 1 {
                    b - 1
                } else if *b == 1 {
                    a - 1
                } else {
                    (*a).max(*b)
                };
                ranks.insert(name.clone(), rank);
            }
        } else if let Some(rank) = operands.first().and_then(|n| ranks.get(n)).copied() {
            ranks.insert(name.clone(), rank);
        }
        operations.push(Operation {
            name: name.clone(),
            operator,
            inputs: operands,
        });
    }
    let output_names = graph
        .outputs
        .iter()
        .map(|v| value_spec(v).map(|s| s.name))
        .collect::<Result<Vec<_>>>()?;
    let mut program = Program::new(
        inputs.clone(),
        operations,
        Tree::Tuple(output_names.iter().cloned().map(Tree::Tensor).collect()),
    );
    let mut metadata = BTreeMap::new();
    for pair in model.metadata {
        if metadata.insert(pair.key, pair.value).is_some() {
            return Err(error("duplicate ONNX metadata property"));
        }
    }
    if let Some(value) = metadata.get("rusttorch.inputs.v1") {
        let declared: Vec<ValueSpec> = serde_json::from_str(value).map_err(mapped)?;
        if declared.len() != inputs.len() {
            return Err(error("ONNX guard metadata count mismatch"));
        }
        for (a, b) in declared.iter().zip(&inputs) {
            if a.name != b.name || a.dtype != b.dtype || a.dimensions.len() != b.dimensions.len() {
                return Err(error("ONNX guard metadata disagrees with tensor type"));
            }
            for (a, b) in a.dimensions.iter().zip(&b.dimensions) {
                if !matches!((a,b),(Dimension::Known(a),Dimension::Known(b)) if a==b)
                    && !matches!((a,b),(Dimension::Symbol{name:a,..},Dimension::Symbol{name:b,..}) if a==b)
                {
                    return Err(error("ONNX dimension guard mismatch"));
                }
            }
        }
        inputs = declared;
        program.inputs = inputs;
    }
    if let Some(value) = metadata.get("rusttorch.input_tree.v1") {
        program.input_tree = serde_json::from_str(value).map_err(mapped)?;
    }
    if let Some(value) = metadata.get("rusttorch.output_tree.v1") {
        let tree: Tree = serde_json::from_str(value).map_err(mapped)?;
        if tree.leaves()? != output_names.iter().map(String::as_str).collect::<Vec<_>>() {
            return Err(error("ONNX output tree mismatch"));
        }
        program.output_tree = tree;
    }
    Model::from_artifact(
        Artifact {
            program,
            storages,
            state,
        },
        Device::Cpu,
    )
}
fn attr_int(name: &str, n: i64) -> Attribute {
    Attribute {
        name: name.into(),
        kind: 2,
        int: Some(n),
        ..Default::default()
    }
}
fn node(name: &str, op: &str, inputs: Vec<String>, attrs: Vec<Attribute>) -> NodeProto {
    NodeProto {
        name: name.into(),
        op: op.into(),
        inputs,
        outputs: vec![name.into()],
        attrs,
        ..Default::default()
    }
}
pub(super) fn save(model: &Model, path: &Path) -> Result<()> {
    let program = model.program();
    if model.device != Device::Cpu {
        return Err(error("ONNX export requires CPU state"));
    }
    let artifact = model.snapshot()?;
    let mut used = BTreeSet::new();
    for s in &artifact.state {
        if !used.insert(s.storage) {
            return Err(error("ONNX cannot preserve shared state storage"));
        }
    }
    let examples = program
        .inputs
        .iter()
        .map(|s| {
            let dims = s
                .dimensions
                .iter()
                .map(|d| match d {
                    Dimension::Known(n) => *n,
                    Dimension::Symbol { min, max, .. } => {
                        max.unwrap_or(i64::MAX).min((*min).max(2))
                    }
                })
                .collect::<Vec<_>>();
            byte_count(&dims, s.dtype)?;
            Tensor::f_ones(&dims, (s.dtype.kind(), Device::Cpu)).map_err(Into::into)
        })
        .collect::<Result<Vec<_>>>()?;
    let values = evaluate(program, &examples, &model.state, Device::Cpu)?;
    let mut initializers = Vec::new();
    for (name, tensor) in &model.state {
        let dtype = DType::from_kind(tensor.kind())?;
        let tensor = tensor.f_contiguous()?;
        let mut raw = vec![0; byte_count(&tensor.size(), dtype)?];
        tensor.f_copy_data_u8(&mut raw, tensor.numel())?;
        initializers.push(TensorProto {
            name: name.clone(),
            dtype: code(dtype),
            dims: tensor.size(),
            raw,
            ..Default::default()
        });
    }
    let mut reserved = values.keys().cloned().collect::<BTreeSet<_>>();
    let mut serial = 0;
    let mut fresh = || loop {
        let name = format!("rusttorch_lowered_{serial}");
        serial += 1;
        if reserved.insert(name.clone()) {
            break name;
        }
    };
    let mut nodes = Vec::new();
    for operation in &program.operations {
        let mut inputs = operation.inputs.clone();
        let mut attrs = Vec::new();
        let op = match &operation.operator {
            Operator::Identity => "Identity",
            Operator::Relu => "Relu",
            Operator::Sigmoid => "Sigmoid",
            Operator::Tanh => "Tanh",
            Operator::Add => "Add",
            Operator::Subtract => "Sub",
            Operator::Multiply => "Mul",
            Operator::Matmul => "MatMul",
            Operator::Linear => {
                let transposed = fresh();
                nodes.push(node(
                    &transposed,
                    "Transpose",
                    vec![inputs[1].clone()],
                    vec![Attribute {
                        name: "perm".into(),
                        kind: 7,
                        ints: vec![1, 0],
                        ..Default::default()
                    }],
                ));
                let product = if inputs.len() == 3 {
                    fresh()
                } else {
                    operation.name.clone()
                };
                nodes.push(node(
                    &product,
                    "MatMul",
                    vec![inputs[0].clone(), transposed],
                    vec![],
                ));
                if inputs.len() == 3 {
                    nodes.push(node(
                        &operation.name,
                        "Add",
                        vec![product, inputs[2].clone()],
                        vec![],
                    ));
                }
                continue;
            }
            Operator::Reshape(shape) => {
                let name = fresh();
                initializers.push(TensorProto {
                    name: name.clone(),
                    dtype: 7,
                    dims: vec![shape.len() as i64],
                    ints64: shape.clone(),
                    ..Default::default()
                });
                inputs.push(name);
                attrs.push(attr_int("allowzero", 1));
                "Reshape"
            }
            Operator::Flatten { start: 1, end: -1 } => {
                attrs.push(attr_int("axis", 1));
                "Flatten"
            }
            Operator::Flatten { .. } => {
                return Err(error("ONNX export supports flatten start1/end-1 only"));
            }
            Operator::Permute(axes) => {
                attrs.push(Attribute {
                    name: "perm".into(),
                    kind: 7,
                    ints: axes.clone(),
                    ..Default::default()
                });
                "Transpose"
            }
            Operator::Transpose { dim0, dim1 } => {
                let rank = values[&inputs[0]].dim() as i64;
                let a = if *dim0 < 0 { rank + *dim0 } else { *dim0 };
                let b = if *dim1 < 0 { rank + *dim1 } else { *dim1 };
                if a < 0 || b < 0 || a >= rank || b >= rank {
                    return Err(error("invalid transpose axes"));
                }
                let mut axes = (0..rank).collect::<Vec<_>>();
                axes.swap(a as usize, b as usize);
                attrs.push(Attribute {
                    name: "perm".into(),
                    kind: 7,
                    ints: axes,
                    ..Default::default()
                });
                "Transpose"
            }
            Operator::Cat(dim) => {
                attrs.push(attr_int("axis", *dim));
                "Concat"
            }
            Operator::If { .. } => return Err(error("ONNX conditional export is unsupported")),
        };
        nodes.push(node(&operation.name, op, inputs, attrs));
    }
    let inferred = infer_specs(model)?;
    let outputs = program
        .output_tree
        .leaves()?
        .iter()
        .map(|name| value_proto(&inferred[*name]))
        .collect();
    let metadata = vec![
        Entry {
            key: "rusttorch.inputs.v1".into(),
            value: serde_json::to_string(&program.inputs).map_err(mapped)?,
        },
        Entry {
            key: "rusttorch.input_tree.v1".into(),
            value: serde_json::to_string(&program.input_tree).map_err(mapped)?,
        },
        Entry {
            key: "rusttorch.output_tree.v1".into(),
            value: serde_json::to_string(&program.output_tree).map_err(mapped)?,
        },
    ];
    let output = ModelProto {
        ir_version: 10,
        producer_name: "RustTorch".into(),
        producer_version: env!("CARGO_PKG_VERSION").into(),
        opset: vec![Opset {
            domain: String::new(),
            version: 18,
        }],
        graph: Some(GraphProto {
            name: "RustTorchProgram".into(),
            nodes,
            initializers,
            inputs: program.inputs.iter().map(value_proto).collect(),
            outputs,
            ..Default::default()
        }),
        metadata,
        ..Default::default()
    }
    .encode_to_vec();
    if output.len() > MAX_BYTES {
        return Err(error("ONNX output exceeds size limit"));
    }
    std::fs::write(path, output).map_err(mapped)
}
