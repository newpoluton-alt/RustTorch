#![doc = include_str!("../../docs/deployment.md")]

mod onnx;
mod pt2;
mod zip_envelope;
pub(crate) use zip_envelope::validate_zip_envelope;

use crate::{Result, RustTorchError};
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet},
    io::{Read, Write},
    path::Path,
};
use tch::{CModule, Device, IValue, Kind, Tensor};

const MAX_BYTES: usize = 256 * 1024 * 1024;
const MAX_NODES: usize = 10_000;
const MAX_DEPTH: usize = 32;

fn error(message: impl Into<String>) -> RustTorchError {
    RustTorchError::GraphValidation(format!("deployment: {}", message.into()))
}
fn mapped(e: impl std::fmt::Display) -> RustTorchError {
    error(e.to_string())
}

/// Dtypes supported by the portable interchange subset.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DType {
    /// IEEE 32-bit floating point.
    Float,
    /// IEEE 64-bit floating point.
    Double,
    /// Signed 64-bit integer.
    Int64,
    /// Boolean tensor.
    Bool,
}
impl DType {
    fn kind(self) -> Kind {
        match self {
            Self::Float => Kind::Float,
            Self::Double => Kind::Double,
            Self::Int64 => Kind::Int64,
            Self::Bool => Kind::Bool,
        }
    }
    fn from_kind(kind: Kind) -> Result<Self> {
        match kind {
            Kind::Float => Ok(Self::Float),
            Kind::Double => Ok(Self::Double),
            Kind::Int64 => Ok(Self::Int64),
            Kind::Bool => Ok(Self::Bool),
            _ => Err(error(format!("unsupported dtype {kind:?}"))),
        }
    }
    fn bytes(self) -> usize {
        self.kind().elt_size_in_bytes()
    }
}

/// A fixed dimension or a named dimension with an inclusive runtime range.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Dimension {
    /// Exact nonnegative size.
    Known(i64),
    /// Equal wherever the name occurs; bounded by `min` and optional `max`.
    Symbol {
        /// Nonempty symbol name.
        name: String,
        /// Inclusive nonnegative lower bound.
        min: i64,
        /// Inclusive upper bound, or no upper bound.
        max: Option<i64>,
    },
}

/// Runtime tensor name, dtype and dimensions. Inputs use the model's device.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ValueSpec {
    /// Unique value name.
    pub name: String,
    /// Required tensor dtype.
    pub dtype: DType,
    /// Required dimensions, in axis order.
    pub dimensions: Vec<Dimension>,
}
impl ValueSpec {
    /// Declares one input, for example `ValueSpec::new("x", DType::Float, vec![Dimension::Known(4)])`.
    pub fn new(name: impl Into<String>, dtype: DType, dimensions: Vec<Dimension>) -> Self {
        Self {
            name: name.into(),
            dtype,
            dimensions,
        }
    }
}

/// Tensor-only calling structure. [`Model::run`] takes/returns flattened leaves.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Tree {
    /// Tensor identified by its graph value name.
    Tensor(String),
    /// Ordered tuple of subtrees.
    Tuple(Vec<Tree>),
    /// Ordered list of subtrees.
    List(Vec<Tree>),
    /// Ordered dictionary with unique string keys.
    Dict(Vec<(String, Tree)>),
}
impl Tree {
    /// Returns leaf names in calling order, rejecting excessive depth or duplicate dictionary keys.
    pub fn leaves(&self) -> Result<Vec<&str>> {
        fn walk<'a>(t: &'a Tree, depth: usize, out: &mut Vec<&'a str>) -> Result<()> {
            if depth > MAX_DEPTH || out.len() > MAX_NODES {
                return Err(error("calling tree exceeds limits"));
            }
            match t {
                Tree::Tensor(n) => out.push(n),
                Tree::Tuple(v) | Tree::List(v) => {
                    for x in v {
                        walk(x, depth + 1, out)?;
                    }
                }
                Tree::Dict(v) => {
                    let mut keys = BTreeSet::new();
                    for (key, x) in v {
                        if !keys.insert(key) {
                            return Err(error("duplicate dictionary key"));
                        }
                        walk(x, depth + 1, out)?;
                    }
                }
            }
            Ok(())
        }
        let mut result = Vec::new();
        walk(self, 0, &mut result)?;
        Ok(result)
    }
}

/// Version-one, functional tensor operators; none mutate their operands.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Operator {
    /// Return the same tensor, preserving its storage alias.
    Identity,
    /// Rectified linear activation.
    Relu,
    /// Logistic activation.
    Sigmoid,
    /// Hyperbolic tangent.
    Tanh,
    /// Broadcast tensor addition.
    Add,
    /// Broadcast tensor subtraction.
    Subtract,
    /// Broadcast tensor multiplication.
    Multiply,
    /// LibTorch matrix multiplication, including supported batch broadcasting.
    Matmul,
    /// Linear transformation with operands input, weight, optional bias.
    Linear,
    /// Reshape to a fixed shape; one `-1` dimension may be inferred.
    Reshape(Vec<i64>),
    /// Flatten the inclusive axis range.
    Flatten {
        /// First axis.
        start: i64,
        /// Last axis.
        end: i64,
    },
    /// Swap two axes.
    Transpose {
        /// First axis.
        dim0: i64,
        /// Second axis.
        dim1: i64,
    },
    /// Permute every axis exactly once.
    Permute(Vec<i64>),
    /// Concatenate operands along an axis.
    Cat(i64),
    /// Execute only the selected branch. First operand is a scalar Boolean;
    /// remaining operands bind each branch's inputs in order. Branches return
    /// exactly one tensor and may read the enclosing model's named state.
    If {
        /// Branch used for a true predicate.
        then_branch: Box<Program>,
        /// Branch used for a false predicate.
        else_branch: Box<Program>,
        /// Common output contract, checked for the selected branch.
        result: ValueSpec,
    },
}

/// One named operator result and its ordered operands.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Operation {
    /// Unique result name.
    pub name: String,
    /// Operation and constant attributes.
    pub operator: Operator,
    /// Earlier value or state names.
    pub inputs: Vec<String>,
}
impl Operation {
    /// Defines an operation, for example `Operation::new("y", Operator::Relu, ["x"])`.
    pub fn new<S: Into<String>>(
        name: impl Into<String>,
        operator: Operator,
        inputs: impl IntoIterator<Item = S>,
    ) -> Self {
        Self {
            name: name.into(),
            operator,
            inputs: inputs.into_iter().map(Into::into).collect(),
        }
    }
}

/// A versioned, explicit tensor program; no implicit tracing of Rust branches.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Program {
    /// Serialization version. Only `1` is supported.
    pub version: u32,
    /// Operator-set version. Only `1` is supported.
    pub operator_version: u32,
    /// Input tensors in flattened calling order.
    pub inputs: Vec<ValueSpec>,
    /// Tensor-only input calling structure, with leaves matching `inputs`.
    pub input_tree: Tree,
    /// Topologically ordered operations.
    pub operations: Vec<Operation>,
    /// Output structure and graph value names.
    pub output_tree: Tree,
}
impl Program {
    /// Creates a version-one program with a tuple of positional inputs.
    pub fn new(inputs: Vec<ValueSpec>, operations: Vec<Operation>, output_tree: Tree) -> Self {
        let input_tree = Tree::Tuple(
            inputs
                .iter()
                .map(|s| Tree::Tensor(s.name.clone()))
                .collect(),
        );
        Self {
            version: 1,
            operator_version: 1,
            inputs,
            input_tree,
            operations,
            output_tree,
        }
    }
    fn validate(&self, state: &BTreeSet<String>, depth: usize, budget: &mut usize) -> Result<()> {
        if self.version != 1 || self.operator_version != 1 {
            return Err(error("unsupported program or operator version"));
        }
        if depth > MAX_DEPTH {
            return Err(error("conditional nesting exceeds limit"));
        }
        *budget = budget
            .checked_add(self.operations.len() + self.inputs.len())
            .ok_or_else(|| error("node limit overflow"))?;
        if *budget > MAX_NODES {
            return Err(error("program exceeds node limit"));
        }
        let mut names = state.clone();
        for input in &self.inputs {
            valid_name(&input.name)?;
            if !names.insert(input.name.clone()) {
                return Err(error(format!("duplicate input/state {}", input.name)));
            }
            if input.dimensions.len() > 64 {
                return Err(error("rank exceeds 64"));
            }
            for d in &input.dimensions {
                match d {
                    Dimension::Known(n) if *n >= 0 => (),
                    Dimension::Symbol { name, min, max }
                        if !name.is_empty() && *min >= 0 && max.is_none_or(|n| n >= *min) => {}
                    _ => return Err(error("invalid dimension or range")),
                }
            }
        }
        if self.input_tree.leaves()?
            != self
                .inputs
                .iter()
                .map(|s| s.name.as_str())
                .collect::<Vec<_>>()
        {
            return Err(error("input tree differs from flattened input order"));
        }
        for node in &self.operations {
            valid_name(&node.name)?;
            if names.contains(&node.name) || node.inputs.iter().any(|n| !names.contains(n)) {
                return Err(error(format!(
                    "{} has a duplicate result, forward reference or missing operand",
                    node.name
                )));
            }
            let arity = node.inputs.len();
            let valid = match &node.operator {
                Operator::Identity
                | Operator::Relu
                | Operator::Sigmoid
                | Operator::Tanh
                | Operator::Transpose { .. }
                | Operator::Flatten { .. } => arity == 1,
                Operator::Reshape(shape) => {
                    arity == 1
                        && shape.len() <= 64
                        && shape.iter().all(|n| *n >= -1)
                        && shape.iter().filter(|n| **n == -1).count() <= 1
                }
                Operator::Permute(axes) => {
                    arity == 1
                        && axes.len() <= 64
                        && axes.iter().copied().collect::<BTreeSet<_>>()
                            == (0..axes.len() as i64).collect()
                }
                Operator::Add | Operator::Subtract | Operator::Multiply | Operator::Matmul => {
                    arity == 2
                }
                Operator::Linear => arity == 2 || arity == 3,
                Operator::Cat(_) => arity > 0,
                Operator::If {
                    then_branch,
                    else_branch,
                    result,
                } => {
                    then_branch.validate(state, depth + 1, budget)?;
                    else_branch.validate(state, depth + 1, budget)?;
                    if then_branch.inputs != else_branch.inputs
                        || then_branch.output_tree.leaves()?.len() != 1
                        || else_branch.output_tree.leaves()?.len() != 1
                    {
                        return Err(error(
                            "conditional branches require identical inputs and one output each",
                        ));
                    }
                    Program::new(
                        vec![result.clone()],
                        vec![],
                        Tree::Tensor(result.name.clone()),
                    )
                    .validate(&BTreeSet::new(), depth + 1, budget)?;
                    arity == then_branch.inputs.len() + 1
                }
            };
            if !valid {
                return Err(error(format!(
                    "invalid operator attributes or arity at {}",
                    node.name
                )));
            }
            names.insert(node.name.clone());
        }
        let outputs = self.output_tree.leaves()?;
        if outputs.is_empty() || outputs.iter().any(|n| !names.contains(*n)) {
            return Err(error("missing or unknown graph outputs"));
        }
        Ok(())
    }
}

/// Meaning of a named state tensor in a portable program.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StateRole {
    /// Trainable model parameter.
    Parameter,
    /// Persistent, non-parameter state.
    Buffer,
    /// Non-persistent buffer or constant tensor.
    Constant,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Storage {
    dtype: DType,
    bytes: Vec<u8>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct State {
    name: String,
    role: StateRole,
    storage: usize,
    shape: Vec<i64>,
    strides: Vec<i64>,
    offset: i64,
    requires_grad: bool,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Artifact {
    program: Program,
    storages: Vec<Storage>,
    state: Vec<State>,
}

/// A validated portable program and its named tensors on one device.
#[derive(Debug)]
pub struct Model {
    artifact: Artifact,
    state: BTreeMap<String, Tensor>,
    device: Device,
}
impl Model {
    /// Builds a CPU model from an explicit program and named `(role, tensor)` state.
    /// Tensors are copied to owned contiguous CPU storage; repeated identical
    /// tensors preserve identity. Use [`Model::to_device`] before accelerator inference.
    pub fn new(
        program: Program,
        state: impl IntoIterator<Item = (String, StateRole, Tensor)>,
    ) -> Result<Self> {
        let mut artifact = Artifact {
            program,
            storages: Vec::new(),
            state: Vec::new(),
        };
        let mut aliases = BTreeMap::new();
        for (name, role, tensor) in state {
            let dtype = DType::from_kind(tensor.kind())?;
            let shape = tensor.size();
            let requires_grad = tensor.requires_grad();
            let identity = (tensor.data_ptr() as usize, shape.clone(), tensor.stride());
            let storage = if tensor.numel() > 0 {
                aliases.get(&identity).copied()
            } else {
                None
            };
            let storage = match storage {
                Some(i) => i,
                None => {
                    let tensor = tensor.f_to_device(Device::Cpu)?.f_contiguous()?;
                    let mut bytes = vec![0; byte_count(&shape, dtype)?];
                    tensor.f_copy_data_u8(&mut bytes, tensor.numel())?;
                    let i = artifact.storages.len();
                    artifact.storages.push(Storage { dtype, bytes });
                    aliases.insert(identity, i);
                    i
                }
            };
            artifact.state.push(State {
                name,
                role,
                storage,
                strides: contiguous_strides(&shape)?,
                shape,
                offset: 0,
                requires_grad,
            });
        }
        Self::from_artifact(artifact, Device::Cpu)
    }
    fn from_artifact(artifact: Artifact, device: Device) -> Result<Self> {
        if cfg!(target_endian = "big") {
            return Err(error(
                "portable artifacts currently require a little-endian host",
            ));
        }
        let mut state_names = BTreeSet::new();
        let mut total = 0_usize;
        if artifact.state.len() > MAX_NODES || artifact.storages.len() > MAX_NODES {
            return Err(error("state count exceeds limit"));
        }
        for storage in &artifact.storages {
            total = total
                .checked_add(storage.bytes.len())
                .ok_or_else(|| error("storage size overflow"))?;
            if total > MAX_BYTES
                || storage.bytes.len() % storage.dtype.bytes() != 0
                || (storage.dtype == DType::Bool && storage.bytes.iter().any(|x| *x > 1))
            {
                return Err(error("invalid or excessive storage size"));
            }
        }
        for state in &artifact.state {
            valid_name(&state.name)?;
            if !state_names.insert(state.name.clone()) {
                return Err(error("duplicate state name"));
            }
            let storage = artifact
                .storages
                .get(state.storage)
                .ok_or_else(|| error("state references missing storage"))?;
            validate_view(state, storage)?;
        }
        artifact.program.validate(&state_names, 0, &mut 0)?;
        let mut bases = Vec::new();
        for storage in &artifact.storages {
            // f_from_data_size does not validate the input slice length itself.
            let count = storage.bytes.len() / storage.dtype.bytes();
            let base =
                Tensor::f_from_data_size(&storage.bytes, &[count as i64], storage.dtype.kind())?
                    .f_to_device(device)?;
            bases.push(base);
        }
        let mut state = BTreeMap::new();
        let mut aliases = BTreeMap::<_, Tensor>::new();
        for descriptor in &artifact.state {
            let key = (
                descriptor.storage,
                descriptor.shape.clone(),
                descriptor.strides.clone(),
                descriptor.offset,
                descriptor.requires_grad,
            );
            let value = if let Some(value) = aliases.get(&key) {
                value.shallow_clone()
            } else {
                let value = bases[descriptor.storage]
                    .f_as_strided(
                        &descriptor.shape,
                        &descriptor.strides,
                        Some(descriptor.offset),
                    )?
                    .f_detach()?
                    .f_set_requires_grad(descriptor.requires_grad)?;
                aliases.insert(key, value.shallow_clone());
                value
            };
            state.insert(descriptor.name.clone(), value);
        }
        Ok(Self {
            artifact,
            state,
            device,
        })
    }
    /// Returns the versioned program and calling-tree metadata.
    pub fn program(&self) -> &Program {
        &self.artifact.program
    }
    /// Lowers a supported existing [`crate::graph::GraphModule`] into portable
    /// operators and owned CPU state. Explicit tensor dimensions and dtype are
    /// required. Evaluation dropout lowers to identity; training mode and
    /// operations outside the portable subset are rejected.
    pub fn from_graph(model: &crate::graph::GraphModule) -> Result<Self> {
        use crate::graph::{Dim, GraphOp};
        if model.is_training() {
            return Err(error(
                "set graph to evaluation mode before deployment lowering",
            ));
        }
        let graph = model.graph();
        let mut inputs = Vec::new();
        let mut operations = Vec::new();
        for (name, value) in graph.inputs() {
            let spec = &graph.nodes()[value.0].spec;
            let dimensions = spec
                .dims()
                .ok_or_else(|| error("graph deployment requires explicit input dimensions"))?
                .iter()
                .map(|d| match d {
                    Dim::Known(n) => Ok(Dimension::Known(*n)),
                    Dim::Symbol(name) => Ok(Dimension::Symbol {
                        name: name.clone(),
                        min: 0,
                        max: None,
                    }),
                    _ => Err(error("unnamed dynamic graph dimensions are unsupported")),
                })
                .collect::<Result<Vec<_>>>()?;
            inputs.push(ValueSpec::new(
                name,
                DType::from_kind(
                    spec.kind_value()
                        .ok_or_else(|| error("graph deployment requires explicit input dtype"))?,
                )?,
                dimensions,
            ));
        }
        for id in graph.topological_order()? {
            let node = &graph.nodes()[id.0];
            if !node.active || matches!(node.op, GraphOp::Input) {
                continue;
            }
            let mut operands = node
                .inputs
                .iter()
                .map(|id| graph.nodes()[id.0].name.clone())
                .collect::<Vec<_>>();
            let operator = match &node.op {
                GraphOp::Identity | GraphOp::Output | GraphOp::Dropout { .. } => Operator::Identity,
                GraphOp::ReLU => Operator::Relu,
                GraphOp::Add => Operator::Add,
                GraphOp::Subtract => Operator::Subtract,
                GraphOp::Multiply => Operator::Multiply,
                GraphOp::Flatten { start_dim, end_dim } => Operator::Flatten {
                    start: *start_dim,
                    end: *end_dim,
                },
                GraphOp::Concatenate { dim } => Operator::Cat(*dim),
                GraphOp::Linear { bias, .. } => {
                    operands.push(format!("{}.weight", node.name));
                    if *bias {
                        operands.push(format!("{}.bias", node.name));
                    }
                    Operator::Linear
                }
                _ => return Err(error(format!("unsupported graph operation {:?}", node.op))),
            };
            operations.push(Operation::new(&node.name, operator, operands));
        }
        let outputs = Tree::Dict(
            graph
                .outputs()
                .iter()
                .map(|(name, id)| (name.clone(), Tree::Tensor(graph.nodes()[id.0].name.clone())))
                .collect(),
        );
        let state = model
            .var_store()
            .variables()
            .into_iter()
            .map(|(name, tensor)| {
                let role = if tensor.requires_grad() {
                    StateRole::Parameter
                } else {
                    StateRole::Buffer
                };
                (name, role, tensor)
            });
        Self::new(Program::new(inputs, operations, outputs), state)
    }
    /// Returns named parameters, buffers and constants without copying tensors.
    pub fn state(&self) -> &BTreeMap<String, Tensor> {
        &self.state
    }
    /// Returns the execution device required for all inputs.
    pub fn device(&self) -> Device {
        self.device
    }
    /// Moves state to a concrete device while preserving shared storage groups.
    pub fn to_device(&mut self, device: Device) -> Result<()> {
        let artifact = self.snapshot()?;
        *self = Self::from_artifact(artifact, device)?;
        Ok(())
    }
    /// Runs the program, validating all input guards before tensor operations.
    /// Leaves follow [`Program::input_tree`] and [`Program::output_tree`].
    pub fn run(&self, inputs: &[Tensor]) -> Result<Vec<Tensor>> {
        execute(&self.artifact.program, inputs, &self.state, self.device)
    }
    /// Serializes versioned graph, state roles, storage aliases and raw bytes as JSON.
    pub fn to_json(&self) -> Result<String> {
        let json = serde_json::to_string(&self.snapshot()?).map_err(mapped)?;
        if json.len() > MAX_BYTES {
            return Err(error("JSON artifact exceeds limit"));
        }
        Ok(json)
    }
    /// Loads the bounded version-one JSON artifact. Unknown fields are rejected.
    pub fn from_json(json: &str) -> Result<Self> {
        if json.len() > MAX_BYTES {
            return Err(error("JSON artifact exceeds limit"));
        }
        Self::from_artifact(serde_json::from_str(json).map_err(mapped)?, Device::Cpu)
    }
    /// Imports a pinned PT2 ExportedProgram subset without Python or pickle.
    pub fn load_pt2(path: impl AsRef<Path>) -> Result<Self> {
        pt2::load(path.as_ref())
    }
    /// Exports the supported functional subset to a PyTorch-readable PT2 archive.
    pub fn save_pt2(&self, path: impl AsRef<Path>) -> Result<()> {
        pt2::save(self, path.as_ref())
    }
    /// Imports the supported ONNX IR10/opset18 inference subset.
    pub fn load_onnx(path: impl AsRef<Path>) -> Result<Self> {
        onnx::load(path.as_ref())
    }
    /// Exports the common ONNX inference subset, rejecting unsupported semantics.
    pub fn save_onnx(&self, path: impl AsRef<Path>) -> Result<()> {
        onnx::save(self, path.as_ref())
    }
    /// Traces a branch-free, single-output program into a native TorchScript module with frozen
    /// state. Input guards are retained by the returned [`TracedModel`]. This
    /// rejects conditional graphs, does not trace arbitrary Rust functions and
    /// makes no optimizer or performance claim.
    pub fn trace(&self, examples: &[Tensor]) -> Result<TracedModel> {
        if self.program().output_tree.leaves()?.len() != 1 {
            return Err(error("native tracing supports exactly one tensor output"));
        }
        if self
            .artifact
            .program
            .operations
            .iter()
            .any(|n| matches!(n.operator, Operator::If { .. }))
        {
            return Err(error("conditional-to-TorchScript tracing is unsupported"));
        }
        self.run(examples)?;
        let state = self
            .state
            .iter()
            .map(|(n, t)| {
                let mut copy = t.f_zeros_like()?;
                copy.f_copy_(&t.f_detach()?)?;
                Ok((n.clone(), copy))
            })
            .collect::<Result<BTreeMap<_, _>>>()?;
        let mut failure = None;
        let mut closure = |inputs: &[Tensor]| match execute(
            &self.artifact.program,
            inputs,
            &state,
            self.device,
        ) {
            Ok(v) => v,
            Err(e) => {
                failure = Some(e);
                vec![inputs[0].shallow_clone()]
            }
        };
        if examples.is_empty() {
            return Err(error("tracing requires at least one input"));
        }
        let mut module =
            CModule::create_by_tracing("RustTorchProgram", "forward", examples, &mut closure)?;
        if let Some(e) = failure {
            return Err(e);
        }
        module.f_set_eval()?;
        Ok(TracedModel {
            module,
            inputs: self.artifact.program.inputs.clone(),
            device: self.device,
        })
    }
    fn snapshot(&self) -> Result<Artifact> {
        let mut artifact = self.artifact.clone();
        for descriptor in &mut artifact.state {
            let tensor = &self.state[&descriptor.name];
            if tensor.size() != descriptor.shape
                || DType::from_kind(tensor.kind())? != artifact.storages[descriptor.storage].dtype
            {
                return Err(error(
                    "live state shape or dtype differs from its declaration",
                ));
            }
            descriptor.requires_grad = tensor.requires_grad();
        }
        // Each storage group is rebuilt from the currently live views. Uncovered
        // storage bytes retain their original value; overlapping views must agree.
        for (i, storage) in artifact.storages.iter_mut().enumerate() {
            let count = storage.bytes.len() / storage.dtype.bytes();
            let base =
                Tensor::f_from_data_size(&storage.bytes, &[count as i64], storage.dtype.kind())?;
            tch::no_grad(|| -> Result<()> {
                for state in artifact.state.iter().filter(|s| s.storage == i) {
                    let mut view =
                        base.f_as_strided(&state.shape, &state.strides, Some(state.offset))?;
                    view.f_copy_(&self.state[&state.name].f_to_device(Device::Cpu)?)?;
                }
                Ok(())
            })?;
            base.f_copy_data_u8(&mut storage.bytes, count)?;
        }
        Ok(artifact)
    }
}

/// A real TorchScript module plus RustTorch input guards.
#[derive(Debug)]
pub struct TracedModel {
    module: CModule,
    inputs: Vec<ValueSpec>,
    device: Device,
}
impl TracedModel {
    /// Executes the saved native graph, checking the retained guards first.
    pub fn run(&self, inputs: &[Tensor]) -> Result<Vec<Tensor>> {
        check_inputs(&self.inputs, inputs, self.device)?;
        let values = inputs
            .iter()
            .map(|x| IValue::Tensor(x.shallow_clone()))
            .collect::<Vec<_>>();
        let output = self.module.forward_is(&values)?;
        fn tensors(value: IValue) -> Result<Vec<Tensor>> {
            match value {
                IValue::Tensor(t) => Ok(vec![t]),
                IValue::Tuple(v) | IValue::GenericList(v) => v
                    .into_iter()
                    .map(tensors)
                    .collect::<Result<Vec<_>>>()
                    .map(|v| v.into_iter().flatten().collect()),
                IValue::TensorList(v) => Ok(v),
                _ => Err(error("TorchScript result is not tensor-only")),
            }
        }
        tensors(output)
    }
    /// Saves a TorchScript file and adjacent `.guards.json` input contract.
    /// Both files are required by [`TracedModel::load`].
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        self.module.save(path)?;
        let metadata = serde_json::to_vec(&(1_u32, &self.inputs)).map_err(mapped)?;
        std::fs::write(guard_path(path), metadata).map_err(mapped)
    }
    /// Loads a trusted TorchScript artifact and its bounded input guards.
    /// Native artifacts contain executable code and must come from a trusted producer.
    pub fn load(path: impl AsRef<Path>, device: Device) -> Result<Self> {
        let path = path.as_ref();
        let bytes = read_bounded(&guard_path(path))?;
        let (version, inputs): (u32, Vec<ValueSpec>) =
            serde_json::from_slice(&bytes).map_err(mapped)?;
        if version != 1 {
            return Err(error("unsupported tracing guard version"));
        }
        let definition = Program::new(
            inputs.clone(),
            vec![],
            Tree::Tuple(
                inputs
                    .iter()
                    .map(|s| Tree::Tensor(s.name.clone()))
                    .collect(),
            ),
        );
        definition.validate(&BTreeSet::new(), 0, &mut 0)?;
        let mut module = CModule::load_on_device(path, device)?;
        module.f_set_eval()?;
        Ok(Self {
            module,
            inputs,
            device,
        })
    }
}

fn guard_path(path: &Path) -> std::path::PathBuf {
    let mut name = path.as_os_str().to_os_string();
    name.push(".guards.json");
    name.into()
}
fn read_bounded(path: &Path) -> Result<Vec<u8>> {
    let file = std::fs::File::open(path).map_err(mapped)?;
    let mut data = Vec::new();
    file.take(MAX_BYTES as u64 + 1)
        .read_to_end(&mut data)
        .map_err(mapped)?;
    if data.len() > MAX_BYTES {
        return Err(error("artifact exceeds 256 MiB limit"));
    }
    Ok(data)
}
fn valid_name(name: &str) -> Result<()> {
    if name.is_empty() || name.len() > 1024 || name.chars().any(char::is_control) {
        Err(error("invalid or excessive value name"))
    } else {
        Ok(())
    }
}
fn byte_count(shape: &[i64], dtype: DType) -> Result<usize> {
    if shape.len() > 64 {
        return Err(error("rank exceeds 64"));
    }
    let mut count = 1_usize;
    for &d in shape {
        count = count
            .checked_mul(usize::try_from(d).map_err(|_| error("negative dimension"))?)
            .ok_or_else(|| error("tensor size overflow"))?;
    }
    let bytes = count
        .checked_mul(dtype.bytes())
        .ok_or_else(|| error("tensor byte size overflow"))?;
    if bytes > MAX_BYTES {
        return Err(error("tensor exceeds size limit"));
    }
    Ok(bytes)
}
fn contiguous_strides(shape: &[i64]) -> Result<Vec<i64>> {
    let mut stride = 1_i64;
    let mut out = vec![0; shape.len()];
    for i in (0..shape.len()).rev() {
        out[i] = stride;
        stride = stride
            .checked_mul(shape[i].max(1))
            .ok_or_else(|| error("stride overflow"))?;
    }
    Ok(out)
}
fn validate_view(state: &State, storage: &Storage) -> Result<()> {
    byte_count(&state.shape, storage.dtype)?;
    if state.strides.len() != state.shape.len()
        || state.offset < 0
        || state.strides.iter().any(|s| *s < 0)
    {
        return Err(error("invalid tensor strides or offset"));
    }
    if state.requires_grad && !matches!(storage.dtype, DType::Float | DType::Double) {
        return Err(error("only floating state supports autograd"));
    }
    let mut axes = state
        .shape
        .iter()
        .zip(&state.strides)
        .filter(|(size, _)| **size > 1)
        .map(|(size, stride)| (*stride, *size))
        .collect::<Vec<_>>();
    axes.sort_unstable();
    let mut span = 1_i64;
    for (stride, size) in axes {
        if stride < span {
            return Err(error("internally overlapping state views are unsupported"));
        }
        span = span
            .checked_add(
                (size - 1)
                    .checked_mul(stride)
                    .ok_or_else(|| error("view stride overflow"))?,
            )
            .ok_or_else(|| error("view stride overflow"))?;
    }
    let count = storage.bytes.len() / storage.dtype.bytes();
    let mut end = state.offset as usize;
    if !state.shape.contains(&0) {
        for (&d, &s) in state.shape.iter().zip(&state.strides) {
            end = end
                .checked_add(
                    (d as usize - 1)
                        .checked_mul(s as usize)
                        .ok_or_else(|| error("view bound overflow"))?,
                )
                .ok_or_else(|| error("view bound overflow"))?;
        }
        end = end
            .checked_add(1)
            .ok_or_else(|| error("view bound overflow"))?;
    }
    if end > count {
        return Err(error("tensor view exceeds raw storage"));
    }
    Ok(())
}
fn check_inputs(specs: &[ValueSpec], inputs: &[Tensor], device: Device) -> Result<()> {
    if specs.len() != inputs.len() {
        return Err(error("input count mismatch"));
    }
    let mut symbols = BTreeMap::new();
    for (spec, input) in specs.iter().zip(inputs) {
        if input.device() != device
            || input.kind() != spec.dtype.kind()
            || input.size().len() != spec.dimensions.len()
        {
            return Err(error(format!(
                "{} dtype, rank or device guard failed",
                spec.name
            )));
        }
        for (dimension, actual) in spec.dimensions.iter().zip(input.size()) {
            match dimension {
                Dimension::Known(n) if *n != actual => {
                    return Err(error(format!("{} fixed shape guard failed", spec.name)));
                }
                Dimension::Symbol { name, min, max }
                    if actual < *min
                        || max.is_some_and(|m| actual > m)
                        || symbols
                            .insert(name, actual)
                            .is_some_and(|prior| prior != actual) =>
                {
                    return Err(error(format!(
                        "{} symbol {name} equality/range guard failed",
                        spec.name
                    )));
                }
                _ => (),
            }
        }
    }
    Ok(())
}
fn execute(
    program: &Program,
    inputs: &[Tensor],
    state: &BTreeMap<String, Tensor>,
    device: Device,
) -> Result<Vec<Tensor>> {
    let values = evaluate(program, inputs, state, device)?;
    Ok(program
        .output_tree
        .leaves()?
        .into_iter()
        .map(|n| values[n].shallow_clone())
        .collect())
}
fn evaluate(
    program: &Program,
    inputs: &[Tensor],
    state: &BTreeMap<String, Tensor>,
    device: Device,
) -> Result<BTreeMap<String, Tensor>> {
    check_inputs(&program.inputs, inputs, device)?;
    let mut values = state
        .iter()
        .map(|(n, t)| (n.clone(), t.shallow_clone()))
        .collect::<BTreeMap<_, _>>();
    for (spec, input) in program.inputs.iter().zip(inputs) {
        values.insert(spec.name.clone(), input.shallow_clone());
    }
    for node in &program.operations {
        let x = node.inputs.iter().map(|n| &values[n]).collect::<Vec<_>>();
        let value = match &node.operator {
            Operator::Identity => x[0].shallow_clone(),
            Operator::Relu => x[0].f_relu()?,
            Operator::Sigmoid => x[0].f_sigmoid()?,
            Operator::Tanh => x[0].f_tanh()?,
            Operator::Add => x[0].f_add(x[1])?,
            Operator::Subtract => x[0].f_sub(x[1])?,
            Operator::Multiply => x[0].f_mul(x[1])?,
            Operator::Matmul => x[0].f_matmul(x[1])?,
            Operator::Linear => x[0].f_linear(x[1], x.get(2).copied())?,
            Operator::Reshape(shape) => x[0].f_reshape(shape)?,
            Operator::Flatten { start, end } => x[0].f_flatten(*start, *end)?,
            Operator::Transpose { dim0, dim1 } => x[0].f_transpose(*dim0, *dim1)?,
            Operator::Permute(axes) => x[0].f_permute(axes)?,
            Operator::Cat(dim) => Tensor::f_cat(&x, *dim)?,
            Operator::If {
                then_branch,
                else_branch,
                result,
            } => {
                if x[0].kind() != Kind::Bool || x[0].dim() != 0 {
                    return Err(error("conditional predicate must be a scalar Boolean"));
                }
                let branch = if x[0].f_int64_value(&[])? != 0 {
                    then_branch
                } else {
                    else_branch
                };
                let mut args = x[1..].iter().map(|t| t.shallow_clone()).collect::<Vec<_>>();
                let output = execute(branch, &args, state, device)?.remove(0);
                let mut specs = branch.inputs.clone();
                specs.push(result.clone());
                args.push(output.shallow_clone());
                check_inputs(&specs, &args, device)?;
                output
            }
        };
        values.insert(node.name.clone(), value);
    }
    Ok(values)
}

// Symbolic lowering uses algebraic rules, never observations at a sample size.
fn infer_specs(model: &Model) -> Result<BTreeMap<String, ValueSpec>> {
    fn axis(d: i64, rank: usize) -> Result<usize> {
        let d = if d < 0 { rank as i64 + d } else { d };
        if d < 0 || d >= rank as i64 {
            Err(error("axis outside tensor rank"))
        } else {
            Ok(d as usize)
        }
    }
    fn same(a: &Dimension, b: &Dimension) -> bool {
        match (a, b) {
            (Dimension::Known(a), Dimension::Known(b)) => a == b,
            (Dimension::Symbol { name: a, .. }, Dimension::Symbol { name: b, .. }) => a == b,
            _ => false,
        }
    }
    fn broadcast(a: &[Dimension], b: &[Dimension]) -> Result<Vec<Dimension>> {
        let mut out = Vec::new();
        for i in 0..a.len().max(b.len()) {
            let x = a.len().checked_sub(i + 1).map(|n| &a[n]);
            let y = b.len().checked_sub(i + 1).map(|n| &b[n]);
            out.push(match (x, y) {
                (None, Some(y)) | (Some(y), None) => y.clone(),
                (Some(Dimension::Known(1)), Some(y)) | (Some(y), Some(Dimension::Known(1))) => {
                    y.clone()
                }
                (Some(x), Some(y)) if same(x, y) => x.clone(),
                _ => return Err(error("cannot prove symbolic broadcasting for export")),
            });
        }
        out.reverse();
        Ok(out)
    }
    fn product(dims: &[Dimension]) -> Result<Dimension> {
        if dims.contains(&Dimension::Known(0)) {
            return Ok(Dimension::Known(0));
        }
        let mut count = 1_i64;
        let mut symbol = None;
        for d in dims {
            match d {
                Dimension::Known(n) => {
                    count = count
                        .checked_mul(*n)
                        .ok_or_else(|| error("symbolic dimension product overflow"))?
                }
                Dimension::Symbol { .. } => {
                    if symbol.replace(d.clone()).is_some() {
                        return Err(error(
                            "products of symbolic dimensions are unsupported for export",
                        ));
                    }
                }
            }
        }
        match symbol {
            None => Ok(Dimension::Known(count)),
            Some(d) if count == 1 => Ok(d),
            _ => Err(error(
                "scaled symbolic dimensions are unsupported for export",
            )),
        }
    }
    let mut specs = model
        .program()
        .inputs
        .iter()
        .map(|s| (s.name.clone(), s.clone()))
        .collect::<BTreeMap<_, _>>();
    for (name, tensor) in &model.state {
        specs.insert(
            name.clone(),
            ValueSpec::new(
                name,
                DType::from_kind(tensor.kind())?,
                tensor.size().into_iter().map(Dimension::Known).collect(),
            ),
        );
    }
    for node in &model.program().operations {
        let inputs = node.inputs.iter().map(|n| &specs[n]).collect::<Vec<_>>();
        let first = inputs
            .first()
            .ok_or_else(|| error("operator has no inputs"))?;
        if inputs.iter().any(|s| s.dtype != first.dtype) {
            return Err(error("export currently requires matching operand dtypes"));
        }
        let dims = match &node.operator {
            Operator::Identity | Operator::Relu | Operator::Sigmoid | Operator::Tanh => {
                first.dimensions.clone()
            }
            Operator::Add | Operator::Subtract | Operator::Multiply => {
                broadcast(&first.dimensions, &inputs[1].dimensions)?
            }
            Operator::Linear => {
                let weight = &inputs[1].dimensions;
                if first.dimensions.is_empty()
                    || weight.len() != 2
                    || !same(first.dimensions.last().expect("rank"), &weight[1])
                {
                    return Err(error("linear dimensions cannot be proven during export"));
                }
                if inputs.len() == 3 && inputs[2].dimensions != weight[..1] {
                    return Err(error("linear bias dimensions differ from weight"));
                }
                let mut shape = first.dimensions.clone();
                *shape.last_mut().expect("rank") = weight[0].clone();
                shape
            }
            Operator::Matmul => {
                let a = &first.dimensions;
                let b = &inputs[1].dimensions;
                if a.is_empty()
                    || b.is_empty()
                    || !same(
                        a.last().expect("rank"),
                        &b[if b.len() > 1 { b.len() - 2 } else { 0 }],
                    )
                {
                    return Err(error(
                        "matmul inner dimensions cannot be proven during export",
                    ));
                }
                let mut shape = broadcast(
                    &a[..a.len().saturating_sub(2)],
                    &b[..b.len().saturating_sub(2)],
                )?;
                if a.len() > 1 {
                    shape.push(a[a.len() - 2].clone());
                }
                if b.len() > 1 {
                    shape.push(b[b.len() - 1].clone());
                }
                shape
            }
            Operator::Transpose { dim0, dim1 } => {
                let mut shape = first.dimensions.clone();
                let a = axis(*dim0, shape.len())?;
                let b = axis(*dim1, shape.len())?;
                shape.swap(a, b);
                shape
            }
            Operator::Permute(axes) => axes
                .iter()
                .map(|a| {
                    first
                        .dimensions
                        .get(*a as usize)
                        .cloned()
                        .ok_or_else(|| error("permutation rank mismatch"))
                })
                .collect::<Result<Vec<_>>>()?,
            Operator::Flatten { start, end } => {
                let start = axis(*start, first.dimensions.len())?;
                let end = axis(*end, first.dimensions.len())?;
                if start > end {
                    return Err(error("flatten axes reversed"));
                }
                let flat = product(&first.dimensions[start..=end])?;
                first.dimensions[..start]
                    .iter()
                    .cloned()
                    .chain([flat])
                    .chain(first.dimensions[end + 1..].iter().cloned())
                    .collect()
            }
            Operator::Reshape(shape) => {
                let mut result = shape
                    .iter()
                    .map(|n| Dimension::Known(*n))
                    .collect::<Vec<_>>();
                if let Some(index) = shape.iter().position(|n| *n == -1) {
                    let divisor = shape
                        .iter()
                        .filter(|n| **n != -1)
                        .try_fold(1_i64, |p, n| p.checked_mul(*n))
                        .ok_or_else(|| error("reshape product overflow"))?;
                    if divisor <= 0 {
                        return Err(error("ambiguous reshape inferred dimension"));
                    }
                    let mut known = 1_i64;
                    let mut symbolic = None;
                    for d in &first.dimensions {
                        match d {
                            Dimension::Known(n) => {
                                known = known
                                    .checked_mul(*n)
                                    .ok_or_else(|| error("reshape size overflow"))?
                            }
                            Dimension::Symbol { .. } => {
                                if symbolic.replace(d.clone()).is_some() {
                                    return Err(error("reshape has multiple symbolic dimensions"));
                                }
                            }
                        }
                    }
                    result[index] = match symbolic {
                        None if known % divisor == 0 => Dimension::Known(known / divisor),
                        Some(symbol) if known == divisor => symbol,
                        _ => return Err(error("cannot express inferred reshape dimension")),
                    };
                } else if product(&first.dimensions)? != product(&result)? {
                    return Err(error("reshape element counts cannot be proven"));
                }
                result
            }
            Operator::Cat(dim) => {
                let dim = axis(*dim, first.dimensions.len())?;
                let mut out = first.dimensions.clone();
                let mut total = Dimension::Known(0);
                for input in &inputs {
                    if input.dimensions.len() != out.len() {
                        return Err(error("cat rank mismatch"));
                    }
                    for (i, (a, b)) in out.iter().zip(&input.dimensions).enumerate() {
                        if i != dim && !same(a, b) {
                            return Err(error("cat non-axis dimensions cannot be proven"));
                        }
                    }
                    total = match (&total, &input.dimensions[dim]) {
                        (Dimension::Known(a), Dimension::Known(b)) => Dimension::Known(
                            a.checked_add(*b)
                                .ok_or_else(|| error("cat size overflow"))?,
                        ),
                        (Dimension::Known(0), d) | (d, Dimension::Known(0)) => d.clone(),
                        _ => return Err(error("cat introduces an unsupported symbolic sum")),
                    };
                }
                out[dim] = total;
                out
            }
            Operator::If { .. } => {
                return Err(error(
                    "conditional export requires an explicit supported control-flow format",
                ));
            }
        };
        specs.insert(
            node.name.clone(),
            ValueSpec::new(&node.name, first.dtype, dims),
        );
    }
    Ok(specs)
}
