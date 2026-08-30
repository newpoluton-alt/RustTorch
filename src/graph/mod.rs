//! Optional, inspectable graph IR executed eagerly through LibTorch.

use std::{
    collections::{BTreeMap, BTreeSet, VecDeque},
    fmt::Write as _,
    path::Path,
};

use tch::{Device, Kind, Tensor, nn::VarStore};

use crate::{
    DeviceSpec, Result, RustTorchError,
    device::{ensure_device, resolve_device},
    interop::{
        LoadOptions, LoadReport, StateDictMapping, load_state_dict, load_state_dict_with_mapping,
        save_state_dict,
    },
    nn::{self, GeluApproximation},
};

/// Stable identifier for a node within one [`Graph`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NodeId(
    /// Zero-based node index.
    pub usize,
);

/// Stable identifier for a value produced within one [`Graph`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ValueId(
    /// Zero-based producing-node index.
    pub usize,
);

/// One dimension in a [`TensorSpec`].
///
/// New dimension forms may be added in future releases.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum Dim {
    /// A statically known non-negative size.
    Known(i64),
    /// A size that is checked only by LibTorch at execution time.
    Dynamic,
    /// A named size that must agree wherever the symbol appears at runtime.
    Symbol(String),
}

/// Optional static facts attached to a graph value.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct TensorSpec {
    dimensions: Option<Vec<Dim>>,
    rank: Option<usize>,
    kind: Option<Kind>,
    device: Option<DeviceSpec>,
}

impl TensorSpec {
    /// Creates a tensor spec with no static constraints.
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    /// Constrains the tensor rank without specifying individual dimensions.
    pub fn rank(mut self, rank: usize) -> Self {
        self.rank = Some(rank);
        self
    }

    #[must_use]
    /// Constrains every dimension and derives the rank from the supplied list.
    pub fn dimensions(mut self, dimensions: impl Into<Vec<Dim>>) -> Self {
        let dimensions = dimensions.into();
        self.rank = Some(dimensions.len());
        self.dimensions = Some(dimensions);
        self
    }

    #[must_use]
    /// Constrains every dimension to a known size.
    pub fn known_dimensions<const N: usize>(self, dimensions: [i64; N]) -> Self {
        self.dimensions(dimensions.into_iter().map(Dim::Known).collect::<Vec<_>>())
    }

    #[must_use]
    /// Constrains the tensor dtype.
    pub fn kind(mut self, kind: Kind) -> Self {
        self.kind = Some(kind);
        self
    }

    #[must_use]
    /// Constrains the tensor device after resolving the device request.
    pub fn device(mut self, device: DeviceSpec) -> Self {
        self.device = Some(device);
        self
    }

    /// Returns the declared or dimension-derived rank, if any.
    pub fn rank_value(&self) -> Option<usize> {
        self.dimensions
            .as_ref()
            .map_or(self.rank, |dims| Some(dims.len()))
    }

    /// Returns the declared dimensions, if any.
    pub fn dims(&self) -> Option<&[Dim]> {
        self.dimensions.as_deref()
    }

    /// Returns the declared dtype, if any.
    pub const fn kind_value(&self) -> Option<Kind> {
        self.kind
    }

    /// Returns the declared device request, if any.
    pub const fn device_value(&self) -> Option<DeviceSpec> {
        self.device
    }

    fn validate(&self, context: &str) -> Result<()> {
        if let (Some(rank), Some(dimensions)) = (self.rank, &self.dimensions)
            && rank != dimensions.len()
        {
            return Err(RustTorchError::GraphValidation(format!(
                "{context} declares rank {rank} but has {} dimensions",
                dimensions.len()
            )));
        }
        if let Some(dimensions) = &self.dimensions {
            for dimension in dimensions {
                match dimension {
                    Dim::Known(value) if *value < 0 => {
                        return Err(RustTorchError::GraphValidation(format!(
                            "{context} contains negative known dimension {value}"
                        )));
                    }
                    Dim::Symbol(symbol) if symbol.is_empty() => {
                        return Err(RustTorchError::GraphValidation(format!(
                            "{context} contains an empty symbolic dimension"
                        )));
                    }
                    Dim::Known(_) | Dim::Dynamic | Dim::Symbol(_) => {}
                }
            }
        }
        Ok(())
    }
}

/// Operation represented by a graph node.
///
/// New operations may be added in future releases, so downstream matches must
/// include a wildcard arm.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum GraphOp {
    /// Named runtime tensor input.
    Input,
    /// Named graph result.
    Output,
    /// Pass-through operation.
    Identity,
    /// Fully connected transformation.
    Linear {
        /// Size of the last input dimension.
        in_features: i64,
        /// Size of the last output dimension.
        out_features: i64,
        /// Whether the operation owns an additive bias.
        bias: bool,
    },
    /// Element-wise rectified linear unit.
    ReLU,
    /// Element-wise Gaussian error linear unit.
    Gelu {
        /// Approximation used by the GELU kernel.
        approximation: GeluApproximation,
    },
    /// Randomly masks elements during training.
    Dropout {
        /// Probability that an element is zeroed.
        probability: f64,
    },
    /// Flattens an inclusive dimension range.
    Flatten {
        /// First dimension to flatten.
        start_dim: i64,
        /// Last dimension to flatten.
        end_dim: i64,
    },
    /// Element-wise addition with LibTorch broadcasting.
    Add,
    /// Element-wise subtraction with LibTorch broadcasting.
    Subtract,
    /// Element-wise multiplication with LibTorch broadcasting.
    Multiply,
    /// Concatenates tensors along one dimension.
    Concatenate {
        /// Dimension along which tensors are concatenated.
        dim: i64,
    },
    /// Mean-reduced mean squared error.
    MseLoss,
    /// Mean-reduced cross-entropy loss.
    CrossEntropyLoss,
}

impl GraphOp {
    fn expected_arity(&self) -> Option<usize> {
        match self {
            Self::Input => Some(0),
            Self::Output
            | Self::Identity
            | Self::Linear { .. }
            | Self::ReLU
            | Self::Gelu { .. }
            | Self::Dropout { .. }
            | Self::Flatten { .. } => Some(1),
            Self::Add
            | Self::Subtract
            | Self::Multiply
            | Self::MseLoss
            | Self::CrossEntropyLoss => Some(2),
            Self::Concatenate { .. } => None,
        }
    }

    fn label(&self) -> &'static str {
        match self {
            Self::Input => "Input",
            Self::Output => "Output",
            Self::Identity => "Identity",
            Self::Linear { .. } => "Linear",
            Self::ReLU => "ReLU",
            Self::Gelu { .. } => "GELU",
            Self::Dropout { .. } => "Dropout",
            Self::Flatten { .. } => "Flatten",
            Self::Add => "Add",
            Self::Subtract => "Subtract",
            Self::Multiply => "Multiply",
            Self::Concatenate { .. } => "Concatenate",
            Self::MseLoss => "MSELoss",
            Self::CrossEntropyLoss => "CrossEntropyLoss",
        }
    }
}

/// One operation and its value edges in a [`Graph`].
///
/// Nodes are constructed by [`GraphBuilder`]; future releases may attach
/// additional metadata.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct Node {
    /// Stable node identifier.
    pub id: NodeId,
    /// Unique user-visible name and parameter path.
    pub name: String,
    /// Operation performed by this node.
    pub op: GraphOp,
    /// Values consumed by this node, in operation argument order.
    pub inputs: Vec<ValueId>,
    /// Value produced by this node.
    pub output: ValueId,
    /// Static facts known about the produced value.
    pub spec: TensorSpec,
    /// Whether graph execution and inspection include this node.
    pub active: bool,
}

/// Training state supplied to the graph executor.
///
/// New execution modes may be added in future releases.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ExecutionMode {
    /// Enables training-only behavior such as dropout.
    Training,
    /// Disables training-only behavior.
    Evaluation,
}

/// Validated, device-independent graph definition.
#[derive(Debug, Clone)]
pub struct Graph {
    nodes: Vec<Node>,
    inputs: BTreeMap<String, ValueId>,
    outputs: BTreeMap<String, ValueId>,
}

impl Graph {
    /// Returns all nodes, including nodes made inactive by graph passes.
    pub fn nodes(&self) -> &[Node] {
        &self.nodes
    }

    /// Returns named graph inputs and their values.
    pub fn inputs(&self) -> &BTreeMap<String, ValueId> {
        &self.inputs
    }

    /// Returns named graph outputs and their values.
    pub fn outputs(&self) -> &BTreeMap<String, ValueId> {
        &self.outputs
    }

    /// Validates names, edges, arity, tensor specs, acyclicity, and outputs.
    pub fn validate(&self) -> Result<()> {
        if self.outputs.is_empty() {
            return Err(RustTorchError::GraphValidation(
                "at least one graph output is required".to_owned(),
            ));
        }
        let mut names = BTreeSet::new();
        let mut values = BTreeSet::new();
        for node in &self.nodes {
            validate_name(&node.name)?;
            if !names.insert(node.name.clone()) {
                return Err(RustTorchError::DuplicateName(node.name.clone()));
            }
            if !values.insert(node.output) {
                return Err(RustTorchError::GraphValidation(format!(
                    "value {:?} has multiple producers",
                    node.output
                )));
            }
            node.spec.validate(&format!("node `{}`", node.name))?;
            if let Some(expected) = node.op.expected_arity()
                && node.inputs.len() != expected
            {
                return Err(RustTorchError::GraphValidation(format!(
                    "node `{}` ({}) expects {expected} input(s), got {}",
                    node.name,
                    node.op.label(),
                    node.inputs.len()
                )));
            }
            if matches!(node.op, GraphOp::Concatenate { .. }) && node.inputs.is_empty() {
                return Err(RustTorchError::GraphValidation(format!(
                    "concatenate node `{}` requires at least one input",
                    node.name
                )));
            }
            for input in &node.inputs {
                if input.0 >= self.nodes.len() {
                    return Err(RustTorchError::GraphValidation(format!(
                        "node `{}` references missing value {:?}",
                        node.name, input
                    )));
                }
            }
            if let GraphOp::Dropout { probability } = node.op {
                nn::functional::validate_dropout(probability)?;
            }
        }
        for (name, value) in &self.outputs {
            if value.0 >= self.nodes.len() {
                return Err(RustTorchError::GraphValidation(format!(
                    "output `{name}` references missing value {value:?}"
                )));
            }
        }
        let order = self.topological_order()?;
        let mut reachable = BTreeSet::new();
        for node_id in order {
            let node = &self.nodes[node_id.0];
            if !node.active {
                continue;
            }
            if matches!(node.op, GraphOp::Input)
                || node.inputs.iter().all(|input| reachable.contains(input))
            {
                reachable.insert(node.output);
            }
        }
        for (name, value) in &self.outputs {
            if !reachable.contains(value) {
                return Err(RustTorchError::GraphValidation(format!(
                    "output `{name}` is not reachable from graph inputs"
                )));
            }
        }
        Ok(())
    }

    /// Returns active nodes in deterministic dependency order.
    pub fn topological_order(&self) -> Result<Vec<NodeId>> {
        let active = self
            .nodes
            .iter()
            .filter(|node| node.active)
            .map(|node| node.id)
            .collect::<BTreeSet<_>>();
        let mut indegree = BTreeMap::new();
        let mut consumers: BTreeMap<NodeId, Vec<NodeId>> = BTreeMap::new();
        for node in self.nodes.iter().filter(|node| node.active) {
            let mut count = 0usize;
            for input in &node.inputs {
                let producer = NodeId(input.0);
                if active.contains(&producer) {
                    count += 1;
                    consumers.entry(producer).or_default().push(node.id);
                }
            }
            indegree.insert(node.id, count);
        }
        let mut ready = indegree
            .iter()
            .filter_map(|(id, degree)| (*degree == 0).then_some(*id))
            .collect::<BTreeSet<_>>();
        let mut order = Vec::with_capacity(active.len());
        while let Some(id) = ready.pop_first() {
            order.push(id);
            if let Some(nodes) = consumers.get(&id) {
                for consumer in nodes {
                    let degree = indegree.get_mut(consumer).ok_or_else(|| {
                        RustTorchError::GraphValidation("invalid consumer edge".to_owned())
                    })?;
                    *degree -= 1;
                    if *degree == 0 {
                        ready.insert(*consumer);
                    }
                }
            }
        }
        if order.len() == active.len() {
            Ok(order)
        } else {
            Err(RustTorchError::GraphValidation(
                "graph contains a cycle".to_owned(),
            ))
        }
    }

    /// Returns active nodes whose values can be produced from graph inputs.
    pub fn reachable_nodes(&self) -> Result<Vec<NodeId>> {
        let mut reachable_values = BTreeSet::new();
        let mut reachable_nodes = Vec::new();
        for node_id in self.topological_order()? {
            let node = &self.nodes[node_id.0];
            if matches!(node.op, GraphOp::Input)
                || node
                    .inputs
                    .iter()
                    .all(|input| reachable_values.contains(input))
            {
                reachable_values.insert(node.output);
                reachable_nodes.push(node_id);
            }
        }
        Ok(reachable_nodes)
    }

    /// Returns a deterministic human-readable table of active nodes.
    pub fn summary(&self) -> String {
        let mut summary = String::from("id  name  op  inputs  shape  kind  parameters\n");
        for node in self.nodes.iter().filter(|node| node.active) {
            let parameters = match node.op {
                GraphOp::Linear {
                    in_features,
                    out_features,
                    bias,
                } => in_features.saturating_mul(out_features) + if bias { out_features } else { 0 },
                _ => 0,
            };
            let _ = writeln!(
                summary,
                "{}  {}  {}  {:?}  {:?}  {:?}  {}",
                node.id.0,
                node.name,
                node.op.label(),
                node.inputs,
                node.spec.dims(),
                node.spec.kind_value(),
                parameters
            );
        }
        summary
    }

    /// Renders active nodes and edges as a Graphviz DOT graph.
    pub fn to_dot(&self) -> String {
        let mut dot = String::from("digraph rusttorch {\n");
        for node in self.nodes.iter().filter(|node| node.active) {
            let _ = writeln!(
                dot,
                "  n{} [label=\"{}\\n{}\"];",
                node.id.0,
                escape_dot(&node.name),
                node.op.label()
            );
        }
        for node in self.nodes.iter().filter(|node| node.active) {
            for input in &node.inputs {
                if self
                    .nodes
                    .get(input.0)
                    .is_some_and(|producer| producer.active)
                {
                    let _ = writeln!(dot, "  n{} -> n{};", input.0, node.id.0);
                }
            }
        }
        dot.push_str("}\n");
        dot
    }

    /// Allocates graph parameters on `device` and creates an executable module.
    pub fn build(self, device: DeviceSpec) -> Result<GraphModule> {
        GraphModule::from_graph(self, device)
    }
}

/// Incremental graph construction with stable node and value identifiers.
#[derive(Debug, Default)]
pub struct GraphBuilder {
    nodes: Vec<Node>,
    names: BTreeSet<String>,
    inputs: BTreeMap<String, ValueId>,
    outputs: BTreeMap<String, ValueId>,
}

impl GraphBuilder {
    /// Creates an empty graph builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a named runtime input with optional tensor constraints.
    pub fn input(&mut self, name: impl Into<String>, spec: TensorSpec) -> Result<ValueId> {
        let name = name.into();
        spec.validate(&format!("input `{name}`"))?;
        let value = self.push_node(name.clone(), GraphOp::Input, Vec::new(), spec)?;
        self.inputs.insert(name, value);
        Ok(value)
    }

    /// Adds a pass-through operation.
    pub fn identity(&mut self, name: impl Into<String>, input: ValueId) -> Result<ValueId> {
        self.unary(name, GraphOp::Identity, input)
    }

    /// Adds a biased linear transformation.
    pub fn linear(
        &mut self,
        name: impl Into<String>,
        input: ValueId,
        in_features: i64,
        out_features: i64,
    ) -> Result<ValueId> {
        self.linear_config(name, input, in_features, out_features, true)
    }

    /// Adds a linear transformation with explicit bias configuration.
    pub fn linear_config(
        &mut self,
        name: impl Into<String>,
        input: ValueId,
        in_features: i64,
        out_features: i64,
        bias: bool,
    ) -> Result<ValueId> {
        if in_features < 0 || out_features < 0 {
            return Err(RustTorchError::InvalidConfiguration {
                field: "graph linear dimensions",
                reason: format!(
                    "in_features and out_features must be non-negative, got {in_features} and {out_features}"
                ),
            });
        }
        let input_spec = self.spec(input)?.clone();
        if let Some(dimensions) = input_spec.dims()
            && let Some(Dim::Known(actual)) = dimensions.last()
            && *actual != in_features
        {
            return Err(RustTorchError::InvalidDimensions {
                context: "graph linear input".to_owned(),
                expected: format!("last dimension {in_features}"),
                actual: actual.to_string(),
            });
        }
        let mut output_spec = input_spec;
        if let Some(dimensions) = &mut output_spec.dimensions
            && let Some(last) = dimensions.last_mut()
        {
            *last = Dim::Known(out_features);
        }
        self.push_node(
            name.into(),
            GraphOp::Linear {
                in_features,
                out_features,
                bias,
            },
            vec![input],
            output_spec,
        )
    }

    /// Adds an element-wise ReLU operation.
    pub fn relu(&mut self, name: impl Into<String>, input: ValueId) -> Result<ValueId> {
        self.unary(name, GraphOp::ReLU, input)
    }

    /// Adds an exact element-wise GELU operation.
    pub fn gelu(&mut self, name: impl Into<String>, input: ValueId) -> Result<ValueId> {
        self.unary(
            name,
            GraphOp::Gelu {
                approximation: GeluApproximation::None,
            },
            input,
        )
    }

    /// Adds dropout with a probability in the inclusive range `[0, 1]`.
    pub fn dropout(
        &mut self,
        name: impl Into<String>,
        input: ValueId,
        probability: f64,
    ) -> Result<ValueId> {
        nn::functional::validate_dropout(probability)?;
        self.unary(name, GraphOp::Dropout { probability }, input)
    }

    /// Adds flattening over the inclusive dimension range.
    pub fn flatten(
        &mut self,
        name: impl Into<String>,
        input: ValueId,
        start_dim: i64,
        end_dim: i64,
    ) -> Result<ValueId> {
        let input_spec = self.spec(input)?.clone();
        let output_spec = infer_flatten_spec(&input_spec, start_dim, end_dim)?;
        self.push_node(
            name.into(),
            GraphOp::Flatten { start_dim, end_dim },
            vec![input],
            output_spec,
        )
    }

    /// Adds element-wise addition with broadcasting.
    pub fn add(
        &mut self,
        name: impl Into<String>,
        left: ValueId,
        right: ValueId,
    ) -> Result<ValueId> {
        self.binary(name, GraphOp::Add, left, right)
    }

    /// Adds element-wise subtraction with broadcasting.
    pub fn subtract(
        &mut self,
        name: impl Into<String>,
        left: ValueId,
        right: ValueId,
    ) -> Result<ValueId> {
        self.binary(name, GraphOp::Subtract, left, right)
    }

    /// Adds element-wise multiplication with broadcasting.
    pub fn multiply(
        &mut self,
        name: impl Into<String>,
        left: ValueId,
        right: ValueId,
    ) -> Result<ValueId> {
        self.binary(name, GraphOp::Multiply, left, right)
    }

    /// Adds concatenation of one or more values along `dim`.
    pub fn concatenate(
        &mut self,
        name: impl Into<String>,
        inputs: impl Into<Vec<ValueId>>,
        dim: i64,
    ) -> Result<ValueId> {
        let inputs = inputs.into();
        if inputs.is_empty() {
            return Err(RustTorchError::GraphValidation(
                "concatenate requires at least one input".to_owned(),
            ));
        }
        let specs = inputs
            .iter()
            .map(|input| self.spec(*input).cloned())
            .collect::<Result<Vec<_>>>()?;
        let output_spec = infer_concat_spec(&specs, dim)?;
        self.push_node(
            name.into(),
            GraphOp::Concatenate { dim },
            inputs,
            output_spec,
        )
    }

    /// Adds a scalar mean-squared-error loss.
    pub fn mse_loss(
        &mut self,
        name: impl Into<String>,
        input: ValueId,
        target: ValueId,
    ) -> Result<ValueId> {
        self.loss(name, GraphOp::MseLoss, input, target)
    }

    /// Adds a scalar cross-entropy loss whose target is an `Int64` tensor.
    pub fn cross_entropy_loss(
        &mut self,
        name: impl Into<String>,
        logits: ValueId,
        target: ValueId,
    ) -> Result<ValueId> {
        if let Some(kind) = self.spec(target)?.kind_value()
            && kind != Kind::Int64
        {
            return Err(RustTorchError::GraphValidation(format!(
                "cross-entropy target must be Int64, got {kind:?}"
            )));
        }
        self.loss(name, GraphOp::CrossEntropyLoss, logits, target)
    }

    /// Adds a named graph output and returns its output-node value.
    pub fn add_output(&mut self, name: impl Into<String>, value: ValueId) -> Result<ValueId> {
        let name = name.into();
        let output = self.unary(name.clone(), GraphOp::Output, value)?;
        self.outputs.insert(name, output);
        Ok(output)
    }

    /// Adds a named output using builder chaining.
    pub fn output(mut self, name: impl Into<String>, value: ValueId) -> Result<Self> {
        self.add_output(name, value)?;
        Ok(self)
    }

    /// Validates and finalizes the device-independent graph.
    ///
    /// Dead-node elimination and shape propagation run before the final validation.
    pub fn finish(self) -> Result<Graph> {
        let mut graph = Graph {
            nodes: self.nodes,
            inputs: self.inputs,
            outputs: self.outputs,
        };
        graph.validate()?;
        DeadNodeElimination.run(&mut graph)?;
        ShapePropagation.run(&mut graph)?;
        graph.validate()?;
        Ok(graph)
    }

    /// Finalizes the graph and allocates an executable module on `device`.
    pub fn build(self, device: DeviceSpec) -> Result<GraphModule> {
        self.finish()?.build(device)
    }

    fn unary(&mut self, name: impl Into<String>, op: GraphOp, input: ValueId) -> Result<ValueId> {
        let spec = self.spec(input)?.clone();
        self.push_node(name.into(), op, vec![input], spec)
    }

    fn binary(
        &mut self,
        name: impl Into<String>,
        op: GraphOp,
        left: ValueId,
        right: ValueId,
    ) -> Result<ValueId> {
        let spec = infer_binary_spec(self.spec(left)?, self.spec(right)?)?;
        self.push_node(name.into(), op, vec![left, right], spec)
    }

    fn loss(
        &mut self,
        name: impl Into<String>,
        op: GraphOp,
        input: ValueId,
        target: ValueId,
    ) -> Result<ValueId> {
        let left = self.spec(input)?;
        let right = self.spec(target)?;
        if let (Some(left), Some(right)) = (left.kind_value(), right.kind_value())
            && matches!(op, GraphOp::MseLoss)
            && left != right
        {
            return Err(RustTorchError::GraphValidation(format!(
                "MSE inputs have incompatible dtypes {left:?} and {right:?}"
            )));
        }
        let spec = TensorSpec::new()
            .known_dimensions([])
            .kind(left.kind_value().unwrap_or(Kind::Float));
        self.push_node(name.into(), op, vec![input, target], spec)
    }

    fn push_node(
        &mut self,
        name: String,
        op: GraphOp,
        inputs: Vec<ValueId>,
        spec: TensorSpec,
    ) -> Result<ValueId> {
        validate_name(&name)?;
        if !self.names.insert(name.clone()) {
            return Err(RustTorchError::DuplicateName(name));
        }
        for input in &inputs {
            self.spec(*input)?;
        }
        let id = NodeId(self.nodes.len());
        let output = ValueId(self.nodes.len());
        self.nodes.push(Node {
            id,
            name,
            op,
            inputs,
            output,
            spec,
            active: true,
        });
        Ok(output)
    }

    fn spec(&self, value: ValueId) -> Result<&TensorSpec> {
        self.nodes
            .get(value.0)
            .map(|node| &node.spec)
            .ok_or_else(|| {
                RustTorchError::GraphValidation(format!("value {value:?} does not exist"))
            })
    }
}

/// Named tensor inputs. Duplicate insertion is rejected.
#[derive(Debug, Default)]
pub struct GraphInputs {
    values: BTreeMap<String, Tensor>,
}

impl GraphInputs {
    /// Creates an empty named input collection.
    pub fn new() -> Self {
        Self::default()
    }

    /// Inserts an input using builder chaining.
    pub fn with(mut self, name: impl Into<String>, tensor: Tensor) -> Result<Self> {
        self.insert(name, tensor)?;
        Ok(self)
    }

    /// Inserts one named tensor, rejecting a duplicate name.
    pub fn insert(&mut self, name: impl Into<String>, tensor: Tensor) -> Result<()> {
        let name = name.into();
        if self.values.insert(name.clone(), tensor).is_some() {
            Err(RustTorchError::DuplicateName(name))
        } else {
            Ok(())
        }
    }
}

/// Named graph results.
#[derive(Debug, Default)]
pub struct GraphOutputs {
    values: BTreeMap<String, Tensor>,
}

impl GraphOutputs {
    /// Returns a named output or [`RustTorchError::MissingGraphOutput`].
    pub fn get(&self, name: &str) -> Result<&Tensor> {
        self.values
            .get(name)
            .ok_or_else(|| RustTorchError::MissingGraphOutput(name.to_owned()))
    }

    /// Iterates over outputs in deterministic name order.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &Tensor)> {
        self.values
            .iter()
            .map(|(name, tensor)| (name.as_str(), tensor))
    }
}

/// Eager graph executor. Tensor operations retain normal LibTorch autograd history.
#[derive(Debug, Default)]
pub struct EagerExecutor;

trait GraphExecutor {
    fn execute(
        &self,
        graph: &Graph,
        linears: &BTreeMap<NodeId, nn::Linear>,
        inputs: GraphInputs,
        mode: ExecutionMode,
        device: Device,
    ) -> Result<GraphOutputs>;
}

impl GraphExecutor for EagerExecutor {
    fn execute(
        &self,
        graph: &Graph,
        linears: &BTreeMap<NodeId, nn::Linear>,
        mut inputs: GraphInputs,
        mode: ExecutionMode,
        device: Device,
    ) -> Result<GraphOutputs> {
        graph.validate()?;
        for name in inputs.values.keys() {
            if !graph.inputs.contains_key(name) {
                return Err(RustTorchError::UnexpectedGraphInput(name.clone()));
            }
        }
        for name in graph.inputs.keys() {
            if !inputs.values.contains_key(name) {
                return Err(RustTorchError::MissingGraphInput(name.clone()));
            }
        }

        let training = matches!(mode, ExecutionMode::Training);
        let mut symbols = BTreeMap::new();
        let mut values = BTreeMap::new();
        for node_id in graph.topological_order()? {
            let node = &graph.nodes[node_id.0];
            let value = match &node.op {
                GraphOp::Input => {
                    let tensor = inputs
                        .values
                        .remove(&node.name)
                        .ok_or_else(|| RustTorchError::MissingGraphInput(node.name.clone()))?;
                    validate_runtime_spec(&node.name, &tensor, &node.spec, device, &mut symbols)?;
                    tensor
                }
                GraphOp::Output | GraphOp::Identity => {
                    input_value(&values, node, 0)?.shallow_clone()
                }
                GraphOp::Linear { .. } => linears
                    .get(&node.id)
                    .ok_or_else(|| {
                        RustTorchError::GraphValidation(format!(
                            "linear parameters for `{}` are missing",
                            node.name
                        ))
                    })?
                    .forward(input_value(&values, node, 0)?)?,
                GraphOp::ReLU => nn::functional::relu(input_value(&values, node, 0)?)?,
                GraphOp::Gelu { approximation } => nn::functional::gelu_with_approximation(
                    input_value(&values, node, 0)?,
                    *approximation,
                )?,
                GraphOp::Dropout { probability } => {
                    nn::functional::dropout(input_value(&values, node, 0)?, *probability, training)?
                }
                GraphOp::Flatten { start_dim, end_dim } => {
                    nn::functional::flatten(input_value(&values, node, 0)?, *start_dim, *end_dim)?
                }
                GraphOp::Add => {
                    input_value(&values, node, 0)?.f_add(input_value(&values, node, 1)?)?
                }
                GraphOp::Subtract => {
                    input_value(&values, node, 0)?.f_sub(input_value(&values, node, 1)?)?
                }
                GraphOp::Multiply => {
                    input_value(&values, node, 0)?.f_mul(input_value(&values, node, 1)?)?
                }
                GraphOp::Concatenate { dim } => {
                    let tensors = node
                        .inputs
                        .iter()
                        .map(|input| {
                            values.get(input).ok_or_else(|| {
                                RustTorchError::GraphValidation(format!(
                                    "node `{}` is missing input value {input:?}",
                                    node.name
                                ))
                            })
                        })
                        .collect::<Result<Vec<_>>>()?;
                    Tensor::f_cat(&tensors, *dim)?
                }
                GraphOp::MseLoss => nn::functional::mse_loss(
                    input_value(&values, node, 0)?,
                    input_value(&values, node, 1)?,
                )?,
                GraphOp::CrossEntropyLoss => nn::functional::cross_entropy(
                    input_value(&values, node, 0)?,
                    input_value(&values, node, 1)?,
                )?,
            };
            ensure_device(format!("graph node `{}` output", node.name), &value, device)?;
            values.insert(node.output, value);
        }

        let mut outputs = BTreeMap::new();
        for (name, value) in &graph.outputs {
            let tensor = values.get(value).ok_or_else(|| {
                RustTorchError::GraphValidation(format!(
                    "graph output `{name}` value {value:?} was not produced"
                ))
            })?;
            outputs.insert(name.clone(), tensor.shallow_clone());
        }
        Ok(GraphOutputs { values: outputs })
    }
}

/// An executable graph with one LibTorch parameter store.
#[derive(Debug)]
pub struct GraphModule {
    graph: Graph,
    var_store: VarStore,
    linears: BTreeMap<NodeId, nn::Linear>,
    executor: EagerExecutor,
    training: bool,
}

impl GraphModule {
    /// Validates `graph`, resolves `device`, and allocates graph parameters.
    pub fn from_graph(graph: Graph, device: DeviceSpec) -> Result<Self> {
        graph.validate()?;
        let device = resolve_device(device)?;
        let var_store = VarStore::new(device);
        let mut linears = BTreeMap::new();
        for node in graph.nodes.iter().filter(|node| node.active) {
            if let GraphOp::Linear {
                in_features,
                out_features,
                bias,
            } = node.op
            {
                let path = var_store.root() / node.name.as_str();
                let layer = nn::LinearConfig::new(in_features, out_features)
                    .bias(bias)
                    .build(&path)?;
                linears.insert(node.id, layer);
            }
        }
        Ok(Self {
            graph,
            var_store,
            linears,
            executor: EagerExecutor,
            training: true,
        })
    }

    /// Executes the graph using the module's current training state.
    pub fn forward(&self, inputs: GraphInputs) -> Result<GraphOutputs> {
        self.forward_t(inputs, self.training)
    }

    /// Executes the graph with an explicit training flag without changing module state.
    pub fn forward_t(&self, inputs: GraphInputs, training: bool) -> Result<GraphOutputs> {
        self.executor.execute(
            &self.graph,
            &self.linears,
            inputs,
            if training {
                ExecutionMode::Training
            } else {
                ExecutionMode::Evaluation
            },
            self.device(),
        )
    }

    /// Enables training behavior for subsequent [`GraphModule::forward`] calls.
    pub fn train(&mut self) {
        self.training = true;
    }

    /// Enables evaluation behavior for subsequent [`GraphModule::forward`] calls.
    pub fn eval(&mut self) {
        self.training = false;
    }

    /// Returns whether default forward calls use training behavior.
    pub const fn is_training(&self) -> bool {
        self.training
    }

    /// Returns the device-independent graph definition.
    pub fn graph(&self) -> &Graph {
        &self.graph
    }

    /// Returns a deterministic human-readable graph summary.
    pub fn summary(&self) -> String {
        self.graph.summary()
    }

    /// Renders the graph as Graphviz DOT.
    pub fn to_dot(&self) -> String {
        self.graph.to_dot()
    }

    /// Returns the device holding this module's parameters.
    pub fn device(&self) -> Device {
        self.var_store.device()
    }

    /// Returns the parameter store used by graph operations.
    pub const fn var_store(&self) -> &VarStore {
        &self.var_store
    }

    /// Moves all graph parameters to the resolved device.
    pub fn to_device(&mut self, device: DeviceSpec) -> Result<()> {
        self.var_store.set_device(resolve_device(device)?);
        Ok(())
    }

    /// Saves graph parameters as a device-neutral SafeTensors file.
    pub fn save_weights(&self, path: impl AsRef<Path>) -> Result<()> {
        save_state_dict(path, &self.var_store)
    }

    /// Strictly loads graph parameters from a SafeTensors file.
    pub fn load_weights(&self, path: impl AsRef<Path>) -> Result<LoadReport> {
        load_state_dict(path, &self.var_store)
    }

    /// Loads graph parameters with explicit key mapping and strictness options.
    pub fn load_weights_with_mapping(
        &self,
        path: impl AsRef<Path>,
        mapping: &StateDictMapping,
        options: LoadOptions,
    ) -> Result<LoadReport> {
        load_state_dict_with_mapping(path, &self.var_store, mapping, options)
    }
}

/// Result of applying a [`GraphPass`].
///
/// Future releases may add pass diagnostics.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct PassReport {
    /// Stable pass name.
    pub pass: &'static str,
    /// Nodes whose state or tensor spec changed.
    pub changed_nodes: Vec<NodeId>,
}

/// A deterministic transformation or validation over a graph.
pub trait GraphPass {
    /// Applies the pass and reports changed nodes.
    fn run(&self, graph: &mut Graph) -> Result<PassReport>;
}

/// Pass that checks graph structure without modifying it.
#[derive(Debug, Default)]
pub struct Validation;

impl GraphPass for Validation {
    fn run(&self, graph: &mut Graph) -> Result<PassReport> {
        graph.validate()?;
        Ok(PassReport {
            pass: "validation",
            changed_nodes: Vec::new(),
        })
    }
}

/// Pass that marks nodes not contributing to outputs as inactive.
#[derive(Debug, Default)]
pub struct DeadNodeElimination;

impl GraphPass for DeadNodeElimination {
    fn run(&self, graph: &mut Graph) -> Result<PassReport> {
        let mut live_values = graph.outputs.values().copied().collect::<BTreeSet<_>>();
        let mut queue = live_values.iter().copied().collect::<VecDeque<_>>();
        while let Some(value) = queue.pop_front() {
            if let Some(node) = graph.nodes.get(value.0) {
                for input in &node.inputs {
                    if live_values.insert(*input) {
                        queue.push_back(*input);
                    }
                }
            }
        }
        let mut changed_nodes = Vec::new();
        for node in &mut graph.nodes {
            let active = live_values.contains(&node.output);
            if node.active != active {
                node.active = active;
                changed_nodes.push(node.id);
            }
        }
        Ok(PassReport {
            pass: "dead-node-elimination",
            changed_nodes,
        })
    }
}

/// Pass that derives output tensor specs from operation inputs.
#[derive(Debug, Default)]
pub struct ShapePropagation;

impl GraphPass for ShapePropagation {
    fn run(&self, graph: &mut Graph) -> Result<PassReport> {
        let order = graph.topological_order()?;
        let mut changed_nodes = Vec::new();
        for node_id in order {
            let node = &graph.nodes[node_id.0];
            let input_specs = node
                .inputs
                .iter()
                .map(|input| graph.nodes[input.0].spec.clone())
                .collect::<Vec<_>>();
            let inferred = infer_node_spec(&node.op, &node.spec, &input_specs)?;
            if inferred != graph.nodes[node_id.0].spec {
                graph.nodes[node_id.0].spec = inferred;
                changed_nodes.push(node_id);
            }
        }
        Ok(PassReport {
            pass: "shape-propagation",
            changed_nodes,
        })
    }
}

fn input_value<'a>(
    values: &'a BTreeMap<ValueId, Tensor>,
    node: &Node,
    index: usize,
) -> Result<&'a Tensor> {
    let value = node.inputs.get(index).ok_or_else(|| {
        RustTorchError::GraphValidation(format!(
            "node `{}` is missing input index {index}",
            node.name
        ))
    })?;
    values.get(value).ok_or_else(|| {
        RustTorchError::GraphValidation(format!(
            "node `{}` is missing input value {value:?}",
            node.name
        ))
    })
}

fn validate_runtime_spec(
    name: &str,
    tensor: &Tensor,
    spec: &TensorSpec,
    model_device: Device,
    symbols: &mut BTreeMap<String, i64>,
) -> Result<()> {
    ensure_device(format!("graph input `{name}`"), tensor, model_device)?;
    if let Some(requested) = spec.device_value() {
        let expected = resolve_device(requested)?;
        ensure_device(
            format!("graph input `{name}` tensor spec"),
            tensor,
            expected,
        )?;
    }
    if let Some(kind) = spec.kind_value()
        && tensor.kind() != kind
    {
        return Err(RustTorchError::DtypeMismatch {
            name: name.to_owned(),
            expected: kind,
            actual: tensor.kind(),
        });
    }
    if let Some(rank) = spec.rank_value()
        && tensor.size().len() != rank
    {
        return Err(RustTorchError::InvalidDimensions {
            context: format!("graph input `{name}`"),
            expected: format!("rank {rank}"),
            actual: tensor.size().len().to_string(),
        });
    }
    if let Some(dimensions) = spec.dims() {
        for (index, (dimension, actual)) in dimensions.iter().zip(tensor.size()).enumerate() {
            match dimension {
                Dim::Known(expected) if *expected != actual => {
                    return Err(RustTorchError::InvalidDimensions {
                        context: format!("graph input `{name}` dimension {index}"),
                        expected: expected.to_string(),
                        actual: actual.to_string(),
                    });
                }
                Dim::Symbol(symbol) => match symbols.get(symbol) {
                    Some(expected) if *expected != actual => {
                        return Err(RustTorchError::InvalidDimensions {
                            context: format!(
                                "graph input `{name}` dimension {index} symbol `{symbol}`"
                            ),
                            expected: expected.to_string(),
                            actual: actual.to_string(),
                        });
                    }
                    Some(_) => {}
                    None => {
                        symbols.insert(symbol.clone(), actual);
                    }
                },
                Dim::Known(_) | Dim::Dynamic => {}
            }
        }
    }
    Ok(())
}

fn infer_node_spec(
    op: &GraphOp,
    current: &TensorSpec,
    inputs: &[TensorSpec],
) -> Result<TensorSpec> {
    match op {
        GraphOp::Input => Ok(current.clone()),
        GraphOp::Output
        | GraphOp::Identity
        | GraphOp::ReLU
        | GraphOp::Gelu { .. }
        | GraphOp::Dropout { .. } => Ok(inputs[0].clone()),
        GraphOp::Linear { out_features, .. } => {
            let mut spec = inputs[0].clone();
            if let Some(dimensions) = &mut spec.dimensions
                && let Some(last) = dimensions.last_mut()
            {
                *last = Dim::Known(*out_features);
            }
            Ok(spec)
        }
        GraphOp::Flatten { start_dim, end_dim } => {
            infer_flatten_spec(&inputs[0], *start_dim, *end_dim)
        }
        GraphOp::Add | GraphOp::Subtract | GraphOp::Multiply => {
            infer_binary_spec(&inputs[0], &inputs[1])
        }
        GraphOp::Concatenate { dim } => infer_concat_spec(inputs, *dim),
        GraphOp::MseLoss | GraphOp::CrossEntropyLoss => Ok(current.clone()),
    }
}

fn infer_binary_spec(left: &TensorSpec, right: &TensorSpec) -> Result<TensorSpec> {
    let kind = match (left.kind_value(), right.kind_value()) {
        (Some(left), Some(right)) if left != right => {
            return Err(RustTorchError::GraphValidation(format!(
                "binary inputs have incompatible dtypes {left:?} and {right:?}"
            )));
        }
        (Some(kind), _) | (_, Some(kind)) => Some(kind),
        (None, None) => None,
    };
    let dimensions = match (left.dims(), right.dims()) {
        (Some(left), Some(right)) => Some(broadcast_dimensions(left, right)?),
        _ => None,
    };
    let rank = dimensions.as_ref().map(Vec::len).or_else(|| {
        match (left.rank_value(), right.rank_value()) {
            (Some(left), Some(right)) => Some(left.max(right)),
            _ => None,
        }
    });
    let device = merge_device(left.device, right.device, "binary")?;
    Ok(TensorSpec {
        dimensions,
        rank,
        kind,
        device,
    })
}

fn broadcast_dimensions(left: &[Dim], right: &[Dim]) -> Result<Vec<Dim>> {
    let rank = left.len().max(right.len());
    let mut output = Vec::with_capacity(rank);
    for offset in 0..rank {
        let left = left
            .len()
            .checked_sub(1 + offset)
            .and_then(|index| left.get(index));
        let right = right
            .len()
            .checked_sub(1 + offset)
            .and_then(|index| right.get(index));
        let dimension = match (left, right) {
            (None, Some(value)) | (Some(value), None) => value.clone(),
            (Some(Dim::Known(1)), Some(value)) | (Some(value), Some(Dim::Known(1))) => {
                value.clone()
            }
            (Some(Dim::Known(left)), Some(Dim::Known(right))) if left == right => Dim::Known(*left),
            (Some(Dim::Known(left)), Some(Dim::Known(right))) => {
                return Err(RustTorchError::GraphValidation(format!(
                    "dimensions {left} and {right} are not broadcast-compatible"
                )));
            }
            (Some(Dim::Symbol(left)), Some(Dim::Symbol(right))) if left == right => {
                Dim::Symbol(left.clone())
            }
            (Some(_), Some(_)) => Dim::Dynamic,
            (None, None) => continue,
        };
        output.push(dimension);
    }
    output.reverse();
    Ok(output)
}

fn infer_concat_spec(inputs: &[TensorSpec], dim: i64) -> Result<TensorSpec> {
    let first = &inputs[0];
    let mut kind = first.kind;
    let mut device = first.device;
    for input in &inputs[1..] {
        if let (Some(left), Some(right)) = (kind, input.kind_value())
            && left != right
        {
            return Err(RustTorchError::GraphValidation(format!(
                "concatenate inputs have incompatible dtypes {left:?} and {right:?}"
            )));
        }
        if let (Some(left), Some(right)) = (first.rank_value(), input.rank_value())
            && left != right
        {
            return Err(RustTorchError::GraphValidation(format!(
                "concatenate inputs have ranks {left} and {right}"
            )));
        }
        kind = kind.or(input.kind);
        device = merge_device(device, input.device, "concatenate")?;
    }
    let Some(rank) = first.rank_value() else {
        return Ok(first.clone());
    };
    let dim = normalize_dim(dim, rank)?;
    let Some(first_dimensions) = first.dims() else {
        return Ok(TensorSpec {
            dimensions: None,
            rank: Some(rank),
            kind,
            device,
        });
    };
    let mut output = first_dimensions.to_vec();
    let mut concatenated = match output[dim] {
        Dim::Known(value) => Some(value),
        Dim::Dynamic | Dim::Symbol(_) => None,
    };
    for input in &inputs[1..] {
        let Some(dimensions) = input.dims() else {
            return Ok(TensorSpec {
                dimensions: None,
                rank: Some(rank),
                kind,
                device,
            });
        };
        for index in 0..rank {
            if index == dim {
                concatenated = match (concatenated, &dimensions[index]) {
                    (Some(left), Dim::Known(right)) => Some(left.saturating_add(*right)),
                    _ => None,
                };
            } else if let (Dim::Known(left), Dim::Known(right)) =
                (&output[index], &dimensions[index])
                && left != right
            {
                return Err(RustTorchError::GraphValidation(format!(
                    "concatenate dimension {index} differs: {left} vs {right}"
                )));
            }
        }
    }
    output[dim] = concatenated.map_or(Dim::Dynamic, Dim::Known);
    Ok(TensorSpec {
        dimensions: Some(output),
        rank: Some(rank),
        kind,
        device,
    })
}

fn merge_device(
    left: Option<DeviceSpec>,
    right: Option<DeviceSpec>,
    context: &str,
) -> Result<Option<DeviceSpec>> {
    match (left, right) {
        (Some(left), Some(right)) if left != right => Err(RustTorchError::GraphValidation(
            format!("{context} inputs declare incompatible devices {left:?} and {right:?}"),
        )),
        (Some(device), _) | (_, Some(device)) => Ok(Some(device)),
        (None, None) => Ok(None),
    }
}

fn infer_flatten_spec(input: &TensorSpec, start_dim: i64, end_dim: i64) -> Result<TensorSpec> {
    let Some(rank) = input.rank_value() else {
        return Ok(input.clone());
    };
    let start = normalize_dim(start_dim, rank)?;
    let end = normalize_dim(end_dim, rank)?;
    if start > end {
        return Err(RustTorchError::GraphValidation(format!(
            "flatten start_dim {start_dim} follows end_dim {end_dim}"
        )));
    }
    let output_rank = rank - (end - start);
    let dimensions = input.dims().map(|dimensions| {
        let flattened = dimensions[start..=end]
            .iter()
            .try_fold(1i64, |product, dimension| match dimension {
                Dim::Known(value) => product.checked_mul(*value),
                Dim::Dynamic | Dim::Symbol(_) => None,
            })
            .map_or(Dim::Dynamic, Dim::Known);
        dimensions[..start]
            .iter()
            .cloned()
            .chain(std::iter::once(flattened))
            .chain(dimensions[end + 1..].iter().cloned())
            .collect::<Vec<_>>()
    });
    Ok(TensorSpec {
        dimensions,
        rank: Some(output_rank),
        kind: input.kind,
        device: input.device,
    })
}

fn normalize_dim(dim: i64, rank: usize) -> Result<usize> {
    let rank = i64::try_from(rank).map_err(|_| {
        RustTorchError::GraphValidation("tensor rank does not fit in i64".to_owned())
    })?;
    let normalized = if dim < 0 { rank + dim } else { dim };
    if (0..rank).contains(&normalized) {
        Ok(normalized as usize)
    } else {
        Err(RustTorchError::GraphValidation(format!(
            "dimension {dim} is invalid for rank {rank}"
        )))
    }
}

fn validate_name(name: &str) -> Result<()> {
    if name.is_empty() {
        Err(RustTorchError::GraphValidation(
            "names must not be empty".to_owned(),
        ))
    } else if name.contains(['.', '/']) {
        Err(RustTorchError::GraphValidation(format!(
            "name `{name}` must not contain '.' or '/'"
        )))
    } else {
        Ok(())
    }
}

fn escape_dot(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
}
