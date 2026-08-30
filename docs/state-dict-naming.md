# State-dict naming

State names are part of the interoperability contract.

## Rules

- A module's local parameters use `weight`, `bias`, or another PyTorch name.
- Nested modules join path components with a dot: `classifier.weight`.
- Sequential children use stable zero-based indices:
  `encoder.0.weight`, `encoder.0.bias`.
- Graph parameter-owning nodes use the stable, unique node name followed by the
  local name: `hidden.weight`, `hidden.bias`.
- Duplicate module, node, parameter, or mapped destination names are errors.

Names must remain stable across save/load, device movement, and train/eval
changes. Renaming a public module path is a state-format compatibility change.

## Mapping

When equivalent models use different paths, callers provide exact mappings,
for example `fc1.weight -> hidden.weight`. Safe prefix mapping is allowed only
when deterministic and collision-free. A dry run reports loaded, missing,
unexpected, and remapped keys before mutation.

Mappings never guess by edit distance and never imply transpose, reshape, or
other tensor transformation. Such transformations require a separate explicit
conversion tool.

Strict loading fails on missing or unexpected required entries. Non-strict
loading returns the full report without treating those two categories as
fatal; shape, unsafe dtype, and duplicate-destination errors remain failures.

## 0.1 state boundary

The state helpers save and load the named tensors returned by
`tch::nn::VarStore::variables`. RustTorch's 0.1 modules register parameters but
do not expose a dedicated persistent-buffer registration API. Persistent-buffer
round trips and tied-parameter alias preservation are therefore not part of the
0.1 compatibility claim.
