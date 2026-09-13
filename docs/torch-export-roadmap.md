# PT2 interoperability

`deployment::Model::load_pt2` and `save_pt2` implement a bounded functional
ExportedProgram subset. The [deployment guide](deployment.md) provides Rust
examples, supported operators and practical format selection.

The reference is PyTorch 2.13.0 commit
`cf30153c4c131c8164ee7798e5022d810682e2cb`:

- `torch/csrc/export/pt2_archive_constants.h`: archive version 0 and member paths.
- `torch/_export/serde/schema.py`: schema 8.20, tensor metadata, state signatures
  and calling-tree version 1.
- `torch/export/pt2_archive/_package.py`: raw storage/configuration payloads.
- `torch/_export/serde/serialize.py`: serialized symbols and empty optional
  sample-input representation.

The importer checks format/version headers, byte order, model count, export
schema and ATen opset. It reads ordinary dense tensor storage and JSON metadata;
it never loads pickle. Tensor subclasses and custom objects are rejected and
sample-input bytes are ignored. Export emits an empty sample-input member, which
the reference deserializer interprets as absent.

Parameters, persistent buffers and tensor constants retain names and roles.
Non-persistent buffers import as constants. Shared storage
and checked strided views retain alias relationships; identical tied parameter
views share their gradient leaf. Tensor-only tuple/list/dictionary calling trees
retain their leaf order. Named symbolic dimensions retain range/equality guards.

Executable guard code, arbitrary symbolic expressions, mutation signatures,
custom operators and higher-order control flow are rejected with an error.
Export derives supported shapes algebraically and emits the same functional
subset. It does not claim arbitrary ExportedProgram, AOTInductor binary or Python
object portability.

`tests/deployment.rs::python_pt2_onnx_and_torchscript_numerical_round_trips`
compares imported values and verifies exported PT2 using the pinned loader,
multiple batch sizes and input gradients. Companion tests cover named state,
shared storage, malformed archives and guards.

Extensions remain for additional schemas/opsets, higher-order operators,
mutation, richer symbolic expressions, non-CPU archive metadata, subclasses and
compiled binary loading. Each needs separate numerical and backend evidence.
