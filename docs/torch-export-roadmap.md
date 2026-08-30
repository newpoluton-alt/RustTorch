# `torch.export` and PT2 roadmap

Native `.pt2` import/export is not part of the MVP. This roadmap targets the
PyTorch 2.13.0 reference and must be revalidated when its archive or export
schema changes.

PyTorch documents `.pt2` as a ZIP archive with format and byte-order headers,
archive version and serialization identity, model JSON under `models/`, and
optional data such as sample inputs or AOTInductor artifacts. An exported
program describes an ATen-oriented graph plus parameters, buffers, constants,
input/output trees, symbolic dimensions, range constraints, and opset/schema
metadata. Higher-order operators represent supported control flow.

Sources:

- [PyTorch `torch.export` documentation](https://docs.pytorch.org/docs/stable/export.html)
- [PT2 archive specification](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/export/pt2_archive.html)
- [Export API reference](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/export/api_reference.html)

## Import milestones

1. Read ZIP headers and reject unsupported archive/schema/opset versions.
2. Parse model metadata without executing Python or loading untrusted pickle.
3. Import parameters, buffers, constants, and input/output trees.
4. Map a small explicit ATen operator subset to RustTorch Graph IR.
5. Import symbolic dimensions and range constraints as graph specs/guards.
6. Reject unsupported control flow, aliasing, mutation, or operators with a
   precise report.
7. Compare forward values, gradients where applicable, state names, and
   CPU/CUDA/MPS behavior against PyTorch.

Export follows only after the importer and versioning rules are stable enough
to round-trip the same subset. A converter must never imply that arbitrary
`ExportedProgram` files, AOTInductor binaries, or Python-specific objects are
portable. PyTorch's own loader may use pickle for parts of this workflow, so
untrusted archives remain outside the RustTorch compatibility contract.
