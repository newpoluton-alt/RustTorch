# Changelog

All notable changes are recorded here. The 0.x series does not yet promise API
or graph-format stability.

## 0.1.0 - 2026-08-30

- Establish the `rusttorch` MVP package and `rusttorch` library crate.
- Target `tch` 0.26.0 and PyTorch/LibTorch 2.13.0.
- Define eager modules, device selection, SafeTensors interchange, and optional
  explicit Graph IR surfaces.
- Add project-local environment, backend inspection, and Python parity tools.
- Add architecture, compatibility, backend, interop, and roadmap documentation.
- Publish complete public API documentation and docs.rs build configuration.
- Verify strict bidirectional SafeTensors loading and deterministic CPU parity
  for forward, gradients, losses, Adam, SGD, and a residual model.
- Verify Rust MPS eager/graph execution, losses, gradients, optimizer steps,
  SafeTensors transfer, and CPU↔MPS movement on the development Mac.
