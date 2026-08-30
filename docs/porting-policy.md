# Porting policy

The behavioral reference is PyTorch v2.13.0 at commit `cf30153`. Relevant
sources include `torch/nn/modules`, `torch/nn/functional.py`,
`torch/nn/init.py`, `torch/optim`, and their official tests. The binding
reference is `tch` 0.26.0.

Port behavior, not Python syntax. Preserve externally meaningful defaults,
validation, initialization, parameter structure and names, train/eval state,
serialization, and error conditions. Express Python keyword arguments as
small Rust configuration types or builders only when optional settings need
them.

Every public API is classified in `compat/pytorch_api.toml`:

- `reexport`: the `tch` type is already the right API.
- `delegate`: RustTorch validates or improves ergonomics, then calls `tch`.
- `composite_port`: meaningful high-level PyTorch logic is adapted while
  tensor calculations remain delegated.
- `interop`: state, metadata, graph, or checkpoint exchange.
- `unsupported`: Python runtime behavior or backend support prevents a safe,
  honest MVP implementation.

Before adding a wrapper, check `tch` and LibTorch. Do not wrap hundreds of
tensor methods, port kernels, reproduce autograd, or copy large comments and
docstrings. Substantially adapted logic needs a short source-path attribution
and an entry in `THIRD_PARTY_NOTICES.md`.

Unsupported features must return a clear error or remain absent. They must not
be accepted and silently ignored.
