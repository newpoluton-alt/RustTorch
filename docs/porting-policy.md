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

The canonical ledger is `compat/pytorch_api.toml`. Its status is one of
`supported`, `partial`, `planned`, `not_supported`, or `python_only`; its
implementation is `libtorch`, `rusttorch`, `mixed`, or `none`. A status applies
only to the exact written scope and named test evidence. It is not a claim
about every option on the referenced symbol.

The separate census assigns every pinned documented object and canonical
ATen schema exactly one disposition. An available schema or generated binding
is only a candidate for implementation, not proof of compatibility. See the
[API census maintenance guide](api-census.md) for reproducible refresh and
offline checks. Update mappings when adding or narrowing a supported scope.

Implementation choices should remain simple: reexport an appropriate native
type, delegate to a safe existing kernel, compose necessary high-level behavior,
or define an explicit artifact boundary. Keep unsupported functionality absent
or return a clear error.

Before adding a wrapper, check `tch` and LibTorch. Do not wrap hundreds of
tensor methods, port kernels, reproduce autograd, or copy large comments and
docstrings. Substantially adapted logic needs a short source-path attribution
and an entry in `THIRD_PARTY_NOTICES.md`.

Unsupported features must return a clear error or remain absent. They must not
be accepted and silently ignored.
