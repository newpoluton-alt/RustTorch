# API census, tensor workflows and usable reference documentation

Issue: [#11](https://github.com/newpoluton-alt/RustTorch/issues/11).

This delivery targets the core contracts for roadmap workstreams 3 and 4. Together
with the previous delivery this would be 4 of 8 workstreams by scoped delivery count,
not a measured percentage of PyTorch APIs or effort. No workstream is marked
delivered until its contracts have executable evidence. The complete census is
an inventory and disposition of APIs, not a claim that all those APIs work.

## Census contracts

- [x] Validate the exact upstream commit, clean source checkout and runtime identity.
- [x] Resolve documented objects semantically through isolated Sphinx/MyST sources,
  without executing upstream presentation configuration or losing included objects.
- [x] Collect canonical ATen schemas through pinned torchgen, including generated
  overloads, runtime-presence metadata and internal/public classification.
- [x] Produce deterministic inventories with stable identities and source locations.
- [x] Assign every inventory identity exactly one compatible ledger disposition;
  incomplete and Python-specific functionality remain visible.
- [x] Validate committed inventory/mapping offline and reject drift/corruption.
- [x] Pin maintainer tooling, document provenance and reproduce full refresh twice.
- [x] Integrate contribution rules, generated coverage and required CI enforcement.

## Tensor and differentiation contracts

- [x] Index/select/gather/scatter/mask use cases with safe error propagation.
- [x] Views/aliasing, broadcasting, conversion and reduction contracts with examples.
- [x] Linear algebra workflows with numerical output and gradient comparisons.
- [x] FFT/complex and special-function workflows with reconstruction examples.
- [x] Sparse/quantized/nested layout boundaries with explicit supported operations.
- [x] Seeded sampling and probability distribution utilities with validation.
- [x] Higher-order gradients, vector products, Jacobians and custom-gradient
  composition using the available safe differentiation API.
- [ ] Runnable task guides, precise API mappings, Python parity, full workspace,
  MSRV, documentation, package and platform validation, signed commits and push.

## Documentation and Windows acceptance

- [x] Replace inventory-dominated docs.rs entry points with task navigation and
  complete examples for tensors, data-to-training, layers, optimizers and persistence.
- [x] Make DataLoader type/builder/facade documentation directly usable: tensor
  batches, custom datasets/collation, epochs, workers, streams and exact resume.
- [x] Compile GitHub guides as doctests and inspect generated Rustdoc navigation.
- [x] Preserve and validate the diagnosed Windows fix. The reported failure is
  an existing 1 ns deadline-test race fixed by `58d1e57`; latest Windows CI passed.
- [x] Prepare publication-ready documentation and distinguish source completion
  from public docs.rs availability, which currently serves the 0.2.0 release.

## Decisions and limits

Ruling: continue the existing clean contribution branch, with separate file
ownership for agents, because this is a continuation of the same requested work;
creating another checkout would separate the unreleased APIs from their guides.

Ruling: use native Tensor capabilities and safe reverse-mode differentiation
before adding wrappers or bridges. Numerical Jacobian-vector products can use
reverse-over-reverse differentiation; they do not claim native dual-level forward
AD, Python custom Function hooks, or arbitrary dtype/layout/backend coverage.
Each such remaining capability receives an explicit inventory disposition.

Ruling: published docs.rs remains old because PR #9 is open and no new crate
version has been published. A release requires the reviewed commit and complete
release gates described in docs/releasing.md. Source documentation alone cannot
replace an existing immutable registry version.

## Validation record

- CPU: 401 workspace tests; 199 documentation tests (including intentional
  compile-fail contracts); all nine pinned Python parity test targets.
- Rust 1.88.0 workspace/all-targets check and stable clippy with warnings denied.
- Native documentation and docs.rs configuration builds with warnings denied.
- Existing Rust MPS parity suite: eager/graph execution, gradients, optimizer
  updates, SafeTensors and device movement pass with native device access.
  CUDA is unavailable and recorded as skipped. New tensor/differentiation
  numerical coverage is CPU only.
- Independent review corrected categorical entropy at extreme logits,
  empty-shape index validation, effective dtype epsilon/quantization scale
  ranges, and added focused regressions. Data documentation examples were
  executed independently before integration. The final documentation review
  checked 1,758 local links across 14 pages with no broken destinations.
- All four 0.3.0 workspace archives pass local package compilation; final
  archive inspection will include the completed inventory.
- 114 Python policy tests pass, including 58 focused inventory/compatibility
  tests; offline inventory, generated coverage, lockfile and release metadata
  checks pass. Latest cross-platform CI, integration and public publication
  remain external gates tracked in PR #9 and the v0.3.0 release.

Ruling: the user explicitly authorized integration and push to `main`.
Future branches use `b/` for features, `f/` for fixes and descriptive prefixes
for other tasks. The existing contribution branch remains until integration;
no new branch with the `codex/` prefix is created.

The full inventory contains 12,462 identities: 9,361 documented Python
object/overload records and 3,101 canonical ATen schemas (2,261 public-name,
840 internal-name, including 517 generated variants). All 99 original ledger
IDs remain; the ledger now has 136 scoped rows. Positive mappings require
exact documented symbols or an explicitly evidenced schema identity. Two fresh
end-to-end generations produced identical bytes: SHA-256
`000eb929953388a15aa8892f7f531a4834f14e0096d7d40e87097e89c0b798cf`.

Census procedure rulings: the pinned root is `index.md`; stage only tracked
source files; use the literal upstream filename map and reviewed ExportDB
prerequisite; retain documented aliases and distinct overload signatures;
reject missing API-bearing pages; normalize unstable address-bearing signature
representations to explicit unavailable values. All 3,101 native schemas are
registered in this wheel, which is availability evidence only, not binding
or RustTorch implementation support.
