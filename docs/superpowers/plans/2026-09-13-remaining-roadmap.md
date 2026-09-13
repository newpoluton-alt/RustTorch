# Distributed training, deployment, domain data and framework tools

Issue: [#18](https://github.com/newpoluton-alt/RustTorch/issues/18).
Base release: `v0.3.0` (`147685b`). Branch: `b/remaining-roadmap`.

The user requests the remaining four scoped roadmap workstreams and updated
GitHub/docs.rs documentation. Completion requires working APIs and executable
acceptance evidence. This count is not complete PyTorch API parity. Existing
extension backlogs remain explicit; none are silently promoted by this delivery.

## Global constraints

- Follow CONTRIBUTING.md, DCO, exact upstream provenance and the canonical ledger.
- Reuse std, RustTorch and safe tch capabilities before adding a dependency or
  native bridge. Native code must have an explicit safety and platform review.
- Every public item needs useful Rust documentation. Application guides explain
  inputs, expected shapes, outputs, failures and real use cases.
- Check tensor byte lengths and all dimensions before native construction.
  Bound untrusted archives, records, messages and decoded allocations.
- Keep unavailable backends and unsupported format/operator variants explicit.
- Do not edit unrelated `.serena` content, existing releases or Dependabot PRs.
- Root coordinates shared manifests, public exports, compatibility, CI and release.

## Distributed training acceptance

- [x] CPU multi-process TCP groups: explicit rank/world/session, finite I/O
  deadlines, bounded payloads, ordered collectives and terminal failure state.
- [x] Broadcast, reductions, gather/scatter, all-gather, reduce-scatter,
  all-to-all, barrier and point-to-point messages with collective validation.
- [x] Replicated parameter/buffer synchronization and explicit averaged-gradient
  updates with accumulation supported between synchronization boundaries.
- [x] Functional sharded training with actual rank-local parameters, gradients
  and optimizer moments; gather one parameter group for forward/backward, then
  reduce-scatter and update shards. Automatic module hooks and overlap are not
  claimed by this Rust API.
- [x] Versioned coordinated checkpoints, consolidation and world-size resharding.
- [x] Multiprocess numerical, resume, disconnect, mismatch and timeout tests;
  pinned CPU Gloo comparisons and practical training/checkpoint examples.

Decision: safe tch exposes no distributed groups or autograd hooks. Native
Gloo headers are incomplete in the installed wheel. A synchronous std TCP
backend and explicit functional sharding satisfy the CPU contracts without
inventing a native binding or claiming hook-driven PyTorch FSDP API parity.

## Compilation and deployment acceptance

- [x] Versioned graph/operator schema with validated IDs, named I/O and state roles.
- [x] Shape/range/dtype/device guards and explicit conditional subgraphs.
- [x] Pinned PT2 archive/schema/ATen/tree validation, bounded raw tensor storage,
  state/input-output trees and the documented operator subset, without pickle or eval.
- [x] PT2 import/export forward/state/gradient round trips against pinned torch.
- [x] Pinned ONNX IR/opset subset, initializers and symbolic batches, with official
  checker/reference execution and import/export numerical round trips.
- [x] Real TorchScript compilation/tracing and native artifact load/save/execute;
  reject unsupported trace/control-flow boundaries explicitly.

Decision: existing CModule provides native TorchScript execution. Portable
graph interpretation remains named as such. AOTInductor and arbitrary Rust
or Python bytecode capture are separate extensions, not aliases for this executor.

## Domain data acceptance

- [x] Shared finite ResourceLimits; domain samples implement loader memory/pinning
  and typed collation contracts without adding another loading engine.
- [x] Vision: bounded image decoding, typed geometry/targets, deterministic
  transforms, ImageFolder/MNIST/CIFAR/COCO local datasets and model-ready batches.
- [x] Codec: explicit FFmpeg system/vcpkg baseline, time/stream/capability model,
  CPU audio/video decode, seek/drain, bounded streaming and native boundary tests.
- [x] Audio: typed waveforms, native Rust decode, resampling, spectral/Mel/MFCC
  features, deterministic augmentations and variable-length batches.
- [x] Text: tokenizer adapter, explicit sequence policies, masks, token-budget
  batch source and distributed/checkpoint integration.
- [x] Tabular: CSV/JSONL, optional Arrow/Parquet, schema/missing-value policy,
  immutable fitted preprocessing and tensor conversion.
- [x] Every package has a real end-to-end pipeline, tiny offline fixtures,
  typed contextual failures, finite limits and direct/facade feature support.
- [x] Native CI uses a recorded dynamically linked LGPL FFmpeg configuration;
  user-selected local GPL builds never count as official artifact evidence.

## Framework tools acceptance

- [x] Tensor comparison diagnostics and bounded finite-difference gradient checks.
- [x] Bounded user-region profiling/trace export and benchmark warmup/statistics
  with explicit device synchronization and no invented kernel-level timing.
- [x] Reproducible runtime configuration and task-local stochastic tensor
  generation with serializable state, independent of LibTorch global RNG.
- [x] Atomic checkpoint publication and validated model/optimizer/application
  state restoration; examples include scheduler/scaler, exact loader position,
  local RNG and deployment export.
- [x] CPU numerical and practical use-case evidence, plus explicit conditional
  backend checks wherever backend behavior is advertised.

## Integration and release

- [x] Review per-task implementation and fix material findings.
- [x] Update ledger/inventory dispositions, notices, tutorials, README, roadmap
  and changelog from verified behavior.
- [x] Expand portable/native feature CI, package inspection and provenance for
  all nine synchronized packages without weakening existing gates.
- [x] Pass full formatting/checks/Clippy/tests/doctests, pinned numerical parity,
  MSRV, backend probes, dependency policy and clean package builds.
- [ ] DCO-signed commits, contribution PR, green required CI and integration to main.
- [ ] Publish the synchronized release, verify registry versions, immutable tag,
  complete provenance assets and actual docs.rs guide/example pages.

## Progress record

- Issue registered and clean feature branch created from released main.
- Independent distributed, deployment and domain investigations completed.
- Implementations assigned with disjoint file ownership; integration work ongoing.

### 0.4.0 integration evidence

- Distributed: 10 multiprocess tests and 11 module/guide doctests; native pinned
  Gloo/DDP/FSDP comparisons, seven optimizer families and 2→3-rank resharding.
- Deployment: 12 tests plus two shared ZIP-envelope regressions; actual pinned
  PT2/ONNX/TorchScript/conditional values, state and gradients. Framework tools
  have seven tests, including native comparison and gradcheck references.
- Checkpoints: six tests cover exact next loader/RNG/scheduler/scaler/optimizer
  update, malformed BOOL bytes, archive bounds, store identity and alias rules.
- Domains: 32 runtime tests across five packages, original fixtures, pinned
  spectral/input-gradient values, bounded sample/batch payloads and schema state.
- Independently reviewed native audio ownership/alignment/count invariants,
  malformed archive allocation bounds and release/provenance permissions.
- Built checksummed FFmpeg 8.1.2 from unmodified source with the recorded LGPL
  dynamic configuration; all four codec tests pass against that profile.
- Full optional features pass Rust 1.88, warnings-denied Clippy, runtime tests,
  runnable Rustdoc examples and docs.rs builds. The complete pinned parity gate
  passes. CPU and MPS probes work; CUDA is unavailable and remains a skip.
- Python lock now includes ONNX 1.22.0 checker tooling; complete artifact/graph
  policy and all 12,462 explicit inventory dispositions validate offline.
- All nine source archive inventories and full native archive verification
  pass (facade about 916 KiB compressed; domain archives about 23–28 KiB before
  adding original license texts). Final repository CI, registry publication
  and live docs checks remain gates below. Versioned docs are never claimed updated merely from a source commit.
- Windows and Linux CI pass the expanded runtime and DataLoader regression.
  Mac CI exposed native MPS bias omission on virtual M1 hardware; the shared
  affine correction passes local exact dtype/layout/gradient, attention,
  deployment and existing backend checks. Required CI must validate the fix.
- Final documentation review corrected two links that broke when guides were
  included in Rustdoc and updated the stale umbrella codec coverage entry.
