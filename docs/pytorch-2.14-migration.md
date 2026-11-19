# PyTorch 2.14 migration decision

**Keep RustTorch on LibTorch 2.13.0 and `tch` 0.26.0.** The 2.14 candidate trained a model and restored its next update on macOS and Linux arm64. The existing PT2 importer rejects a 2.14 export, Windows execution is outstanding, and the matching Rust binding is an experimental Git revision. These are release blockers. The qualification branch contains the candidate dependency locks and three-platform smoke test. [Qualification commit](https://github.com/newpoluton-alt/RustTorch/commit/18f0edba544b8839a38e0eed46feca32b5644e52)

## Source and binary inputs

| Input | Exact revision or checksum |
|---|---|
| Shipping PyTorch 2.13.0 | `cf30153c4c131c8164ee7798e5022d810682e2cb` |
| Proposed PyTorch 2.14.0 source tag | `2b3ec34829036a65cd9d1398ea72a0167dc37470` |
| Installed macOS 2.14.0 wheel source | `08187d9e0fba026dc8217405802ab5381dc88d90` |
| macOS arm64 CPython 3.14 wheel SHA-256 | `d0f78744000a7863cc3f1eafaf511d4b9944f079f08277de6fefe4b01f4d41b9` |
| macOS wheel `torch.__config__.show()` SHA-256 | `7087d129a9b4d0dad0f49ccb08b46952ceac534033de37a2791e4a7870f75bd2` |
| Documented and native inventory from the macOS wheel and its matching checkout, SHA-256 | `3278f4febdbe45d9088efad363fec4d28a39708e3bec60e0088f662a5399ef70` |

The release tag adds one `SDPAParams` import in `test/test_transformers.py` after the wheel revision. A Git tree comparison found no other file change. The native schema comparison below parses the exact release tag. The Sphinx and runtime observations use the wheel's own source revision. The candidate wheel has 3,103 exact ATen signatures in common with the tagged source, including all generated variants. [Upstream tree comparison](https://github.com/pytorch/pytorch/compare/08187d9e0fba026dc8217405802ab5381dc88d90...2b3ec34829036a65cd9d1398ea72a0167dc37470)

## API and binding changes

| Frozen inventory | 2.13 | 2.14 | Added | Removed | Changed source or runtime signature among retained IDs |
|---|---:|---:|---:|---:|---:|
| Documented Python identities | 9,361 | 9,363 | 73 | 71 | 30 |
| ATen schemas | 3,101 | 3,103 | 6 | 4 | 5 |
| Total | 12,462 | 12,466 | 79 | 75 | 35 |

Another comparison found **100 retained Python identities with changed documented signatures**. These can overlap the 30 changed Python source or runtime signatures. Missing signatures stay explicit in the inventory. All six added native schemas were registered in the macOS 2.14 wheel. All four removed schemas were registered in the 2.13 wheel. The recorded registration flag was unchanged for every common native schema.

- **Linear algebra.** The new native schemas are `linalg_polar`, `linalg_matrix_sqrth` and their `.out` variants. The other additions are `_philox_key_fold_in.Tensor` and `_scaled_grouped_mm_v2.out`. The removed schemas are `cholesky`, `cholesky.out`, `qr` and `qr.Q`. Their four shipping Rust methods, `f_cholesky`, `f_cholesky_out`, `f_qr` and `f_qr_q`, are absent from the candidate generated wrapper. Callers need an evaluated migration through the corresponding `linalg` operations, including their output and mode conventions.
- **Backward and attention kernels.** Both `_cdist_backward` and `_pdist_backward`, including their out variants, rename the `grad` argument to `grad_output`. These four have shipping binding traces. The fifth changed schema, `_cudnn_attention_forward`, adds optional keyword-only `seqused_k` and `block_table` tensors. Each changed overload needs a fresh trace and argument tests for the target.
- **Python frontends.** Added identities include accelerator RNG accessors, CUDA graph lifecycle hooks, compiler precompilation and autograd node creation hooks. Of the 71 removed documented identities, 61 have the `torch.distributed._dist2` prefix. Examples of changes among retained signatures include the `torch.compile` parameter change from `shapes_spec` to `dynamic_shapes`, and `include_python_functions=False` on profiler `key_averages`. A documented identity leaving the inventory requires a disposition specific to that target.
- **Non-object capabilities.** The two macOS wheels expose 46 and 47 distinct canonical dtype values respectively. The addition is `torch.bcomplex32`. The same eight layout values, four memory formats and native `DeviceType` enum values were present. Dtype representation, device availability and per-kernel execution each need their own evidence. The existing shipping capability manifest does not yet describe `bcomplex32`.

The four audited Rust frontend identities (linear, ReLU and two gradient-mode aliases) and the identity used by both index-select behavior cases retain their recorded signatures. Their shipping evidence stays attached to 2.13. For 2.14, mark each added ID for initial binding and behavior review, retire each removed ID from the candidate denominator with its replacement or removal history, and reopen changed contracts. The comparison command below prints the exact IDs and before/after signatures for that reconciliation. The shipping dispatcher map and performance, dtype and backend claims require a target-specific audit. [Shipping bindings](../compat/pytorch_bindings.toml), [shipping behavior cases](../compat/pytorch_behavior.toml), [upstream release changes and deprecations](https://github.com/pytorch/pytorch/releases/tag/v2.14.0)

## Numerical and artifact qualification

A scalar lower bound and a differentiable tensor bound exercise different backward rules at equality. Both macOS wheels were tested on CPU with the input and lower bound equal to zero. [Upstream boundary-gradient change](https://github.com/pytorch/pytorch/pull/191142)

| `clamp_min` case | Observed 2.13 gradient | Observed 2.14 gradient |
|---|---|---|
| Scalar bound, input gradient | `1` | `0` |
| Tensor bound, gradients of input and bound | `[1, 0]` | `[0.5, 0.5]` |

Both wheels could export and reload a one-input ReLU on CPU, producing `[0, 2]` from `[-1, 2]`. Both archives reported schema 8.20 and ATen opset 10. They identified their producer as 2.13.0 and 2.14.0 respectively. Running the RustTorch classifier interoperability test with the candidate binding and a 2.14-generated fixture failed at `Model::load_pt2` with `PT2 requires PyTorch2.13.0/schema8.20/ATen10`. This prevented the subsequent ONNX and TorchScript checks in that test. Qualify the supported classifier graph, state names, bounded shapes, gradients, both directions of interchange, and malformed archives before extending the importer. Preserve reading of shipping artifacts. The 2.14 Python runtime also issued a TorchScript tracing warning on CPython 3.14. [Rust PT2 reader and writer](../src/deployment/pt2.rs), [interchange fixture](../tests/python_reference/deployment.py)

| Candidate runtime on CPU | Training, native linkage, checkpoint save/load and restored next update |
|---|---|
| macOS 27 arm64, CPython 3.14.7, Rust 1.88.0, Apple Clang 21 | Passed |
| Linux 6.12.76 arm64 VM, glibc 2.36, GCC 12.2, CPython 3.14.7, Rust 1.88.0, 2.14.0+cpu | Passed |
| Windows x64 and hosted Linux x64 | Unverified. GitHub rejected the matrix before the jobs started because of account billing. |

The example tests backward, an optimizer update, serialization of RustTorch model and training state, and continuation from the saved epoch. The resumed update reproduced the weights, optimizer moments, schedule and scale. Inter-version checkpoint import, general Python pickle, compiled packages, accelerator execution and multi-node training remain separate obligations. [Training/checkpoint example](../examples/training_checkpoint.rs), [three-OS qualification run](https://github.com/newpoluton-alt/RustTorch/actions/runs/44619597057)

The published `torch-sys` 0.26.0 build targets PyTorch 2.13.0. The published 0.27.0 build targets 2.15.0. The candidate Git revision for upstream PR #1072 declares package version 0.27.0 but targets 2.14.0 and requests C++20. The experiment pins its full Git revision for both Rust crates and keeps the native version check enabled. Migrating the nine published RustTorch crates needs a publishable matching binding, successful Windows and release CI qualification, versioned evidence for changed behavior, and a reviewed archive migration. [0.26.0 build script](https://github.com/LaurentMazare/tch-rs/blob/4227b89b72059b0651ff83a38637693574e0a2d6/torch-sys/build.rs), [published 0.27.0 build script](https://github.com/LaurentMazare/tch-rs/blob/14a568ac3122eb4e387bce934ed91f82adde9edf/torch-sys/build.rs), [2.14 candidate build script](https://github.com/RikHeijdens/tch-rs/blob/b75d71aa421ab38983184bd71cd1708dac669fc8/torch-sys/build.rs), [upstream PR #1072](https://github.com/LaurentMazare/tch-rs/pull/1072)

## Reproduce the comparison

From the shipping RustTorch checkout on macOS or Linux, create temporary checkouts for the tagged source, the wheel's source and the qualification branch. The candidate lock supplies the 2.14 runtime and pinned Sphinx tooling. The collector needs temporary disk space for a copy of the upstream sources and the documentation build. All generated files below stay in the temporary directory.

```sh
work=$(mktemp -d)
git clone --depth 1 --branch v2.14.0 https://github.com/pytorch/pytorch.git "$work/release"
git -C "$work/release" fetch --depth 1 origin 08187d9e0fba026dc8217405802ab5381dc88d90
git -C "$work/release" worktree add --detach "$work/wheel-source" 08187d9e0fba026dc8217405802ab5381dc88d90
git clone --single-branch --branch test/qualify-libtorch-214 https://github.com/newpoluton-alt/RustTorch.git "$work/probe"
git -C "$work/probe" checkout --detach 18f0edba544b8839a38e0eed46feca32b5644e52
uv sync --locked --project "$work/probe" --python 3.14.7

"$work/probe/.venv/bin/python" scripts/compare-pytorch-schemas.py \
    --collect-wheel-census "$work/wheel-source" > "$work/wheel-inventory.json"
.venv/bin/python scripts/compare-pytorch-schemas.py --source "$work/release" \
    --candidate-inventory "$work/wheel-inventory.json" > "$work/migration-delta.json"
```

The final command compares every candidate native signature to the tagged source before reporting documented identities, signatures and runtime registrations. To compare just the canonical native schemas, invoke it with `--source` alone. The complete inventory includes a binary fingerprint, so the full-file checksum above applies to the recorded macOS build. Operating system differences can also affect resolved Python objects. See the [baseline census guide](api-census.md) for the underlying collection rules.
