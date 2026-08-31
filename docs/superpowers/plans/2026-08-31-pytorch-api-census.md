# PyTorch API Census Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give every documented PyTorch 2.13 public symbol and every canonical ATen native schema an immutable inventory identity and exactly one explicit RustTorch compatibility disposition, while marking internal operator names as non-public implementation inventory, so work can proceed file by file without omissions or unsupported “100%” claims.

**Architecture:** A maintainer-only synchronizer builds the exact pinned PyTorch documentation sources with an isolated, locked semantic Sphinx configuration and a small collector extension, consumes Sphinx's resolved Python-domain objects/doctrees, reconciles those objects with the pinned runtime and ATen schemas, and writes a deterministic JSON inventory. The collector never imports PyTorch's presentation-oriented `conf.py`. A separate hand-maintained TOML map assigns each inventory item to one existing compatibility-ledger row. Ordinary CI is offline: it validates the committed snapshot, exact one-to-one mapping, executable evidence, and generated documentation without cloning PyTorch or rebuilding Sphinx.

**Tech Stack:** Python 3.14 from the repository's UV environment, locked Sphinx/MyST/YAML semantic tooling, pinned PyTorch 2.13 runtime introspection, resolved Sphinx Python-domain objects/doctrees, canonical `native_functions.yaml` schemas, TOML/JSON, existing compatibility checker and GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-08-30-pytorch-compatibility-program.md`

## Global Constraints

- The sole upstream reference is PyTorch `2.13.0`, commit `cf30153c4c131c8164ee7798e5022d810682e2cb`.
- The runtime reference is the locked `.venv` PyTorch `2.13.0`; `torch.version.git_version` must equal the pinned commit before synchronization.
- Only resolved documented Python symbols and canonical schemas from the pinned `native_functions.yaml` enter the required inventory. Schemas whose operator basename begins with `_` are retained with `visibility = "internal"` and are never presented as public compatibility promises; runtime-only JIT registrations are build diagnostics, not required inventory IDs.
- Every inventory ID maps to exactly one `compat/pytorch_api.toml` row; no missing or duplicate mapping is accepted.
- A mapping is a disposition, not proof of support. Existing row status, scope, implementation kind, and executable evidence rules remain authoritative.
- The inventory never calculates or publishes a support percentage.
- Ordinary validation is offline and reads committed files only. Network access and a PyTorch checkout are required only for an explicit maintainer refresh.
- The synchronizer never edits Rust implementation files, never downloads artifacts itself, and never accepts an uncommitted or wrong-commit upstream tree.
- Generated output is deterministic, sorted, atomically replaced, and tested against Sphinx include/glob/autosummary/module-context/conditional-content, alias, overload, platform, import, signature-default, and memory-address edge cases.
- The workflow uses no paid service.

---

### Task 1: Define the pinned inventory and mapping contracts

**Files:**
- Create: `compat/pytorch_reference.toml`
- Create: `scripts/sync-pytorch-inventory.py`
- Create: `scripts/pytorch_inventory_conf.py`
- Create: `scripts/pytorch_inventory_sphinx.py`
- Create: `tests/test_pytorch_inventory.py`
- Modify: `scripts/check-python-lock.py`
- Modify: `tests/test_python_lock.py`
- Modify: `pyproject.toml` to add exact development pins `sphinx==9.1.0`, `pyyaml==6.0.3`, `myst-parser==5.1.0`, and `myst-nb==1.4.0`
- Modify: `uv.lock`

**Interfaces:**
- Produces: `load_reference`, `verify_runtime`, `verify_source_checkout`, `build_sphinx_domain_export`, `collect_documented_symbols`, `collect_aten_schemas`, `render_inventory`, and `validate_mapping`.
- Produces CLI: `--source PATH --build-sphinx --write`, `--source PATH --check-upstream`, and offline `--check`.
- Produces immutable inventory format version 1 and mapping format version 1.

- [ ] **Step 1: Write schema, pin, sort, and atomic-write tests**

Use temporary fixtures and assert the reference file has exactly these nine
top-level fields (including the `source_suffixes` table):

```toml
format_version = 1
pytorch_version = "2.13.0"
pytorch_commit = "cf30153c4c131c8164ee7798e5022d810682e2cb"
documentation_root = "docs/source"
sphinx_builder = "dummy"
sphinx_inventory_format = 1
sphinx_tags = ["cpu", "linux"]
python_namespaces = ["torch"]

[source_suffixes]
".ipynb" = "myst-nb"
".md" = "myst-nb"
".rst" = "restructuredtext"
```

Test rejection of a missing/extra/wrong source-suffix mapping, wrong runtime version, wrong `torch.version.git_version`, wrong or dirty checkout HEAD, unknown fields, unsorted/unsupported Sphinx tags/namespaces, duplicate/unsorted inventory IDs, invalid kinds, absolute/upward source paths, missing mappings, mappings to unknown ledger rows, duplicate mapped IDs, and mappings to unknown inventory IDs. Simulate an interrupted write and prove existing bytes remain unchanged. Extend Python-lock tests to require the four exact Sphinx/MyST/YAML direct pins, their fully locked transitive graph/artifacts, and rejection of any unpinned or alternate-index documentation dependency.

- [ ] **Step 2: Run tests and verify the synchronizer is absent**

Run:

```sh
.venv/bin/python -m unittest tests.test_pytorch_inventory -v
```

Expected: import fails because `scripts/sync-pytorch-inventory.py` does not exist.

- [ ] **Step 3: Implement exact reference and offline validation**

Use these inventory objects:

```json
{
  "format_version": 1,
  "pytorch_version": "2.13.0",
  "pytorch_commit": "cf30153c4c131c8164ee7798e5022d810682e2cb",
  "scope": ["torch.utils.data"],
  "symbols": [
    {
      "id": "python:torch.utils.data.DataLoader",
      "kind": "class",
      "module": "torch.utils.data",
      "name": "DataLoader",
      "signature": "(dataset, batch_size=1, shuffle=None, sampler=None, batch_sampler=None, num_workers=0, collate_fn=None, pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None, multiprocessing_context=None, generator=None, *, prefetch_factor=None, persistent_workers=False, pin_memory_device='', in_order=True)",
      "source": "torch/utils/data/dataloader.py"
    }
  ]
}
```

Allowed kinds are `class`, `function`, `method`, `constant`, `module`, and `aten_schema`. Mapping uses exact string keys:

```toml
format_version = 1

[mapping]
"python:torch.utils.data.DataLoader" = "data.loader"
```

Offline `--check` validates committed JSON/TOML bytes, declared-scope completeness, mapping completeness, and target ledger IDs without importing torch or opening the network.

Add the four exact direct dependencies, update the fail-closed package/artifact
expectations in `scripts/check-python-lock.py`, and regenerate the lock once:

```sh
uv lock --no-cache
```

The lock records every Sphinx transitive dependency and PyYAML wheel for the
supported platforms; ordinary CI remains frozen/offline.

- [ ] **Step 4: Run unit tests and commit the validator**

All tests use complete temporary inventory/map fixtures. Repository inventory files are created with the first real `torch.utils.data` synchronization in Task 2, so no empty production inventory is committed.

```sh
python3 scripts/check-python-lock.py
uv lock --check --offline --no-cache
uv sync --frozen --no-cache
.venv/bin/python -m unittest tests.test_pytorch_inventory -v
git add compat/pytorch_reference.toml scripts/sync-pytorch-inventory.py scripts/pytorch_inventory_conf.py scripts/pytorch_inventory_sphinx.py scripts/check-python-lock.py tests/test_pytorch_inventory.py tests/test_python_lock.py pyproject.toml uv.lock
git commit -s -m "feat(compat): define the pinned PyTorch inventory"
```

### Task 2: Inventory and map the complete classic data surface

**Files:**
- Modify: `scripts/sync-pytorch-inventory.py`
- Modify: `scripts/pytorch_inventory_conf.py`
- Modify: `scripts/pytorch_inventory_sphinx.py`
- Modify: `tests/test_pytorch_inventory.py`
- Create generated content: `compat/pytorch_inventory.json`
- Create: `compat/pytorch_inventory_map.toml`
- Modify: `compat/pytorch_api.toml`
- Modify: `scripts/check-compatibility.py`
- Modify: `tests/test_compatibility_script.py`
- Regenerate: `docs/api-coverage.md`
- Regenerate: `crates/rusttorch-core/COMPATIBILITY.md`
- Regenerate: `crates/rusttorch-data/COMPATIBILITY.md`

**Interfaces:**
- Collects every documented symbol from pinned `torch.utils.data`, dataset, sampler, distributed, DataLoader, collation, worker-info, and DataPipe pages.
- Maps each item to an exact data ledger row.
- Makes the existing compatibility checker fail when an inventory mapping or ledger target drifts.

- [ ] **Step 1: Write resolved-Sphinx-domain and runtime-export tests**

Build tiny Sphinx fixtures covering:

```rst
.. automodule:: torch.utils.data
   :members:

.. autoclass:: torch.utils.data.DataLoader
   :members:

.. autofunction:: torch.utils.data.default_collate
```

Include nested `include`, globbed `toctree`, `autosummary`-generated stubs, module context, conditional `only` content, class/protocol members, aliases, inherited methods, MyST Markdown, and a MyST notebook. Exercise content-preserving neutral replacements for presentation-only directives and assert an unknown or API-bearing unsupported directive fails closed. Mock runtime `__all__`, signatures, a C extension without an inspectable signature, and a documented symbol absent on the host platform. Include a `random_split`-style default object whose normal `repr` contains a process address; generate twice in separate subprocesses and assert byte-identical address-free output. Assert canonical IDs are stable, aliases are retained as separate public IDs, private underscore names are excluded unless present in the resolved Python domain, source paths are relative, and unavailable documented symbols remain in inventory with `signature = null`.

- [ ] **Step 2: Export and consume Sphinx's resolved Python domain**

Build through `scripts/pytorch_inventory_conf.py`, copied byte-for-byte to a temporary Sphinx config directory as `conf.py`; copy `scripts/pytorch_inventory_sphinx.py` beside it and load it by the importable module name `pytorch_inventory_sphinx`. Never import or execute the pinned checkout's `docs/source/conf.py`. Enable only the locked semantic extensions needed for Python API discovery (`autodoc`, `autosummary`, `napoleon`, `myst_nb`, and `pytorch_inventory_sphinx`); `myst_nb` activates the pinned MyST parser, so do not also activate `myst_parser`. Set the pinned checkout as the source root, and set the exact source-suffix mapping, tags, and namespaces from the validated reference manifest. Launch with the temporary config directory as the sole added `PYTHONPATH` entry. Register content-preserving neutral directives/roles only for a reviewed allowlist of presentation extensions; unknown directives and any neutral directive that could hide a Python API object fail the refresh. After Sphinx resolves includes, glob toctrees, autosummary pages, module context, conditionals, MyST sources, and autodoc members, export every `py`-domain object with canonical name, object type, document name, node ID, alias flag, resolved signature node, and source location. The synchronizer consumes only that versioned JSON export; it does not claim completeness from a handwritten RST parser.

Keep Sphinx inputs, config, output, and doctrees in temporary directories, but invoke the build as `uv run --project <verified-repository-root> --frozen ...` so UV always selects this repository's lock and environment regardless of the caller's working directory. Use the locked development dependencies, a finite timeout, deterministic locale/hash seed, and the canonical Linux/CPU documentation tags recorded in the reference manifest. Add a real subprocess smoke test that starts from an unrelated temporary working directory, invokes this exact project-qualified frozen command and sanitized environment against the tiny mixed RST/MyST fixture, proves the collector module imports from the copied config directory, and asserts upstream `conf.py` side effects never run. Resolve implementation source with the pinned checkout rather than the installed wheel. Runtime introspection enriches signatures and export aliases but never removes a resolved Sphinx object. For Python definitions, reconstruct signatures from the pinned source AST and `ast.unparse` defaults/annotations. For extension objects, parse `__text_signature__` or structurally encode only primitive defaults and qualified type/enum/singleton names; never serialize an arbitrary `repr`, and emit `signature = null` with a reason when stable encoding is impossible. Normalize overloads as `python:fully.qualified.name#overload-N` in signature order. Store the exact address-free `DataLoader` signature including `pin_memory_device` and `in_order`.

- [ ] **Step 3: Generate and map the data inventory**

Create the read-only reference checkout at a fixed temporary path if it is not
already present, then detach it at the exact commit:

```sh
git clone --filter=blob:none --no-checkout https://github.com/pytorch/pytorch.git /private/tmp/rusttorch-pytorch-cf30153
git -C /private/tmp/rusttorch-pytorch-cf30153 fetch --depth 1 origin cf30153c4c131c8164ee7798e5022d810682e2cb
git -C /private/tmp/rusttorch-pytorch-cf30153 checkout --detach cf30153c4c131c8164ee7798e5022d810682e2cb
uv run --frozen python scripts/sync-pytorch-inventory.py --source /private/tmp/rusttorch-pytorch-cf30153 --build-sphinx --write --namespace torch.utils.data
```

Map every emitted ID to a precise row. Add rows where the current broad ledger cannot distinguish dataset adapters, sampler variants, collation, worker info, argument behavior, DataPipes, Python multiprocessing, and deprecated pin-device behavior. `python_only`, `not_supported`, and Rust-native replacement rows are valid mappings; omission is not.

- [ ] **Step 4: Enforce mapping through the compatibility checker**

Make `scripts/check-compatibility.py --check` load the inventory/map and validate each mapped ledger target before rendering. Render an exact “Pinned inventory” count per ledger row without percentages and link to the mapping file.

```sh
.venv/bin/python -m unittest tests.test_pytorch_inventory tests.test_compatibility_script -v
.venv/bin/python scripts/sync-pytorch-inventory.py --check
.venv/bin/python scripts/check-compatibility.py --write
.venv/bin/python scripts/check-compatibility.py --check
```

- [ ] **Step 5: Commit**

```sh
git add compat scripts/sync-pytorch-inventory.py scripts/pytorch_inventory_conf.py scripts/pytorch_inventory_sphinx.py scripts/check-compatibility.py tests docs/api-coverage.md crates/rusttorch-core/COMPATIBILITY.md crates/rusttorch-data/COMPATIBILITY.md
git commit -s -m "feat(compat): inventory the PyTorch data API"
```

### Task 3: Inventory all documented Python APIs and canonical ATen schemas

**Files:**
- Modify: `scripts/sync-pytorch-inventory.py`
- Modify: `tests/test_pytorch_inventory.py`
- Replace generated content: `compat/pytorch_inventory.json`
- Modify exhaustively: `compat/pytorch_inventory_map.toml`
- Expand: `compat/pytorch_api.toml`
- Regenerate: `docs/api-coverage.md`
- Regenerate: `crates/rusttorch-core/COMPATIBILITY.md`
- Regenerate: `crates/rusttorch-data/COMPATIBILITY.md`
- Modify attribution: `THIRD_PARTY_NOTICES.md`

**Interfaces:**
- Collects all public Python objects present in the pinned, resolved PyTorch Sphinx Python domain.
- Collects all schemas produced from the pinned canonical `aten/src/ATen/native/native_functions.yaml` by the matching installed `torchgen` parser, including declared `autogen` functional/out variants, while preserving overload names, normalized schema text, and public/internal-name classification.
- Records whether each canonical schema is present in the pinned runtime build without treating build-dependent runtime-only registrations as public API.
- Requires every symbol/schema to map to exactly one subsystem ledger row.

- [ ] **Step 1: Write whole-tree and ATen normalization tests**

Fixture tests cover nested includes, globbed Sphinx toctrees, duplicate references, generated `autosummary`, `automodule`, module context, conditional content, documented class/protocol members, explicitly documented underscore symbols, overloaded functions, aliases, removed/deprecated entries, platform-only entries, stable default normalization, and deterministic source ordering. ATen tests cover overload names, alias annotations, optional/list/Tensor arguments, keyword-only markers, multiple returns, internal underscore names, `autogen` out/functional variants, runtime-present/runtime-absent flags, and duplicate schema rejection.

- [ ] **Step 2: Build the official docs sources and read resolved domain objects**

Build from pinned `docs/source/index.rst` with the same repository-owned isolated configuration and copied collector proven in Task 2; never load the upstream configuration. Collect only resolved Python-domain objects whose canonical namespace is explicitly present in the reference manifest. For this core plan, enable only `torch`; domain repositories receive separate pinned manifests. Fail the refresh if the Sphinx build or collector reports an unresolved generated page, duplicate canonical object, or an empty declared namespace.

Do not import arbitrary files from the checkout in the synchronizer process. Resolve runtime members/signatures in isolated `.venv/bin/python` subprocesses with a finite timeout, sanitized environment, captured stderr, and one module per process. A failed optional import records the documented symbol without a runtime signature and does not erase it.

- [ ] **Step 3: Collect canonical source schemas and runtime availability**

Use `torchgen.gen.parse_native_yaml` from the verified PyTorch wheel, passing the
pinned checkout's `native_functions.yaml` and `tags.yaml` with native-function
generation enabled. This is the canonical expansion used by PyTorch codegen and
includes `autogen` variants that are not literal `func` entries. Reject a
`torchgen` module not installed beside the verified `torch` package. Normalize
every returned `NativeFunction.func` and emit IDs such as:

```text
aten:aten::add.Tensor
aten:aten::add.Scalar
```

Store the normalized schema string, exact source path/base line, generated-from
name when applicable, and `visibility` based on the operator basename. Test an
`autogen: name.out` fixture and require both the base and generated IDs. In the
pinned subprocess, collect `torch._C._jit_get_all_schemas()` only to set
`runtime_present` for matching canonical IDs and record the runtime build
fingerprint. Ignore runtime-only registrations for completeness and never assign
them the YAML source path. Treat every schema as a binding candidate, not proof
that `tch` or RustTorch exposes it; internal schemas map to an explicit
non-public ledger disposition.

- [ ] **Step 4: Map every item and split ledger scopes where needed**

Set the inventory scope to `["torch"]`. Map generated operators to exact generated-binding families and map Python frontend symbols to subsystem rows: tensor/operators, autograd, nn modules/functionals, optimizers/schedulers, AMP, serialization, distributions, linalg/fft/special, sparse/quantized/nested, distributed, compiler/export/ONNX, profiler/testing/utilities, and data/domain packages. Split a row whenever one status/scope cannot truthfully describe every mapped item.

The synchronizer exits nonzero until every ID in the declared `torch` scope is mapped.

- [ ] **Step 5: Run deterministic regeneration twice and commit**

```sh
uv run --frozen python scripts/sync-pytorch-inventory.py --source /private/tmp/rusttorch-pytorch-cf30153 --build-sphinx --write
cp compat/pytorch_inventory.json /tmp/rusttorch-pytorch-inventory-first.json
uv run --frozen python scripts/sync-pytorch-inventory.py --source /private/tmp/rusttorch-pytorch-cf30153 --build-sphinx --write
cmp /tmp/rusttorch-pytorch-inventory-first.json compat/pytorch_inventory.json
.venv/bin/python scripts/sync-pytorch-inventory.py --check
.venv/bin/python scripts/check-compatibility.py --write
.venv/bin/python scripts/check-compatibility.py --check
git add compat scripts tests docs/api-coverage.md crates/rusttorch-core/COMPATIBILITY.md crates/rusttorch-data/COMPATIBILITY.md THIRD_PARTY_NOTICES.md
git commit -s -m "feat(compat): census the pinned PyTorch API"
```

### Task 4: Make inventory drift and subsystem planning part of free CI

**Files:**
- Modify: `.github/workflows/ci.yml`
- Modify: `CONTRIBUTING.md`
- Modify: `docs/pytorch-compatibility.md`
- Modify: `docs/porting-policy.md`
- Modify: `docs/maintainer-guide.md`
- Modify: `tests/test_community_health.py`

**Interfaces:**
- Adds an offline required inventory check to existing CI.
- Defines the one-feature workflow from inventory ID to ledger row, scoped spec/plan, implementation, executable evidence, docs, and status update.
- Adds no scheduled network job, paid scanner, or premium ruleset dependency.

- [ ] **Step 1: Write CI/policy contract tests**

Assert stable CI runs both:

```sh
.venv/bin/python scripts/sync-pytorch-inventory.py --check
.venv/bin/python scripts/check-compatibility.py --check
```

Assert neither command contains a URL, clone, download, package install, unpinned Python executable, or alternate environment. Assert contribution docs require inventory mapping for new compatibility claims and forbid hand-editing generated inventory/docs.

- [ ] **Step 2: Run policy tests and verify missing inventory enforcement**

Run:

```sh
.venv/bin/python -m unittest tests.test_community_health -v
```

Expected: new assertions fail because CI and contribution docs do not mention the inventory.

- [ ] **Step 3: Add the offline CI gate and maintainer workflow**

Place inventory validation immediately before compatibility validation in the UV-backed quality job. Document the maintainer refresh command with an explicit local source path, exact clean-commit requirement, mapping reconciliation, two-pass deterministic check, and review of added/removed symbols.

Document the implementation loop exactly:

```text
inventory ID -> ledger disposition -> scoped design/plan -> failing test
-> Rust/LibTorch implementation -> parity evidence -> rustdoc/example
-> ledger status update -> generated coverage -> CI
```

One pull request may cover a coherent family, but every included inventory ID remains visible and mapped.

- [ ] **Step 4: Run the complete policy and compatibility gate**

```sh
.venv/bin/python -m unittest discover -s tests -p 'test_*.py' -v
.venv/bin/python scripts/sync-pytorch-inventory.py --check
.venv/bin/python scripts/check-compatibility.py --check
cargo fmt --all -- --check
```

Expected: all checks pass offline after the locked `.venv` already exists.

- [ ] **Step 5: Commit**

```sh
git add .github/workflows/ci.yml CONTRIBUTING.md docs tests/test_community_health.py
git commit -s -m "ci: enforce the pinned PyTorch census"
```
