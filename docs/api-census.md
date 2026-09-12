# Pinned PyTorch API census

The committed inventory assigns one stable identity to each resolved documented
Python object in the pinned `torch` namespace and each canonical ATen native
schema. Every identity has exactly one explicit disposition in the compatibility
ledger. A mapping is not proof of support: each ledger row's written scope,
status, Rust surface, and executable evidence determine the claim.

The sole reference is PyTorch **2.13.0**, commit
`cf30153c4c131c8164ee7798e5022d810682e2cb`. The reference manifest is
[`compat/pytorch_reference.toml`](../compat/pytorch_reference.toml); the generated
snapshot is [`compat/pytorch_inventory.json`](../compat/pytorch_inventory.json),
and its hand-maintained dispositions are
[`compat/pytorch_inventory_map.toml`](../compat/pytorch_inventory_map.toml).

The completed snapshot contains **12,462 identities**: 9,361 documented Python
identities and 3,101 canonical ATen schemas. The ATen set includes 2,261 public-name
and 840 internal-name schemas, with 517 generated variants. These are inventory
counts, not a compatibility percentage. The snapshot records the actual
`darwin/arm64` wheel fingerprint separately from the pinned CPU/Linux
documentation selection.

## Ordinary offline checks

After creating the frozen project environment, run from the repository root:

```sh
.venv/bin/python scripts/sync-pytorch-inventory.py --check
.venv/bin/python scripts/check-compatibility.py --check
```

These commands use only committed files and the Python standard library. They
do not import PyTorch, rebuild documentation, fetch source, or use the network.
Missing IDs, duplicate IDs, extra mappings, unknown ledger targets, unsafe
source paths, changed pins, noncanonical JSON bytes, and stale generated
coverage fail validation.

## Explicit maintainer refresh

Prepare an external, clean PyTorch checkout at the exact commit above. The
synchronizer does not download source or artifacts. Install the project tooling
with `uv sync --frozen --no-cache`; the exact documentation development pins are
Sphinx 9.1.0, MyST-Parser 5.1.0, MyST-NB 1.4.0, and PyYAML 6.0.3, including their
locked transitive dependencies and artifacts. The installed PyTorch wheel must
report the same version and source commit as the manifest.

Run from the RustTorch repository root, substituting the explicit local source
path:

```sh
.venv/bin/python scripts/sync-pytorch-inventory.py \
  --source /absolute/path/to/pytorch --check-upstream
uv run --frozen python scripts/sync-pytorch-inventory.py \
  --source /absolute/path/to/pytorch --build-sphinx --write
```

The refresh writes the actual inventory atomically, then exits nonzero if its
mapping needs reconciliation. Review added and removed IDs against the pinned
source. Edit the mapping and, where needed, add narrowly scoped ledger rows;
do not hand-edit the generated inventory or generated coverage pages. An
unimplemented, Python-only, or deliberately unsupported disposition is valid.
A new support claim requires implementation and evidence. Supported or partial
Python mappings require the exact canonical name in the target row's
`python_symbols`, including documented aliases. Positive ATen mappings require
an explicitly evidenced full schema ID; internal schemas remain unsupported
implementation inventory.

After reconciling the map, prove deterministic generation:

```sh
cp compat/pytorch_inventory.json /tmp/rusttorch-inventory-first.json
uv run --frozen python scripts/sync-pytorch-inventory.py \
  --source /absolute/path/to/pytorch --build-sphinx --write
cmp /tmp/rusttorch-inventory-first.json compat/pytorch_inventory.json
.venv/bin/python scripts/sync-pytorch-inventory.py --check
.venv/bin/python scripts/check-compatibility.py --write
.venv/bin/python scripts/check-compatibility.py --check
```

Review and commit the manifest, inventory, map, ledger, evidence, and generated
coverage together. Keep the external checkout, build directories, and downloaded
wheels out of Git.

## Collection boundaries

The isolated Sphinx build uses the pinned CPU/Linux documentation tags and the
upstream RST, MyST Markdown, and notebook sources. At this commit the root is
`docs/source/index.md`. Notebook execution is disabled. The collector observes
Sphinx's resolved Python domain and doctrees, including includes, autosummary,
module context, class members, aliases, and conditional content. It never imports
or executes upstream `conf.py`; only its literal filename map is read to preserve
case-insensitive filesystem compatibility. Upstream autosummary templates are
used as templates, not documentation pages.

Reviewed layout directives preserve nested content; Mermaid remains literal
content and cannot contain API directives. The pinned source's mixed-delimiter
CSV headers are normalized without changing table cells. Optional z3/TensorBoard
module definitions are represented from their source AST for Sphinx discovery
when unavailable in the tooling wheel; their source code is never executed.
Unresolved imports, unknown directives, missing generated pages, and duplicate
canonical objects fail the refresh. Five known same-page duplicate property descriptions
(`Transform.sign` and four `PackedSequence` fields) resolve to their single
canonical object; all other duplicate descriptions fail.

Only Git-tracked upstream files are staged, so ignored generated pages cannot
contaminate the build. The exact tracked ExportDB generator runs in a separate,
isolated pinned-wheel process before Sphinx, as required by the upstream
Makefile. Its example code and exported graphs remain temporary documentation
content; they make no RustTorch support claim.

Runtime enrichment runs one module per isolated, timed subprocess. Python
signatures come from pinned-source AST expressions; extension text signatures
are used only when stable. Missing signatures remain explicit `null` values
with a reason, and arbitrary object representations are never serialized. Each
overload also keeps its distinct resolved `documented_signature`, separately
from its runtime/source signature. Unstable documented representations are
explicitly unavailable with a reason.
Source paths refer to pinned implementation files or the original document that
requested a generated API page.

ATen schemas come from the matching installed `torchgen` parser applied to the
pinned `native_functions.yaml` and `tags.yaml`, with native generation enabled.
Declared generated functional/out variants retain their origin. Operator
basenames beginning with `_` are **internal implementation inventory**, not
public compatibility promises. Runtime registration is recorded only as build
availability; runtime-only registrations do not enter the required inventory.
The snapshot includes its runtime build fingerprint. None of these facts proves
that `tch` or RustTorch exposes a schema.

Generated coverage shows exact per-row inventory counts and links to the map.
Counts have unequal scope and are never converted into a support percentage.

## Implementing one family

```text
inventory ID -> ledger disposition -> scoped design/plan -> failing test
-> Rust/LibTorch implementation -> parity evidence -> rustdoc/example
-> ledger status update -> generated coverage -> CI
```

One coherent change may cover several IDs; every included identity remains
visible and mapped. Broader domain repositories require their own separately
pinned manifests and are outside this `torch` census.
