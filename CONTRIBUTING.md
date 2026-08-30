# Contributing to RustTorch

Thank you for helping RustTorch. By participating, you agree to the
[Code of Conduct](CODE_OF_CONDUCT.md). Usage questions belong in the channels
described by the [support policy](SUPPORT.md); vulnerabilities must follow the
[security policy](SECURITY.md).

The project is governed as described in [governance](GOVERNANCE.md). Every
original contribution is accepted under `MIT OR Apache-2.0` and must satisfy
the [Developer Certificate of Origin](DCO.md).

## Before you start

Search existing issues and pull requests first. Open an issue before work that
changes public API or architecture, adds a dependency or `unsafe` code, changes
compatibility scope, or is likely to require several files. Small documentation
fixes and narrow bug fixes may go directly to a pull request.

An issue records direction; it does not reserve work indefinitely. Maintainers
may redirect proposals that duplicate existing functionality, belong in `tch`,
or exceed the current compatibility milestone.

## Development workflow

1. Fork the repository and create a focused branch from current `main`.
2. Make the smallest complete change and add the narrowest test that fails
   without it.
3. Update public documentation, the changelog, and compatibility records when
   behavior or supported scope changes.
4. Commit with DCO sign-off and open a pull request using the repository
   template.

Set up the project-local Python/LibTorch environment, then run the core checks
from the repository root:

```sh
. scripts/dev-env.sh
python3 scripts/check-compatibility.py --check
cargo fmt --all -- --check
cargo check --workspace --all-targets
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-targets
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps
cargo check -p rusttorch --all-targets --no-default-features --features tch/doc-only
python3 -m unittest tests/test_community_health.py -v
```

After committing, inspect both package file lists:

```sh
cargo package -p rusttorch --locked --no-verify --list
cargo package -p rusttorch-cli --locked --list
```

Run `scripts/run-python-parity.sh` for changes to PyTorch-visible defaults,
initialization, gradients, optimizers, state naming, or serialization. Run
`scripts/check-backends.sh` for backend claims and report unavailable hardware
as skipped, never passed.

## Compatibility and public API

RustTorch does not claim full PyTorch parity. A compatibility entry is valid
only within its written scope and evidence. Edit the canonical
[`compat/pytorch_api.toml`](compat/pytorch_api.toml), then regenerate and
verify the public view; do not edit generated coverage by hand:

```sh
python3 scripts/check-compatibility.py --write
python3 scripts/check-compatibility.py --check
```

Changes to public behavior must:

- update the compatibility row's stable ID, status, implementation, symbols,
  exact scope, pinned source, evidence, and notes as applicable;
- provide complete rustdoc and an ergonomic example for every new public item;
- add a narrow Rust test and record it as
  `repo-relative-test-file::exact_test_name` evidence;
- cite the pinned PyTorch symbol, source path, tag, and commit when adapting
  upstream behavior;
- add Rust tests and, where Python-visible behavior is matched, parity evidence;
- include backend evidence for every backend newly claimed; and
- document intentional differences caused by Rust or unavailable Python
  semantics.

Backward-incompatible 0.x changes still require an issue, migration note,
changelog entry, and complete rustdoc. Avoid public abstractions added only for
speculative future use.

## Performance, safety, and dependencies

Performance claims need a reproducible benchmark, named baseline, hardware and
software versions, build profile, warm-up, sample count, raw results, and
variance. Compare equivalent work and synchronize accelerators when required.
Rust alone is not evidence of a speedup.

New dependencies and `unsafe` code require issue-first review. Explain why
existing code, the standard library, platform functionality, or an installed
dependency is insufficient. Document every safety invariant and keep the
unsafe boundary as small as possible. Dependency changes must include license,
maintenance, security, and package-size review.

## Provenance and generated work

Third-party code, models, datasets, schemas, and generated files must identify
their source, version or commit, license, and transformation or generation
method. Update `THIRD_PARTY_NOTICES.md` when required. Do not submit material
whose license is incompatible with `MIT OR Apache-2.0` or whose redistribution
rights are uncertain.

AI-assisted contributors remain responsible for every submitted line. Review
generated output for correctness, provenance, licensing, security, private
data, and hidden dependencies. Tool use does not transfer authorship
responsibility and does not weaken the DCO certification.

Never commit credentials, personal data, virtual environments, downloaded
LibTorch or CUDA artifacts, generated model files, untrusted pickle data,
vendored PyTorch source, or unrelated build output. If a secret is exposed,
revoke it immediately and follow the security policy; deleting one commit is
not sufficient remediation.

## Sign-off

Every commit must carry a `Signed-off-by` trailer matching its author identity:

```sh
git commit --signoff
git commit --amend --signoff
```

The sign-off certifies DCO 1.1, not merely authorship. The requirement applies
to maintainers as well as contributors. It is project policy now; automated
pull-request enforcement is added before required repository checks are
activated.

## Pull-request review

Keep pull requests focused and fill out every applicable checklist item.
Maintainers may mark work as needing triage, request changes, approve it, defer
it to a milestone, close stale work after notice, or reject changes that are
unsafe, unsupported, insufficiently evidenced, or out of scope. Approval is
not a promise to merge if later checks or review uncover a problem.

Contributors retain copyright in their work while licensing it to recipients
under the repository's MIT or Apache-2.0 terms.
