# RustTorch Open-Source Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn RustTorch into a professional, understandable, policy-driven
public project with hardened CI and verifiable release provenance.

**Architecture:** Markdown and GitHub-native community files define the human
contract. A least-privilege Rust CI workflow enforces objective contribution
rules. A tag-only workflow packages final Cargo archives once and delegates
provenance signing to OpenSSF's SLSA Generic generator.

**Tech Stack:** Markdown, Cargo, Python standard library, GitHub Actions,
GitHub CLI, OpenSSF SLSA Generic generator.

**Spec:** `docs/superpowers/specs/2026-08-30-open-source-foundation.md`

## Global Constraints

- Keep every public claim within implemented scope; never advertise full
  PyTorch parity or unmeasured performance.
- Use the repository owner `@newpoluton-alt`; do not invent an email address.
- Accept contributions under `MIT OR Apache-2.0` with DCO 1.1 sign-off.
- Ordinary actions use verified immutable SHAs. The SLSA reusable workflow's
  required `@v2.1.0` tag is the sole documented exception.
- Workflows grant the minimum permissions per job and never expose secrets to
  pull-request code.
- Never reuse the previously exposed crates.io credential.

---

### Task 1: Redesign the public README

**Files:**
- Replace: `README.md`

**Interfaces:**
- Produces: the GitHub and crates.io landing page for the current source tree.

- [ ] **Step 1: Inventory every public claim and link**

Confirm install commands against the workspace manifests, compile the eager
example against the public API, and list only capabilities present in code and
tests. Treat the current unpublished CLI as a source install.

- [ ] **Step 2: Write the professional landing page**

Use a centered project heading, credo, factual crates.io/docs/license/MSRV
badges, compact navigation, a prominent early-stage notice, a source quick
start, a short eager example, scoped capability table, three setup modes,
documentation map, concise limitations, contribution/security links, and
dual-license footer. Keep detailed backend evidence in its existing docs.

- [ ] **Step 3: Validate content**

Run rustdoc doctests with the network-free `tch/doc-only` feature, assert every
relative Markdown link points to a tracked path, and manually inspect rendered
heading order, code fences, badge destinations, and line length.

- [ ] **Step 4: Commit**

```sh
git add README.md
git commit -m "docs: redesign the RustTorch project page"
```

### Task 2: Define community, legal, security, and governance policy

**Files:**
- Replace: `CONTRIBUTING.md`
- Modify: `README.md`
- Create: `CODE_OF_CONDUCT.md`
- Create: `DCO.md`
- Create: `SECURITY.md`
- Create: `SUPPORT.md`
- Create: `GOVERNANCE.md`
- Create: `docs/maintainer-guide.md`
- Create: `.github/CODEOWNERS`
- Create: `.github/ISSUE_TEMPLATE/bug.yml`
- Create: `.github/ISSUE_TEMPLATE/feature.yml`
- Create: `.github/ISSUE_TEMPLATE/compatibility.yml`
- Create: `.github/ISSUE_TEMPLATE/performance.yml`
- Create: `.github/ISSUE_TEMPLATE/config.yml`
- Create: `.github/pull_request_template.md`
- Create: `.github/dependabot.yml`

**Interfaces:**
- Produces: the complete contributor contract and structured intake surface.

- [ ] **Step 1: Write policy fixtures before policy files**

Create `tests/test_community_health.py` using only Python's standard library.
Assert every required file exists; template YAML has the expected `name`,
`description`, and required acknowledgements; CODEOWNERS protects `.github/`
and itself; contribution docs link DCO, security, conduct, governance, and
support; and no placeholder such as `INSERT`, `TODO`, or invented contact
address remains.

- [ ] **Step 2: Run the test and verify it fails**

```sh
python3 -m unittest tests/test_community_health.py -v
```

Expected: FAIL because the community files do not yet exist.

- [ ] **Step 3: Add the minimum complete community files**

Adopt Contributor Covenant 2.1 and DCO 1.1 verbatim where their licenses
require it. Route vulnerabilities to GitHub private vulnerability reporting.
For conduct reports, point to the maintainer's published profile contact and
warn reporters not to disclose sensitive details publicly; do not promise a
confidential channel until one is separately configured. Define scopes,
labels, reproduction fields, PyTorch reference and backend fields, benchmark
methodology, PR checklist, code ownership, and weekly Dependabot updates for
Cargo and GitHub Actions.

- [ ] **Step 4: Expand contributor and maintainer workflows**

Document issue-first changes, fork/branch/commit/sign-off flow, local gates,
public API/ledger rules, dependency and unsafe-code review, third-party
provenance, AI-assisted contribution responsibility, review states, release
authority, conflict resolution, and maintainer succession. Replace the
README's interim GitHub security-page link with the committed security policy.

- [ ] **Step 5: Validate and commit**

Run the standard-library community-health test for required files and textual
form fields, validate issue-form YAML through GitHub after push, and check every
relative link. Do not claim local YAML syntax validation without a declared
parser. Then:

```sh
git add README.md CONTRIBUTING.md CODE_OF_CONDUCT.md DCO.md SECURITY.md SUPPORT.md \
  GOVERNANCE.md docs/maintainer-guide.md .github tests/test_community_health.py
git commit -m "docs: establish the RustTorch contributor contract"
```

### Task 3: Harden the Rust contribution workflow

**Prerequisite:** Compatibility-ledger Task 4 has created the base CI workflow.

**Files:**
- Modify: `.github/workflows/ci.yml`
- Create: `scripts/check-dco.py`
- Create: `tests/test_check_dco.py`
- Modify: `CONTRIBUTING.md`
- Modify: `README.md`

**Interfaces:**
- Consumes: GitHub pull-request base/head SHAs and the compatibility checker.
- Produces: `CI / required`, the stable repository-rules status check.

- [ ] **Step 1: Test the DCO checker**

Use temporary Git repositories to prove that an unsigned commit fails, a
matching `Signed-off-by` trailer passes, a mismatched signatory fails, multiple
commits are all checked, and commits reachable from the base are excluded.
Also cover body-line spoofing, malformed trailers, multiple sign-offs,
co-authors, unsigned merge commits, invalid revisions, and the narrow verified
Dependabot exception.

- [ ] **Step 2: Implement the standard-library DCO checker**

Accept `--base` and `--head`, validate both revisions, enumerate `base..head`,
and parse only Git's trailer block. Require a `Signed-off-by: Name <email>`
trailer matching the commit author and every `Co-authored-by` identity under a
documented case-insensitive normalization rule. Dependabot is exempt only when
an explicit trusted workflow-actor flag and its exact verified identity both
match; do not waive DCO for other bots. Include merge commits, fail closed, and
print short commit IDs plus exact sign-off recovery commands. Document policy
for maintainer-applied patches, verified automation, and GitHub update-branch
merge commits.

- [ ] **Step 3: Restructure CI with least privilege**

Use these exact action pins:

```text
actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1 # v7.0.1
actions/setup-python@5fda3b95a4ea91299a34e894583c3862153e4b97 # v7.0.0
actions/dependency-review-action@595b5aeba73380359d98a5e087f648dbb0edce1b # v4.7.3
actions-rust-lang/setup-rust-toolchain@46268bd060767258de96ed93c1251119784f2ab6 # v1.16.1
```

Set workflow `name: CI` and name the aggregator job exactly `required`, yielding
the stable `CI / required` status. Run DCO and dependency review only for pull
requests. DCO checkout uses `fetch-depth: 0` and checks
`${{ github.event.pull_request.base.sha }}..${{ github.event.pull_request.head.sha }}`.
Every checkout preceding contributor-controlled execution sets
`persist-credentials: false`. Dependency Review uses only `contents: read` and
explicitly sets `comment-summary-in-pr: never`. Run the library quality suite
on Ubuntu stable, compile both packages on Rust 1.88, and test the CLI on
Ubuntu, macOS, and Windows. Preserve Python 3.14, the project virtual
environment, the pinned CPU PyTorch wheel from its CPU index installed
separately from SafeTensors and NumPy, the exported LibTorch path, and Python
parity. Run standard-library Python test discovery, package-specific rustdoc,
and both package-list and archive-verification checks. Add concurrency
cancellation, read-only default permissions, and `if: always()` logic that
needs every quality/MSRV/platform/PR-only job, requires PR-only jobs on pull
requests, accepts only their intentional skip on push/manual runs, and rejects
failed, cancelled, or unexpectedly skipped applicable prerequisites.

- [ ] **Step 4: Document and validate**

Add the CI badge only after the workflow exists. Run DCO unit tests, workflow
structure tests in `test_community_health.py`, all local network-free gates,
and `git diff --check`.

- [ ] **Step 5: Commit**

```sh
git add .github/workflows/ci.yml scripts/check-dco.py tests/test_check_dco.py \
  tests/test_community_health.py CONTRIBUTING.md README.md
git commit -m "ci: enforce the RustTorch contribution gates"
```

### Task 4: Generate SLSA3 provenance for release artifacts

**Files:**
- Create: `.github/workflows/release.yml`
- Create: `scripts/check-release.py`
- Create: `tests/test_release_workflow.py`
- Create: `docs/releasing.md`
- Modify: `docs/maintainer-guide.md`
- Modify: `README.md`

**Interfaces:**
- Consumes: tag `vX.Y.Z`, both Cargo manifests, lockfile, changelog, ledger.
- Produces: exact `.crate` archives, SHA-256 subjects, an in-toto provenance
  file, and a GitHub release containing those same bytes.

- [ ] **Step 1: Test release invariants and workflow structure**

Test matching and mismatched tags/versions/changelog, deterministic byte-exact
subject formatting, tag-only triggers, non-cancelling per-tag concurrency, job
permission maps, immutable ordinary action pins, the single SLSA tag exception,
absence of pull-request and secret access, exact build/provenance/release needs,
and proof that release jobs download rather than rebuild artifacts. Assert the
draft release, checksum/base64 revalidation, exact asset set, and publish-last
sequence.

- [ ] **Step 2: Implement the release preflight**

Use Python's standard library to strictly parse a `vX.Y.Z` tag, validate exact
root and CLI package names and equal versions, require exactly one matching
lockfile entry for each package, require an exact changelog release heading,
and validate the compatibility ledger before packaging. Reject malformed,
prerelease, build-metadata, and leading-zero tags under this release policy.

- [ ] **Step 3: Build exact release subjects once**

In a read-only build job, install pinned CPU PyTorch, set
`LIBTORCH_USE_PYTORCH=1` and its native-library path, then run locked Cargo
package verification. Copy only both generated `.crate` files into `dist/`.
Start with an empty `dist/`, require the exact versioned root and CLI archive
names, and reject any extra archive. From that directory hash the two explicit
filenames in deterministic order into `subjects.txt`; require the GNU
`64hex + two spaces + basename + newline` form and base64-encode the file
unchanged, failing closed if the job output is empty. Upload exactly both
archives plus `subjects.txt` as one named,
immutable, bounded-retention workflow artifact, with missing files treated as
an error and overwrite disabled, using:

```text
actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02 # v4.6.2
```

- [ ] **Step 4: Add isolated provenance and release jobs**

Call the reusable generator exactly as:

```yaml
uses: slsa-framework/slsa-github-generator/.github/workflows/generator_generic_slsa3.yml@v2.1.0
```

Pass `base64-subjects`, set a versioned `provenance-name`, and enable
`upload-assets`; also pass `draft-release: "true"` so immutable-release assets
can be completed before publication. Set workflow permissions to `{}`, build
to `contents: read`, provenance exactly to `actions: read`, `id-token: write`,
and `contents: write`, and release to `contents: write`. Download the named
package artifact without checkout using:

```text
actions/download-artifact@d3f86a106a0bac45b974a628896c90dbdf5c8093 # v4.3.0
```

Revalidate `sha256sum --check subjects.txt` and require its base64 to match the
exact build output supplied to provenance. With narrowly scoped `GH_TOKEN` and
`GH_REPO: ${{ github.repository }}`, require the generator-created release to
still be a draft, use
`gh release upload "$GITHUB_REF_NAME" dist/*.crate --clobber`, verify that the
complete asset set is exactly both crates plus the versioned provenance, then
run `gh release edit "$GITHUB_REF_NAME" --draft=false` as the final command.
Do not create a second release: the generator owns the draft and provenance.
Never rebuild or publish to crates.io from this workflow; a partial failure
must leave a recoverable draft and a published immutable release must fail
closed.

- [ ] **Step 5: Document verification and commit**

Document installing `slsa-verifier` v2.7.1 and verifying both exact crate
subjects in one invocation with
`--source-uri github.com/newpoluton-alt/RustTorch` and
`--source-tag vX.Y.Z`. State that GitHub's automatic source archives are not
covered subjects. State
that the requested Generic Generator is no longer actively maintained and
record GitHub artifact attestations as the migration path. Run release tests
and package dry runs, then commit.

### Task 5: Apply and verify public repository settings

**Prerequisite:** The final workflows and community files are pushed and CI
has reported its stable check name.

**Files:**
- Modify: `docs/maintainer-guide.md`

**Interfaces:**
- Produces: GitHub-native protections whose read-back matches documented policy.

- [ ] **Step 1: Inspect current settings and authentication**

Read repository settings, existing rulesets/branch protection, Actions
permissions, vulnerability-reporting status, and authenticated account scopes.
Do not replace an existing stricter rule without explicit review.

- [ ] **Step 2: Enable collaboration and safe defaults**

Enable Discussions, private vulnerability reporting, Dependabot alerts and
security updates, secret scanning and push protection where available,
automatic deletion of merged branches, squash merging, web commit sign-off,
and read-only workflow token defaults. Web sign-off covers only GitHub web
commits; CLI commits remain enforced by DCO CI. Record unavailable controls.

- [ ] **Step 3: Protect main and release tags**

Create rulesets for `main` and `v*` that prevent deletion and force pushes.
While there is one maintainer, require pull requests, resolved conversations,
and `CI / required` for `main`, but keep required approvals at zero and record
an owner user bypass in `pull_request` mode. After governance records a second
active maintainer, raise approvals to one and require code-owner review. Give
the owner a documented tag-creation bypass for `v*`; do not use team-required-
reviewer rules in this user-owned repository.

- [ ] **Step 4: Read back, document, and commit**

Fetch every changed setting/ruleset, compare it with the intended policy,
record its name and recovery procedure in the maintainer guide, and commit the
documentation update. If GitHub plan or authentication prevents a setting,
record the exact unavailable control and the closest safe alternative.
