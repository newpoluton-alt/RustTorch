# RustTorch Open-Source Foundation

**Status:** Approved on 2026-08-30

## Purpose

RustTorch should be easy to evaluate, safe to contribute to, and difficult to
release incorrectly. Repository policy cannot guarantee that no contributor
ever violates a rule, but it can make the rules unambiguous, automate the
objective checks, protect sensitive paths, and give maintainers a documented
response process.

This foundation covers the public project page, contributor/legal policy,
governance, security reporting, contribution intake, continuous integration,
release provenance, and GitHub repository protections. It does not claim that
the current API has full PyTorch coverage.

## Public project page

The README is a concise product page, not a test report or a promise about
future features. It must:

- lead with the credo: “easy to use and easy to implement, but crazy fast”;
- explain that RustTorch is an unofficial Rust frontend over LibTorch;
- provide one copyable source install and one compiling eager example;
- show current, scoped capabilities and a prominent early-stage/parity notice;
- link to API docs, compatibility coverage, platform setup, interoperability,
  contributing, security, governance, and licenses;
- use only badges backed by a live public page or committed workflow; and
- avoid unsupported performance, compatibility, release, or affiliation
  claims.

Detailed host-specific parity output and low-level setup rules remain in the
dedicated documentation rather than overwhelming the landing page.

## Contribution and legal contract

All original contributions are accepted under `MIT OR Apache-2.0`, matching
the project. Contributors certify the right to submit each commit using the
Developer Certificate of Origin 1.1 and a `Signed-off-by` trailer. CI checks
the commits introduced by each pull request.

The contribution guide defines:

- issue-first expectations for large API, unsafe, dependency, architecture,
  and compatibility changes;
- formatting, test, rustdoc, compatibility-ledger, parity, package, and
  security gates;
- public API compatibility and performance-evidence requirements;
- third-party code, model, dataset, and generated-code provenance rules;
- a prohibition on credentials, personal data, downloaded runtimes, generated
  model artifacts, and incompatible source entering Git;
- review, changelog, documentation, and DCO requirements; and
- how maintainers may request changes, close stale work, or reject unsafe or
  out-of-scope contributions.

The project adopts Contributor Covenant 2.1. Conduct reports and vulnerability
reports use GitHub private vulnerability reporting so no private email address
is invented or published.

## Governance and support

Governance documents the current maintainer, contributor and maintainer roles,
consensus-seeking decisions, final maintainer responsibility, conflict-of-
interest disclosure, security embargo authority, release authority, and the
path for adding future maintainers. A support policy routes usage questions to
GitHub Discussions and reserves Issues for actionable defects and accepted
work.

`CODEOWNERS` assigns the repository owner to all paths and explicitly protects
workflows, ownership policy, security policy, release policy, licenses, and
the compatibility ledger. Code ownership complements repository rules; it is
not itself an access-control boundary.

## Contribution intake

GitHub receives structured issue forms for bugs, features, PyTorch
compatibility, and performance reports, plus a pull-request template that
checks scope, tests, docs, compatibility evidence, safety, provenance, and
sign-off. Blank issues are disabled, while security reports are redirected to
the private reporting flow. Dependabot opens weekly Cargo and GitHub Actions
updates with conservative limits.

## Continuous integration

The Rust workflow runs on pull requests, pushes to `main`, and manual dispatch.
It grants `contents: read` by default, cancels superseded runs, and pins every
ordinary third-party action to a full immutable commit SHA with a version
comment.

Required checks cover:

- DCO sign-off for pull-request commits;
- dependency review on pull requests;
- compatibility-ledger freshness;
- formatting, checks, Clippy with warnings denied, tests, and rustdoc;
- the minimum supported Rust version 1.88 and stable Rust;
- the bootstrap CLI on Linux, macOS, and Windows;
- package file lists and archive safety; and
- a stable final `CI / required` result suitable for repository rules.

Python PyTorch CPU wheels are installed from PyTorch's CPU index separately
from ordinary Python dependencies. Documentation-only validation disables
runtime downloading.

## Releases and SLSA provenance

Release automation runs only for a `vX.Y.Z` tag. The tag, both Cargo package
versions, lockfile, changelog, and compatibility ledger must agree before any
asset is built.

The build job has read-only repository permissions and creates each final
`.crate` archive exactly once under `dist/`. It hashes those exact bytes and
uploads them once as workflow artifacts. The isolated provenance job delegates
to OpenSSF's Generic SLSA3 generator at
`slsa-framework/slsa-github-generator/.github/workflows/generator_generic_slsa3.yml@v2.1.0`
with only `actions: read`, `id-token: write`, and `contents: write`.

The semantic tag is a deliberate, documented exception to the full-SHA rule:
the generator's verification contract requires a tagged reusable workflow.
The release job downloads the same package bytes and provenance and uploads
them with the preinstalled GitHub CLI; it never rebuilds them. Release docs
include `slsa-verifier` instructions tied to the source repository and tag.

Publishing to crates.io remains a separate, protected step until a new secure
credential is configured. The previously exposed credential is never reused,
stored in GitHub, or copied into a command.

## Repository protections

After the workflows exist and have produced their check names, GitHub settings
enable Discussions, private vulnerability reporting, automatic branch cleanup,
and squash merging. Repository rules protect `main` and `v*` tags from deletion
and force-pushes, require pull requests and the stable required check on
`main`, and require code-owner review where the hosting plan permits it.

The initial sole maintainer retains an explicit emergency bypass so the
repository cannot be permanently locked before a second maintainer exists.
Every applied setting is read back and recorded in the maintainer guide.

## Acceptance

The foundation is complete when all committed links resolve, local equivalents
of required CI gates pass, the release workflow packages exact bytes and has a
valid SLSA job shape, community-health files are recognized by GitHub, public
repository settings are read back, and no secret or native runtime artifact is
present in the Git history or package archives.
