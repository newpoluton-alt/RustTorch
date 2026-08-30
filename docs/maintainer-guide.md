# Maintainer guide

This guide applies the project's [governance](../GOVERNANCE.md),
[contribution rules](../CONTRIBUTING.md), [security policy](../SECURITY.md),
and [Code of Conduct](../CODE_OF_CONDUCT.md).

## Triage and review

Classify new work as needs triage, changes requested, approved, deferred, or
closed. Confirm scope, reproduction, tests, documentation, compatibility
evidence, safety, provenance, DCO sign-off, and changelog impact. Use
CODEOWNERS to request review of sensitive paths, but remember that ownership
files are routing rules, not access control.

Ask for the smallest complete revision. Close duplicates and clearly
out-of-scope or unsafe work with a reason. Stale work may be closed after a
public notice and reasonable chance to respond; it can be reopened when the
missing evidence or implementation arrives.

## Policy and conflicts

Apply the same documented requirements to maintainer work. Disclose conflicts
and seek an independent reviewer when practical. Record material technical or
governance decisions in the relevant issue or pull request. Do not turn private
conduct or embargoed security information into public rationale.

## Security response

Keep vulnerability work in GitHub's private advisory until coordinated
disclosure. Establish affected versions, severity, dependency involvement,
tests, fix and backport scope, credits, and disclosure timing. Never copy
credentials, reporter identity, or unnecessary proof-of-concept details into a
public issue. Rotate exposed credentials rather than relying on Git history
rewrites.

## Releases

Only the maintainer authorizes releases. Before publication, reconcile both
Cargo package versions, `Cargo.lock`, `CHANGELOG.md`, compatibility metadata,
tests, package contents, and release notes. Build final package archives once
through the documented release process; never substitute rebuilt bytes after
provenance is generated. Keep publication credentials out of Git and pull
request execution.

## Repository administration

Inspect existing GitHub settings before changing them and preserve stricter
controls. Required status checks are configured only after their committed
workflows have reported stable names. While the project has one maintainer,
retain a documented owner emergency path without weakening deletion,
force-push, or required-check protection. Read back every applied setting and
record unavailable controls and recovery steps in the same policy change.

Weekly dependency pull requests are review inputs, not automatic approval.
Verify changelogs, licenses, lockfile scope, transitive changes, security
impact, MSRV, package size, and the complete test surface before merging.
