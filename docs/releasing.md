# Release provenance

RustTorch releases use a tag-only GitHub Actions workflow to package the two
Cargo crates once, generate provenance for those exact archive bytes, and
publish a GitHub release only after every expected asset is present. The
workflow does not publish to crates.io and does not receive a crates.io
credential.

## Maintainer sequence

Only the maintainer may authorize a release. Work from a clean, reviewed
commit on `main` and complete the project-wide release gate before changing
any public registry state.

1. Set the same stable `X.Y.Z` version in `rusttorch`, `rusttorch-cli`, and
   `Cargo.lock`. Add exactly one `## X.Y.Z - YYYY-MM-DD` changelog heading.
2. Run the complete CI-equivalent test, documentation, compatibility, parity,
   and package checks. Locally confirm the tag metadata contract with:

   ```sh
   .venv/bin/python scripts/check-release.py --tag vX.Y.Z
   ```

3. Record the exact release commit. Publish `rusttorch-cli` and then
   `rusttorch` to crates.io from that commit using a newly configured,
   protected credential. Never reuse an exposed credential.
4. Confirm both versions are public before creating and pushing the immutable
   `vX.Y.Z` tag for the recorded commit. Do not create a GitHub release by
   hand; the tag workflow owns it.
5. Require the `Release provenance` workflow to finish successfully and check
   that the published release has exactly these three assets:

   ```text
   rusttorch-X.Y.Z.crate
   rusttorch-cli-X.Y.Z.crate
   rusttorch-X.Y.Z.intoto.jsonl
   ```

The build job uses the committed `pyproject.toml` and `uv.lock` with frozen,
cache-free synchronization, then packages each Cargo crate exactly once. Its
standard-library release checker rejects mismatched metadata, stale
compatibility documentation, unexpected archive names, unsafe archive
members, and generated or native environment files. It writes two GNU-format
SHA-256 subject lines in a deterministic order.

The OpenSSF Generic Generator creates a draft release and uploads the signed
provenance. A separate job downloads the single package artifact without a
source checkout, rechecks both SHA-256 digests and the byte-exact base64
subjects, uploads the same two `.crate` files, and verifies the complete asset
set. Publishing the draft is its final command. A failure before that command
leaves an unpublished draft for diagnosis; never move the tag or replace the
attested bytes. If the tagged source is wrong, leave that version unpublished
and prepare a new patch version.

## Verify downloaded assets

Install [`slsa-verifier`](https://github.com/slsa-framework/slsa-verifier)
v2.7.1 from its checksummed release binary, or compile that exact version with
Go:

```sh
go install github.com/slsa-framework/slsa-verifier/v2/cli/slsa-verifier@v2.7.1
```

Download all three release assets into one directory, then verify both crates
in one invocation:

```sh
slsa-verifier verify-artifact \
  rusttorch-X.Y.Z.crate rusttorch-cli-X.Y.Z.crate \
  --provenance-path rusttorch-X.Y.Z.intoto.jsonl \
  --source-uri github.com/newpoluton-alt/RustTorch \
  --source-tag vX.Y.Z
```

Verification covers the exact `.crate` files downloaded from the GitHub
release. GitHub's automatically generated source archives are not provenance
subjects. A crates.io download is also not covered by this statement unless
its SHA-256 digest is first shown to equal the corresponding attested GitHub
release asset.

## Generator maintenance status

The requested
[`slsa-github-generator`](https://github.com/slsa-framework/slsa-github-generator)
Generic Generator is no longer actively maintained. Its `@v2.1.0` semantic
tag is the sole exception to this repository's immutable-SHA action policy
because the generator's verification contract requires a release tag. This
workflow produces verifiable provenance for its two named subjects, but the
project does not make a broader SLSA compliance claim.

GitHub recommends
[`artifact attestations`](https://docs.github.com/en/actions/concepts/security/artifact-attestations)
as the migration path. Maintainers should track that path and plan a reviewed
transition before a GitHub Actions runtime deprecation affects v2.1.0. That
migration changes consumer verification from `slsa-verifier` to
`gh attestation verify`, so documentation and release gates must change
together.
