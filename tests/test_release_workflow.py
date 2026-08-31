import base64
import importlib.util
import io
import re
import tarfile
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/check-release.py"
WORKFLOW = ROOT / ".github/workflows/release.yml"

CHECKOUT = (
    "actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1 # v7.0.1"
)
SETUP_PYTHON = (
    "actions/setup-python@5fda3b95a4ea91299a34e894583c3862153e4b97 # v7.0.0"
)
SETUP_UV = (
    "astral-sh/setup-uv@c771a70e6277c0a99b617c7a806ffedaca235ff9 # v9.0.0"
)
SETUP_RUST = (
    "actions-rust-lang/setup-rust-toolchain@"
    "46268bd060767258de96ed93c1251119784f2ab6 # v1.16.1"
)
UPLOAD_ARTIFACT = (
    "actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02 # v4.6.2"
)
DOWNLOAD_ARTIFACT = (
    "actions/download-artifact@d3f86a106a0bac45b974a628896c90dbdf5c8093 # v4.3.0"
)
SLSA_GENERATOR = (
    "slsa-framework/slsa-github-generator/.github/workflows/"
    "generator_generic_slsa3.yml@v2.1.0"
)
PYTHON_LOCK_CHECK = "python3 scripts/check-python-lock.py"
APPROVED_RELEASE_ACTIONS = {
    value.split(" #", 1)[0]
    for value in (
        CHECKOUT,
        SETUP_PYTHON,
        SETUP_UV,
        SETUP_RUST,
        UPLOAD_ARTIFACT,
        DOWNLOAD_ARTIFACT,
    )
} | {SLSA_GENERATOR}

LIBRARY_ARCHIVE = bytes.fromhex(
    "1f8b08000000000002ffedcf310e41411405d0a9adc2067c6fe4875a42a9b1838f828448c657d8bda1"
    "12b548c439cd7db9cdcd2bd74bdf9fcb763f8a2637315e2fe78bd5b239edd2e744356ddb6756ef19915"
    "fee479ff32c266918e90beaff5da9f3e93f1d0f9bd295db20010000000000000000f053eee7ad92b600"
    "280000"
)
CLI_ARCHIVE = bytes.fromhex(
    "1f8b08000000000002ffedd1310ac2401005d0ad3d851748dc84d55e30a58d3790582828c21aefef6025"
    "d62288ef357ff8cd304cbddfa6e95ac763339e4f4d6ebb362f76c37ab31ddacb217d460eab529e19de33"
    "949739faae2fcb3ecd73fa82b87f5f637dfa4ff1f559020000000000000000e0e73c00c4164e890028"
    "0000"
)
EXPECTED_SUBJECTS = (
    b"105eb2e7eeab9fd3250acec60b43acb34a904f6788b0e305b7bd934e17abcca2"
    b"  rusttorch-0.1.0.crate\n"
    b"2dbed70cf9dc3f093d70cf9701f6f79d8bd9db29f50d2bd5cc4aff4e18cf5564"
    b"  rusttorch-cli-0.1.0.crate\n"
)


def load_release_module():
    spec = importlib.util.spec_from_file_location("check_release", SCRIPT)
    if spec is None or spec.loader is None:
        raise AssertionError(f"cannot load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ReleasePreflightTests(unittest.TestCase):
    def setUp(self) -> None:
        self.assertTrue(SCRIPT.is_file(), "scripts/check-release.py must exist")
        self.release = load_release_module()
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.write_valid_repository()

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def write(self, relative: str, contents: str) -> None:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents, encoding="utf-8")

    def write_valid_repository(self) -> None:
        self.write(
            "Cargo.toml",
            '[package]\nname = "rusttorch"\nversion = "0.1.0"\n',
        )
        self.write(
            "crates/rusttorch-cli/Cargo.toml",
            '[package]\nname = "rusttorch-cli"\nversion = "0.1.0"\n',
        )
        self.write(
            "Cargo.lock",
            'version = 4\n\n[[package]]\nname = "rusttorch"\nversion = "0.1.0"\n'
            '\n[[package]]\nname = "rusttorch-cli"\nversion = "0.1.0"\n',
        )
        self.write(
            "CHANGELOG.md",
            "# Changelog\n\n## Unreleased\n\n## 0.1.0 - 2026-08-30\n\n- Initial release.\n",
        )
        self.write("scripts/check-compatibility.py", "raise SystemExit(0)\n")

    def assert_invalid(self, message: str) -> None:
        with self.assertRaisesRegex(self.release.ReleaseError, message):
            self.release.validate_release(self.root, "v0.1.0")

    def test_matching_tag_manifests_lock_changelog_and_ledger_pass(self) -> None:
        self.assertEqual(self.release.validate_release(self.root, "v0.1.0"), "0.1.0")

    def test_release_tags_are_strict_stable_semver(self) -> None:
        for tag in (
            "0.1.0",
            "v01.1.0",
            "v1.01.0",
            "v1.1.00",
            "v1.2",
            "v1.2.3.4",
            "v1.2.3-rc.1",
            "v1.2.3+build",
            "v1.2.3\n",
        ):
            with self.subTest(tag=tag):
                with self.assertRaisesRegex(self.release.ReleaseError, "release tag"):
                    self.release.parse_tag(tag)

    def test_package_names_and_versions_must_match_the_tag(self) -> None:
        mutations = {
            "root package name": ("Cargo.toml", "rusttorch", "RustTorch"),
            "CLI package name": (
                "crates/rusttorch-cli/Cargo.toml",
                "rusttorch-cli",
                "rusttorch_setup",
            ),
            "CLI version": (
                "crates/rusttorch-cli/Cargo.toml",
                'version = "0.1.0"',
                'version = "0.1.1"',
            ),
        }
        for name, (relative, old, new) in mutations.items():
            with self.subTest(mutation=name):
                self.write_valid_repository()
                path = self.root / relative
                path.write_text(
                    path.read_text(encoding="utf-8").replace(old, new, 1),
                    encoding="utf-8",
                )
                with self.assertRaises(self.release.ReleaseError):
                    self.release.validate_release(self.root, "v0.1.0")

        self.write_valid_repository()
        with self.assertRaisesRegex(self.release.ReleaseError, "does not match"):
            self.release.validate_release(self.root, "v0.1.1")

    def test_lockfile_requires_one_exact_entry_per_workspace_package(self) -> None:
        lock_path = self.root / "Cargo.lock"
        lock_path.write_text(
            lock_path.read_text(encoding="utf-8")
            + '\n[[package]]\nname = "rusttorch"\nversion = "0.1.0"\n',
            encoding="utf-8",
        )
        self.assert_invalid("exactly one")

        self.write_valid_repository()
        lock_path.write_text(
            lock_path.read_text(encoding="utf-8").replace(
                'name = "rusttorch-cli"\nversion = "0.1.0"',
                'name = "rusttorch-cli"\nversion = "0.1.1"',
                1,
            ),
            encoding="utf-8",
        )
        self.assert_invalid("lockfile")

    def test_changelog_requires_one_exact_dated_release_heading(self) -> None:
        self.write(
            "CHANGELOG.md",
            "# Changelog\n\n## Unreleased\n\n## v0.1.0\n\n- Initial release.\n",
        )
        self.assert_invalid("changelog")

        self.write_valid_repository()
        changelog = self.root / "CHANGELOG.md"
        changelog.write_text(
            changelog.read_text(encoding="utf-8")
            + "\n## 0.1.0 - 2026-08-30\n\n- Duplicate.\n",
            encoding="utf-8",
        )
        self.assert_invalid("exactly one")

        self.write_valid_repository()
        changelog.write_text(
            changelog.read_text(encoding="utf-8").replace(
                "2026-08-30", "2026-02-30", 1
            ),
            encoding="utf-8",
        )
        self.assert_invalid("date")

    def test_compatibility_validation_failure_stops_the_release(self) -> None:
        self.write("scripts/check-compatibility.py", "raise SystemExit(7)\n")
        self.assert_invalid("compatibility")


class ReleaseSubjectTests(unittest.TestCase):
    def setUp(self) -> None:
        self.assertTrue(SCRIPT.is_file(), "scripts/check-release.py must exist")
        self.release = load_release_module()
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.dist = self.root / "dist"
        self.dist.mkdir()
        (self.dist / "rusttorch-0.1.0.crate").write_bytes(LIBRARY_ARCHIVE)
        (self.dist / "rusttorch-cli-0.1.0.crate").write_bytes(CLI_ARCHIVE)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_subjects_are_byte_exact_gnu_sha256_in_package_order(self) -> None:
        output = self.root / "subjects.txt"
        encoded = self.release.write_subjects(self.dist, "0.1.0", output)
        self.assertEqual(output.read_bytes(), EXPECTED_SUBJECTS)
        self.assertEqual(encoded, base64.b64encode(EXPECTED_SUBJECTS).decode("ascii"))

    def test_subject_generation_rejects_extra_or_missing_dist_entries(self) -> None:
        (self.dist / "unexpected.txt").write_text("stale", encoding="utf-8")
        with self.assertRaisesRegex(self.release.ReleaseError, "exactly"):
            self.release.write_subjects(
                self.dist, "0.1.0", self.root / "subjects.txt"
            )

        (self.dist / "unexpected.txt").unlink()
        (self.dist / "rusttorch-cli-0.1.0.crate").unlink()
        with self.assertRaisesRegex(self.release.ReleaseError, "exactly"):
            self.release.write_subjects(
                self.dist, "0.1.0", self.root / "subjects.txt"
            )

    def test_subject_generation_rejects_archive_symlinks(self) -> None:
        external_archive = self.root / "external.crate"
        external_archive.write_bytes(LIBRARY_ARCHIVE)
        archive_path = self.dist / "rusttorch-0.1.0.crate"
        archive_path.unlink()
        archive_path.symlink_to(external_archive)
        with self.assertRaisesRegex(self.release.ReleaseError, "regular file"):
            self.release.write_subjects(
                self.dist, "0.1.0", self.root / "subjects.txt"
            )

    def test_subject_generation_rejects_unsafe_archive_contents(self) -> None:
        archive_path = self.dist / "rusttorch-0.1.0.crate"
        with tarfile.open(archive_path, "w:gz") as archive:
            data = b"native"
            member = tarfile.TarInfo("rusttorch-0.1.0/.venv/libtorch.so")
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))
        with self.assertRaisesRegex(self.release.ReleaseError, "unsafe"):
            self.release.write_subjects(
                self.dist, "0.1.0", self.root / "subjects.txt"
            )

    def test_subject_generation_rejects_invalid_or_empty_archives(self) -> None:
        archive_path = self.dist / "rusttorch-0.1.0.crate"
        archive_path.write_bytes(b"not a tar archive")
        with self.assertRaisesRegex(self.release.ReleaseError, "archive"):
            self.release.write_subjects(
                self.dist, "0.1.0", self.root / "subjects.txt"
            )


class ReleaseWorkflowTests(unittest.TestCase):
    def read_workflow(self) -> str:
        self.assertTrue(WORKFLOW.is_file(), ".github/workflows/release.yml must exist")
        return WORKFLOW.read_text(encoding="utf-8")

    def jobs(self, text: str) -> dict[str, str]:
        jobs_text = text.split("\njobs:\n", 1)[1]
        matches = list(re.finditer(r"(?m)^  ([a-z0-9_-]+):$", jobs_text))
        return {
            match.group(1): jobs_text[
                match.start() : (
                    matches[index + 1].start()
                    if index + 1 < len(matches)
                    else len(jobs_text)
                )
            ]
            for index, match in enumerate(matches)
        }

    def assert_security_contract(self, text: str) -> None:
        normalized_text = re.sub(r"\\\r?\n[ \t]*", "", text)
        self.assertNotRegex(text, r'''(?i)["'][a-z_][a-z0-9_-]*["']\s*:''')
        self.assertNotRegex(text, r"(?m)(?:^\s*(?:-\s+)?|[{\[,]\s*)\?(?:\s|$)")
        expected_trigger = (
            "  push:\n    tags:\n      - \"v[0-9]+.[0-9]+.[0-9]+\"\n"
        )
        trigger_text = text.split("\non:\n", 1)[1].split("\npermissions:", 1)[0]
        self.assertEqual(trigger_text, expected_trigger)
        self.assertRegex(text, r"(?m)^permissions: \{\}$")
        self.assertNotRegex(text, r"(?i)\bsecrets\b")

        jobs = self.jobs(text)
        self.assertEqual(set(jobs), {"build", "provenance", "release"})
        jobs_text = text.split("\njobs:\n", 1)[1]
        self.assertEqual(
            re.findall(r"(?m)^  ([a-z0-9_-]+)\s*:", jobs_text),
            ["build", "provenance", "release"],
        )

        canonical_actions = re.findall(r"(?m)^\s+uses:\s+([^\s#]+)", text)
        all_uses = re.findall(
            r"(?i)(?<![a-z0-9_-])uses\s*:\s*([^\s,}\n]+)", text
        )
        self.assertEqual(all_uses, canonical_actions)
        self.assertEqual(set(canonical_actions), APPROVED_RELEASE_ACTIONS)
        self.assertEqual(len(canonical_actions), len(APPROVED_RELEASE_ACTIONS))
        ordinary_actions = [action for action in canonical_actions if action != SLSA_GENERATOR]
        for action in ordinary_actions:
            self.assertRegex(action, r"@[0-9a-f]{40}$")
        self.assertEqual(canonical_actions.count(SLSA_GENERATOR), 1)

        self.assertRegex(
            jobs["build"],
            r"(?m)^    permissions:\n      contents: read\n    outputs:\n",
        )
        self.assertRegex(
            jobs["provenance"],
            r"(?m)^    permissions:\n"
            r"      actions: read\n"
            r"      id-token: write\n"
            r"      contents: write\n"
            r"    uses:",
        )
        self.assertRegex(
            jobs["release"],
            r"(?m)^    permissions:\n      contents: write\n    runs-on:",
        )
        self.assertEqual(text.count("permissions:"), 4)
        self.assertEqual(text.count("persist-credentials: false"), 1)
        self.assertNotRegex(text, r"(?i)(?:^|/)cache@")
        self.assertEqual(
            len(re.findall(r"(?i)(?<![a-z0-9_-])enable-cache\s*:", text)), 1
        )
        self.assertEqual(
            len(re.findall(r"(?m)^\s+enable-cache: false$", text)), 1
        )
        self.assertEqual(
            len(re.findall(r"(?i)(?<![a-z0-9_-])cache\s*:", text)), 1
        )
        self.assertEqual(len(re.findall(r"(?m)^\s+cache: false$", text)), 1)
        self.assertNotRegex(normalized_text, r"(?i)\bpip(?:3)?\b")
        self.assertEqual(len(re.findall(r"(?i)\buv\s+lock\b", normalized_text)), 1)
        self.assertEqual(len(re.findall(r"(?i)\buv\s+sync\b", normalized_text)), 1)
        self.assertNotIn("cargo publish", normalized_text)
        self.assertNotIn("gh release create", normalized_text)
        self.assertNotIn("GH_TOKEN:", jobs["build"] + jobs["provenance"])
        self.assertNotIn("GH_REPO:", jobs["build"] + jobs["provenance"])
        lock_policy_step = (
            "      - name: Validate Python lock artifacts\n"
            f"        run: {PYTHON_LOCK_CHECK}\n"
        )
        lock_policy_boundary = (
            lock_policy_step
            + "\n      - name: Set up UV\n"
            + f"        uses: {SETUP_UV}\n"
        )
        build = jobs["build"]
        normalized_build = re.sub(r"\\\r?\n[ \t]*", "", build)
        self.assertEqual(build.count(lock_policy_step), 1)
        self.assertEqual(build.count(lock_policy_boundary), 1)
        self.assertEqual(build.count("scripts/check-python-lock.py"), 1)
        self.assertEqual(normalized_build.count(PYTHON_LOCK_CHECK), 1)
        self.assertLess(
            build.index(f"uses: {SETUP_PYTHON}"), build.index(lock_policy_boundary)
        )
        self.assertLess(
            build.index(lock_policy_boundary), build.index(f"uses: {SETUP_UV}")
        )
        self.assertLess(
            build.index(lock_policy_boundary),
            build.index("uv lock --check --offline --no-cache"),
        )
        self.assertLess(
            build.index(lock_policy_boundary),
            build.index("uv sync --frozen --no-cache"),
        )
        normalized_release = re.sub(r"\\\r?\n[ \t]*", "", jobs["release"])
        self.assertNotRegex(normalized_release, r"(?i)\bcargo\s")
        self.assertNotIn("scripts/", jobs["release"])
        self.assertNotIn("actions/checkout@", jobs["release"])

    def test_release_runs_only_for_tags_without_cross_tag_cancellation(self) -> None:
        text = self.read_workflow()
        self.assert_security_contract(text)
        self.assertRegex(text, r"(?m)^name: Release provenance$")
        self.assertIn(
            "concurrency:\n"
            "  group: release-${{ github.ref }}\n"
            "  cancel-in-progress: false\n",
            text,
        )

    def test_build_uses_locked_python_and_packages_each_archive_once(self) -> None:
        text = self.read_workflow()
        build = self.jobs(text)["build"]
        self.assertIn('LIBTORCH_USE_PYTORCH: "1"', build)
        for action in (CHECKOUT, SETUP_PYTHON, SETUP_UV, SETUP_RUST, UPLOAD_ARTIFACT):
            self.assertIn(f"uses: {action}", build)
        self.assertIn('python-version: "3.14"', build)
        lock_policy_step = (
            "      - name: Validate Python lock artifacts\n"
            f"        run: {PYTHON_LOCK_CHECK}\n"
        )
        self.assertEqual(build.count(lock_policy_step), 1)
        self.assertEqual(build.count("scripts/check-python-lock.py"), 1)
        self.assertIn('version: "0.12.3"', build)
        self.assertIn("enable-cache: false", build)
        self.assertIn("cache: false", build)
        self.assertIn("uv lock --check --offline --no-cache", build)
        self.assertIn("uv sync --frozen --no-cache", build)
        self.assertLess(build.index(f"uses: {SETUP_PYTHON}"), build.index(lock_policy_step))
        self.assertLess(build.index(lock_policy_step), build.index(f"uses: {SETUP_UV}"))
        self.assertLess(
            build.index(lock_policy_step),
            build.index("uv lock --check --offline --no-cache"),
        )
        self.assertLess(
            build.index(lock_policy_step),
            build.index("uv sync --frozen --no-cache"),
        )
        self.assertIn('echo "$PWD/.venv/bin" >> "$GITHUB_PATH"', build)
        self.assertIn('echo "VIRTUAL_ENV=$PWD/.venv" >> "$GITHUB_ENV"', build)
        self.assertIn("LD_LIBRARY_PATH", build)
        self.assertIn(
            '.venv/bin/python scripts/check-release.py --tag "$GITHUB_REF_NAME"',
            build,
        )
        self.assertIn("test ! -e dist", build)
        self.assertIn("mkdir dist", build)
        for package in ("rusttorch", "rusttorch-cli"):
            self.assertEqual(
                len(
                    re.findall(
                        rf"cargo package -p {re.escape(package)} --locked(?:\s|$)",
                        re.sub(r"\\\r?\n[ \t]*", "", build),
                    )
                ),
                1,
            )
        self.assertIn('target/package/rusttorch-$VERSION.crate', build)
        self.assertIn('target/package/rusttorch-cli-$VERSION.crate', build)
        self.assertIn(
            '--dist dist --subjects-output dist/subjects.txt',
            build,
        )
        self.assertIn('test -n "$base64_subjects"', build)
        self.assertIn('echo "base64-subjects=$base64_subjects" >> "$GITHUB_OUTPUT"', build)
        upload = build.split(f"uses: {UPLOAD_ARTIFACT}", 1)[1]
        self.assertIn("name: release-packages", upload)
        self.assertIn("dist/rusttorch-${{ steps.preflight.outputs.version }}.crate", upload)
        self.assertIn(
            "dist/rusttorch-cli-${{ steps.preflight.outputs.version }}.crate", upload
        )
        self.assertIn("dist/subjects.txt", upload)
        self.assertIn("if-no-files-found: error", upload)
        self.assertIn("compression-level: 0", upload)
        self.assertIn("overwrite: false", upload)
        self.assertIn("include-hidden-files: false", upload)
        self.assertRegex(upload, r"(?m)^          retention-days: [1-9][0-9]?$",)

    def test_provenance_has_only_the_required_oidc_contract(self) -> None:
        text = self.read_workflow()
        provenance = self.jobs(text)["provenance"]
        self.assertRegex(provenance, r"(?m)^    needs: build$")
        self.assertIn(f"uses: {SLSA_GENERATOR}", provenance)
        self.assertIn(
            "base64-subjects: ${{ needs.build.outputs.base64-subjects }}", provenance
        )
        self.assertIn(
            "provenance-name: rusttorch-${{ needs.build.outputs.version }}.intoto.jsonl",
            provenance,
        )
        self.assertIn("upload-assets: true", provenance)
        self.assertIn('draft-release: "true"', provenance)
        self.assertNotIn("runs-on:", provenance)
        self.assertNotIn("steps:", provenance)

    def test_release_downloads_revalidates_and_publishes_last(self) -> None:
        text = self.read_workflow()
        release = self.jobs(text)["release"]
        self.assertIn("needs:\n      - build\n      - provenance\n", release)
        self.assertIn(f"uses: {DOWNLOAD_ARTIFACT}", release)
        download = release.split(f"uses: {DOWNLOAD_ARTIFACT}", 1)[1].split(
            "\n      - name:", 1
        )[0]
        self.assertIn("name: release-packages", download)
        self.assertIn("path: dist", download)
        self.assertIn("sha256sum --check subjects.txt", release)
        self.assertIn("base64 -w 0 subjects.txt", release)
        self.assertIn('test "$actual_base64" = "$EXPECTED_BASE64_SUBJECTS"', release)
        self.assertIn("GH_TOKEN: ${{ github.token }}", release)
        self.assertIn("GH_REPO: ${{ github.repository }}", release)
        draft_check = (
            'test "$(gh release view "$GITHUB_REF_NAME" --json isDraft --jq .isDraft)" '
            '= "true"'
        )
        self.assertIn(draft_check, release)
        upload = 'gh release upload "$GITHUB_REF_NAME" dist/*.crate --clobber'
        self.assertIn(upload, release)
        self.assertIn(".assets[].name", release)
        for asset in (
            "rusttorch-$VERSION.crate",
            "rusttorch-cli-$VERSION.crate",
            "rusttorch-$VERSION.intoto.jsonl",
        ):
            self.assertIn(asset, release)
        self.assertIn('test "$actual_assets" = "$expected_assets"', release)
        publish = 'gh release edit "$GITHUB_REF_NAME" --draft=false'
        self.assertLess(release.index(draft_check), release.index(upload))
        self.assertLess(release.index(upload), release.index("actual_assets="))
        self.assertLess(release.index("actual_assets="), release.index(publish))
        self.assertEqual(text.rstrip().splitlines()[-1].strip(), publish)

    def test_release_security_contract_fails_closed_on_workflow_drift(self) -> None:
        text = self.read_workflow()
        lock_policy_step = (
            "      - name: Validate Python lock artifacts\n"
            f"        run: {PYTHON_LOCK_CHECK}\n"
        )
        setup_uv_step = f'''      - name: Set up UV
        uses: {SETUP_UV}
        with:
          version: "0.12.3"
          enable-cache: false
'''
        late_policy = text.replace(lock_policy_step, "", 1)
        late_policy = late_policy.replace(
            setup_uv_step, setup_uv_step + "\n" + lock_policy_step, 1
        )
        mutations = {
            "broad tag trigger": text.replace(
                '"v[0-9]+.[0-9]+.[0-9]+"',
                '"v[0-9]*.[0-9]*.[0-9]*"',
                1,
            ),
            "pull request trigger": text.replace(
                "  push:\n", "  pull_request:\n  push:\n", 1
            ),
            "manual trigger": text.replace(
                "  push:\n", "  workflow_dispatch:\n  push:\n", 1
            ),
            "floating ordinary action": text.replace(CHECKOUT, "actions/checkout@v7", 1),
            "second generator": text.replace(
                f"uses: {DOWNLOAD_ARTIFACT}",
                f"uses: {SLSA_GENERATOR}",
                1,
            ),
            "unapproved flow action": text.replace(
                "    steps:\n", "    steps:\n      - {name: Backdoor, uses: evil/action@main}\n", 1
            ),
            "quoted flow action": text.replace(
                "    steps:\n",
                '    steps:\n      - {name: Backdoor, "uses": evil/action@main}\n',
                1,
            ),
            "extra flow job": text.replace(
                "  release:\n",
                "  shadow: {runs-on: ubuntu-latest}\n\n  release:\n",
                1,
            ),
            "quoted job": text.replace(
                "  release:\n",
                '  "shadow": {runs-on: ubuntu-latest}\n\n  release:\n',
                1,
            ),
            "secret context": text.replace(
                "name: Release provenance",
                "name: Release provenance\n# ${{ secrets.CARGO_TOKEN }}",
                1,
            ),
            "broad build permission": text.replace("contents: read", "contents: write", 1),
            "inherited reusable secrets": text.replace(
                "    with:\n      base64-subjects:",
                "    secrets: inherit\n    with:\n      base64-subjects:",
                1,
            ),
            "persisted checkout credential": text.replace(
                "persist-credentials: false", "persist-credentials: true", 1
            ),
            "UV cache": text.replace("enable-cache: false", "enable-cache: true", 1),
            "Rust cache": text.replace("cache: false", "cache: true", 1),
            "pip install": text.replace(
                "uv sync --frozen --no-cache",
                "uv sync --frozen --no-cache\n          python -m pip install wheel",
                1,
            ),
            "continued pip install": text.replace(
                "uv sync --frozen --no-cache",
                "uv sync --frozen --no-cache\n          python -m p\\\n            ip install wheel",
                1,
            ),
            "missing artifact policy": text.replace(lock_policy_step, "", 1),
            "artifact policy after setup UV": late_policy,
            "artifact policy through venv": text.replace(
                PYTHON_LOCK_CHECK,
                ".venv/bin/python scripts/check-python-lock.py",
                1,
            ),
            "continued artifact policy": text.replace(
                PYTHON_LOCK_CHECK,
                "python3 scripts/check-python-\\\n          lock.py",
                1,
            ),
            "aliased artifact policy": text.replace(
                f"        run: {PYTHON_LOCK_CHECK}",
                "        run: |\n"
                "          alias python3=true\n"
                f"          {PYTHON_LOCK_CHECK}",
                1,
            ),
            "ignored artifact policy failure": text.replace(
                PYTHON_LOCK_CHECK, f"{PYTHON_LOCK_CHECK} || true", 1
            ),
            "artifact policy continue on error": text.replace(
                lock_policy_step,
                lock_policy_step + "        continue-on-error: true\n",
                1,
            ),
            "artifact policy false condition": text.replace(
                lock_policy_step,
                lock_policy_step + "        if: ${{ false }}\n",
                1,
            ),
            "artifact policy neutralized shell": text.replace(
                lock_policy_step,
                lock_policy_step + "        shell: echo {0}\n",
                1,
            ),
            "duplicate artifact policy neutralizer": text.replace(
                lock_policy_step,
                lock_policy_step
                + "        continue-on-error: false\n"
                + "        continue-on-error: true\n",
                1,
            ),
            "flow artifact policy neutralizer": text.replace(
                lock_policy_step,
                "      - {name: Validate Python lock artifacts, "
                f"run: {PYTHON_LOCK_CHECK}, continue-on-error: true}}\n",
                1,
            ),
            "explicit artifact policy neutralizer": text.replace(
                lock_policy_step,
                lock_policy_step
                + "        ? continue-on-error # neutralize validation\n"
                + "        : true\n",
                1,
            ),
            "quoted artifact policy key": text.replace(
                f"        run: {PYTHON_LOCK_CHECK}",
                f'        "run": {PYTHON_LOCK_CHECK}',
                1,
            ),
            "explicit artifact policy key": text.replace(
                f"        run: {PYTHON_LOCK_CHECK}",
                "        ? run # validate before environment creation\n"
                f"        : {PYTHON_LOCK_CHECK}",
                1,
            ),
            "second unlocked sync": text.replace(
                "uv sync --frozen --no-cache",
                "uv sync --frozen --no-cache\n          uv sync",
                1,
            ),
            "Cargo publish": text.replace(
                "cargo package -p rusttorch --locked",
                "cargo package -p rusttorch --locked\n          cargo publish -p rusttorch",
                1,
            ),
            "second GitHub release": text.replace(
                "gh release upload", "gh release create \"$GITHUB_REF_NAME\"\n          gh release upload", 1
            ),
            "release rebuild": text.replace(
                "sha256sum --check subjects.txt",
                "cargo package -p rusttorch --locked\n          sha256sum --check subjects.txt",
                1,
            ),
            "release checkout": text.replace(
                f"      - name: Download package artifact\n        uses: {DOWNLOAD_ARTIFACT}",
                f"      - name: Check out source\n        uses: {CHECKOUT}\n\n"
                f"      - name: Download package artifact\n        uses: {DOWNLOAD_ARTIFACT}",
                1,
            ),
        }
        for name, mutation in mutations.items():
            with self.subTest(mutation=name):
                self.assertNotEqual(mutation, text)
                with self.assertRaises(AssertionError):
                    self.assert_security_contract(mutation)


if __name__ == "__main__":
    unittest.main()
