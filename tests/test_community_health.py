import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

CHECKOUT = (
    "actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1 # v7.0.1"
)
SETUP_PYTHON = (
    "actions/setup-python@5fda3b95a4ea91299a34e894583c3862153e4b97 # v7.0.0"
)
SETUP_RUST = (
    "actions-rust-lang/setup-rust-toolchain@"
    "46268bd060767258de96ed93c1251119784f2ab6 # v1.16.1"
)
DEPENDENCY_REVIEW = (
    "actions/dependency-review-action@"
    "595b5aeba73380359d98a5e087f648dbb0edce1b # v4.7.3"
)
APPROVED_CI_ACTIONS = {
    action.split(" #", 1)[0]
    for action in (CHECKOUT, SETUP_PYTHON, SETUP_RUST, DEPENDENCY_REVIEW)
}

REQUIRED_FILES = (
    "CODE_OF_CONDUCT.md",
    "DCO.md",
    "SECURITY.md",
    "SUPPORT.md",
    "GOVERNANCE.md",
    "docs/maintainer-guide.md",
    ".github/CODEOWNERS",
    ".github/ISSUE_TEMPLATE/bug.yml",
    ".github/ISSUE_TEMPLATE/feature.yml",
    ".github/ISSUE_TEMPLATE/compatibility.yml",
    ".github/ISSUE_TEMPLATE/performance.yml",
    ".github/ISSUE_TEMPLATE/config.yml",
    ".github/pull_request_template.md",
    ".github/dependabot.yml",
)

ISSUE_FORMS = {
    "bug.yml": (
        "Bug report",
        "Report a reproducible RustTorch defect",
        ("version", "environment", "backend", "reproduction", "expected", "actual", "acknowledgements"),
    ),
    "feature.yml": (
        "Feature request",
        "Propose a scoped RustTorch improvement",
        ("problem", "proposal", "scope", "alternatives", "pytorch_reference", "acknowledgements"),
    ),
    "compatibility.yml": (
        "PyTorch compatibility",
        "Report a scoped PyTorch compatibility gap",
        ("pytorch_symbols", "pytorch_source", "rusttorch_scope", "backend", "evidence", "acknowledgements"),
    ),
    "performance.yml": (
        "Performance report",
        "Report a reproducible RustTorch performance issue",
        ("summary", "environment", "benchmark", "methodology", "results", "acknowledgements"),
    ),
}


class CommunityHealthTests(unittest.TestCase):
    def read(self, relative: str) -> str:
        return (ROOT / relative).read_text(encoding="utf-8")

    def assert_ci_security_contract(self, text: str) -> None:
        self.assertNotRegex(text, r'''(?i)["'][a-z_][a-z0-9_-]*["']\s*:''')
        self.assertRegex(text, r"(?m)^name: CI$")
        jobs_text = text.split("\njobs:\n", 1)[1]
        job_matches = list(re.finditer(r"(?m)^  ([a-z0-9_-]+):$", jobs_text))
        canonical_job_names = [match.group(1) for match in job_matches]
        self.assertEqual(
            re.findall(r"(?m)^  ([a-z0-9_-]+)\s*:", jobs_text),
            canonical_job_names,
        )
        jobs = {
            match.group(1): jobs_text[
                match.start() : (
                    job_matches[index + 1].start()
                    if index + 1 < len(job_matches)
                    else len(jobs_text)
                )
            ]
            for index, match in enumerate(job_matches)
        }
        self.assertEqual(
            set(jobs),
            {"dco", "dependency-review", "quality", "msrv", "cli", "required"},
        )
        self.assertRegex(jobs["required"], r"(?m)^    name: required$")

        actions = re.findall(r"(?m)^\s+uses:\s+([^\s#]+)", text)
        self.assertTrue(actions)
        self.assertEqual(set(actions), APPROVED_CI_ACTIONS)
        self.assertEqual(
            len(re.findall(r"(?i)(?<![a-z0-9_-])uses\s*:", text)),
            len(actions),
        )
        self.assertNotRegex(text, r"(?i)\bsecrets\b")
        cache_key_pattern = r"(?i)(?<![a-z0-9_-])cache(?:-[a-z0-9_-]+)?\s*:"
        setup_rust_sections = text.split(f"uses: {SETUP_RUST}")
        self.assertEqual(len(setup_rust_sections) - 1, 3)
        for section in setup_rust_sections[1:]:
            step = section.split("\n      - name:", 1)[0]
            self.assertEqual(len(re.findall(cache_key_pattern, step)), 1)
            self.assertRegex(step, r"(?m)^          cache: false$")
        self.assertEqual(len(re.findall(cache_key_pattern, text)), 3)
        self.assertEqual(text.count("          cache: false"), 3)
        self.assertNotRegex(text, r"(?i)(?:^|/)cache@")
        self.assertNotRegex(text, r"(?m)^\s*[^#\n]*:\s*write(?:-all)?\s*$")
        self.assertNotRegex(text, r"(?m)^\s*permissions:.*\bwrite\b")
        self.assertNotRegex(
            text,
            r'''(?i)(?<![a-z0-9_-])[a-z][a-z0-9_-]*\s*:\s*["']?'''
            r'''write(?:-all)?["']?(?=[\s,}#]|$)''',
        )

        self.assertEqual(text.count("permissions:"), 3)
        self.assertRegex(
            text,
            r"(?m)^permissions:\n  contents: read\n\nconcurrency:\n",
        )
        for job_name, section in jobs.items():
            if job_name in {"dco", "dependency-review"}:
                self.assertIn(
                    "    permissions:\n      contents: read\n    steps:\n",
                    section,
                )
                self.assertEqual(section.count("permissions:"), 1)
            else:
                self.assertNotIn("permissions:", section)

    def test_required_files_exist(self) -> None:
        for relative in REQUIRED_FILES:
            with self.subTest(path=relative):
                self.assertTrue((ROOT / relative).is_file(), relative)

    def test_issue_forms_have_identity_and_required_fields(self) -> None:
        forms = ROOT / ".github/ISSUE_TEMPLATE"
        for filename, (name, description, field_ids) in ISSUE_FORMS.items():
            with self.subTest(form=filename):
                text = (forms / filename).read_text(encoding="utf-8")
                self.assertRegex(text, rf"(?m)^name: {re.escape(name)}$")
                self.assertRegex(text, rf"(?m)^description: {re.escape(description)}$")
                for field_id in field_ids:
                    self.assertRegex(text, rf"(?m)^\s+id: {field_id}$")
                self.assertRegex(text, r"(?m)^\s+id: acknowledgements$")
                self.assertIn("Code of Conduct", text)
                self.assertIn("security", text.lower())
                self.assertGreaterEqual(text.count("required: true"), len(field_ids))

    def test_issue_intake_disables_blank_reports_and_routes_security(self) -> None:
        text = self.read(".github/ISSUE_TEMPLATE/config.yml")
        self.assertRegex(text, r"(?m)^blank_issues_enabled: false$")
        self.assertIn("https://github.com/newpoluton-alt/RustTorch/security", text)

    def test_codeowners_covers_sensitive_policy(self) -> None:
        text = self.read(".github/CODEOWNERS")
        for pattern in (
            "* @newpoluton-alt",
            "/.github/** @newpoluton-alt",
            "/.github/workflows/** @newpoluton-alt",
            "/.github/CODEOWNERS @newpoluton-alt",
            "/SECURITY.md @newpoluton-alt",
            "/GOVERNANCE.md @newpoluton-alt",
            "/docs/releasing.md @newpoluton-alt",
            "/LICENSE-MIT @newpoluton-alt",
            "/LICENSE-APACHE @newpoluton-alt",
            "/compat/pytorch_api.toml @newpoluton-alt",
        ):
            with self.subTest(pattern=pattern):
                self.assertIn(pattern, text)

    def test_contributing_links_the_project_contract(self) -> None:
        text = self.read("CONTRIBUTING.md")
        for link in (
            "[Developer Certificate of Origin](DCO.md)",
            "[security policy](SECURITY.md)",
            "[Code of Conduct](CODE_OF_CONDUCT.md)",
            "[governance](GOVERNANCE.md)",
            "[support policy](SUPPORT.md)",
        ):
            with self.subTest(link=link):
                self.assertIn(link, text)

    def test_dependabot_checks_cargo_and_actions_weekly(self) -> None:
        text = self.read(".github/dependabot.yml")
        self.assertRegex(text, r"(?m)^version: 2$")
        self.assertIn('package-ecosystem: "cargo"', text)
        self.assertIn('package-ecosystem: "github-actions"', text)
        self.assertEqual(text.count('interval: "weekly"'), 2)
        self.assertEqual(text.count("open-pull-requests-limit: 5"), 2)

    def test_pull_request_template_covers_review_contract(self) -> None:
        text = self.read(".github/pull_request_template.md")
        for phrase in (
            "Signed-off-by",
            "compatibility ledger",
            "security",
            "provenance",
            "documentation",
        ):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, text)

    def test_relative_markdown_links_resolve(self) -> None:
        paths = (
            "README.md",
            "CONTRIBUTING.md",
            "CODE_OF_CONDUCT.md",
            "DCO.md",
            "SECURITY.md",
            "SUPPORT.md",
            "GOVERNANCE.md",
            "docs/maintainer-guide.md",
            ".github/pull_request_template.md",
        )
        for relative in paths:
            path = ROOT / relative
            text = path.read_text(encoding="utf-8")
            for target in re.findall(r"\[[^]]+\]\(([^)]+)\)", text):
                if target.startswith(("https://", "http://", "#")):
                    continue
                destination = path.parent / target.split("#", 1)[0]
                with self.subTest(source=relative, target=target):
                    self.assertTrue(destination.is_file(), destination)

    def test_policy_files_have_no_placeholders_or_invented_email(self) -> None:
        paths = [ROOT / relative for relative in REQUIRED_FILES]
        paths.extend((ROOT / "CONTRIBUTING.md", ROOT / "README.md"))
        for path in paths:
            if not path.is_file():
                continue
            text = path.read_text(encoding="utf-8")
            with self.subTest(path=path.relative_to(ROOT)):
                self.assertNotRegex(text, r"\b(?:INSERT|TODO|TBD|CHANGEME)\b")
                self.assertNotRegex(
                    text,
                    r"[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
                )

    def test_ci_uses_least_privilege_and_immutable_actions(self) -> None:
        text = self.read(".github/workflows/ci.yml")
        self.assert_ci_security_contract(text)
        self.assertRegex(text, r"(?m)^permissions:\n  contents: read$")
        for action in (CHECKOUT, SETUP_PYTHON, SETUP_RUST, DEPENDENCY_REVIEW):
            with self.subTest(action=action):
                self.assertIn(f"uses: {action}", text)
        self.assertEqual(
            text.count("persist-credentials: false"), text.count(f"uses: {CHECKOUT}")
        )
        self.assertIn("comment-summary-in-pr: never", text)
        dependency_job = text.split("  dependency-review:\n", 1)[1].split("\n  quality:", 1)[0]
        self.assertRegex(dependency_job, r"(?m)^    permissions:\n      contents: read$")
        self.assertNotIn("pull-requests: write", dependency_job)

    def test_ci_security_contract_rejects_privilege_and_identity_drift(self) -> None:
        text = self.read(".github/workflows/ci.yml")
        mutations = {
            "floating action": text.replace(CHECKOUT, "actions/checkout@v7", 1),
            "flow action": text.replace(
                "    steps:\n",
                "    steps:\n      - { name: Unapproved, uses: evil/example@main }\n",
                1,
            ),
            "quoted flow action": text.replace(
                "    steps:\n",
                '    steps:\n      - { name: Unapproved, "uses": evil/example@main }\n',
                1,
            ),
            "secret context": text.replace(
                "name: CI", "name: CI\n# ${{ secrets.CARGO_TOKEN }}", 1
            ),
            "spaced secret context": text.replace(
                "        env:\n",
                '        env:\n          TOKEN: ${{ secrets ["CARGO_TOKEN"] }}\n',
                1,
            ),
            "cache config": text.replace(
                "          toolchain: stable",
                "          toolchain: stable\n          cache: cargo",
                1,
            ),
            "Rust action cache enabled": text.replace(
                "          cache: false", "          cache: true", 1
            ),
            "flow cache config": text.replace(
                "        with:\n          python-version: \"3.14\"",
                '        with: { python-version: "3.14", cache: pip }',
                1,
            ),
            "quoted flow cache config": text.replace(
                "        with:\n          python-version: \"3.14\"",
                '        with: { python-version: "3.14", "cache": pip }',
                1,
            ),
            "write permission": text.replace("contents: read", "contents: write", 1),
            "flow write permission": text.replace(
                "    permissions:\n      contents: read\n    steps:",
                "    permissions: { contents: write }\n    steps:",
                1,
            ),
            "quoted flow write permission": text.replace(
                "\n  dependency-review:\n",
                '\n    "permissions": { "contents": "write" }\n\n  dependency-review:\n',
                1,
            ),
            "extra read permission": text.replace(
                "    permissions:\n      contents: read\n    steps:",
                "    permissions:\n      contents: read\n      issues: read\n    steps:",
                1,
            ),
            "workflow rename": text.replace("name: CI", "name: Build", 1),
            "required rename": text.replace("    name: required", "    name: optional", 1),
            "quoted flow job": text.replace(
                "\n  quality:\n",
                '\n  "shadow": { name: Shadow, runs-on: ubuntu-latest }\n\n  quality:\n',
                1,
            ),
            "unquoted flow job": text.replace(
                "\n  quality:\n",
                "\n  shadow: { name: Shadow, runs-on: ubuntu-latest }\n\n  quality:\n",
                1,
            ),
        }
        for name, mutation in mutations.items():
            with self.subTest(mutation=name):
                with self.assertRaises(AssertionError):
                    self.assert_ci_security_contract(mutation)

    def test_ci_checks_dco_dependency_changes_msrv_and_cli_platforms(self) -> None:
        text = self.read(".github/workflows/ci.yml")
        for job in ("dco", "dependency-review", "quality", "msrv", "cli", "required"):
            self.assertRegex(text, rf"(?m)^  {re.escape(job)}:$")
        dco_job = text.split("  dco:\n", 1)[1].split("\n  dependency-review:", 1)[0]
        self.assertIn("if: github.event_name == 'pull_request'", dco_job)
        self.assertIn("fetch-depth: 0", dco_job)
        self.assertIn("github.event.pull_request.base.sha", dco_job)
        self.assertIn("github.event.pull_request.head.sha", dco_job)
        self.assertIn("github.actor == 'dependabot[bot]'", dco_job)
        self.assertIn("--trusted-dependabot-actor", dco_job)
        dependency_job = text.split("  dependency-review:\n", 1)[1].split("\n  quality:", 1)[0]
        self.assertIn("if: github.event_name == 'pull_request'", dependency_job)
        msrv_job = text.split("  msrv:\n", 1)[1].split("\n  cli:", 1)[0]
        self.assertIn('toolchain: "1.88"', msrv_job)
        self.assertIn("cargo check -p rusttorch ", msrv_job)
        self.assertIn("cargo check -p rusttorch-cli ", msrv_job)
        cli_job = text.split("  cli:\n", 1)[1].split("\n  required:", 1)[0]
        for operating_system in ("ubuntu-latest", "macos-latest", "windows-latest"):
            self.assertIn(operating_system, cli_job)
        self.assertIn("cargo test -p rusttorch-cli --all-targets --locked", cli_job)

    def test_ci_preserves_real_parity_docs_and_archive_verification(self) -> None:
        text = self.read(".github/workflows/ci.yml")
        quality = text.split("  quality:\n", 1)[1].split("\n  msrv:", 1)[0]
        self.assertIn('python-version: "3.14"', quality)
        self.assertIn("python3 -m venv .venv", quality)
        self.assertIn("'torch==2.13.0'", quality)
        self.assertIn("https://download.pytorch.org/whl/cpu", quality)
        self.assertIn("'safetensors==0.8.0' 'numpy==2.5.2'", quality)
        torch_step = quality.split("- name: Install PyTorch CPU runtime", 1)[1].split(
            "- name: Install parity dependencies", 1
        )[0]
        self.assertNotIn("safetensors", torch_step)
        self.assertIn("LD_LIBRARY_PATH", quality)
        self.assertIn("scripts/run-python-parity.sh", quality)
        self.assertIn("python -m unittest discover", quality)
        self.assertIn("cargo doc -p rusttorch --no-deps", quality)
        self.assertIn("cargo doc -p rusttorch-cli --no-deps", quality)
        for package in ("rusttorch", "rusttorch-cli"):
            with self.subTest(package=package):
                self.assertIn(f"cargo package -p {package} --locked --list", quality)
                self.assertRegex(
                    quality,
                    rf"(?m)^\s+run: cargo package -p {re.escape(package)} --locked$",
                )
        self.assertIn("Reject unsafe package contents", quality)

    def test_required_ci_result_handles_pr_only_skips_explicitly(self) -> None:
        text = self.read(".github/workflows/ci.yml")
        required = text.split("  required:\n", 1)[1]
        self.assertIn("if: always()", required)
        for job in ("dco", "dependency-review", "quality", "msrv", "cli"):
            with self.subTest(job=job):
                self.assertRegex(required, rf"(?m)^      - {re.escape(job)}$")
        for result in (
            "DCO_RESULT",
            "DEPENDENCY_REVIEW_RESULT",
            "QUALITY_RESULT",
            "MSRV_RESULT",
            "CLI_RESULT",
        ):
            self.assertIn(result, required)
        self.assertIn('if [ "$EVENT_NAME" = "pull_request" ]; then', required)
        self.assertIn('test "$DCO_RESULT" = "success"', required)
        self.assertIn('test "$DEPENDENCY_REVIEW_RESULT" = "success"', required)
        self.assertIn('test "$DCO_RESULT" = "skipped"', required)
        self.assertIn('test "$DEPENDENCY_REVIEW_RESULT" = "skipped"', required)


if __name__ == "__main__":
    unittest.main()
