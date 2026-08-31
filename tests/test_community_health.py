import re
import subprocess
import tomllib
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
SETUP_UV = (
    "astral-sh/setup-uv@c771a70e6277c0a99b617c7a806ffedaca235ff9 # v9.0.0"
)
PYTHON_LOCK_CHECK = "python3 scripts/check-python-lock.py"
DEPENDENCY_REVIEW = (
    "actions/dependency-review-action@"
    "595b5aeba73380359d98a5e087f648dbb0edce1b # v4.7.3"
)
APPROVED_CI_ACTIONS = {
    action.split(" #", 1)[0]
    for action in (
        CHECKOUT,
        SETUP_PYTHON,
        SETUP_RUST,
        SETUP_UV,
        DEPENDENCY_REVIEW,
    )
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
        self.assertNotRegex(
            text,
            r"(?m)(?:^\s*(?:-\s+)?|[{\[,]\s*)\?(?:\s|$)",
        )
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

    def assert_python_ci_contract(self, text: str) -> None:
        quality = text.split("  quality:\n", 1)[1].split("\n  msrv:", 1)[0]
        normalized_text = re.sub(r"\\\r?\n[ \t]*", "", text)
        normalized_quality = normalized_text.split("  quality:\n", 1)[1].split(
            "\n  msrv:", 1
        )[0]
        self.assertEqual(text.count(f"uses: {SETUP_UV}"), 1)
        setup_uv = quality.split(f"uses: {SETUP_UV}", 1)[1].split(
            "\n      - name:", 1
        )[0]
        self.assertIn(
            '\n        with:\n          version: "0.12.3"\n          enable-cache: false\n',
            setup_uv,
        )
        self.assertEqual(len(re.findall(r"(?i)\benable-cache\s*:", text)), 1)
        self.assertEqual(text.count("          enable-cache: false"), 1)
        self.assertEqual(
            len(re.findall(r"(?i)\buv\s+lock\b", normalized_quality)), 1
        )
        self.assertRegex(
            quality, r"(?m)^        run: uv lock --check --offline --no-cache$"
        )
        self.assertEqual(
            len(re.findall(r"(?i)\buv\s+sync\b", normalized_quality)), 1
        )
        self.assertRegex(quality, r"(?m)^        run: uv sync --frozen --no-cache$")
        self.assertNotRegex(normalized_text, r"(?i)\bpip\b")
        self.assertNotIn("python3 -m venv", quality)

        lock_policy_step = (
            "      - name: Validate Python lock artifacts\n"
            f"        run: {PYTHON_LOCK_CHECK}\n"
        )
        lock_policy_boundary = (
            lock_policy_step
            + "\n      - name: Set up UV\n"
            + f"        uses: {SETUP_UV}\n"
        )
        self.assertEqual(quality.count(lock_policy_step), 1)
        self.assertEqual(quality.count(lock_policy_boundary), 1)
        self.assertEqual(quality.count("scripts/check-python-lock.py"), 1)
        self.assertEqual(normalized_quality.count(PYTHON_LOCK_CHECK), 1)

        path_export = 'echo "$PWD/.venv/bin" >> "$GITHUB_PATH"'
        environment_export = 'echo "VIRTUAL_ENV=$PWD/.venv" >> "$GITHUB_ENV"'
        self.assertIn(path_export, quality)
        self.assertIn(environment_export, quality)
        setup_python_position = quality.index(f"uses: {SETUP_PYTHON}")
        lock_policy_position = quality.index(lock_policy_boundary)
        setup_uv_position = quality.index(f"uses: {SETUP_UV}")
        lock_position = quality.index("uv lock --check --offline --no-cache")
        sync_position = quality.index("uv sync --frozen --no-cache")
        self.assertLess(setup_python_position, lock_policy_position)
        self.assertLess(lock_policy_position, setup_uv_position)
        self.assertLess(setup_uv_position, lock_position)
        self.assertLess(lock_position, sync_position)
        self.assertLess(sync_position, quality.index(path_export))
        self.assertIn(
            ".venv/bin/python -m unittest discover -s tests -p 'test_*.py' -v",
            quality,
        )
        self.assertIn(
            ".venv/bin/python scripts/check-compatibility.py --check", quality
        )
        self.assertIn(
            ".venv/bin/python -c 'from pathlib import Path; import torch", quality
        )
        self.assertIn("scripts/run-python-parity.sh", quality)
        self.assertIn(r"pyproject\.toml|uv\.lock", quality)

    def assert_cargo_package_contract(self, text: str) -> None:
        manifest = tomllib.loads(text)
        package = manifest["package"]
        self.assertNotIn("include", package)
        self.assertIn("/pyproject.toml", package["exclude"])
        self.assertIn("/uv.lock", package["exclude"])

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

    def test_contributing_uses_the_frozen_python_environment(self) -> None:
        text = self.read("CONTRIBUTING.md")
        self.assertIn("uv sync --frozen --no-cache", text)
        for platform_scope in (
            "Linux x86_64, AArch64, and s390x",
            "macOS arm64",
            "Windows x86_64",
            "only on Ubuntu x86_64",
            "Intel macOS or Windows ARM64",
        ):
            with self.subTest(platform_scope=platform_scope):
                self.assertIn(platform_scope, text)

    def test_uv_lock_is_current_without_network_or_cache(self) -> None:
        result = subprocess.run(
            ["uv", "lock", "--check", "--offline", "--no-cache"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_cargo_packages_exclude_python_tooling_metadata(self) -> None:
        manifest = self.read("Cargo.toml")
        self.assert_cargo_package_contract(manifest)
        mutation = manifest.replace(
            "\n[lib]\n", '\ninclude = ["**/*"]\n\n[lib]\n', 1
        )
        self.assertNotEqual(mutation, manifest)
        with self.assertRaises(AssertionError):
            self.assert_cargo_package_contract(mutation)

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
        self.assert_python_ci_contract(text)
        self.assertRegex(text, r"(?m)^permissions:\n  contents: read$")
        for action in (
            CHECKOUT,
            SETUP_PYTHON,
            SETUP_RUST,
            SETUP_UV,
            DEPENDENCY_REVIEW,
        ):
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
            "explicit action key": text.replace(
                "    steps:\n",
                "    steps:\n"
                "      - name: Unapproved explicit action\n"
                "        ? uses # invoke an unapproved action\n"
                "        : evil/example@main\n",
                1,
            ),
            "flow explicit action key": text.replace(
                "    steps:\n",
                "    steps:\n"
                "      - { ? uses # invoke action\n"
                "          : evil/example@main }\n",
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
            "explicit cache key": text.replace(
                '          python-version: "3.14"',
                '          python-version: "3.14"\n'
                "          ? cache # enable a package cache\n"
                "          : pip",
                1,
            ),
            "flow explicit cache key": text.replace(
                '        with:\n          python-version: "3.14"',
                "        with: { ? cache # enable wheels\n"
                '                  : pip, python-version: "3.14" }',
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

    def test_ci_python_contract_rejects_unlocked_or_imperative_setup(self) -> None:
        text = self.read(".github/workflows/ci.yml")
        python_step = f'''      - name: Set up Python
        uses: {SETUP_PYTHON}
        with:
          python-version: "3.14"
'''
        sync_step = '''      - name: Synchronize locked Python environment
        run: uv sync --frozen --no-cache
'''
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
        reordered = text.replace(python_step, "__PYTHON_STEP__", 1)
        reordered = reordered.replace(sync_step, python_step, 1)
        reordered = reordered.replace("__PYTHON_STEP__", sync_step, 1)
        late_policy = text.replace(lock_policy_step, "", 1)
        late_policy = late_policy.replace(
            setup_uv_step, setup_uv_step + "\n" + lock_policy_step, 1
        )
        mutations = {
            "floating setup action": text.replace(
                SETUP_UV, "astral-sh/setup-uv@v9", 1
            ),
            "UV version drift": text.replace(
                'version: "0.12.3"', 'version: "0.12.4"', 1
            ),
            "persistent cache": text.replace(
                "enable-cache: false", "enable-cache: true", 1
            ),
            "unfrozen sync": text.replace(
                "uv sync --frozen --no-cache", "uv sync --no-cache", 1
            ),
            "cached sync": text.replace(
                "uv sync --frozen --no-cache", "uv sync --frozen", 1
            ),
            "missing lock check": text.replace(
                "uv lock --check --offline --no-cache", "uv lock --offline --no-cache", 1
            ),
            "second unlocked sync": text.replace(
                "        run: uv sync --frozen --no-cache",
                "        run: uv sync --frozen --no-cache\n\n"
                "      - name: Resynchronize without the lock\n"
                "        run: |\n"
                "          uv sync",
                1,
            ),
            "continued unlocked sync": text.replace(
                "        run: uv sync --frozen --no-cache",
                "        run: uv sync --frozen --no-cache\n\n"
                "      - name: Resynchronize through a shell continuation\n"
                "        run: |\n"
                "          uv \\\n"
                "          sync",
                1,
            ),
            "inline pip install": text.replace(
                "uv sync --frozen --no-cache",
                "uv sync --frozen --no-cache\n          python -m pip install wheel",
                1,
            ),
            "split inline pip install": text.replace(
                "uv sync --frozen --no-cache",
                "uv sync --frozen --no-cache\n          python -m pip \\\n"
                "            install wheel",
                1,
            ),
            "continued inline pip install": text.replace(
                "        run: uv sync --frozen --no-cache",
                "        run: uv sync --frozen --no-cache\n\n"
                "      - name: Install through a shell continuation\n"
                "        run: |\n"
                "          python -m p\\\n"
                "          ip install wheel",
                1,
            ),
            "sync before Python": reordered,
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
            "archive scanner removed": text.replace(
                r"pyproject\.toml|uv\.lock", "metadata.toml", 1
            ),
        }
        for name, mutation in mutations.items():
            with self.subTest(mutation=name):
                self.assertNotEqual(mutation, text)
                with self.assertRaises(AssertionError):
                    self.assert_python_ci_contract(mutation)

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
        self.assertIn("uv sync --frozen --no-cache", quality)
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
