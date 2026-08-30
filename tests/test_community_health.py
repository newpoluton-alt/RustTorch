import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

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


if __name__ == "__main__":
    unittest.main()
