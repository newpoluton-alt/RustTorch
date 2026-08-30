from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "check-compatibility.py"

SPEC = importlib.util.spec_from_file_location("check_compatibility", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
CHECKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECKER)


VALID_ROW = {
    "id": "nn.linear",
    "python_symbols": ["torch.nn.Linear"],
    "rust_symbols": ["rusttorch::nn::Linear"],
    "status": "supported",
    "implementation": "mixed",
    "scope": "Forward, parameters, gradients, and documented defaults.",
    "source": "torch/nn/modules/linear.py",
    "evidence": [
        "tests/eager.rs::linear_bias_and_no_bias_have_expected_shapes_and_values"
    ],
    "notes": "Rust configuration replaces Python keyword arguments.",
}

VALID_LEDGER = {
    "format_version": 2,
    "package": "rusttorch",
    "crate": "rusttorch",
    "tch": "0.26.0",
    "pytorch_reference": "v2.13.0",
    "pytorch_commit": "cf30153",
    "api": [VALID_ROW],
}

VALID_LEDGER_TOML = """\
format_version = 2
package = "rusttorch"
crate = "rusttorch"
tch = "0.26.0"
pytorch_reference = "v2.13.0"
pytorch_commit = "cf30153"

[[api]]
id = "nn.linear"
python_symbols = ["torch.nn.Linear"]
rust_symbols = ["rusttorch::nn::Linear"]
status = "supported"
implementation = "mixed"
scope = "Forward, parameters, gradients, and documented defaults."
source = "torch/nn/modules/linear.py"
evidence = ["tests/eager.rs::linear_bias_and_no_bias_have_expected_shapes_and_values"]
notes = "Rust configuration replaces Python keyword arguments."
"""


class CompatibilityScriptTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        (self.root / "tests").mkdir()
        (self.root / "tests" / "eager.rs").write_text(
            "#[test]\n"
            "fn linear_bias_and_no_bias_have_expected_shapes_and_values() {}\n",
            encoding="utf-8",
        )
        (self.root / "Cargo.toml").write_text(
            '[package]\nname = "rusttorch"\nversion = "9.9.9"\n\n'
            '[lib]\nname = "rusttorch"\n\n'
            '[dependencies]\ntch = "0.26.0"\n',
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def ledger(self) -> dict[str, object]:
        return copy.deepcopy(VALID_LEDGER)

    def errors(self, ledger: dict[str, object]) -> list[str]:
        return CHECKER.validate_ledger(self.root, ledger)

    def assert_error_contains(
        self, ledger: dict[str, object], *fragments: str
    ) -> None:
        errors = "\n".join(self.errors(ledger))
        self.assertTrue(errors, "validation unexpectedly succeeded")
        for fragment in fragments:
            self.assertIn(fragment, errors)

    def test_minimal_supported_row_is_valid(self) -> None:
        self.assertEqual(self.errors(self.ledger()), [])

    def test_ids_must_be_sorted_and_unique(self) -> None:
        duplicate = self.ledger()
        duplicate["api"].append(copy.deepcopy(VALID_ROW))
        self.assert_error_contains(duplicate, "nn.linear", "duplicate")

        unsorted = self.ledger()
        earlier_row = copy.deepcopy(VALID_ROW)
        earlier_row["id"] = "core.tensor"
        unsorted["api"].append(earlier_row)
        self.assert_error_contains(unsorted, "core.tensor", "sorted")

    def test_unknown_top_level_and_row_fields_are_rejected(self) -> None:
        top_level = self.ledger()
        top_level["percentage"] = 100
        self.assert_error_contains(top_level, "percentage")

        row_field = self.ledger()
        row_field["api"][0]["reference"] = "moving target"
        self.assert_error_contains(row_field, "nn.linear", "reference")

    def test_status_and_implementation_enums_are_exact(self) -> None:
        invalid_status = self.ledger()
        invalid_status["api"][0]["status"] = "complete"
        self.assert_error_contains(invalid_status, "nn.linear", "status")

        invalid_implementation = self.ledger()
        invalid_implementation["api"][0]["implementation"] = "delegate"
        self.assert_error_contains(
            invalid_implementation, "nn.linear", "implementation"
        )

    def test_supported_rows_require_executable_evidence(self) -> None:
        ledger = self.ledger()
        ledger["api"][0]["evidence"] = []
        self.assert_error_contains(ledger, "nn.linear", "evidence")

    def test_evidence_file_and_exact_declaration_must_exist(self) -> None:
        missing_file = self.ledger()
        missing_file["api"][0]["evidence"] = ["tests/missing.rs::linear_works"]
        self.assert_error_contains(missing_file, "nn.linear", "tests/missing.rs")

        missing_test = self.ledger()
        missing_test["api"][0]["evidence"] = ["tests/eager.rs::linear_works"]
        self.assert_error_contains(missing_test, "nn.linear", "linear_works")

        malformed = self.ledger()
        malformed["api"][0]["evidence"] = ["tests/eager.rs"]
        self.assert_error_contains(malformed, "nn.linear", "evidence")

    def test_python_evidence_accepts_an_exact_ordinary_test_declaration(self) -> None:
        (self.root / "tests" / "test_linear.py").write_text(
            "class LinearTests:\n"
            "    def test_linear(self) -> None:\n"
            "        pass\n",
            encoding="utf-8",
        )
        ledger = self.ledger()
        ledger["api"][0]["evidence"] = [
            "tests/test_linear.py::test_linear"
        ]
        self.assertEqual(self.errors(ledger), [])

    def test_manifest_package_library_and_tch_metadata_must_match(self) -> None:
        for field, replacement, fragment in (
            ("package", "wrong-package", "package"),
            ("crate", "wrong_crate", "crate"),
            ("tch", "0.25.0", "tch"),
        ):
            with self.subTest(field=field):
                ledger = self.ledger()
                ledger[field] = replacement
                self.assert_error_contains(ledger, fragment)

    def test_release_version_is_not_part_of_compatibility_validation(self) -> None:
        self.assertEqual(self.errors(self.ledger()), [])

    def test_pinned_schema_and_pytorch_reference_are_required(self) -> None:
        for field, replacement in (
            ("format_version", 1),
            ("pytorch_reference", "main"),
            ("pytorch_commit", "deadbeef"),
        ):
            with self.subTest(field=field):
                ledger = self.ledger()
                ledger[field] = replacement
                self.assert_error_contains(ledger, field)

    def test_source_is_a_safe_nonempty_upstream_relative_path(self) -> None:
        for source in ("", "/torch/nn/modules/linear.py", "../linear.py", "torch\\linear.py"):
            with self.subTest(source=source):
                ledger = self.ledger()
                ledger["api"][0]["source"] = source
                self.assert_error_contains(ledger, "nn.linear", "source")

    def test_rust_symbol_and_implementation_rules_follow_status(self) -> None:
        no_supported_surface = self.ledger()
        no_supported_surface["api"][0]["rust_symbols"] = []
        self.assert_error_contains(no_supported_surface, "nn.linear", "rust_symbols")

        unsupported_with_implementation = self.ledger()
        row = unsupported_with_implementation["api"][0]
        row["status"] = "not_supported"
        row["rust_symbols"] = []
        row["implementation"] = "mixed"
        row["evidence"] = []
        self.assert_error_contains(
            unsupported_with_implementation, "nn.linear", "implementation"
        )

        python_only_with_rust = self.ledger()
        row = python_only_with_rust["api"][0]
        row["status"] = "python_only"
        row["evidence"] = []
        self.assert_error_contains(
            python_only_with_rust, "nn.linear", "rust_symbols"
        )

        planned_symbol_without_implementation = self.ledger()
        row = planned_symbol_without_implementation["api"][0]
        row["status"] = "planned"
        row["implementation"] = "none"
        row["evidence"] = []
        self.assert_error_contains(
            planned_symbol_without_implementation, "nn.linear", "implementation"
        )

        planned = self.ledger()
        row = planned["api"][0]
        row["status"] = "planned"
        row["rust_symbols"] = []
        row["implementation"] = "none"
        row["evidence"] = []
        self.assertEqual(self.errors(planned), [])

    def test_load_ledger_reads_toml(self) -> None:
        path = self.root / "ledger.toml"
        path.write_text(VALID_LEDGER_TOML, encoding="utf-8")
        self.assertEqual(CHECKER.load_ledger(path), VALID_LEDGER)

    def test_render_markdown(self) -> None:
        ledger = self.ledger()
        supported = ledger["api"][0]
        supported["implementation"] = "libtorch"
        rows = [supported]
        for row_id, status, implementation in (
            ("core.tensor", "partial", "mixed"),
            ("data.loader", "planned", "none"),
            ("compiler.compile", "python_only", "none"),
            ("serialization.pickle", "not_supported", "none"),
        ):
            row = copy.deepcopy(VALID_ROW)
            row["id"] = row_id
            row["status"] = status
            row["implementation"] = implementation
            if implementation == "none":
                row["rust_symbols"] = []
                row["evidence"] = []
            rows.append(row)
        ledger["api"] = list(reversed(rows))

        first = CHECKER.render_markdown(ledger)
        second = CHECKER.render_markdown(ledger)
        self.assertEqual(first, second)
        self.assertTrue(
            first.startswith(
                "<!-- Generated by scripts/check-compatibility.py. Do not edit directly. -->\n"
                "# API coverage\n"
            )
        )
        section_positions = [
            first.index(f"## {heading}")
            for heading in (
                "Supported",
                "Partial",
                "Planned",
                "Python-only",
                "Not supported",
            )
        ]
        self.assertEqual(section_positions, sorted(section_positions))
        self.assertIn("`nn.linear`", first)
        self.assertIn("**PyTorch:** `torch.nn.Linear`", first)
        self.assertIn("**RustTorch:** `rusttorch::nn::Linear`", first)
        self.assertIn("**Implementation:** Delegated to LibTorch", first)
        self.assertIn(
            "**Scope:** Forward, parameters, gradients, and documented defaults.",
            first,
        )
        self.assertIn(
            "[`torch/nn/modules/linear.py`](https://github.com/pytorch/pytorch/blob/cf30153/torch/nn/modules/linear.py)",
            first,
        )
        self.assertIn(
            "[`tests/eager.rs::linear_bias_and_no_bias_have_expected_shapes_and_values`](../tests/eager.rs)",
            first,
        )
        self.assertIn(
            "**Notes:** Rust configuration replaces Python keyword arguments.",
            first,
        )
        self.assertIn("PyTorch `v2.13.0` at commit `cf30153`", first)
        self.assertIn("`tch` `0.26.0`", first)
        self.assertIn("**RustTorch:** —", first)
        self.assertIn("**Evidence:** —", first)
        self.assertNotIn("%", first)
        self.assertNotIn("100%", first)

    def test_render_markdown_escapes_ledger_text_and_link_destinations(self) -> None:
        ledger = self.ledger()
        row = ledger["api"][0]
        row["python_symbols"] = ["torch.fn`name|[link]"]
        row["rust_symbols"] = ["rusttorch::fn`name"]
        row["scope"] = "A *scope* with [link](bad), <tag>, & pipe | and slash \\."
        row["source"] = "torch/odd folder/(linear).py"
        row["evidence"] = ["tests/odd file.rs::linear_test"]
        row["notes"] = "Line_one\nLine *two*."

        rendered = CHECKER.render_markdown(ledger)

        self.assertIn("``torch.fn`name|[link]``", rendered)
        self.assertIn("``rusttorch::fn`name``", rendered)
        self.assertIn(
            r"A \*scope\* with \[link\](bad), &lt;tag&gt;, &amp; pipe \| and slash \\.",
            rendered,
        )
        self.assertIn("torch/odd%20folder/%28linear%29.py", rendered)
        self.assertIn("../tests/odd%20file.rs", rendered)
        self.assertIn(r"Line\_one<br>Line \*two\*.", rendered)

    def test_real_ledger_validates(self) -> None:
        ledger = CHECKER.load_ledger(ROOT / "compat" / "pytorch_api.toml")
        self.assertEqual(CHECKER.validate_ledger(ROOT, ledger), [])

    def cli_root(self) -> Path:
        (self.root / "scripts").mkdir()
        (self.root / "compat").mkdir()
        (self.root / "docs").mkdir()
        shutil.copyfile(SCRIPT, self.root / "scripts" / SCRIPT.name)
        (self.root / "compat" / "pytorch_api.toml").write_text(
            VALID_LEDGER_TOML, encoding="utf-8"
        )
        return self.root

    def run_cli(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(self.root / "scripts" / SCRIPT.name), *arguments],
            cwd=self.root,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_cli_check_accepts_current_generated_bytes(self) -> None:
        self.cli_root()
        (self.root / "docs" / "api-coverage.md").write_text(
            CHECKER.render_markdown(self.ledger()), encoding="utf-8"
        )
        result = self.run_cli("--check")
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_cli_check_reports_stale_docs_without_changing_them(self) -> None:
        self.cli_root()
        coverage = self.root / "docs" / "api-coverage.md"
        coverage.write_text("stale\n", encoding="utf-8")
        result = self.run_cli("--check")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("stale", result.stderr)
        self.assertIn("python3 scripts/check-compatibility.py --write", result.stderr)
        self.assertEqual(coverage.read_text(encoding="utf-8"), "stale\n")

    def test_atomic_write_failure_preserves_existing_bytes_and_cleans_temp_file(self) -> None:
        docs = self.root / "docs"
        docs.mkdir()
        coverage = docs / "api-coverage.md"
        coverage.write_bytes(b"preserve me\r\n")

        with mock.patch.object(CHECKER.os, "fsync", side_effect=OSError("disk full")):
            with self.assertRaisesRegex(OSError, "disk full"):
                CHECKER._atomic_write(coverage, "replacement\n")

        self.assertEqual(coverage.read_bytes(), b"preserve me\r\n")
        self.assertEqual(list(docs.iterdir()), [coverage])

    def test_cli_write_atomically_generates_then_check_accepts_exact_bytes(self) -> None:
        self.cli_root()
        coverage = self.root / "docs" / "api-coverage.md"
        coverage.write_text("preserve me\n", encoding="utf-8")
        first_write = self.run_cli("--write")
        self.assertEqual(first_write.returncode, 0, first_write.stderr)
        first_bytes = coverage.read_bytes()
        self.assertEqual(first_bytes, CHECKER.render_markdown(self.ledger()).encode())

        check = self.run_cli("--check")
        self.assertEqual(check.returncode, 0, check.stderr)

        second_write = self.run_cli("--write")
        self.assertEqual(second_write.returncode, 0, second_write.stderr)
        self.assertEqual(coverage.read_bytes(), first_bytes)

    def test_cli_rejects_missing_conflicting_and_unknown_arguments(self) -> None:
        self.cli_root()
        for arguments in ((), ("--check", "--write"), ("--wat",)):
            with self.subTest(arguments=arguments):
                result = self.run_cli(*arguments)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("usage:", result.stderr)


if __name__ == "__main__":
    unittest.main()
