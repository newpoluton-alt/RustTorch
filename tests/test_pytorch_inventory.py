"""Offline contract checks and a frozen isolated semantic collector smoke test."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("inventory", ROOT / "scripts/sync-pytorch-inventory.py")
inventory = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(inventory)


class InventoryTests(unittest.TestCase):
    def fixture(self):
        return dict(format_version=1, pytorch_version=inventory.VERSION, pytorch_commit=inventory.COMMIT,
                    scope=["torch"], symbols=[dict(id="python:torch.example", kind="function", module="torch",
                    name="example", signature="(value=None)", source="torch/example.py")])

    def test_exact_reference_rejects_all_pin_and_field_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reference.toml"
            original = (ROOT / "compat/pytorch_reference.toml").read_text()
            path.write_text(original)
            self.assertEqual(inventory.load_reference(path), inventory.REFERENCE)
            mutations = [original.replace('"myst-nb"', '"markdown"', 1), original.replace('".md" = "myst-nb"\n', ''),
                         original + '\n".txt" = "restructuredtext"\n', original.replace('"cpu", "linux"', '"linux", "cpu"'),
                         original.replace('["torch"]', '["torch", "torchvision"]'), original.replace('format_version = 1', 'format_version = true'),
                         'unknown = 1\n' + original, original.replace(inventory.COMMIT, 'a' * 40)]
            for text in mutations:
                with self.subTest(text=text):
                    path.write_text(text)
                    with self.assertRaises(ValueError):
                        inventory.load_reference(path)

    def test_runtime_pin_is_exact(self):
        inventory.verify_runtime(runtime={"version": inventory.VERSION, "commit": inventory.COMMIT})
        for runtime in ({"version": "2.12.0", "commit": inventory.COMMIT}, {"version": inventory.VERSION, "commit": "a" * 40}):
            with self.assertRaises(ValueError):
                inventory.verify_runtime(runtime=runtime)

    def test_inventory_schema_paths_sort_duplicates_and_scope(self):
        self.assertEqual(inventory.validate_inventory(self.fixture()), [])
        for field, value in (("kind", "unknown"), ("source", "/tmp/source.py"), ("source", "../source.py"),
                             ("source", "torch/../source.py"), ("signature", "(value=<Generator at 0x123abcd>)"), ("id", "python:numpy.add")):
            fixture = self.fixture()
            fixture["symbols"][0][field] = value
            with self.subTest(field=field, value=value):
                self.assertTrue(inventory.validate_inventory(fixture))
        fixture = self.fixture()
        fixture["symbols"] *= 2
        self.assertTrue(inventory.validate_inventory(fixture))
        fixture = self.fixture()
        fixture["scope"] = ["torch.utils.data"]
        self.assertTrue(inventory.validate_inventory(fixture))
        fixture = self.fixture()
        fixture["extra"] = True
        self.assertTrue(inventory.validate_inventory(fixture))

    def runtime_build(self):
        return dict(wheel_version=inventory.VERSION, git_commit=inventory.COMMIT,
                    platform="linux", machine="x86_64", cuda=None, hip=None,
                    debug=False, torch_config_sha256="a" * 64)

    def aten_fixture(self):
        return dict(id="aten:aten::example", kind="aten_schema", module="aten", name="example",
                    signature="aten::example(Tensor self) -> Tensor",
                    source="aten/src/ATen/native/native_functions.yaml", line=10,
                    visibility="public", runtime_present=False)

    def test_runtime_fingerprint_rejects_spoofed_fields_and_types(self):
        fixture = self.fixture()
        fixture["runtime_build"] = self.runtime_build()
        self.assertEqual(inventory.validate_inventory(fixture), [])
        for key, value in (("wheel_version", "1.0.0"), ("wheel_version", None),
                           ("git_commit", "b" * 40), ("platform", "unknown"),
                           ("machine", 123), ("machine", ""), ("cuda", True),
                           ("hip", "garbage"), ("debug", 0),
                           ("torch_config_sha256", "x" * 64), ("torch_config_sha256", 1)):
            bad = copy.deepcopy(fixture)
            bad["runtime_build"][key] = value
            with self.subTest(key=key, value=value):
                self.assertTrue(inventory.validate_inventory(bad))

    def test_python_identity_and_provenance_are_internally_consistent(self):
        for key, value in (("module", "numpy"), ("module", "torch.other"),
                           ("name", "other"), ("id", "python:torch.example#overload-garbage"),
                           ("id", "python:torch.example#overload-0"),
                           ("id", "python:torch.example#overload-1"),
                           ("alias", "false"), ("documentation", "../escape.rst"),
                           ("documentation", "/tmp/api.rst"), ("documentation", "torch/api.py"),
                           ("source", "."), ("line", True), ("kind", []),
                           ("signature_reason", "not actually missing"), ("runtime_present", False)):
            bad = self.fixture()
            bad["symbols"][0][key] = value
            with self.subTest(key=key, value=value):
                self.assertTrue(inventory.validate_inventory(bad))
        fixture = self.fixture()
        first = fixture["symbols"][0]
        first["id"] += "#overload-1"
        second = copy.deepcopy(first)
        second["id"] = "python:torch.example#overload-2"
        fixture["symbols"].append(second)
        self.assertEqual(inventory.validate_inventory(fixture), [])
        second["id"] = "python:torch.example#overload-3"
        self.assertTrue(inventory.validate_inventory(fixture))

    def test_aten_schema_identity_and_origin_cannot_be_spoofed(self):
        fixture = self.fixture()
        fixture["symbols"].insert(0, self.aten_fixture())
        self.assertEqual(inventory.validate_inventory(fixture), [])
        for key, value in (("signature", None), ("signature", "aten::unrelated(Tensor x) -> Tensor"),
                           ("signature", "aten::example(Tensor x)"), ("source", "torch/example.py"),
                           ("runtime_present", 1), ("visibility", "internal"), ("generated_from", []),
                           ("generated_from", "missing"), ("generated_from", "example"),
                           ("alias", False), ("documentation", "docs/source/api.rst")):
            bad = copy.deepcopy(fixture)
            bad["symbols"][0][key] = value
            with self.subTest(key=key, value=value):
                self.assertTrue(inventory.validate_inventory(bad))
        del fixture["symbols"][0]["line"]
        self.assertTrue(inventory.validate_inventory(fixture))

    def test_production_snapshot_requires_full_scope_runtime_and_aten(self):
        fixture = self.fixture()
        fixture["runtime_build"] = self.runtime_build()
        fixture["symbols"].insert(0, self.aten_fixture())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            compat = root / "compat"
            compat.mkdir()
            (compat / "pytorch_reference.toml").write_text((ROOT / "compat/pytorch_reference.toml").read_text())
            ledger = {"api": [{"id": "core.example", "status": "planned", "implementation": "none", "source": "aten/src/ATen/native/native_functions.yaml"}]}
            def write(value):
                (compat / "pytorch_inventory.json").write_text(inventory.render_inventory(value))
                lines = ["format_version = 1", "", "[mapping]"]
                lines.extend(json.dumps(row["id"]) + ' = "core.example"' for row in value["symbols"])
                (compat / "pytorch_inventory_map.toml").write_text("\n".join(lines) + "\n")
            write(fixture)
            inventory.load_snapshot(root, ledger)
            for invalid_root in (None, [], True, "not an inventory"):
                (compat / "pytorch_inventory.json").write_text(json.dumps(invalid_root))
                with self.subTest(invalid_root=invalid_root), self.assertRaises(ValueError):
                    inventory.load_snapshot(root, ledger)
            for mutation in ("scope", "runtime", "aten"):
                bad = copy.deepcopy(fixture)
                if mutation == "scope":
                    bad["scope"] = ["torch.example"]
                elif mutation == "runtime":
                    del bad["runtime_build"]
                else:
                    bad["symbols"] = bad["symbols"][1:]
                self.assertEqual(inventory.validate_inventory(bad), [])
                write(bad)
                with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                    inventory.load_snapshot(root, ledger)

    def test_mapping_requires_exact_total_disposition(self):
        fixture = self.fixture()
        ledger = {"api": [{"id": "core.example", "status": "planned", "implementation": "none", "source": "aten/src/ATen/native/native_functions.yaml"}]}
        mapping = {"format_version": 1, "mapping": {"python:torch.example": "core.example"}}
        self.assertEqual(inventory.validate_mapping(fixture, mapping, ledger), [])
        for entries in ({}, {"python:torch.example": "missing"}, {"python:torch.example": "core.example", "python:torch.extra": "core.example"}):
            self.assertTrue(inventory.validate_mapping(fixture, {"format_version": 1, "mapping": entries}, ledger))
        import tomllib
        with self.assertRaises(tomllib.TOMLDecodeError):
            tomllib.loads('[mapping]\n"python:torch.example"="one"\n"python:torch.example"="two"\n')

    def test_atomic_write_keeps_existing_bytes_on_interruption(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "inventory.json"
            path.write_text("original")
            with patch.object(inventory.os, "replace", side_effect=OSError("interrupted")):
                with self.assertRaises(OSError):
                    inventory.atomic_write(path, "replacement")
            self.assertEqual(path.read_text(), "original")
            self.assertEqual(list(path.parent.iterdir()), [path])

    def test_source_ast_defaults_never_use_runtime_repr(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "sample.py").write_text('class Example:\n def __init__(self, value=default_generator, *, mode=None): pass\n')
            self.assertEqual(inventory.source_definition(root, "sample.py", "Example"), "(value=default_generator, *, mode=None)")

    def test_canonical_autogen_and_runtime_availability(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "aten/src/ATen/native"
            path.mkdir(parents=True)
            (path / "tags.yaml").write_text("[]\n")
            (path / "native_functions.yaml").write_text(
                "- func: census_(Tensor(a!) self, Tensor? other=None, *, int[] dims=[]) -> Tensor(a!)\n"
                "  autogen: census, census.out\n  dispatch:\n    CPU: census_cpu\n"
                "- func: _private_census(Tensor self) -> (Tensor, Tensor)\n"
            )
            result = inventory.collect_aten_schemas(directory)
            rows = {row["id"]: row for row in result}
            self.assertEqual(set(rows), {"aten:aten::census", "aten:aten::census_", "aten:aten::census.out", "aten:aten::_private_census"})
            self.assertEqual(rows["aten:aten::census.out"]["generated_from"], "census_")
            self.assertIn("Tensor(a!) out", rows["aten:aten::census.out"]["signature"])
            self.assertEqual(rows["aten:aten::_private_census"]["visibility"], "internal")
            self.assertFalse(any(row["runtime_present"] for row in result))
            with (path / "native_functions.yaml").open("a") as stream:
                stream.write("- func: _private_census(Tensor self) -> (Tensor, Tensor)\n")
            with self.assertRaises(ValueError):
                inventory.collect_aten_schemas(directory)

    def test_unavailable_symbols_aliases_and_overloads_remain_explicit(self):
        export = {"format_version": 1, "objects": [dict(name=name, module="torch", kind="function", source="api.rst", alias=name.endswith("alias"), signatures=["fn(x)", "fn(x,y)"]) for name in ("torch.unavailable", "torch.alias")]}
        with tempfile.TemporaryDirectory() as directory, patch.object(inventory, "run_python", return_value={}):
            result = inventory.collect_documented_symbols(export, directory)
        self.assertEqual([row["id"] for row in result], ["python:torch.alias#overload-1", "python:torch.alias#overload-2", "python:torch.unavailable#overload-1", "python:torch.unavailable#overload-2"])
        self.assertTrue(all(row["signature"] is None and row["signature_reason"] for row in result))
        self.assertTrue(result[0]["alias"])
        self.assertEqual([row["documented_signature"] for row in result[:2]], ["fn(x)", "fn(x,y)"])
        fixture = self.fixture()
        fixture["symbols"] = result
        self.assertEqual(inventory.validate_inventory(fixture), [])
        result[0]["documented_signature_reason"] = None
        self.assertTrue(inventory.validate_inventory(fixture))

    def test_canonical_alias_module_and_unstable_documented_defaults(self):
        export = {"format_version": 1, "objects": [dict(name="torch.Alias", module="torch._C", kind="class", source="api.rst", alias=True, signatures=["Alias(value=<object object at 0x123abc>)"])]}
        with tempfile.TemporaryDirectory() as directory, patch.object(inventory, "run_python", return_value={}):
            result = inventory.collect_documented_symbols(export, directory)
        self.assertEqual((result[0]["module"], result[0]["name"]), ("torch", "Alias"))
        self.assertIsNone(result[0]["documented_signature"])
        self.assertTrue(result[0]["documented_signature_reason"])
        fixture = self.fixture()
        fixture["symbols"] = result
        self.assertNotIn("0x123abc", inventory.render_inventory(fixture))

    def test_mapping_cannot_promote_unrelated_or_internal_symbols(self):
        fixture = self.fixture()
        row = {"id": "nn.linear", "status": "supported", "implementation": "mixed", "python_symbols": ["torch.nn.Linear"], "evidence": ["tests/example.rs::test_example"]}
        mapping = {"format_version": 1, "mapping": {"python:torch.example": "nn.linear"}}
        self.assertTrue(inventory.validate_mapping(fixture, mapping, {"api": [row]}))
        row["python_symbols"].append("torch.example")
        self.assertEqual(inventory.validate_mapping(fixture, mapping, {"api": [row]}), [])
        schema = self.aten_fixture()
        fixture["symbols"] = [schema]
        mapping["mapping"] = {schema["id"]: "nn.linear"}
        self.assertTrue(inventory.validate_mapping(fixture, mapping, {"api": [row]}))
        row["python_symbols"].append(schema["id"])
        self.assertEqual(inventory.validate_mapping(fixture, mapping, {"api": [row]}), [])
        schema.update(id="aten:aten::_internal", name="_internal", visibility="internal", signature="aten::_internal(Tensor x) -> Tensor")
        mapping["mapping"] = {schema["id"]: "nn.linear"}
        row["python_symbols"].append(schema["id"])
        self.assertTrue(inventory.validate_mapping(fixture, mapping, {"api": [row]}))

    def test_missing_semantic_sources_fail_refresh(self):
        with tempfile.TemporaryDirectory() as directory:
            docs = Path(directory)
            for directive in (".. toctree::\n\n   generated/torch.missing", ".. toctree::\n   :glob:\n\n   absent/*", ".. include:: missing.inc"):
                (docs / "index.rst").write_text("Fixture\n=======\n\n.. py:function:: torch.existing()\n\n" + directive + "\n")
                with self.subTest(directive=directive), self.assertRaisesRegex(ValueError, "Sphinx census failed"):
                    inventory.build_sphinx_domain_export(docs, source_root=docs, timeout=120)

    def test_ignored_upstream_pages_are_not_staged(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            docs = source / "docs/source"
            (docs / "generated").mkdir(parents=True)
            (docs / "index.rst").write_text("Fixture\n=======\n\n.. py:function:: torch.tracked()\n")
            (docs / "generated/stale.rst").write_text(".. py:function:: torch.ignored()\n")
            generator = docs / "scripts/exportdb/generate_example_rst.py"
            generator.parent.mkdir(parents=True)
            generator.write_text("import sys\nassert sys.flags.isolated\n")
            run = subprocess.run
            def tracked_run(command, **kwargs):
                if command[0] == "git":
                    return subprocess.CompletedProcess(command, 0, "docs/source/index.rst\0docs/source/scripts/exportdb/generate_example_rst.py\0", "")
                return run(command, **kwargs)
            with patch.object(inventory.subprocess, "run", side_effect=tracked_run):
                exported = inventory.build_sphinx_domain_export(source, timeout=120)
            self.assertEqual([row["name"] for row in exported["objects"]], ["torch.tracked"])

    def test_untracked_implementation_cannot_supply_signature(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            (source / "torch").mkdir()
            (source / "torch/ignored.py").write_text("def example(secret=123): pass\n")
            exported = {"format_version": 1, "objects": [dict(name="torch.example", module="torch", kind="function", source="api.rst", alias=False, signatures=["example()"])]}
            with patch.object(inventory, "run_python", return_value={"example": {"source": "torch/ignored.py", "qualname": "example"}}):
                result = inventory.collect_documented_symbols(exported, source, tracked_paths={"docs/source/api.rst"})
            self.assertEqual(result[0]["source"], "docs/source/api.rst")
            self.assertIsNone(result[0]["signature"])

    def test_checkout_requires_clean_exact_commit(self):
        good = subprocess.CompletedProcess([], 0, inventory.COMMIT + "\n", "")
        dirty = subprocess.CompletedProcess([], 0, "?? stray.txt\n", "")
        with patch.object(inventory.subprocess, "run", side_effect=[good, dirty]):
            with self.assertRaisesRegex(ValueError, "clean"):
                inventory.verify_source_checkout(ROOT)
        wrong = subprocess.CompletedProcess([], 0, "a" * 40, "")
        with patch.object(inventory.subprocess, "run", return_value=wrong):
            with self.assertRaisesRegex(ValueError, "exact"):
                inventory.verify_source_checkout(ROOT)

    def test_resolved_mixed_sources_and_upstream_conf_is_not_executed(self):
        with tempfile.TemporaryDirectory() as directory:
            docs = Path(directory)
            marker = docs / "conf-executed"
            (docs / "conf.py").write_text(f'raise RuntimeError("upstream conf executed: {marker}")\n')
            (docs / "index.rst").write_text('Census\n======\n\n.. include:: nested.inc\n\n.. toctree::\n   :glob:\n\n   page*\n\n.. autosummary::\n   :toctree: generated\n\n   torch.add\n')
            (docs / "nested.inc").write_text('.. currentmodule:: torch\n\n.. py:function:: fixture(value=None)\n\n.. only:: cpu\n\n   .. py:function:: cpu_fixture()\n\n.. only:: cuda\n\n   .. py:function:: hidden_fixture()\n\n.. dropdown:: Presentation\n\n   .. py:function:: nested_fixture()\n')
            (docs / "page.md").write_text('# Markdown\n\n```{py:function} torch.markdown_fixture(value=None)\n```\n')
            (docs / "page-notebook.ipynb").write_text(json.dumps({"nbformat": 4, "nbformat_minor": 5, "metadata": {}, "cells": [{"id": "api", "cell_type": "markdown", "metadata": {}, "source": ["# Notebook\n", "```{py:function} torch.notebook_fixture()\n", "```\n"]}]}))
            export = inventory.build_sphinx_domain_export(docs, source_root=docs, timeout=120)
            names = {obj["name"] for obj in export["objects"]}
            self.assertTrue({"torch.fixture", "torch.cpu_fixture", "torch.nested_fixture", "torch.markdown_fixture", "torch.notebook_fixture", "torch.add"} <= names, names)
            self.assertNotIn("torch.hidden_fixture", names)
            self.assertFalse(marker.exists())
            (docs / "page.md").write_text('# Bad\n\n```{unknown-api-directive} torch.bad\n```\n')
            with self.assertRaisesRegex(ValueError, "Sphinx census failed"):
                inventory.build_sphinx_domain_export(docs, source_root=docs, timeout=120)


if __name__ == "__main__":
    unittest.main()
