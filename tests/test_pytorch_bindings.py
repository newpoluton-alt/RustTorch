import hashlib
import importlib.util
from pathlib import Path
import tempfile
import runpy
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("bindings", ROOT / "scripts/pytorch_bindings.py")
bindings = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bindings)

ATEN = "aten:aten::relu"
PYTHON = "python:torch.relu"
TORCH = "cf30153c4c131c8164ee7798e5022d810682e2cb"
TCH = "4227b89b72059b0651ff83a38637693574e0a2d6"
HEADER = "inventory_id\trust_symbol\tffi_symbol\tdispatcher\n"
ROW = f"{ATEN}\ttch::Tensor::f_relu\tatg_relu\trelu\n"


class BindingTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        (self.root / "compat").mkdir()
        self.trace = self.root / "compat/pytorch_tch_dispatch.tsv"
        self.trace.write_text(HEADER + ROW, encoding="utf-8")
        self.inventory = {
            "pytorch_commit": TORCH,
            "symbols": [{"id": ATEN, "kind": "aten_schema"}, {"id": PYTHON, "kind": "function"}],
        }
        self.mapping = {"mapping": {ATEN: "nn.relu", PYTHON: "nn.relu"}}
        self.ledger = {"api": [{"id": "nn.relu"}], "tch": "0.26.0"}

    def manifest(self):
        return f'''format_version = 1
pytorch_commit = "{TORCH}"
tch_version = "0.26.0"
tch_commit = "{TCH}"
capabilities_sha256 = "{bindings.unresolved_digest(['capability:device:meta'])}"
trace_sha256 = "{hashlib.sha256(self.trace.read_text(encoding="utf-8").encode("utf-8")).hexdigest()}"

[unresolved.aten]
ids_sha256 = "{bindings.unresolved_digest([])}"
reason = "No exact dispatcher trace."

[unresolved.python]
ids_sha256 = "{bindings.unresolved_digest([PYTHON])}"
reason = "Frontend contract needs review."

[[capability]]
family = "device"
values = ["meta"]
ledger = "nn.relu"
source = "torch/headeronly/core/DeviceType.h"
route = "unresolved"
limitations = "No audited native device encoding."
'''

    def check(self, manifest=None):
        (self.root / "compat/pytorch_bindings.toml").write_text(
            self.manifest() if manifest is None else manifest, encoding="utf-8"
        )
        return bindings.load_bindings(self.root, self.inventory, self.mapping, self.ledger)

    def test_exact_dispatch_and_frozen_unresolved_set(self):
        self.assertEqual(self.check(), {
            ATEN: "existing_tch", PYTHON: "unresolved", "capability:device:meta": "unresolved",
        })

    def test_missing_classification_is_an_error(self):
        self.inventory["symbols"].append({"id": "aten:aten::relu_", "kind": "aten_schema"})
        self.mapping["mapping"]["aten:aten::relu_"] = "nn.relu"
        with self.assertRaisesRegex(ValueError, "unresolved.aten"):
            self.check()

    def test_empty_bindings_fail_completeness(self):
        with self.assertRaisesRegex(ValueError, "binding"):
            self.check("format_version = 1\n")

    def test_malformed_nested_tables_fail_as_validation_errors(self):
        valid = self.manifest()
        for broken in [
            valid.replace('route = "unresolved"', 'route = []'),
            valid.replace('family = "device"', 'family = 7'),
            valid.replace('ledger = "nn.relu"', 'ledger = {}'),
            valid.replace('[[capability]]', '[capability]'),
            valid.replace('[unresolved.aten]', 'unresolved.aten = 7\n[unused]'),
        ]:
            with self.subTest(broken=broken), self.assertRaises(ValueError):
                self.check(broken)

    def test_duplicate_trace_is_rejected_even_with_current_source_digest(self):
        self.trace.write_text(HEADER + ROW + ROW, encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.check()

    def test_unknown_ledger_and_inventory_targets(self):
        with self.assertRaisesRegex(ValueError, "ledger"):
            self.check(self.manifest().replace('ledger = "nn.relu"', 'ledger = "absent"'))
        self.trace.write_text(HEADER + ROW.replace(ATEN, "aten:aten::absent"), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "inventory"):
            self.check()

    def test_regeneration_detects_a_rehashed_invented_binding_and_stale_digest(self):
        verify = runpy.run_path(str(ROOT / "scripts/trace-tch-dispatch.py"))["verify_trace"]
        generated = HEADER + ROW
        self.check()
        verify(self.root, generated)
        self.trace.write_text(generated.replace("f_relu", "f_made_up"), encoding="utf-8")
        self.check()  # Offline indexes alone cannot disprove rehashed forged metadata.
        with self.assertRaisesRegex(ValueError, "rows differ"):
            verify(self.root, generated)
        self.trace.write_text(generated, encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "digest differs"):
            verify(self.root, generated)

    def test_invented_rust_method_breaks_pinned_trace(self):
        original = self.manifest()
        self.trace.write_text(HEADER + ROW.replace("f_relu", "f_made_up"), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "digest"):
            self.check(original)

    def test_deleting_an_obligation_cannot_reduce_the_gap_silently(self):
        with self.assertRaisesRegex(ValueError, "capabilit"):
            self.check(self.manifest().replace('["meta"]', '["xpu"]'))

    def test_nonexistent_frontend_symbol_fails_even_with_a_valid_named_test(self):
        test_dir = self.root / "tests"
        test_dir.mkdir()
        (test_dir / "witness.rs").write_text(
            '#[test]\nfn works() {\n    /* nested /* comment */ rusttorch::nn::invented */\n'
            '    // rusttorch::nn::invented\n    let _ = r#"a "rusttorch::nn::invented" b"#;\n}\n',
            encoding="utf-8",
        )
        self.ledger["api"][0]["rust_symbols"] = ["rusttorch::nn::invented"]
        extra = f'''[[binding]]
id = "{PYTHON}"
route = "rust"
rust_symbol = "rusttorch::nn::invented"
source = "tests/witness.rs"
source_commit = "{TCH}"
evidence = "tests/witness.rs::works"
limitations = "CPU only."
'''
        with self.assertRaisesRegex(ValueError, "absent from binding evidence"):
            self.check(self.manifest() + extra)

    def test_duplicate_capability_and_unproven_route_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.check(self.manifest().replace('["meta"]', '["meta", "meta"]'))
        with self.assertRaisesRegex(ValueError, "route"):
            self.check(self.manifest().replace('route = "unresolved"', 'route = "assumed"'))
        with self.assertRaisesRegex(ValueError, "proof|evidence"):
            self.check(self.manifest().replace('route = "unresolved"', 'route = "rust"'))


class DispatchTraceTests(unittest.TestCase):
    def test_requires_a_single_dispatcher_and_exact_native_declaration(self):
        collect = runpy.run_path(str(ROOT / "scripts/trace-tch-dispatch.py"))["collect_routes"]
        schema = "aten::add.Tensor(Tensor self, Tensor other) -> Tensor"
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            def write(path, contents):
                path = root / path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(contents, encoding="utf-8")
            write("third_party/pytorch/Declarations-v2.13.0.yaml",
                  f"- name: add\n  schema_string: {schema}\n")
            write("ATen/ops/add_ops.h",
                  'struct TORCH_API add_Tensor {\n'
                  f'  static constexpr const char* schema_str = "{schema[6:]}";\n}};\n')
            names = ("add", "ambiguous", "decomposed", "undeclared")
            write("src/c_generated.rs", "".join(
                f"    pub fn atg_{name}();\n" for name in names if name != "undeclared"
            ))
            write("src/wrappers/tensor_fallible_generated.rs", "".join(
                f"    pub fn f_{name}() {{\n        atg_{name}();\n    }}\n"
                for name in names
            ))
            write("libtch/torch_api_generated.cpp", "".join(
                f"void atg_{name}() {{\n    auto outputs__ = torch::{native}();\n}}\n"
                for name, native in zip(names, ("add", "add", "mul", "add"))
            ))
            ir = "".join(
                f"define void @atg_{name}() {{\n" + "".join(
                    f"  call void @_ZN2at4_ops10{op}4callE()\n" for op in calls
                ) + "}\n"
                for name, calls in zip(names, (["add_Tensor"], ["add_Tensor", "mul_Tensor"],
                                               ["add_Tensor"], ["add_Tensor"]))
            )
            symbols = [{"kind": "aten_schema", "signature": schema, "id": "aten:aten::add.Tensor"}]
            self.assertEqual(
                collect(root, root, root, ir, {"symbols": symbols}),
                HEADER + "aten:aten::add.Tensor\ttch::Tensor::f_add\tatg_add\tadd_Tensor\n",
            )
            symbols[0]["signature"] = schema.replace("other", "right")
            with self.assertRaisesRegex(ValueError, "no exact matching"):
                collect(root, root, root, ir, {"symbols": symbols})


if __name__ == "__main__":
    unittest.main()
