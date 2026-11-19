from copy import deepcopy
from pathlib import Path
import runpy
import unittest


ROOT = Path(__file__).resolve().parents[1]
MIGRATION = runpy.run_path(str(ROOT / "scripts/compare-pytorch-schemas.py"))
WHEEL = MIGRATION["WHEEL_COMMIT"]


class MigrationDeltaTests(unittest.TestCase):
    def setUp(self):
        self.id = "aten:aten::relu"
        self.schema = "aten::relu(Tensor self) -> Tensor"
        self.candidate = {
            "format_version": 1, "pytorch_version": "2.14.0", "pytorch_commit": WHEEL, "scope": ["torch"],
            "runtime_build": {"wheel_version": "2.14.0", "git_commit": WHEEL, "platform": "darwin",
                              "machine": "arm64", "cuda": None, "hip": None, "debug": False,
                              "torch_config_sha256": "0" * 64},
            "symbols": [
                {"id": self.id, "module": "aten", "kind": "aten_schema", "name": "relu",
                 "signature": self.schema, "source": "aten/src/ATen/native/native_functions.yaml",
                 "line": 1, "visibility": "public", "runtime_present": True},
                {"id": "python:torch.add", "module": "torch", "name": "add", "kind": "function",
                 "source": "torch/_torch_docs.py", "signature": "(input, other)",
                 "documented_signature": "(input, other, *, alpha=1)"},
            ],
        }

    def test_python_defaults_and_native_registration_have_independent_deltas(self):
        baseline = deepcopy(self.candidate)
        baseline["symbols"][0]["runtime_present"] = False
        baseline["symbols"][1]["documented_signature"] = "(input, other)"
        result = MIGRATION["semantic_delta"](baseline, self.candidate, {self.id: self.schema})
        self.assertEqual(result["added"], [])
        self.assertEqual(result["removed"], [])
        self.assertEqual(result["changed"]["signature"], {})
        self.assertEqual(result["changed"]["runtime_present"], {self.id: {"before": False, "after": True}})
        self.assertEqual(result["changed"]["documented_signature"]["python:torch.add"]["after"],
                         "(input, other, *, alpha=1)")

    def test_candidate_must_match_both_binary_and_exact_tagged_source(self):
        compare = MIGRATION["semantic_delta"]
        with self.assertRaisesRegex(ValueError, "canonical target"):
            compare(self.candidate, self.candidate, {self.id: "different"})
        invalid = deepcopy(self.candidate)
        invalid["runtime_build"]["git_commit"] = MIGRATION["TARGET"]
        with self.assertRaisesRegex(ValueError, "runtime commit"):
            compare(self.candidate, invalid, {self.id: self.schema})


if __name__ == "__main__":
    unittest.main()
