from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import tomllib
import unittest


ROOT = Path(__file__).resolve().parents[1]


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / filename)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CHECKER = load_module('checker', 'check-compatibility.py')
BEHAVIOR = load_module('behavior', 'pytorch_behavior.py')


class BehaviorTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.document = tomllib.loads((ROOT / 'compat/pytorch_behavior.toml').read_text())
        self.case = next(case for case in self.document["case"]
                         if case["id"] == "index-select.repeated-rows.backward")
        case_id = self.case["id"]
        self.document["case"] = [self.case]
        self.inventory = {
            "pytorch_commit": self.document["pytorch_commit"],
            "pytorch_version": "2.13.0",
            "symbols": [{"id": self.case["inventory_id"]}],
        }
        self.mapping = {self.case["inventory_id"]: self.case["ledger"]}
        receipt_path = "compat/results/index-select-macos-arm64.json"
        self.document["observations"] = [receipt_path]
        self.receipt = self.root / receipt_path
        paths = [
            "src/tensor.rs", "Cargo.lock", receipt_path, self.case["documentation"],
            self.case["rust_test"].split("::")[0], self.case["reference_test"].split("::")[0],
        ]
        for path in paths:
            destination = self.root / path
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / path, destination)
        receipt = json.loads(self.receipt.read_text())
        for field in ["cases", "fingerprints"]:
            receipt[field] = {case_id: receipt[field][case_id]}
        self.receipt.write_text(json.dumps(receipt))

    def errors(self):
        errors = BEHAVIOR.validate_behavior(
            self.root, self.document, self.inventory, self.mapping, CHECKER._validate_evidence
        )
        return "\n".join(errors)

    def test_snapshot_is_valid(self):
        self.assertEqual(self.errors(), '')

    def test_rejects_duplicate_and_missing_case_ids(self):
        self.document['case'].append(copy.deepcopy(self.case))
        self.assertIn('duplicate', self.errors())
        del self.case['id']
        self.assertIn('id', self.errors())

    def test_rejects_unknown_inventory_and_ledger(self):
        self.case['ledger'] = 'unknown'
        self.assertIn('ledger', self.errors())
        self.case['inventory_id'] = 'unknown'
        self.assertIn('inventory', self.errors())

    def test_device_obligations_cannot_be_dropped(self):
        self.document["case"] = [self.case]
        self.document["observations"] = []
        self.case["devices"] = {"cpu": "untested"}
        self.assertIn("device matrix", self.errors())

    def test_requires_runnable_evidence(self):
        self.case['rust_test'] += '_absent'
        self.case['reference_test'] += '_absent'
        errors = self.errors()
        self.assertIn('test declaration', errors)
        self.assertIn('reference', errors)

    def test_rejects_hardware_skip_as_pass(self):
        receipt = json.loads(self.receipt.read_text())
        receipt['cases'][self.case['id']]['cpu'] = {'outcome': 'skipped', 'reason': 'hardware unavailable'}
        self.receipt.write_text(json.dumps(receipt))
        self.assertIn('untested', self.errors())

    def test_rejects_wrong_hardware_and_changed_numeric_output(self):
        receipt = json.loads(self.receipt.read_text())
        cell = receipt['cases'][self.case['id']]['cpu']
        cell['rust']['device'] = 'cuda'
        cell['rust']['values']['gradient'][0] += 1.0
        self.receipt.write_text(json.dumps(receipt))
        self.assertIn('fail', self.errors())

    def test_rejects_unbound_or_stale_proof(self):
        receipt = json.loads(self.receipt.read_text())
        receipt['environment']['torch_commit'] = '0' * 40
        self.receipt.write_text(json.dumps(receipt))
        self.assertIn('torch', self.errors())
        self.receipt.unlink()
        self.assertIn('observation', self.errors())

    def test_untested_does_not_mean_upstream_inapplicable(self):
        self.case['devices']['cuda'] = 'upstream_inapplicable'
        self.assertIn('restriction', self.errors())

    def test_rejects_invalid_tolerance(self):
        self.case['tolerance']['atol'] = float('nan')
        self.assertIn('tolerance', self.errors())

    def test_rejects_reference_path_outside_repository(self):
        self.case['reference_test'] = '../outside.py::test_repeated_rows'
        self.assertIn('malformed', self.errors())

    def test_rejects_execution_on_gpu_missing_from_environment(self):
        receipt = json.loads(self.receipt.read_text())
        del receipt["environment"]["devices"]["mps"]
        self.receipt.write_text(json.dumps(receipt))
        self.assertIn("untested hardware", self.errors())

    def test_requires_all_observation_fields_on_both_sides(self):
        receipt = json.loads(self.receipt.read_text())
        cell = receipt["cases"][self.case["id"]]["cpu"]
        for side in ["rust", "reference"]:
            del cell[side]["values"]["gradient"]
        self.receipt.write_text(json.dumps(receipt))
        self.assertIn("fail", self.errors())

    def test_changed_production_source_invalidates_previous_pass(self):
        path = self.root / "src/tensor.rs"
        path.write_text(path.read_text() + "\n// implementation changed\n")
        self.assertIn("changed case", self.errors())

    def test_rejects_matching_but_truncated_values(self):
        receipt = json.loads(self.receipt.read_text())
        cell = receipt["cases"][self.case["id"]]["cpu"]
        for side in ["rust", "reference"]:
            cell[side]["values"]["gradient"] = []
        self.receipt.write_text(json.dumps(receipt))
        self.assertIn("fail", self.errors())

    def test_does_not_reuse_receipt_after_test_changed(self):
        source = self.root / self.case["reference_test"].split("::")[0]
        source.write_text(source.read_text() + "\n# changed\n")
        self.assertIn("changed case", self.errors())
