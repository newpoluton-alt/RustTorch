"""Check frozen behavior cases offline, or run their paired Rust/Python tests."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import tempfile
import tomllib

ROOT = Path(__file__).resolve().parents[1]
DEVICES = {"cpu", "mps", "cuda"}
STATUSES = {"pass", "fail", "untested", "upstream_inapplicable"}
CASE_FIELDS = {
    "id", "inventory_id", "ledger", "profile", "dtype", "layout", "options",
    "expected", "observe", "tolerance", "rust_test", "reference_test", "documentation", "devices",
}


def local_file(root, relative):
    if not isinstance(relative, str) or not relative or "\\" in relative or ":" in relative:
        raise ValueError(f"malformed repository path {relative!r}")
    path = root / relative
    if ".." in Path(relative).parts or Path(relative).is_absolute() or not path.resolve().is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError(f"not a repository file: {relative}")
    return path


def fingerprint(root, case):
    # Status and device availability may change between runs of the same case.
    contract = {k: v for k, v in case.items() if k not in {"devices", "upstream_restrictions"}}
    inputs = [json.dumps(contract, sort_keys=True).encode()]
    for field in ["rust_test", "reference_test"]:
        path = local_file(root, case[field].split("::")[0])
        inputs.append(path.read_bytes().replace(b"\r\n", b"\n"))
    return hashlib.sha256(b"\0".join(inputs)).hexdigest()


def finite_number(value):
    return type(value) in (int, float) and -sys.float_info.max <= value <= sys.float_info.max


def compare_observations(case, device, actual, reference):
    for result in [actual, reference]:
        if not isinstance(result, dict) or result.keys() != {"device", "dtype", "layout", "metadata", "values"}:
            return False
        if any(result[key] != expected for key, expected in (
            ("device", device), ("dtype", case["dtype"]), ("layout", case["layout"])
        )):
            return False
        for field in ["metadata", "values"]:
            if not isinstance(result[field], dict) or result[field].keys() != set(case["observe"][field]):
                return False
    # Distinguish booleans, integers and floats when comparing metadata.
    if json.dumps(actual["metadata"], sort_keys=True) != json.dumps(reference["metadata"], sort_keys=True):
        return False
    for key, expected in reference["values"].items():
        observed = actual["values"][key]
        if not isinstance(expected, list) or not isinstance(observed, list) or len(expected) != len(observed):
            return False
        for x, y in zip(observed, expected):
            if not finite_number(x) or not finite_number(y):
                return False
            if abs(x - y) > case["tolerance"]["atol"] + case["tolerance"]["rtol"] * abs(y):
                return False
    return True


def nonempty(value):
    return isinstance(value, str) and bool(value.strip())


def validate_case(root, case, ids, mapping, commit, validate_evidence):
    if not isinstance(case, dict):
        return ["behavior case must be a table"]
    case_id = case.get("id")
    missing = CASE_FIELDS - case.keys()
    unknown = case.keys() - CASE_FIELDS - {"upstream_restrictions"}
    if missing or unknown:
        return [f"{case_id}: missing fields {sorted(missing)}, unknown fields {sorted(unknown)}"]
    errors = []
    for field in CASE_FIELDS - {"devices", "tolerance", "observe"}:
        if not nonempty(case[field]):
            errors.append(f"{case_id}: {field} requires a nonempty string")
    if errors:
        return errors
    if not re.fullmatch(r"[a-z0-9]+(?:[.-][a-z0-9]+)*", case_id):
        errors.append(f"{case_id}: invalid stable case id")
    if case["inventory_id"] not in ids:
        errors.append(f"{case_id}: unknown inventory identity {case['inventory_id']}")
    if mapping.get(case["inventory_id"]) != case["ledger"]:
        errors.append(f"{case_id}: ledger does not match inventory disposition")
    if case["profile"] not in {"native", "python_interop"}:
        errors.append(f"{case_id}: unknown execution profile")
    observe = case["observe"]
    if (not isinstance(observe, dict) or observe.keys() != {"metadata", "values"}
            or any(not isinstance(names, list) or not names or not all(nonempty(n) for n in names)
                   or len(names) != len(set(names)) for names in observe.values())):
        errors.append(f"{case_id}: observe requires exact metadata and value field names")
    tolerance = case["tolerance"]
    if (not isinstance(tolerance, dict) or tolerance.keys() != {"rtol", "atol"}
            or any(not finite_number(v) or v < 0 for v in tolerance.values())):
        errors.append(f"{case_id}: tolerance requires nonnegative finite rtol and atol")
    for field, suffix in [("rust_test", ".rs"), ("reference_test", ".py")]:
        entry = case[field]
        if not entry.partition("::")[0].endswith(suffix):
            errors.append(f"{case_id}: {field} must point to a {suffix} test")
        try:
            local_file(root, entry.partition("::")[0])
            errors.extend(validate_evidence(root, f"{case_id}.{field}", [entry]))
        except (OSError, ValueError) as error:
            errors.append(f"{case_id}: malformed {field}: {error}")
    try:
        local_file(root, case["documentation"])
    except (OSError, ValueError) as error:
        errors.append(f"{case_id}: documentation: {error}")
    devices = case["devices"]
    if not isinstance(devices, dict) or not devices or devices.keys() - DEVICES:
        return [*errors, f"{case_id}: devices must name applicable hardware"]
    if any(not isinstance(s, str) or s not in STATUSES for s in devices.values()):
        errors.append(f"{case_id}: unknown device result status")
    inapplicable = {d for d, s in devices.items() if s == "upstream_inapplicable"}
    restrictions = case.get("upstream_restrictions", {})
    if not isinstance(restrictions, dict) or restrictions.keys() != inapplicable:
        errors.append(f"{case_id}: upstream-inapplicable cells require matching restrictions")
    else:
        for device, why in restrictions.items():
            if (not isinstance(why, dict) or why.keys() != {"source", "reason"}
                    or not nonempty(why["source"]) or not nonempty(why["reason"])
                    or not why["source"].startswith(f"https://github.com/pytorch/pytorch/blob/{commit}/")):
                errors.append(f"{case_id}/{device}: expected a pinned upstream restriction and reason")
    return errors


def recorded_status(case, device, cell):
    if isinstance(cell, dict):
        if cell.keys() == {"outcome", "reason"} and nonempty(cell["reason"]):
            if cell["outcome"] == "skipped":
                return "untested"
            if cell["outcome"] == "error":
                return "fail"
        if cell.keys() == {"outcome", "rust", "reference"} and cell["outcome"] == "executed":
            return "pass" if compare_observations(case, device, cell["rust"], cell["reference"]) else "fail"
    raise ValueError(f"{case['id']}/{device}: malformed observation")


def locked_tch(root):
    with (root / "Cargo.lock").open("rb") as lock:
        packages = tomllib.load(lock).get("package", [])
    versions = [p.get("version") for p in packages if isinstance(p, dict) and p.get("name") == "tch"]
    if len(versions) != 1 or not nonempty(versions[0]):
        raise ValueError("Cargo.lock must contain one tch version")
    return versions[0]


def validate_behavior(root, document, inventory, mapping, validate_evidence):
    """Validate sidecar claims against exact inventory dispositions and saved paired runs."""
    if not isinstance(document, dict) or document.keys() != {"format_version", "pytorch_commit", "observations", "case"}:
        return ["behavior: missing or unknown top-level fields"]
    if type(document["format_version"]) is not int or document["format_version"] != 1:
        return ["behavior: unsupported format version"]
    if document["pytorch_commit"] != inventory["pytorch_commit"]:
        return ["behavior: wrong PyTorch commit"]
    cases = document["case"]
    if not isinstance(cases, list) or not cases:
        return ["behavior: at least one case is required"]
    errors = []
    ids = {item["id"] for item in inventory["symbols"]}
    by_id = {}
    for case in cases:
        issues = validate_case(root, case, ids, mapping, inventory["pytorch_commit"], validate_evidence)
        errors.extend(issues)
        if not issues:
            if case["id"] in by_id:
                errors.append(f"duplicate behavior case id {case['id']}")
            by_id[case["id"]] = case
    if errors:
        return errors
    observations = document["observations"]
    if (not isinstance(observations, list) or not all(nonempty(item) for item in observations)
            or len(set(observations)) != len(observations)):
        return ["behavior: observations must be a list of unique run files"]
    statuses = defaultdict(set)
    for filename in observations:
        try:
            report = json.loads(local_file(root, filename).read_text(encoding="utf-8"))
            if (not isinstance(report, dict) or report.keys() != {"format_version", "environment", "cases", "fingerprints"}
                    or type(report["format_version"]) is not int or report["format_version"] != 1):
                raise ValueError("malformed run")
            env = report["environment"]
            if (not isinstance(env, dict)
                    or any(not nonempty(env.get(k)) for k in ["platform", "python", "rustc", "revision", "torch", "torch_commit"])
                    or re.fullmatch(r"[0-9a-f]{40}", env["revision"]) is None
                    or env["torch"].split("+")[0] != inventory["pytorch_version"]
                    or env["torch_commit"] != inventory["pytorch_commit"]
                    or env.get("tch") != locked_tch(root)):
                raise ValueError("missing environment fingerprint or wrong torch runtime")
            hardware = env.get("devices")
            if (not isinstance(hardware, dict) or "cpu" not in hardware or hardware.keys() - DEVICES
                    or not all(nonempty(v) for v in hardware.values())
                    or ("cuda" in hardware and not nonempty(env.get("cuda")))):
                raise ValueError("missing or inconsistent device fingerprint")
            if (not isinstance(report["cases"], dict) or not isinstance(report["fingerprints"], dict)
                    or report["cases"].keys() != report["fingerprints"].keys()):
                raise ValueError("case fingerprints do not cover the run")
            for case_id, cells in report["cases"].items():
                case = by_id.get(case_id)
                if not case or fingerprint(root, case) != report["fingerprints"][case_id]:
                    raise ValueError(f"unknown or changed case {case_id}")
                if not isinstance(cells, dict) or cells.keys() - case["devices"].keys():
                    raise ValueError(f"{case_id}: unknown device cells")
                for device, cell in cells.items():
                    status = recorded_status(case, device, cell)
                    if status == "pass" and device not in hardware:
                        raise ValueError(f"{case_id}/{device}: untested hardware in this environment")
                    statuses[case_id, device].add(status)
        except (OSError, ValueError) as error:
            errors.append(f"behavior observation {filename}: {error}")
    for case_id, case in by_id.items():
        for device, expected in case["devices"].items():
            seen = statuses[case_id, device]
            if expected == "upstream_inapplicable" and seen <= {"untested"}:
                continue
            actual = "fail" if "fail" in seen else "pass" if "pass" in seen else "untested"
            if actual != expected:
                errors.append(f"{case_id}/{device}: claimed {expected}, evidence establishes {actual}")
    return errors


def execute_cell(root, case, device, directory):
    path, name = case["reference_test"].split("::")
    spec = importlib.util.spec_from_file_location("reference_probe", local_file(root, path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    reference = getattr(module, name)(device)

    path, name = case["rust_test"].split("::")
    local_file(root, path)
    binary = Path(path).stem
    output = directory / "rust.json"
    env = {**os.environ, "RUSTTORCH_BEHAVIOR_DEVICE": device, "RUSTTORCH_BEHAVIOR_OUTPUT": str(output)}
    subprocess.run(["cargo", "test", "--locked", "-p", "rusttorch", "--test", binary,
                    "--", "--exact", name], cwd=root, env=env, check=True)
    rust = json.loads(output.read_text(encoding="utf-8"))
    return {"outcome": "executed", "rust": rust, "reference": reference}


def run(root, output):
    # Disable opt-in MPS CPU fallback before importing torch or starting LibTorch.
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
    import torch

    document = tomllib.loads((root / "compat/pytorch_behavior.toml").read_text(encoding="utf-8"))
    pinned = tomllib.loads((root / "compat/pytorch_reference.toml").read_text(encoding="utf-8"))
    if (torch.__version__.split("+")[0] != pinned["pytorch_version"]
            or torch.version.git_version != pinned["pytorch_commit"]
            or document["pytorch_commit"] != pinned["pytorch_commit"]):
        raise ValueError("behavior cases require the exact pinned PyTorch build")
    hardware = {"cpu": platform.machine()}
    if torch.backends.mps.is_available():
        hardware["mps"] = torch.backends.mps.get_name()
    if torch.cuda.is_available() and torch.version.cuda:
        hardware["cuda"] = torch.cuda.get_device_name(0)
    environment = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "rustc": subprocess.check_output(["rustc", "-V"], text=True).strip(),
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "torch": torch.__version__,
        "torch_commit": torch.version.git_version,
        "cuda": torch.version.cuda,
        "hip": torch.version.hip,
        "tch": locked_tch(root),
        "devices": hardware,
    }
    report = {"format_version": 1, "environment": environment, "fingerprints": {}, "cases": {}}
    failures = 0
    for case in document["case"]:
        case_id = case["id"]
        report["fingerprints"][case_id] = fingerprint(root, case)
        cells = report["cases"][case_id] = {}
        for device, status in case["devices"].items():
            if status == "upstream_inapplicable":
                continue
            if device not in hardware:
                cells[device] = {"outcome": "skipped", "reason": f"{device} unavailable in this runtime"}
                print(f"{case_id} {device}: untested", flush=True)
                continue
            try:
                with tempfile.TemporaryDirectory() as temporary:
                    cell = execute_cell(root, case, device, Path(temporary))
                passed = compare_observations(case, device, cell["rust"], cell["reference"])
            except (OSError, ValueError, RuntimeError, AssertionError, subprocess.CalledProcessError) as error:
                cell = {"outcome": "error", "reason": str(error) or type(error).__name__}
                passed = False
            cells[device] = cell
            print(f"{case_id} {device}: {'pass' if passed else 'fail'}", flush=True)
            if not passed:
                failures += 1
                print(json.dumps(cell), file=sys.stderr)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return int(bool(failures))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, action="store_true", help="execute every locally available case")
    parser.add_argument("--output", type=Path, default=ROOT / "target/pytorch-behavior.json")
    arguments = parser.parse_args()
    sys.exit(run(ROOT, arguments.output))
