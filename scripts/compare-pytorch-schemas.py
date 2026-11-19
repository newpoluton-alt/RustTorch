#!/usr/bin/env python3
"""Compare shipping ATen contracts with schemas generated from the frozen 2.14 source."""

import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
WHEEL_COMMIT = "08187d9e0fba026dc8217405802ab5381dc88d90"
TARGET = "2b3ec34829036a65cd9d1398ea72a0167dc37470"


def read_target(source):
    """Use torchgen from the requested checkout, independent of an installed wheel."""
    for args, expected in [(["rev-parse", "HEAD"], TARGET), (["status", "--porcelain"], "")]:
        actual = subprocess.check_output(["git", "-C", source, *args], text=True).strip()
        if actual != expected:
            raise ValueError(f"expected clean upstream checkout at {TARGET}")
    code = """
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
sys.path.insert(0, str(root))
from torchgen.gen import parse_native_yaml
import torchgen
assert pathlib.Path(torchgen.__file__).resolve().parent == root / 'torchgen'
native = root / 'aten/src/ATen/native'
entries = parse_native_yaml(str(native / 'native_functions.yaml'), str(native / 'tags.yaml'))
print(json.dumps([['aten:aten::' + str(fn.func.name), 'aten::' + str(fn.func)]
                  for fn in entries.native_functions]))
"""
    rows = json.loads(subprocess.check_output([sys.executable, "-I", "-c", code, source], text=True))
    if len(dict(rows)) != len(rows):
        raise ValueError("duplicate generated schema identity")
    return dict(rows)


def census_tools():
    spec = importlib.util.spec_from_file_location("pytorch_inventory", ROOT / "scripts/sync-pytorch-inventory.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def wheel_reference(tools):
    return dict(tools.REFERENCE, pytorch_version="2.14.0", pytorch_commit=WHEEL_COMMIT)


def collect_wheel_census(source):
    """Run the shipping collectors against an independent, exact wheel/source pair."""
    tools = census_tools()
    reference = wheel_reference(tools)
    if sys.prefix == sys.base_prefix:
        raise ValueError("collect the wheel census using the candidate Python virtual environment")
    with tempfile.TemporaryDirectory(prefix="rusttorch-wheel-census-") as temp:
        # The collectors normally resolve .venv and the Sphinx helpers relative to
        # the repository root. Isolate those inputs and leave production pins alone.
        tools.ROOT = Path(temp)
        project = Path(sys.prefix).parent
        for filename in ("pyproject.toml", "uv.lock"):
            # Sphinx runs through uv --frozen --offline against this lock.
            shutil.copyfile(project / filename, tools.ROOT / filename)
        (tools.ROOT / ".venv").symlink_to(Path(sys.prefix), target_is_directory=True)
        (tools.ROOT / "scripts").symlink_to(ROOT / "scripts", target_is_directory=True)
        runtime = tools.verify_runtime(reference)
        source = tools.verify_source_checkout(source, reference)
        export = tools.build_sphinx_domain_export(source, reference)
        tracked = subprocess.check_output(["git", "-C", source, "ls-files", "-z"], text=True)
        symbols = tools.collect_documented_symbols(export, source, tracked_paths=set(tracked.split("\0")))
        symbols += tools.collect_aten_schemas(source)
        fingerprint = tools.run_python("""import hashlib, json, platform, sys, torch
print(json.dumps(dict(wheel_version=torch.__version__, git_commit=torch.version.git_version,
    platform=sys.platform, machine=platform.machine(), cuda=torch.version.cuda, hip=torch.version.hip,
    debug=torch.version.debug, torch_config_sha256=hashlib.sha256(torch.__config__.show().encode()).hexdigest())))
""")
        tools.verify_source_checkout(source, reference)
    inventory = dict(format_version=1, pytorch_version=runtime["version"], pytorch_commit=runtime["commit"],
                     scope=["torch"], symbols=sorted(symbols, key=lambda item: item["id"]), runtime_build=fingerprint)
    return render_wheel_census(inventory)


def render_wheel_census(inventory):
    tools = census_tools()
    return tools.render_inventory(inventory, wheel_reference(tools))


def semantic_delta(baseline, candidate, schemas):
    tools = census_tools()
    reference = wheel_reference(tools)
    errors = tools.validate_inventory(candidate, reference)
    if errors:
        raise ValueError("; ".join(errors))
    if (candidate["scope"] != ["torch"] or candidate.get("runtime_build", {}).get("git_commit") != WHEEL_COMMIT):
        raise ValueError("candidate inventory needs complete torch scope and a matching wheel fingerprint")
    records = {s["id"]: s for s in candidate["symbols"]}
    runtime_schemas = {key: r["signature"] for key, r in records.items() if r["kind"] == "aten_schema"}
    if runtime_schemas != schemas:
        raise ValueError("wheel-census ATen schemas differ from canonical target source")
    base = {s["id"]: s for s in baseline["symbols"]}
    common = records.keys() & base.keys()
    changed = {}
    for field in ("kind", "signature", "documented_signature", "runtime_present", "visibility", "generated_from"):
        changed[field] = {
            key: {"before": base[key].get(field), "after": records[key].get(field)}
            for key in common if base[key].get(field) != records[key].get(field)
        }
    return {
        "runtime": candidate["runtime_build"], "baseline_total": len(base), "target_total": len(records),
        "added": sorted(records.keys() - base.keys()), "removed": sorted(base.keys() - records.keys()),
        "changed": changed,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--source", type=Path, help="clean checkout at the exact frozen release commit")
    mode.add_argument("--collect-wheel-census", type=Path, metavar="WHEEL_SOURCE",
                      help="print the semantic census using the matching PyTorch 2.14 wheel and source")
    parser.add_argument("--candidate-inventory", type=Path,
                        help="include the documented API and registration delta from a wheel census")
    args = parser.parse_args()
    if args.collect_wheel_census:
        if args.candidate_inventory:
            parser.error("--candidate-inventory requires --source")
        sys.stdout.write(collect_wheel_census(args.collect_wheel_census.resolve()))
        return
    target = read_target(args.source.resolve())
    snapshot = json.loads((ROOT / "compat/pytorch_inventory.json").read_bytes())
    base = {s["id"]: s["signature"] for s in snapshot["symbols"] if s["kind"] == "aten_schema"}
    diff = {
        "baseline_commit": snapshot["pytorch_commit"], "target_commit": TARGET,
        "baseline_total": len(base), "target_total": len(target),
        "added": {key: target[key] for key in target.keys() - base.keys()},
        "removed": {key: base[key] for key in base.keys() - target.keys()},
        "changed": {key: {"before": base[key], "after": target[key]}
                    for key in base.keys() & target.keys() if base[key] != target[key]},
    }
    if args.candidate_inventory:
        candidate = json.loads(args.candidate_inventory.read_bytes())
        diff["semantic_census"] = semantic_delta(snapshot, candidate, target)
    print(json.dumps(diff, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        sys.exit(f"error: {error}")
