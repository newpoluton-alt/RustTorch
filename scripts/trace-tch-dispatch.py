#!/usr/bin/env python3
"""Trace generated tch wrappers through a Clang-compiled bridge to ATen overloads.

Run with the pinned torch installed and the unpacked tch and torch-sys crates.
Output is one representative Rust method per exact, unambiguous dispatcher schema.
Only source availability is recorded. The IR stays in a temporary directory.
"""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile
import tomllib

ROOT = Path(__file__).resolve().parents[1]
TCH_COMMIT = "4227b89b72059b0651ff83a38637693574e0a2d6"
HEADER = "inventory_id\trust_symbol\tffi_symbol\tdispatcher\n"


def collect_routes(tch, sys_crate, include, ir, inventory):
    signatures = {
        item["signature"]: item["id"]
        for item in inventory["symbols"] if item["kind"] == "aten_schema"
    }
    # gen.ml reads these declaration fields when selecting functions. Some
    # C++ names share multiple schemas, so this match alone is insufficient.
    declarations = (tch / "third_party/pytorch/Declarations-v2.13.0.yaml").read_text()
    cpp_names = defaultdict(set)
    for block in declarations.split("- name: ")[1:]:
        schema = re.search(r"^  schema_string: (aten::[^\n]*)$", block, re.MULTILINE)
        if schema:
            cpp_names[schema[1]].add(block.splitlines()[0])

    headers = {}
    for header in (include / "ATen/ops").glob("*_ops.h"):
        for name, schema in re.findall(
            r'struct TORCH_API (\w+) \{[^}]*? schema_str = "([^"\n]*)";',
            header.read_text(), re.DOTALL,
        ):
            headers[name] = "aten::" + schema

    bridge = (sys_crate / "libtch/torch_api_generated.cpp").read_text()
    cpp_calls = {}
    for name, body in re.findall(r"^\w+[ *]+(atg_\w+)\([^\n]*\) \{(.*?)^\}", bridge, re.M | re.S):
        primary = re.search(r"auto outputs__ = (?:torch::|self->)(\w+)\(", body)
        if primary:
            cpp_calls[name] = primary[1]

    ffi = (sys_crate / "src/c_generated.rs").read_text()
    externs = set(re.findall(r"^    pub fn (atg_\w+)\(", ffi, re.M))
    rust = (tch / "src/wrappers/tensor_fallible_generated.rs").read_text()
    methods = {}
    for method, body in re.findall(r"^    pub fn (f_\w+)\b(.*?)^    \}", rust, re.M | re.S):
        calls = set(re.findall(r"\b(atg_\w+)\(", body))
        if len(calls) == 1:
            methods[calls.pop()] = method

    routes = defaultdict(list)
    for c_fn, body in re.findall(r"^define [^\n]*?@(atg_\w+)\([^\n]*\{(.*?)^\}", ir, re.M | re.S):
        # In the Itanium C++ ABI these name components encode
        # at::_ops::<dispatcher>::call; the compiler selected the overload.
        calls = set(re.findall(r"@_ZN2at4_ops\d+(\w+?)4callE", body))
        if len(calls) != 1 or c_fn not in externs or c_fn not in methods:
            continue
        dispatcher = calls.pop()
        schema = headers.get(dispatcher)
        if schema in signatures and cpp_calls.get(c_fn) in cpp_names[schema]:
            routes[signatures[schema]].append((f"tch::Tensor::{methods[c_fn]}", c_fn, dispatcher))
    if not routes:
        raise ValueError("no exact matching dispatch traces; check the compiler ABI and source pins")
    return HEADER + "".join(
        "\t".join((key, *min(routes[key]))) + "\n" for key in sorted(routes)
    )


def verify_trace(root, generated):
    """Verify both the table and its digest against a new compiler trace."""
    published = (root / "compat/pytorch_tch_dispatch.tsv").read_text(encoding="utf-8")
    manifest = tomllib.loads((root / "compat/pytorch_bindings.toml").read_text(encoding="utf-8"))
    if generated != published:
        raise ValueError("generated dispatcher rows differ from the committed table")
    if hashlib.sha256(generated.encode("utf-8")).hexdigest() != manifest["trace_sha256"]:
        raise ValueError("generated dispatcher digest differs from the binding manifest")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tch-source", type=Path, required=True)
    parser.add_argument("--torch-sys-source", type=Path, required=True)
    parser.add_argument("--clang", default="clang++")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output", type=Path, help="write the new trace for review")
    mode.add_argument("--check", action="store_true", help="compare with the reviewed trace and digest")
    args = parser.parse_args()
    import torch  # Only regeneration needs the pinned runtime and its headers.

    inventory = json.loads((ROOT / "compat/pytorch_inventory.json").read_text())
    if torch.version.git_version != inventory["pytorch_commit"]:
        parser.error("torch wheel must match the inventory commit")
    for src in (args.tch_source, args.torch_sys_source):
        if json.loads((src / ".cargo_vcs_info.json").read_text())["git"]["sha1"] != TCH_COMMIT:
            parser.error(f"wrong tch source commit in {src}")
    include = Path(torch.__file__).parent / "include"
    with tempfile.TemporaryDirectory() as temp:
        ir = Path(temp) / "bridge.ll"
        subprocess.run(
            [args.clang, "-std=c++17", "-Wno-deprecated-declarations", "-O1", "-S", "-emit-llvm",
             "-I", str(include), "-I", str(include / "torch/csrc/api/include"),
             str(args.torch_sys_source / "libtch/torch_api_generated.cpp"), "-o", str(ir)],
            check=True,
        )
        contents = collect_routes(args.tch_source, args.torch_sys_source, include, ir.read_text(), inventory)
    if args.check:
        try:
            verify_trace(ROOT, contents)
        except ValueError as error:
            parser.exit(1, f"error: {error}\n")
        print(f"verified {len(contents.splitlines()) - 1} dispatcher routes")
    else:
        args.output.write_bytes(contents.encode("utf-8"))


if __name__ == "__main__":
    main()
