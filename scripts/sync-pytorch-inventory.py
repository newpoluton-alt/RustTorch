#!/usr/bin/env python3
"""Refresh from a clean pinned source checkout, or validate committed bytes offline."""
from __future__ import annotations

import argparse
import ast
import copy
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib

ROOT = Path(__file__).resolve().parents[1]
VERSION = "2.13.0"
COMMIT = "cf30153c4c131c8164ee7798e5022d810682e2cb"
REFERENCE = dict(format_version=1, pytorch_version=VERSION, pytorch_commit=COMMIT,
                 documentation_root="docs/source", sphinx_builder="dummy",
                 sphinx_inventory_format=1, sphinx_tags=["cpu", "linux"],
                 python_namespaces=["torch"], source_suffixes={".ipynb": "myst-nb", ".md": "myst-nb", ".rst": "restructuredtext"})
KINDS = {"class", "function", "method", "constant", "module", "aten_schema"}


def load_reference(path=ROOT / "compat/pytorch_reference.toml"):
    value = tomllib.loads(Path(path).read_text())
    if value != REFERENCE or any(type(value[k]) is not type(v) for k, v in REFERENCE.items()):
        raise ValueError("reference must exactly match the pinned version-1 manifest")
    return value


def safe_path(value):
    if not isinstance(value, str) or not value or "\\" in value:
        return False
    path = PurePosixPath(value)
    return not path.is_absolute() and ".." not in path.parts and path.as_posix() == value


def validate_inventory(inventory, reference=REFERENCE):
    errors = []
    expected = {"format_version", "pytorch_version", "pytorch_commit", "scope", "symbols"}
    if not isinstance(inventory, dict) or not expected <= set(inventory) or set(inventory) - expected - {"runtime_build"}:
        return ["inventory fields do not match format 1"]
    if "runtime_build" in inventory:
        build = inventory["runtime_build"]
        fields = {"wheel_version", "git_commit", "platform", "machine", "cuda", "hip", "debug", "torch_config_sha256"}
        if not isinstance(build, dict) or set(build) != fields:
            errors.append("invalid runtime build fingerprint fields")
        else:
            version = build["wheel_version"]
            if not isinstance(version, str) or not re.fullmatch(re.escape(reference["pytorch_version"]) + r"(?:\+[A-Za-z0-9][A-Za-z0-9._-]*)?", version):
                errors.append("runtime wheel version differs from reference")
            if build["git_commit"] != reference["pytorch_commit"]:
                errors.append("runtime commit differs from reference")
            if build["platform"] not in ("linux", "darwin", "win32"):
                errors.append("invalid runtime platform")
            if not isinstance(build["machine"], str) or not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*", build["machine"]):
                errors.append("invalid runtime machine")
            for field in ("cuda", "hip"):
                value = build[field]
                if value is not None and (not isinstance(value, str) or not re.fullmatch(r"[0-9]+(?:\.[0-9]+)+(?:[A-Za-z0-9._+-]*)?", value)):
                    errors.append(f"invalid runtime {field} version")
            if type(build["debug"]) is not bool:
                errors.append("runtime debug must be Boolean")
            if not isinstance(build["torch_config_sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", build["torch_config_sha256"]):
                errors.append("invalid runtime configuration SHA256")
    for key in ("format_version", "pytorch_version", "pytorch_commit"):
        if type(inventory[key]) is not type(reference[key]) or inventory[key] != reference[key]:
            errors.append(f"inventory {key} differs from reference")
    scope = inventory["scope"]
    def qualified(value):
        return isinstance(value, str) and all(part.isidentifier() for part in value.split("."))
    if not isinstance(scope, list) or not scope or any(not qualified(n) or not (n == "torch" or n.startswith("torch.")) for n in scope) or scope != sorted(set(scope)):
        errors.append("inventory scope must be sorted unique torch namespaces")
        scope = []
    symbols = inventory["symbols"]
    if not isinstance(symbols, list) or not symbols:
        return errors + ["inventory symbols must be nonempty"]
    ids, overloads, generated = [], {}, []
    for symbol in symbols:
        if not isinstance(symbol, dict):
            errors.append("inventory symbol must be an object")
            continue
        required = {"id", "kind", "module", "name", "signature", "source"}
        allowed = required | {"line", "documentation", "alias", "signature_reason", "documented_signature", "documented_signature_reason", "visibility", "runtime_present", "generated_from"}
        if not required <= set(symbol) or set(symbol) - allowed:
            errors.append("invalid symbol fields")
            continue
        ident = symbol["id"]
        if not isinstance(ident, str):
            errors.append("inventory ID must be a string")
            continue
        ids.append(ident)
        kind = symbol["kind"]
        if not isinstance(kind, str) or kind not in KINDS:
            errors.append(f"{ident}: invalid kind")
            continue
        if not safe_path(symbol["source"]) or symbol["source"] == ".":
            errors.append(f"{ident}: invalid source path")
        signature = symbol["signature"]
        if signature is not None and (not isinstance(signature, str) or not signature.strip() or re.search(r"0x[0-9a-fA-F]{6,}", signature)):
            errors.append(f"{ident}: unstable signature")
        if "documentation" in symbol and (not safe_path(symbol["documentation"]) or not symbol["documentation"].startswith("docs/source/") or symbol["documentation"] == "docs/source/"):
            errors.append(f"{ident}: invalid documentation path")
        if "alias" in symbol and type(symbol["alias"]) is not bool:
            errors.append(f"{ident}: alias must be Boolean")
        if "signature_reason" in symbol and (signature is not None or not isinstance(symbol["signature_reason"], str) or not symbol["signature_reason"].strip()):
            errors.append(f"{ident}: invalid missing-signature reason")
        if "documented_signature" in symbol:
            documented = symbol["documented_signature"]
            reason = symbol.get("documented_signature_reason")
            if documented is None:
                if not isinstance(reason, str) or not reason.strip():
                    errors.append(f"{ident}: missing documented-signature reason")
            elif not isinstance(documented, str) or not documented.strip() or re.search(r"0x[0-9a-fA-F]+|<[^>]*\bobject\b[^>]*>", documented) or "documented_signature_reason" in symbol:
                errors.append(f"{ident}: invalid documented signature")
        elif "documented_signature_reason" in symbol:
            errors.append(f"{ident}: documented-signature reason without signature field")
        if "line" in symbol and (type(symbol["line"]) is not int or symbol["line"] < 1):
            errors.append(f"{ident}: invalid source line")
        if kind == "aten_schema":
            name = symbol["name"]
            valid_name = isinstance(name, str) and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)?", name)
            if not valid_name or ident != "aten:aten::" + name or symbol["module"] != "aten":
                errors.append(f"{ident}: malformed ATen identity")
            if not valid_name or not isinstance(signature, str) or not signature.startswith("aten::" + name + "(") or not re.search(r"\) -> .+\Z", signature):
                errors.append(f"{ident}: schema signature does not match identity")
            if symbol["source"] != "aten/src/ATen/native/native_functions.yaml" or "line" not in symbol:
                errors.append(f"{ident}: missing canonical ATen source location")
            visibility = "internal" if isinstance(name, str) and name.startswith("_") else "public"
            if symbol.get("visibility") != visibility or type(symbol.get("runtime_present")) is not bool:
                errors.append(f"{ident}: missing ATen visibility/runtime availability")
            if any(field in symbol for field in ("documentation", "alias", "signature_reason", "documented_signature", "documented_signature_reason")):
                errors.append(f"{ident}: Python-only provenance fields on ATen schema")
            if "generated_from" in symbol:
                origin = symbol["generated_from"]
                if not isinstance(origin, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)?", origin):
                    errors.append(f"{ident}: invalid generated schema origin")
                else:
                    generated.append((ident, "aten:aten::" + origin))
        else:
            match = re.fullmatch(r"python:([^#]+)(?:#overload-([1-9][0-9]*))?", ident)
            name = match[1] if match else ""
            if not match or not qualified(name) or not any(name == n or name.startswith(n + ".") for n in scope):
                errors.append(f"{ident}: malformed Python identity or outside declared scope")
            module, short = symbol["module"], symbol["name"]
            if not qualified(module) or not qualified(short) or not (module == "torch" or module.startswith("torch.")):
                errors.append(f"{ident}: invalid name/module")
            elif (name != module or kind != "module" or short != module) and name != module + "." + short:
                errors.append(f"{ident}: Python name/module does not match identity")
            if any(field in symbol for field in ("visibility", "runtime_present", "generated_from")):
                errors.append(f"{ident}: ATen-only provenance fields on Python object")
            if match:
                overloads.setdefault(name, []).append(int(match[2]) if match[2] else None)
    if ids != sorted(set(ids)):
        errors.append("inventory IDs must be sorted and unique")
    for name, numbers in overloads.items():
        if numbers != [None] and (None in numbers or len(numbers) < 2 or sorted(numbers) != list(range(1, len(numbers) + 1))):
            errors.append(f"{name}: overload identities must be a contiguous sequence of at least two")
    known = set(ids)
    for ident, origin in generated:
        if origin == ident or origin not in known:
            errors.append(f"{ident}: generated schema origin is absent or self-referential")
    for namespace in scope:
        if not any(i == "python:" + namespace or i.startswith("python:" + namespace + ".") for i in ids):
            errors.append(f"empty declared namespace: {namespace}")
    return errors

def validate_mapping(inventory, mapping, ledger):
    if not isinstance(mapping, dict) or set(mapping) != {"format_version", "mapping"} or type(mapping["format_version"]) is not int or mapping["format_version"] != 1 or not isinstance(mapping["mapping"], dict):
        return ["mapping fields do not match format 1"]
    entries = mapping["mapping"]
    ids = {s["id"] for s in inventory["symbols"]}
    rows = {r["id"]: r for r in ledger["api"]}
    symbols = {s["id"]: s for s in inventory["symbols"]}
    errors = []
    for ident in sorted(ids - entries.keys()):
        errors.append(f"missing inventory mapping: {ident}")
    for ident in sorted(entries.keys() - ids):
        errors.append(f"unknown mapped inventory ID: {ident}")
    for ident, row in entries.items():
        if not isinstance(row, str) or row not in rows:
            errors.append(f"{ident}: unknown ledger target {row!r}")
            continue
        symbol = symbols.get(ident)
        if symbol is None:
            continue
        target = rows[row]
        if symbol["kind"] == "aten_schema":
            if symbol.get("visibility") == "internal":
                if row != "census.aten.internal" or target.get("status") != "not_supported" or target.get("implementation") != "none":
                    errors.append(f"{ident}: internal schema requires the non-public internal disposition")
            elif row == "census.aten.internal":
                errors.append(f"{ident}: public schema cannot use an internal disposition")
            elif target.get("status") in {"supported", "partial"}:
                if ident not in target.get("python_symbols", []) or not target.get("evidence"):
                    errors.append(f"{ident}: implemented ATen target requires its exact schema ID and evidence")
            elif target.get("status") not in {"planned", "not_supported"}:
                errors.append(f"{ident}: unimplemented ATen target must be planned or not_supported")
            elif target.get("source") != "aten/src/ATen/native/native_functions.yaml":
                errors.append(f"{ident}: unimplemented ATen target must explicitly scope canonical schemas")
        elif target.get("status") in {"supported", "partial"}:
            canonical = ident.removeprefix("python:").split("#overload-")[0]
            if canonical not in target.get("python_symbols", []):
                errors.append(f"{ident}: implemented target must explicitly list the exact Python symbol")
    if list(entries) != sorted(entries):
        errors.append("mapping IDs must be sorted")
    return errors


def inventory_counts(mapping):
    return Counter(mapping["mapping"].values())


def render_inventory(inventory):
    errors = validate_inventory(inventory)
    if errors:
        raise ValueError("; ".join(errors))
    return json.dumps(inventory, indent=2, ensure_ascii=False) + "\n"


def load_snapshot(root=ROOT, ledger=None):
    root = Path(root)
    reference = load_reference(root / "compat/pytorch_reference.toml")
    path = root / "compat/pytorch_inventory.json"
    inventory = json.loads(path.read_text())
    errors = validate_inventory(inventory, reference)
    if errors:
        raise ValueError("; ".join(errors))
    if inventory.get("scope") != reference["python_namespaces"]:
        errors.append("production inventory scope must equal the reference namespaces")
    if "runtime_build" not in inventory:
        errors.append("production inventory requires the runtime build fingerprint")
    symbols = inventory.get("symbols", [])
    if not isinstance(symbols, list) or not any(isinstance(row, dict) and row.get("kind") == "aten_schema" for row in symbols):
        errors.append("production inventory requires canonical ATen schemas")
    if errors:
        raise ValueError("; ".join(errors))
    if path.read_bytes() != render_inventory(inventory).encode():
        raise ValueError("inventory JSON bytes are not canonical")
    mapping = tomllib.loads((root / "compat/pytorch_inventory_map.toml").read_text())
    if ledger is None:
        ledger = tomllib.loads((root / "compat/pytorch_api.toml").read_text())
    errors = validate_mapping(inventory, mapping, ledger)
    if errors:
        raise ValueError("; ".join(errors))
    return inventory, mapping


def atomic_write(path, contents):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix="." + path.name, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(contents.encode())
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def clean_environment():
    return {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": os.environ.get("HOME", "/tmp"),
            "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8", "PYTHONHASHSEED": "0", "PYTHONNOUSERSITE": "1",
            "UV_OFFLINE": "1", "UV_NO_CACHE": "1", "TOKENIZERS_PARALLELISM": "false"}


def run_python(code, *args, timeout=120):
    process = subprocess.run([str(ROOT / ".venv/bin/python"), "-I", "-c", code, *map(str, args)],
                             cwd=tempfile.gettempdir(), env=clean_environment(), text=True,
                             capture_output=True, timeout=timeout)
    if process.returncode:
        raise ValueError("pinned runtime subprocess failed: " + process.stderr[-4000:])
    return json.loads(process.stdout)


def verify_runtime(reference=REFERENCE, runtime=None):
    runtime = runtime if runtime is not None else run_python("import json,torch; print(json.dumps({'version':torch.__version__.split('+')[0], 'commit':torch.version.git_version}))")
    if runtime != {"version": reference["pytorch_version"], "commit": reference["pytorch_commit"]}:
        raise ValueError("installed torch version/commit differs from pinned reference")
    return runtime


def verify_source_checkout(source, reference=REFERENCE):
    source = Path(source).resolve()
    for args, expected in [(["rev-parse", "HEAD"], reference["pytorch_commit"]), (["status", "--porcelain", "--untracked-files=all"], "")]:
        result = subprocess.run(["git", "-C", str(source), *args], text=True, capture_output=True, timeout=60)
        if result.returncode or result.stdout.strip() != expected:
            raise ValueError("upstream source must be clean and at the exact pinned commit")
    if not any((source / ("docs/source/index" + suffix)).is_file() for suffix in reference["source_suffixes"]):
        raise ValueError("upstream checkout missing documentation index")
    for path in ("aten/src/ATen/native/native_functions.yaml", "aten/src/ATen/native/tags.yaml"):
        if not (source / path).is_file():
            raise ValueError(f"upstream checkout missing {path}")
    return source


def build_sphinx_domain_export(source, reference=REFERENCE, *, source_root=None, timeout=1800):
    """Copy docs so autosummary never modifies the verified upstream checkout."""
    docs = Path(source_root) if source_root else Path(source) / reference["documentation_root"]
    with tempfile.TemporaryDirectory(prefix="rusttorch-sphinx-") as directory:
        work = Path(directory)
        config = work / "config"
        config.mkdir()
        shutil.copyfile(ROOT / "scripts/pytorch_inventory_conf.py", config / "conf.py")
        shutil.copyfile(ROOT / "scripts/pytorch_inventory_sphinx.py", config / "pytorch_inventory_sphinx.py")
        copied_docs = work / "tree/docs/source"
        if source_root is None:
            tracked = subprocess.run(["git", "-C", str(source), "ls-files", "-z"], check=True,
                                     text=True, capture_output=True, timeout=60).stdout.split("\0")
            # Copy tracked files only, preserving relative include paths. Ignored
            # generated pages or snippets can never contaminate the pinned build.
            for relative in filter(None, tracked):
                if not safe_path(relative):
                    raise ValueError("unsafe tracked upstream path")
                original = Path(source) / relative
                if original.is_file() or original.is_symlink():
                    destination = work / "tree" / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(original, destination, follow_symlinks=False)
            document_sources = sorted(p.removeprefix("docs/source/") for p in tracked if p.startswith("docs/source/"))
        else:
            shutil.copytree(docs, copied_docs)
            document_sources = sorted(p.relative_to(docs).as_posix() for p in docs.rglob("*") if p.is_file())
        (config / "document-sources.json").write_text(json.dumps(document_sources))
        env = clean_environment()
        if source_root is None:
            # The pinned Makefile generates ExportDB before Sphinx. Run only this
            # reviewed tracked generator, isolated from the source import path;
            # its examples come from the verified wheel and outputs stay staged.
            generator = copied_docs / "scripts/exportdb/generate_example_rst.py"
            generated = subprocess.run([str(ROOT / ".venv/bin/python"), "-I", str(generator)],
                                       cwd=work, env=env, text=True, capture_output=True, timeout=300)
            if generated.returncode:
                diagnostic = work.with_suffix(".exportdb.log")
                diagnostic.write_text(generated.stdout + "\n" + generated.stderr)
                raise ValueError(f"pinned ExportDB generator failed; diagnostics: {diagnostic}")
        filenames = {}
        upstream_config = docs / "conf.py"
        if upstream_config.is_file():
            # Read only a literal filename map: never import/execute upstream configuration.
            for node in ast.parse(upstream_config.read_text()).body:
                if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "autosummary_filename_map" for t in node.targets):
                    filenames = ast.literal_eval(node.value)
            if not isinstance(filenames, dict) or any(not isinstance(k, str) or not safe_path(v) for k, v in filenames.items()):
                raise ValueError("unsafe upstream autosummary filename map")
        env.update(PYTHONPATH=str(config), RUSTTORCH_SOURCE_SUFFIXES=json.dumps(reference["source_suffixes"]),
                   RUSTTORCH_NAMESPACES=json.dumps(reference["python_namespaces"]),
                   RUSTTORCH_UPSTREAM_SOURCE=str(Path(source).resolve()),
                   RUSTTORCH_DOCUMENTATION_SOURCE=str(docs.resolve()),
                   RUSTTORCH_DOCUMENT_SOURCES=str(config / "document-sources.json"),
                   RUSTTORCH_AUTOSUMMARY_FILENAMES=json.dumps(filenames))
        command = ["uv", "run", "--project", str(ROOT), "--frozen", "--offline", "python", "-m", "sphinx", "-b", reference["sphinx_builder"], "-c", str(config), "-d", str(work / "doctrees")]
        for tag in reference["sphinx_tags"]:
            command.extend(["-t", tag])
        command.extend([str(copied_docs), str(work / "output")])
        process = subprocess.run(command, cwd=work, env=env, text=True, capture_output=True, timeout=timeout)
        semantic_failure = re.search(r"Unknown directive|Unknown interpreted text role|myst\.directive_unknown|autosummary.*stub file not found|failed to import|duplicate object description|\[toc\.(?:not_readable|glob)\]|toctree.*(?:nonexisting|non-existing|didn't match|excluded document)|(?:include|literalinclude).*?(?:not found|failed|does not exist)|ERROR:", process.stderr, re.IGNORECASE)
        diagnostic = work.with_suffix(".log")
        diagnostic.write_text(process.stdout + "\n" + process.stderr)
        export_path = work / "output/pytorch-domain.json"
        if export_path.is_file():
            # Keep the raw export for diagnosing schema normalization even when
            # semantic validation fails; it is never a committed snapshot.
            work.with_suffix(".raw.json").write_text(export_path.read_text())
        if process.returncode or semantic_failure:
            # Preserve diagnostics, not downloaded source or build artifacts.
            reason = process.stderr[-3000:]
            if semantic_failure:
                reason = process.stderr[semantic_failure.start():].splitlines()[0] + "\n" + reason
            raise ValueError(f"Sphinx census failed; diagnostics: {diagnostic}\n" + reason)
        exported = (work / "output/pytorch-domain.json").read_text()
        work.with_suffix(".json").write_text(exported)
        return json.loads(exported)


def source_definition(source, path, qualname):
    """Extract default expressions without calling repr on arbitrary objects."""
    file = Path(source) / path
    if not file.is_file() or file.suffix != ".py":
        return None
    try:
        tree = parse_source(file)
    except (SyntaxError, UnicodeError):
        return None
    names = qualname.split(".")
    current = tree
    for name in names:
        current = next((n for n in current.body if isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name), None)
        if current is None:
            return None
    if isinstance(current, ast.ClassDef):
        current = next((n for n in current.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == "__init__"), None)
    if current is None:
        return None
    args = copy.deepcopy(current.args)
    for arg in [*args.posonlyargs, *args.args, *args.kwonlyargs]:
        arg.annotation = None
    if args.vararg:
        args.vararg.annotation = None
    if args.kwarg:
        args.kwarg.annotation = None
    if args.args and args.args[0].arg in {"self", "cls"}:
        args.args.pop(0)
    return "(" + ast.unparse(args) + ")"


@lru_cache(maxsize=256)
def parse_source(path):
    return ast.parse(path.read_text())


def stable_text_signature(value):
    if not isinstance(value, str) or re.search(r"0x[0-9a-fA-F]+", value):
        return None
    value = re.sub(r"^\(\$(?:self|module),\s*/(?:,\s*|(?=\)))", "(", value.strip())
    value = re.sub(r"^\(\$(?:self|module),\s*", "(", value)
    try:
        ast.parse("def _signature" + value + ":\n pass")
    except (SyntaxError, ValueError):
        return None
    return value


RUNTIME_MEMBERS = r'''
import importlib,inspect,json,pathlib,sys,types
module,names=json.loads(sys.argv[1]); result={}
try:
    imported=importlib.import_module(module)
except Exception:
    imported=None
for name in names:
    try:
        obj=imported
        for part in name.split('.'):
            obj=getattr(obj,part)
        defining=inspect.getmodule(obj)
        file=inspect.getsourcefile(obj)
        if file and '/torch/' in file.replace('\\','/'):
            file='torch/'+file.replace('\\','/').split('/torch/',1)[1]
        else:
            file=None
        result[name]={'source':file,'qualname':getattr(obj,'__qualname__',name),'text_signature':getattr(obj,'__text_signature__',None)}
    except Exception:
        result[name]={}
print(json.dumps(result))
'''


def collect_documented_symbols(export, source, namespace="torch", *, tracked_paths=None):
    if not isinstance(export, dict) or set(export) != {"format_version", "objects"} or export["format_version"] != 1:
        raise ValueError("invalid Sphinx domain export")
    objects = [o for o in export["objects"] if o["name"] == namespace or o["name"].startswith(namespace + ".")]
    if not objects:
        raise ValueError("empty declared documentation namespace")
    # Sphinx canonical aliases can point at a signature rendered in a different
    # module. Identity fields always describe the exported name, not that target.
    objects = [dict(obj, module=(obj.get("module") if obj.get("module") and
               (obj["name"].startswith(obj["module"] + ".") or obj["kind"] == "module" and obj["name"] == obj["module"])
               else obj["name"].rpartition(".")[0] or "torch")) for obj in objects]
    # One isolated import per documented module. An optional import never drops an ID.
    groups = {}
    for obj in objects:
        module = obj.get("module") or obj["name"].rpartition(".")[0] or "torch"
        groups.setdefault(module, []).append(obj["name"].removeprefix(module + "."))
    def inspect_module(item):
        module, names = item
        try:
            return module, run_python(RUNTIME_MEMBERS, json.dumps([module, sorted(set(names))]), timeout=60)
        except (ValueError, subprocess.TimeoutExpired):
            return module, {}
    with ThreadPoolExecutor(max_workers=4) as workers:
        runtime = dict(workers.map(inspect_module, sorted(groups.items())))
    symbols = []
    for obj in objects:
        name = obj["name"]
        module = obj.get("module") or name.rpartition(".")[0] or "torch"
        short = name.removeprefix(module + ".")
        detail = runtime.get(module, {}).get(short, {})
        path = detail.get("source")
        if (not path or not safe_path(path) or not (Path(source) / path).is_file()
                or tracked_paths is not None and path not in tracked_paths):
            path = "docs/source/" + obj["source"]
        signature = source_definition(source, path, detail.get("qualname", short))
        reason = None
        if signature is None:
            text_signature = stable_text_signature(detail.get("text_signature"))
            if text_signature is not None:
                signature = text_signature
            else:
                reason = "No stable source AST or extension text signature in the pinned build."
        signatures = obj.get("signatures") or [None]
        for number, documented in enumerate(signatures, 1):
            ident = "python:" + name + (f"#overload-{number}" if len(signatures) > 1 else "")
            record = dict(id=ident, kind=obj["kind"], module=module, name=short, signature=signature,
                          source=path, documentation="docs/source/" + obj["source"], alias=obj["alias"])
            if documented is not None and not re.search(r"0x[0-9a-fA-F]+|<[^>]*\bobject\b[^>]*>", documented):
                record["documented_signature"] = documented
            else:
                record["documented_signature"] = None
                record["documented_signature_reason"] = "No resolved callable signature." if documented is None else "Resolved signature contains an unstable runtime object representation."
            if reason:
                record["signature_reason"] = reason
            symbols.append(record)
    if len({s["id"] for s in symbols}) != len(symbols):
        raise ValueError("duplicate canonical documented object")
    return sorted(symbols, key=lambda s: s["id"])


ATEN_COLLECTOR = r'''
import json,pathlib,sys,torch,torchgen,torchgen.gen
root=pathlib.Path(sys.argv[1]); torchroot=pathlib.Path(torch.__file__).resolve().parent.parent
if pathlib.Path(torchgen.__file__).resolve().parent.parent != torchroot:
    raise ValueError('torchgen must be installed beside pinned torch')
yaml=root/'aten/src/ATen/native/native_functions.yaml'
parsed=torchgen.gen.parse_native_yaml(str(yaml),str(root/'aten/src/ATen/native/tags.yaml'),skip_native_fns_gen=False)
runtime={s.name+('.'+s.overload_name if s.overload_name else '') for s in torch._C._jit_get_all_schemas()}
functions=parsed.native_functions
origins={str(g):str(f.func.name) for f in functions for g in f.autogen}
records=[]
for f in functions:
    name=str(f.func.name); schema='aten::'+str(f.func)
    record=dict(id='aten:aten::'+name,kind='aten_schema',module='aten',name=name,signature=schema,
                source='aten/src/ATen/native/native_functions.yaml',line=f.loc.line,
                visibility='internal' if name.startswith('_') else 'public',runtime_present='aten::'+name in runtime)
    if name in origins: record['generated_from']=origins[name]
    records.append(record)
print(json.dumps(records))
'''


def collect_aten_schemas(source):
    records = run_python(ATEN_COLLECTOR, source)
    if len({s["id"] for s in records}) != len(records):
        raise ValueError("duplicate canonical ATen schema")
    return sorted(records, key=lambda s: s["id"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check-upstream", action="store_true")
    parser.add_argument("--source", type=Path)
    parser.add_argument("--build-sphinx", action="store_true")
    parser.add_argument("--namespace", default="torch")
    args = parser.parse_args(argv)
    try:
        reference = load_reference()
        if args.check:
            if args.source or args.build_sphinx:
                raise ValueError("offline --check cannot use upstream inputs")
            inventory, _ = load_snapshot()
            print(f"pinned inventory: {len(inventory['symbols'])} mapped identities; offline validation passed")
            return 0
        if not args.source:
            raise ValueError("refresh requires --source with an explicit clean checkout")
        verify_runtime(reference)
        source = verify_source_checkout(args.source, reference)
        if args.check_upstream:
            print("pinned source and runtime match")
            return 0
        if not args.build_sphinx:
            raise ValueError("--write requires --build-sphinx")
        export = build_sphinx_domain_export(source, reference)
        tracked_paths = set(subprocess.run(["git", "-C", str(source), "ls-files", "-z"], check=True,
                                          text=True, capture_output=True, timeout=60).stdout.split("\0"))
        symbols = collect_documented_symbols(export, source, args.namespace, tracked_paths=tracked_paths)
        if args.namespace == "torch":
            symbols += collect_aten_schemas(source)
        inventory = dict(format_version=1, pytorch_version=VERSION, pytorch_commit=COMMIT,
                         scope=[args.namespace], symbols=sorted(symbols, key=lambda s: s["id"]))
        inventory["runtime_build"] = run_python("import hashlib,json,platform,sys,torch; print(json.dumps(dict(wheel_version=torch.__version__,git_commit=torch.version.git_version,platform=sys.platform,machine=platform.machine(),cuda=torch.version.cuda,hip=torch.version.hip,debug=torch.version.debug,torch_config_sha256=hashlib.sha256(torch.__config__.show().encode()).hexdigest())))")
        verify_source_checkout(source, reference)
        atomic_write(ROOT / "compat/pytorch_inventory.json", render_inventory(inventory))
        # Write the actual census before reporting required hand-maintained reconciliation.
        load_snapshot()
        print(f"wrote {len(symbols)} pinned inventory identities")
        return 0
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
