"""Validate implementation routes independently of behavior claims (offline)."""

import hashlib
from pathlib import PurePosixPath
import re
import tomllib


FORMAT_FIELDS = {"format_version", "pytorch_commit", "tch_version", "tch_commit",
                 "trace_sha256", "unresolved", "capability", "capabilities_sha256"}
TCH_COMMIT = "4227b89b72059b0651ff83a38637693574e0a2d6"
TRACE_HEADER = "inventory_id\trust_symbol\tffi_symbol\tdispatcher"
ROUTES = {"existing_tch", "tch_extension", "stable_abi", "pinned_cpp", "rust", "python_interop", "unresolved"}


def unresolved_digest(ids):
    """Fingerprint exact IDs with UTF-8, lexicographic ordering and LF terminators."""
    return hashlib.sha256("".join(f"{key}\n" for key in sorted(ids)).encode()).hexdigest()


def _text(value):
    return isinstance(value, str) and bool(value.strip())


def _source_path(value):
    return (_text(value) and "\\" not in value and ":" not in value
            and not PurePosixPath(value).is_absolute() and ".." not in PurePosixPath(value).parts
            and PurePosixPath(value).as_posix() == value and value != ".")


def _witness(root, evidence, symbols):
    """The compiled integration test must use each exact path in its named test."""
    if not isinstance(evidence, str) or evidence.count("::") != 1:
        raise ValueError("binding evidence must name a Rust test")
    path, test = evidence.split("::")
    if not _source_path(path) or not path.startswith("tests/") or not path.endswith(".rs"):
        raise ValueError("binding evidence requires a repo-relative Rust integration test")
    if not re.fullmatch(r"[A-Za-z_]\w*", test):
        raise ValueError("binding evidence contains an invalid test name")
    source = (root / path).read_text(encoding="utf-8")
    declaration = r"^#\[test\][ \t]*\n(?:#\[[^\n]*\][ \t]*\n)*fn "
    body = re.search(declaration + re.escape(test) + r"\b(.*?)^\}", source, re.M | re.S)
    if not body:
        raise ValueError(f"missing binding evidence test {evidence}")
    for symbol in symbols:
        if not isinstance(symbol, str) or not re.fullmatch(r"[A-Za-z_]\w*(?:::[A-Za-z_]\w*)+", symbol):
            raise ValueError(f"invalid Rust symbol in {evidence}")
        if not re.search(r"(?<![\w:])" + re.escape(symbol) + r"\b", body[1]):
            raise ValueError(f"{symbol} is absent from binding evidence {evidence}")


def _load_dispatch(root, expected_digest, kinds):
    data = (root / "compat/pytorch_tch_dispatch.tsv").read_bytes()
    # Check logical lines; checkouts with core.autocrlf must give the same result.
    lines = data.decode("utf-8").splitlines()
    canonical = ("\n".join(lines) + "\n").encode("utf-8")
    if hashlib.sha256(canonical).hexdigest() != expected_digest:
        raise ValueError("tch dispatch trace digest differs from the reviewed source trace")
    if not lines or lines[0] != TRACE_HEADER:
        raise ValueError("invalid tch dispatch trace header")
    routes = {}
    methods = set()
    for line in lines[1:]:
        fields = line.split("\t")
        if len(fields) != 4:
            raise ValueError("malformed tch dispatch trace")
        key, method, ffi, dispatcher = fields
        if key in routes or method in methods:
            raise ValueError(f"duplicate tch dispatch trace: {key}, {method}")
        if kinds.get(key) != "aten_schema":
            raise ValueError(f"unknown ATen inventory schema {key}")
        if (not re.fullmatch(r"tch::Tensor::f_\w+", method)
                or not re.fullmatch(r"atg_\w+", ffi)
                or not re.fullmatch(r"\w+", dispatcher)):
            raise ValueError(f"invalid Rust/FFI/dispatcher symbol in {key}")
        routes[key] = "existing_tch"
        methods.add(method)
    if list(routes) != sorted(routes):
        raise ValueError("tch dispatch traces must be sorted by exact inventory ID")
    return routes


def load_bindings(root, inventory, mapping, ledger):
    """Return exact inventory and capability IDs mapped to validated route names."""
    with (root / "compat/pytorch_bindings.toml").open("rb") as file:
        manifest = tomllib.load(file)
    if set(manifest) - {"binding"} != FORMAT_FIELDS:
        raise ValueError("binding manifest must match format 1")
    if type(manifest["format_version"]) is not int or manifest["format_version"] != 1:
        raise ValueError("binding format_version must be 1")
    for key in ("binding", "capability"):
        rows = manifest.get(key, [])
        if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
            raise ValueError(f"{key} must be an array of binding tables")
    if (manifest["tch_commit"] != TCH_COMMIT or manifest["tch_version"] != ledger["tch"]
            or manifest["pytorch_commit"] != inventory["pytorch_commit"]):
        raise ValueError("binding source pins differ from the ledger or frozen inventory")
    kinds = {s["id"]: s["kind"] for s in inventory["symbols"]}
    ledger_rows = {row["id"]: row for row in ledger["api"]}
    targets = mapping["mapping"]
    if kinds.keys() != targets.keys() or set(targets.values()) - ledger_rows.keys():
        raise ValueError("binding inventory has missing or unknown ledger targets")
    routes = _load_dispatch(root, manifest["trace_sha256"], kinds)

    for row in manifest.get("binding", []):
        key = row.get("id")
        fields = {"id", "route", "rust_symbol", "source", "source_commit", "evidence", "limitations"}
        if set(row) != fields or not _text(key) or key not in kinds or key in routes:
            raise ValueError(f"invalid, duplicate or unknown manual binding {key}")
        symbol = row["rust_symbol"]
        if (not _text(row["route"]) or row["route"] not in ROUTES - {"unresolved"}
                or symbol not in ledger_rows[targets[key]]["rust_symbols"]):
            raise ValueError(f"{key}: unknown binding route or Rust symbol for ledger target")
        if (not _source_path(row["source"]) or not _text(row["source_commit"])
                or not re.fullmatch(r"[0-9a-f]{40}", row["source_commit"])
                or not _text(row["limitations"])):
            raise ValueError(f"{key}: incomplete source or limitations")
        if row["route"] == "rust" and not (root / row["source"]).is_file():
            raise ValueError(f"{key}: Rust binding source does not exist")
        _witness(root, row["evidence"], [symbol])
        routes[key] = row["route"]

    groups = manifest["unresolved"]
    if not isinstance(groups, dict) or set(groups) != {"python", "aten"}:
        raise ValueError("binding manifest needs frozen unresolved ATen and Python groups")
    for group, entry in groups.items():
        missing = {key for key, kind in kinds.items() if key not in routes
                   and (kind == "aten_schema") == (group == "aten")}
        if (not isinstance(entry, dict) or set(entry) != {"ids_sha256", "reason"} or not _text(entry["reason"])
                or unresolved_digest(missing) != entry["ids_sha256"]):
            raise ValueError(f"unresolved.{group}: identities or reason differ from the reviewed group")
        routes.update(dict.fromkeys(missing, "unresolved"))

    capability_ids = set()
    for row in manifest["capability"]:
        base = {"family", "values", "ledger", "source", "route", "limitations"}
        resolved = row.get("route") != "unresolved"
        extra = {"symbols", "binding_source", "evidence"} if resolved else set()
        if not _text(row.get("route")) or row["route"] not in ROUTES:
            raise ValueError(f"unknown capability route {row.get('route')}")
        if set(row) != base | extra:
            raise ValueError("invalid capability fields; a resolved route needs symbol and source evidence")
        if not _text(row["ledger"]) or row["ledger"] not in ledger_rows:
            raise ValueError(f"unknown capability ledger target {row['ledger']}")
        if (not _text(row["family"]) or not re.fullmatch(r"[a-z_]+", row["family"]) or not _source_path(row["source"])
                or not _text(row["limitations"])):
            raise ValueError("invalid capability family, source or limitations")
        values = row["values"]
        if not isinstance(values, list) or not values or not all(_text(v) for v in values):
            raise ValueError("capabilities need explicit nonempty values")
        if resolved:
            if row["route"] != "existing_tch" or not _source_path(row["binding_source"]):
                raise ValueError("capability proof requires a source path at the pinned tch commit")
            if not isinstance(row["symbols"], dict) or set(values) != row["symbols"].keys():
                raise ValueError("capability symbol evidence must cover every value")
            _witness(root, row["evidence"], row["symbols"].values())
        for value in values:
            key = f"capability:{row['family']}:{value}"
            if key in routes:
                raise ValueError(f"duplicate capability {key}")
            capability_ids.add(key)
            routes[key] = row["route"]
    if not capability_ids or unresolved_digest(capability_ids) != manifest["capabilities_sha256"]:
        raise ValueError("capability IDs differ from the reviewed obligations")
    return routes
