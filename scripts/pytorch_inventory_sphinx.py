"""Export resolved Python-domain objects from the isolated semantic docs build."""
from __future__ import annotations

import json
import os
import ast
import importlib
import sys
import types
import csv
import io
from pathlib import Path

from docutils import nodes
from docutils.parsers.rst import Directive, directives
from docutils.parsers.rst.directives.tables import CSVTable
from sphinx import addnodes
from sphinx.errors import SphinxError
from sphinx.domains.python import PythonDomain


class Presentation(Directive):
    """Keep all nested semantic content while discarding presentation options."""
    has_content = True
    optional_arguments = 10
    final_argument_whitespace = True
    option_spec = {key: directives.unchanged for key in (
        "class", "name", "title", "icon", "color", "link", "link-type", "img-top",
        "img-bottom", "img-alt", "text-align", "padding", "margin", "gutter",
        "columns", "reverse", "animate", "sync", "open", "id", "width", "height",
    )}

    def run(self):
        container = nodes.container()
        self.state.nested_parse(self.content, self.content_offset, container)
        return [container]


class Diagram(Directive):
    """Mermaid source is literal content, and may not hide API directives."""
    has_content = True

    def run(self):
        text = "\n".join(self.content)
        if ".. py:" in text or "{py:" in text or ".. auto" in text:
            raise SphinxError("API directive inside a presentation-only Mermaid block")
        return [nodes.literal_block(text, text)]


class CompatibleCSVTable(CSVTable):
    """PyTorch uses comma-separated headers with semicolon-separated table bodies."""
    def run(self):
        delimiter = self.options.get("delim", self.options.get("delimiter", ","))
        header = self.options.get("header", "")
        if delimiter != "," and '", "' in header:
            fields = next(csv.reader([header], skipinitialspace=True))
            output = io.StringIO()
            csv.writer(output, delimiter=delimiter).writerow(fields)
            self.options["header"] = output.getvalue().strip()
        return super().run()


def source_members(body):
    for node in body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            yield node
        elif isinstance(node, (ast.If, ast.Try)):
            yield from source_members(node.body)
            yield from source_members(node.orelse)
            if isinstance(node, ast.Try):
                for handler in node.handlers:
                    yield from source_members(handler.body)


def source_object(node, module, qualname):
    """Represent unavailable definitions for autodoc without executing source code."""
    if isinstance(node, ast.ClassDef):
        obj = type(node.name, (), {n.name: source_object(n, module, qualname + "." + n.name) for n in source_members(node.body)})
    else:
        def obj(*args, **kwargs):
            raise RuntimeError("source-only documentation placeholder")
    obj.__name__ = node.name
    obj.__qualname__ = qualname
    obj.__module__ = module
    obj.__doc__ = ast.get_docstring(node)
    return obj


def optional_source_objects(app):
    root = Path(os.environ["RUSTTORCH_UPSTREAM_SOURCE"])
    # These documented modules require optional z3/tensorboard packages absent from
    # the CPU tooling lock. AST fallback supplies definitions, never dependency mocks.
    modules = (
        "torch.fx.experimental.migrate_gradual_types.transform_to_z3",
        "torch.fx.experimental.validator",
        "torch.utils.tensorboard.writer",
        "torch.utils.tensorboard",
    )
    for name in modules:
        path = root / (name.replace(".", "/") + ".py")
        if not path.is_file():
            path = root / name.replace(".", "/") / "__init__.py"
        if not path.is_file():
            continue
        try:
            module = importlib.import_module(name)
        except ImportError:
            module = types.ModuleType(name)
            module.__file__ = str(path)
            if path.name == "__init__.py":
                module.__path__ = [str(path.parent)]
            sys.modules[name] = module
        tree = ast.parse(path.read_text())
        for node in source_members(tree.body):
            if not hasattr(module, node.name):
                setattr(module, node.name, source_object(node, name, node.name))
        if name == "torch.utils.tensorboard":
            writer = sys.modules["torch.utils.tensorboard.writer"]
            module.SummaryWriter = writer.SummaryWriter
            module.FileWriter = writer.FileWriter
        parent, _, short = name.rpartition(".")
        if parent in sys.modules:
            setattr(sys.modules[parent], short, module)


class CensusPythonDomain(PythonDomain):
    """Retain one identity for reviewed same-page autodoc/docstring repetitions."""
    def note_object(self, name, objtype, node_id, aliased=False, location=None):
        repeated = {
            "torch.distributions.transforms.Transform.sign": "distributions",
            **{"torch.nn.utils.rnn.PackedSequence." + field: "generated/torch.nn.utils.rnn.PackedSequence"
               for field in ("batch_sizes", "data", "sorted_indices", "unsorted_indices")},
        }
        other = self.objects.get(name)
        if (other is not None and name in repeated and other.docname == self.env.current_document.docname == repeated[name]
                and other.objtype in {"attribute", "property"} and objtype in {"attribute", "property"}):
            return
        super().note_object(name, objtype, node_id, aliased, location)


def presentation_role(name, rawtext, text, lineno, inliner, options=None, content=None):
    return [nodes.inline(rawtext, text)], []


def unresolved_doctree(app, tree):
    docname = app.env.current_document.docname
    for conditional in tree.findall(addnodes.only):
        if not app.tags.eval_condition(conditional["expr"]):
            for node in conditional.findall(nodes.Element):
                app._pytorch_excluded.update((docname, ident) for ident in node.get("ids", []))


def resolved_doctree(app, tree, docname):
    for node in tree.findall(nodes.Element):
        app._pytorch_targets.update((docname, ident) for ident in node.get("ids", []))
    for message in tree.findall(nodes.system_message):
        if message.get("level", 0) >= 3:
            raise SphinxError(f"unresolved semantic directive in {docname}: {message.astext()[:500]}")
    for signature in tree.findall(addnodes.desc_signature):
        parent = signature.parent
        if parent.get("domain") != "py":
            continue
        for ident in signature.get("ids", []):
            app._pytorch_descriptions[(docname, ident)] = [
                {"module": n.get("module"), "text": n.astext()}
                for n in parent.children if isinstance(n, addnodes.desc_signature)
            ]


def collect(app, exception):
    if exception is not None:
        return
    namespaces = json.loads(os.environ["RUSTTORCH_NAMESPACES"])
    domain = app.env.domains["py"]
    descriptions = app._pytorch_descriptions
    document_sources = set(json.loads(Path(os.environ["RUSTTORCH_DOCUMENT_SOURCES"]).read_text()))
    parents = {}
    for parent, children in app.env.toctree_includes.items():
        for child in children:
            parents.setdefault(child, []).append(parent)

    def source_document(docname):
        pending, visited = [docname], set()
        while pending:
            current = pending.pop(0)
            if current in visited:
                continue
            visited.add(current)
            relative = Path(app.env.doc2path(current)).relative_to(app.srcdir).as_posix()
            if relative in document_sources:
                return relative
            pending.extend(sorted(parents.get(current, [])))
        raise SphinxError(f"generated API page has no pinned source document: {docname}")
    objects = []
    kind_map = {"class": "class", "exception": "class", "function": "function", "method": "method",
                "classmethod": "method", "staticmethod": "method", "property": "constant",
                "attribute": "constant", "data": "constant", "type": "constant", "module": "module"}
    for name, entry in sorted(domain.objects.items()):
        if not any(name == ns or name.startswith(ns + ".") for ns in namespaces):
            continue
        signatures = descriptions.get((entry.docname, entry.node_id), [])
        if (entry.docname, entry.node_id) not in app._pytorch_targets:
            # Objects in a false `only` branch were registered before resolution.
            if (entry.docname, entry.node_id) in app._pytorch_excluded:
                continue
            raise SphinxError(f"unresolved Python-domain object: {name}")
        if not signatures and entry.objtype != "module":
            raise SphinxError(f"Python-domain object has no resolved signature node: {name}")
        if entry.objtype not in kind_map:
            raise SphinxError(f"unsupported Python-domain kind {entry.objtype}: {name}")
        docpath = source_document(entry.docname)
        module = signatures[0].get("module") if signatures else name
        unique_signatures = list(dict.fromkeys(s["text"] for s in signatures))
        objects.append(dict(name=name, kind=kind_map[entry.objtype], module=module,
                            docname=entry.docname, node_id=entry.node_id, alias=bool(entry.aliased),
                            signatures=unique_signatures, source=docpath))
    if not objects:
        raise SphinxError("resolved Python domain is empty")
    output = Path(app.outdir) / "pytorch-domain.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"format_version": 1, "objects": objects}, indent=2) + "\n")


def setup(app):
    app._pytorch_descriptions = {}
    app._pytorch_excluded = set()
    app._pytorch_targets = set()
    app.add_domain(CensusPythonDomain, override=True)
    # Reviewed layout-only sphinx-design/tabs directives preserve their entire body.
    for name in ("grid", "grid-item", "grid-item-card", "card", "dropdown", "tab-set", "tab-set-code", "tab-item", "tabs", "tab", "group-tab", "container"):
        app.add_directive(name, Presentation, override=True)
    for name in ("bdg-primary", "bdg-secondary", "bdg-info", "bdg-success", "bdg-warning", "bdg-danger", "octicon", "fas"):
        app.add_role(name, presentation_role)
    app.add_directive("mermaid", Diagram)
    app.add_directive("csv-table", CompatibleCSVTable, override=True)
    app.connect("builder-inited", optional_source_objects, priority=100)
    app.connect("doctree-read", unresolved_doctree)
    app.connect("doctree-resolved", resolved_doctree)
    app.connect("build-finished", collect)
    return {"version": "1", "parallel_read_safe": False, "parallel_write_safe": False}
