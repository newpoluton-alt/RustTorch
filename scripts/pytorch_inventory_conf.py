"""Isolated semantic configuration; never import the upstream conf.py."""
import json
import os

project = "Pinned PyTorch API census"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "myst_nb",
    "pytorch_inventory_sphinx",
]
_suffixes = json.loads(os.environ["RUSTTORCH_SOURCE_SUFFIXES"])
# Autosummary emits RST using the first suffix, even when other parsers coexist.
source_suffix = {".rst": _suffixes[".rst"], **_suffixes}
root_doc = "index"
exclude_patterns = ["conf.py", "**/.ipynb_checkpoints/**"]
templates_path = ["_templates"]
autosummary_filename_map = json.loads(os.environ.get("RUSTTORCH_AUTOSUMMARY_FILENAMES", "{}"))
autosummary_generate = True
autosummary_generate_overwrite = True
autosummary_imported_members = False
autodoc_typehints = "none"
autodoc_member_order = "bysource"
autodoc_inherit_docstrings = False
autodoc_default_options = {"exclude-members": "from_bytes, to_bytes"}
nb_execution_mode = "off"
nb_render_image_options = {"alt": ""}
myst_enable_extensions = ["colon_fence", "deflist", "dollarmath", "fieldlist"]
nitpicky = False
html_theme = "basic"
