#!/usr/bin/env python3
"""Validate RustTorch's locked Python tooling before creating an environment."""

from __future__ import annotations

import argparse
import re
import sys
import tomllib
from pathlib import Path, PurePosixPath
from typing import Any, Iterator
from urllib.parse import unquote, urlsplit


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PYPI_REGISTRY = "https://pypi.org/simple"
PYPI_ARTIFACT_HOST = "files.pythonhosted.org"
PYTORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"
PYTORCH_ARTIFACT_HOST = "download-r2.pytorch.org"
PYTHON_DEPENDENCIES = [
    "numpy==2.5.2",
    "safetensors==0.8.0",
    "torch==2.13.0",
]
DOCUMENTATION_DEPENDENCIES = ["myst-nb==1.4.0", "myst-parser==5.1.0", "pyyaml==6.0.3", "sphinx==9.1.0"]
EXPECTED_PACKAGES = [('alabaster', '1.0.0'),
 ('appnope', '1.0.0'),
 ('asttokens', '3.0.2'),
 ('attrs', '26.1.0'),
 ('babel', '2.18.0'),
 ('certifi', '2026.7.22'),
 ('cffi', '2.1.1'),
 ('charset-normalizer', '3.5.1'),
 ('click', '8.5.0'),
 ('colorama', '0.4.6'),
 ('comm', '0.2.3'),
 ('debugpy', '1.8.21'),
 ('docutils', '0.22.4'),
 ('executing', '2.2.1'),
 ('fastjsonschema', '2.22.2'),
 ('filelock', '3.32.4'),
 ('fsspec', '2026.7.0'),
 ('greenlet', '3.5.5'),
 ('idna', '3.19'),
 ('imagesize', '2.0.1'),
 ('importlib-metadata', '9.0.1'),
 ('ipykernel', '7.3.0'),
 ('ipython', '9.17.1'),
 ('ipython-pygments-lexers', '1.1.1'),
 ('jedi', '0.20.0'),
 ('jinja2', '3.1.6'),
 ('jsonschema', '4.26.0'),
 ('jsonschema-specifications', '2025.9.1'),
 ('jupyter-cache', '1.0.1'),
 ('jupyter-client', '8.10.0'),
 ('jupyter-core', '5.9.1'),
 ('markdown-it-py', '4.2.0'),
 ('markupsafe', '3.0.3'),
 ('matplotlib-inline', '0.2.2'),
 ('mdit-py-plugins', '0.6.1'),
 ('mdurl', '0.1.2'),
 ('mpmath', '1.3.0'),
 ('myst-nb', '1.4.0'),
 ('myst-parser', '5.1.0'),
 ('nbclient', '0.11.0'),
 ('nbformat', '5.11.1'),
 ('nest-asyncio2', '1.7.2'),
 ('networkx', '3.6.1'),
 ('numpy', '2.5.2'),
 ('packaging', '26.3'),
 ('parso', '0.8.7'),
 ('pexpect', '4.9.0'),
 ('platformdirs', '4.11.8'),
 ('prompt-toolkit', '3.0.53'),
 ('psutil', '7.2.2'),
 ('ptyprocess', '0.7.0'),
 ('pure-eval', '0.2.4'),
 ('pycparser', '3.0'),
 ('pygments', '2.21.0'),
 ('python-dateutil', '2.9.0.post0'),
 ('pyyaml', '6.0.3'),
 ('pyzmq', '27.2.0'),
 ('referencing', '0.37.0'),
 ('requests', '2.34.2'),
 ('roman-numerals', '4.1.0'),
 ('rpds-py', '2026.6.3'),
 ('rusttorch-tooling', '0.0.0'),
 ('safetensors', '0.8.0'),
 ('setuptools', '84.0.0'),
 ('six', '1.17.0'),
 ('snowballstemmer', '3.1.1'),
 ('sphinx', '9.1.0'),
 ('sphinxcontrib-applehelp', '2.0.0'),
 ('sphinxcontrib-devhelp', '2.0.0'),
 ('sphinxcontrib-htmlhelp', '2.1.0'),
 ('sphinxcontrib-jsmath', '1.0.1'),
 ('sphinxcontrib-qthelp', '2.0.0'),
 ('sphinxcontrib-serializinghtml', '2.0.0'),
 ('sqlalchemy', '2.0.52'),
 ('stack-data', '0.6.3'),
 ('sympy', '1.14.0'),
 ('tabulate', '0.10.0'),
 ('torch', '2.13.0'),
 ('torch', '2.13.0+cpu'),
 ('tornado', '6.5.8'),
 ('traitlets', '5.16.1'),
 ('typing-extensions', '4.16.0'),
 ('urllib3', '2.7.0'),
 ('wcwidth', '0.8.3'),
 ('zipp', '4.1.0')]
EXPECTED_DEPENDENCIES = {'cffi': [{'name': 'pycparser'}],
 'importlib-metadata': [{'name': 'zipp'}],
 'ipykernel': [{'marker': "sys_platform == 'darwin'", 'name': 'appnope'},
               {'name': 'comm'},
               {'name': 'debugpy'},
               {'name': 'ipython'},
               {'name': 'jupyter-client'},
               {'name': 'jupyter-core'},
               {'name': 'matplotlib-inline'},
               {'name': 'nest-asyncio2'},
               {'name': 'packaging'},
               {'name': 'psutil'},
               {'name': 'pyzmq'},
               {'name': 'tornado'},
               {'name': 'traitlets'}],
 'ipython': [{'marker': "sys_platform == 'win32'", 'name': 'colorama'},
             {'name': 'ipython-pygments-lexers'},
             {'name': 'jedi'},
             {'name': 'matplotlib-inline'},
             {'marker': "sys_platform != 'emscripten' and sys_platform != "
                        "'win32'",
              'name': 'pexpect'},
             {'name': 'prompt-toolkit'},
             {'marker': "sys_platform != 'cygwin' and sys_platform != "
                        "'emscripten'",
              'name': 'psutil'},
             {'name': 'pygments'},
             {'name': 'stack-data'},
             {'name': 'traitlets'}],
 'ipython-pygments-lexers': [{'name': 'pygments'}],
 'jedi': [{'name': 'parso'}],
 'jinja2': [{'name': 'markupsafe'}],
 'jsonschema': [{'name': 'attrs'},
                {'name': 'jsonschema-specifications'},
                {'name': 'referencing'},
                {'name': 'rpds-py'}],
 'jsonschema-specifications': [{'name': 'referencing'}],
 'jupyter-cache': [{'name': 'attrs'},
                   {'name': 'click'},
                   {'name': 'importlib-metadata'},
                   {'name': 'nbclient'},
                   {'name': 'nbformat'},
                   {'name': 'pyyaml'},
                   {'name': 'sqlalchemy'},
                   {'name': 'tabulate'}],
 'jupyter-client': [{'name': 'jupyter-core'},
                    {'name': 'python-dateutil'},
                    {'name': 'pyzmq'},
                    {'name': 'tornado'},
                    {'name': 'traitlets'},
                    {'name': 'typing-extensions'}],
 'jupyter-core': [{'name': 'platformdirs'}, {'name': 'traitlets'}],
 'markdown-it-py': [{'name': 'mdurl'}],
 'matplotlib-inline': [{'name': 'traitlets'}],
 'mdit-py-plugins': [{'name': 'markdown-it-py'}],
 'myst-nb': [{'name': 'importlib-metadata'},
             {'name': 'ipykernel'},
             {'name': 'ipython'},
             {'name': 'jupyter-cache'},
             {'name': 'myst-parser'},
             {'name': 'nbclient'},
             {'name': 'nbformat'},
             {'name': 'pyyaml'},
             {'name': 'sphinx'},
             {'name': 'typing-extensions'}],
 'myst-parser': [{'name': 'docutils'},
                 {'name': 'jinja2'},
                 {'name': 'markdown-it-py'},
                 {'name': 'mdit-py-plugins'},
                 {'name': 'pyyaml'},
                 {'name': 'sphinx'}],
 'nbclient': [{'name': 'jupyter-client'},
              {'name': 'jupyter-core'},
              {'name': 'nbformat'},
              {'name': 'traitlets'}],
 'nbformat': [{'name': 'fastjsonschema'},
              {'name': 'jsonschema'},
              {'name': 'jupyter-core'},
              {'name': 'traitlets'}],
 'pexpect': [{'name': 'ptyprocess'}],
 'prompt-toolkit': [{'name': 'wcwidth'}],
 'python-dateutil': [{'name': 'six'}],
 'pyzmq': [{'marker': "implementation_name == 'pypy'", 'name': 'cffi'}],
 'referencing': [{'name': 'attrs'}, {'name': 'rpds-py'}],
 'requests': [{'name': 'certifi'},
              {'name': 'charset-normalizer'},
              {'name': 'idna'},
              {'name': 'urllib3'}],
 'sphinx': [{'name': 'alabaster'},
            {'name': 'babel'},
            {'marker': "sys_platform == 'win32'", 'name': 'colorama'},
            {'name': 'docutils'},
            {'name': 'imagesize'},
            {'name': 'jinja2'},
            {'name': 'packaging'},
            {'name': 'pygments'},
            {'name': 'requests'},
            {'name': 'roman-numerals'},
            {'name': 'snowballstemmer'},
            {'name': 'sphinxcontrib-applehelp'},
            {'name': 'sphinxcontrib-devhelp'},
            {'name': 'sphinxcontrib-htmlhelp'},
            {'name': 'sphinxcontrib-jsmath'},
            {'name': 'sphinxcontrib-qthelp'},
            {'name': 'sphinxcontrib-serializinghtml'}],
 'sqlalchemy': [{'marker': "platform_machine == 'AMD64' or platform_machine == "
                           "'WIN32' or platform_machine == 'aarch64' or "
                           "platform_machine == 'amd64' or platform_machine == "
                           "'ppc64le' or platform_machine == 'win32' or "
                           "platform_machine == 'x86_64'",
                 'name': 'greenlet'},
                {'name': 'typing-extensions'}],
 'stack-data': [{'name': 'asttokens'},
                {'name': 'executing'},
                {'name': 'pure-eval'}],
 'sympy': [{'name': 'mpmath'}],
 'torch': [{'name': 'filelock'},
           {'name': 'fsspec'},
           {'name': 'jinja2'},
           {'name': 'networkx'},
           {'name': 'setuptools'},
           {'name': 'sympy'},
           {'name': 'typing-extensions'}]}
EXPECTED_TORCH_WHEEL_BASENAMES = {
    "2.13.0": {
        "torch-2.13.0-cp314-cp314-macosx_14_0_arm64.whl",
        "torch-2.13.0-cp314-cp314t-macosx_14_0_arm64.whl",
    },
    "2.13.0+cpu": {
        "torch-2.13.0+cpu-cp314-cp314-linux_s390x.whl",
        "torch-2.13.0+cpu-cp314-cp314-manylinux_2_28_aarch64.whl",
        "torch-2.13.0+cpu-cp314-cp314-manylinux_2_28_x86_64.whl",
        "torch-2.13.0+cpu-cp314-cp314-win_amd64.whl",
        "torch-2.13.0+cpu-cp314-cp314t-linux_s390x.whl",
        "torch-2.13.0+cpu-cp314-cp314t-manylinux_2_28_aarch64.whl",
        "torch-2.13.0+cpu-cp314-cp314t-manylinux_2_28_x86_64.whl",
        "torch-2.13.0+cpu-cp314-cp314t-win_amd64.whl",
    },
}
EXPECTED_ARTIFACT_COUNTS = {('alabaster', '1.0.0'): 2,
 ('appnope', '1.0.0'): 2,
 ('asttokens', '3.0.2'): 2,
 ('attrs', '26.1.0'): 2,
 ('babel', '2.18.0'): 2,
 ('certifi', '2026.7.22'): 2,
 ('cffi', '2.1.1'): 25,
 ('charset-normalizer', '3.5.1'): 55,
 ('click', '8.5.0'): 2,
 ('colorama', '0.4.6'): 2,
 ('comm', '0.2.3'): 2,
 ('debugpy', '1.8.21'): 6,
 ('docutils', '0.22.4'): 2,
 ('executing', '2.2.1'): 2,
 ('fastjsonschema', '2.22.2'): 2,
 ('filelock', '3.32.4'): 2,
 ('fsspec', '2026.7.0'): 2,
 ('greenlet', '3.5.5'): 16,
 ('idna', '3.19'): 2,
 ('imagesize', '2.0.1'): 2,
 ('importlib-metadata', '9.0.1'): 2,
 ('ipykernel', '7.3.0'): 2,
 ('ipython', '9.17.1'): 2,
 ('ipython-pygments-lexers', '1.1.1'): 2,
 ('jedi', '0.20.0'): 2,
 ('jinja2', '3.1.6'): 2,
 ('jsonschema', '4.26.0'): 2,
 ('jsonschema-specifications', '2025.9.1'): 2,
 ('jupyter-cache', '1.0.1'): 2,
 ('jupyter-client', '8.10.0'): 2,
 ('jupyter-core', '5.9.1'): 2,
 ('markdown-it-py', '4.2.0'): 2,
 ('markupsafe', '3.0.3'): 23,
 ('matplotlib-inline', '0.2.2'): 2,
 ('mdit-py-plugins', '0.6.1'): 2,
 ('mdurl', '0.1.2'): 2,
 ('mpmath', '1.3.0'): 2,
 ('myst-nb', '1.4.0'): 2,
 ('myst-parser', '5.1.0'): 2,
 ('nbclient', '0.11.0'): 2,
 ('nbformat', '5.11.1'): 2,
 ('nest-asyncio2', '1.7.2'): 2,
 ('networkx', '3.6.1'): 2,
 ('numpy', '2.5.2'): 22,
 ('packaging', '26.3'): 2,
 ('parso', '0.8.7'): 2,
 ('pexpect', '4.9.0'): 2,
 ('platformdirs', '4.11.8'): 2,
 ('prompt-toolkit', '3.0.53'): 2,
 ('psutil', '7.2.2'): 15,
 ('ptyprocess', '0.7.0'): 2,
 ('pure-eval', '0.2.4'): 2,
 ('pycparser', '3.0'): 2,
 ('pygments', '2.21.0'): 2,
 ('python-dateutil', '2.9.0.post0'): 2,
 ('pyyaml', '6.0.3'): 19,
 ('pyzmq', '27.2.0'): 24,
 ('referencing', '0.37.0'): 2,
 ('requests', '2.34.2'): 2,
 ('roman-numerals', '4.1.0'): 2,
 ('rpds-py', '2026.6.3'): 30,
 ('safetensors', '0.8.0'): 17,
 ('setuptools', '84.0.0'): 2,
 ('six', '1.17.0'): 2,
 ('snowballstemmer', '3.1.1'): 2,
 ('sphinx', '9.1.0'): 2,
 ('sphinxcontrib-applehelp', '2.0.0'): 2,
 ('sphinxcontrib-devhelp', '2.0.0'): 2,
 ('sphinxcontrib-htmlhelp', '2.1.0'): 2,
 ('sphinxcontrib-jsmath', '1.0.1'): 2,
 ('sphinxcontrib-qthelp', '2.0.0'): 2,
 ('sphinxcontrib-serializinghtml', '2.0.0'): 2,
 ('sqlalchemy', '2.0.52'): 10,
 ('stack-data', '0.6.3'): 2,
 ('sympy', '1.14.0'): 2,
 ('tabulate', '0.10.0'): 2,
 ('torch', '2.13.0'): 2,
 ('torch', '2.13.0+cpu'): 8,
 ('tornado', '6.5.8'): 10,
 ('traitlets', '5.16.1'): 2,
 ('typing-extensions', '4.16.0'): 2,
 ('urllib3', '2.7.0'): 2,
 ('wcwidth', '0.8.3'): 2,
 ('zipp', '4.1.0'): 2}
SHA256 = re.compile(r"sha256:[0-9a-f]{64}")


class PythonLockError(ValueError):
    """Raised when the Python tooling manifest or lock is unsafe."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--pyproject", type=Path)
    parser.add_argument("--lock", type=Path)
    return parser.parse_args(argv)


def select_inputs(args: argparse.Namespace) -> tuple[Path, Path]:
    explicit_paths = args.pyproject is not None or args.lock is not None
    if args.root is not None and explicit_paths:
        raise PythonLockError("--root cannot be combined with --pyproject or --lock")
    if (args.pyproject is None) != (args.lock is None):
        raise PythonLockError("--pyproject and --lock must be provided together")
    if explicit_paths:
        return args.pyproject.resolve(), args.lock.resolve()
    root = REPOSITORY_ROOT if args.root is None else args.root.resolve()
    return root / "pyproject.toml", root / "uv.lock"


def load_toml(path: Path) -> dict[str, object]:
    with path.open("rb") as source:
        value = tomllib.load(source)
    if not isinstance(value, dict):
        raise PythonLockError(f"{path} must contain a TOML table")
    return value


def require_table(value: object, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PythonLockError(f"{context} must be a table")
    return value


def require_list(value: object, context: str) -> list[Any]:
    if not isinstance(value, list):
        raise PythonLockError(f"{context} must be an array")
    return value


def values_match(actual: object, expected: object) -> bool:
    if type(actual) is not type(expected):
        return False
    if isinstance(actual, dict) and isinstance(expected, dict):
        return set(actual) == set(expected) and all(
            values_match(actual[key], expected[key]) for key in actual
        )
    if isinstance(actual, (list, tuple)) and isinstance(expected, (list, tuple)):
        return len(actual) == len(expected) and all(
            values_match(actual_value, expected_value)
            for actual_value, expected_value in zip(actual, expected)
        )
    return actual == expected


def require_equal(actual: object, expected: object, context: str) -> None:
    if not values_match(actual, expected):
        raise PythonLockError(f"{context} does not match the locked policy")


def require_keys(table: dict[str, Any], expected: set[str], context: str) -> None:
    require_equal(set(table), expected, f"{context} fields")


def validate_manifest(pyproject: dict[str, object]) -> None:
    require_keys(pyproject, {"project", "tool", "dependency-groups"}, "pyproject")
    require_equal(pyproject["dependency-groups"], {"dev": DOCUMENTATION_DEPENDENCIES}, "documentation dependencies")
    project = require_table(pyproject["project"], "project")
    require_keys(
        project,
        {"name", "version", "requires-python", "dependencies"},
        "project",
    )
    require_equal(project["name"], "rusttorch-tooling", "project name")
    require_equal(project["version"], "0.0.0", "project version")
    require_equal(
        project["requires-python"], ">=3.14,<3.15", "project Python range"
    )
    require_equal(project["dependencies"], PYTHON_DEPENDENCIES, "project dependencies")

    tool = require_table(pyproject["tool"], "tool")
    require_keys(tool, {"uv"}, "tool")
    uv = require_table(tool["uv"], "tool.uv")
    require_keys(uv, {"package", "sources", "index"}, "tool.uv")
    require_equal(uv["package"], False, "tool.uv package mode")
    require_equal(
        uv["sources"], {"torch": {"index": "pytorch-cpu"}}, "tool.uv sources"
    )
    require_equal(
        uv["index"],
        [{"name": "pytorch-cpu", "url": PYTORCH_CPU_INDEX, "explicit": True}],
        "tool.uv index",
    )


def iter_artifact_records(
    value: object, context: str
) -> Iterator[tuple[str, dict[str, Any]]]:
    if isinstance(value, dict):
        if "url" in value or "hash" in value:
            yield context, value
            return
        for key, nested in value.items():
            yield from iter_artifact_records(nested, f"{context}.{key}")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            yield from iter_artifact_records(nested, f"{context}[{index}]")


def validate_artifact(
    artifact: dict[str, Any], context: str, expected_host: str
) -> None:
    allowed_fields = {"url", "hash", "size", "upload-time"}
    if not {"url", "hash"}.issubset(artifact) or not set(artifact).issubset(
        allowed_fields
    ):
        raise PythonLockError(f"{context} must contain only a URL, hash, and metadata")
    url = artifact["url"]
    digest = artifact["hash"]
    if not isinstance(url, str) or not isinstance(digest, str):
        raise PythonLockError(f"{context} URL and hash must be strings")
    try:
        parsed_url = urlsplit(url)
    except ValueError as error:
        raise PythonLockError(f"{context} has an invalid URL") from error
    if (
        parsed_url.scheme != "https"
        or parsed_url.netloc != expected_host
        or parsed_url.query
        or parsed_url.fragment
    ):
        raise PythonLockError(f"{context} must use HTTPS from {expected_host}")
    if SHA256.fullmatch(digest) is None:
        raise PythonLockError(f"{context} must use a lowercase SHA-256 digest")
    if "size" in artifact and (
        type(artifact["size"]) is not int or artifact["size"] <= 0
    ):
        raise PythonLockError(f"{context} size must be a positive integer")
    if "upload-time" in artifact and (
        not isinstance(artifact["upload-time"], str) or not artifact["upload-time"]
    ):
        raise PythonLockError(f"{context} upload time must be a nonempty string")


def validate_dependency_names(
    package: dict[str, Any], name: str, context: str
) -> None:
    expected_names = EXPECTED_DEPENDENCIES.get(name)
    if expected_names is None:
        if "dependencies" in package:
            raise PythonLockError(f"{context} has unexpected dependencies")
        return
    dependencies = require_list(package["dependencies"], f"{context} dependencies")
    require_equal(dependencies, expected_names, f"{context} dependencies")


def validate_tooling_package(package: dict[str, Any]) -> None:
    context = "rusttorch-tooling package"
    require_keys(
        package,
        {"name", "version", "source", "dependencies", "dev-dependencies", "metadata"},
        context,
    )
    require_equal(package["source"], {"virtual": "."}, f"{context} source")
    require_equal(
        package["dependencies"],
        [
            {"name": "numpy"},
            {"name": "safetensors"},
            {
                "name": "torch",
                "version": "2.13.0",
                "source": {"registry": PYTORCH_CPU_INDEX},
                "marker": "sys_platform == 'darwin'",
            },
            {
                "name": "torch",
                "version": "2.13.0+cpu",
                "source": {"registry": PYTORCH_CPU_INDEX},
                "marker": "sys_platform != 'darwin'",
            },
        ],
        f"{context} dependencies",
    )
    require_equal(package["dev-dependencies"], {"dev": [{"name": pin.split("==")[0]} for pin in DOCUMENTATION_DEPENDENCIES]}, f"{context} development dependencies")
    require_equal(
        package["metadata"],
        {
            "requires-dev": {"dev": [{"name": pin.split("==")[0], "specifier": "==" + pin.split("==")[1]} for pin in DOCUMENTATION_DEPENDENCIES]},
            "requires-dist": [
                {"name": "numpy", "specifier": "==2.5.2"},
                {"name": "safetensors", "specifier": "==0.8.0"},
                {
                    "name": "torch",
                    "specifier": "==2.13.0",
                    "index": PYTORCH_CPU_INDEX,
                },
            ]
        },
        f"{context} metadata",
    )
    if list(iter_artifact_records(package, context)):
        raise PythonLockError(f"{context} cannot contain downloadable artifacts")


def validate_registry_package(package: dict[str, Any], name: str, version: str) -> None:
    context = f"{name} {version} package"
    is_torch = name == "torch"
    fields = {"name", "version", "source", "wheels"}
    if is_torch:
        fields.update({"resolution-markers", "dependencies"})
    else:
        fields.add("sdist")
        if name in EXPECTED_DEPENDENCIES:
            fields.add("dependencies")
    require_keys(package, fields, context)
    expected_source = {
        "registry": PYTORCH_CPU_INDEX if is_torch else PYPI_REGISTRY
    }
    require_equal(package["source"], expected_source, f"{context} source")
    validate_dependency_names(package, name, context)

    wheels = require_list(package["wheels"], f"{context} wheels")
    if not wheels:
        raise PythonLockError(f"{context} must contain wheels")
    if is_torch:
        expected_marker = (
            ["sys_platform == 'darwin'"]
            if version == "2.13.0"
            else ["sys_platform != 'darwin'"]
        )
        require_equal(
            package["resolution-markers"], expected_marker, f"{context} markers"
        )

    artifacts = list(iter_artifact_records(package, context))
    expected_artifact_count = EXPECTED_ARTIFACT_COUNTS[(name, version)]
    require_equal(
        len(artifacts), expected_artifact_count, f"{context} artifact count"
    )
    expected_host = PYTORCH_ARTIFACT_HOST if is_torch else PYPI_ARTIFACT_HOST
    for artifact_context, artifact in artifacts:
        validate_artifact(artifact, artifact_context, expected_host)

    if is_torch:
        wheel_basenames = []
        for index, wheel_value in enumerate(wheels):
            wheel = require_table(wheel_value, f"{context} wheel {index}")
            url = wheel.get("url")
            if not isinstance(url, str):
                raise PythonLockError(f"{context} wheel {index} URL must be a string")
            parsed_url = urlsplit(url)
            path = PurePosixPath(parsed_url.path)
            basename = unquote(path.name)
            if path.parent.as_posix() != "/whl/cpu":
                raise PythonLockError(f"{context} wheels must use the CPU path")
            wheel_basenames.append(basename)
        expected_basenames = EXPECTED_TORCH_WHEEL_BASENAMES[version]
        if len(wheel_basenames) != len(expected_basenames) or set(
            wheel_basenames
        ) != expected_basenames:
            raise PythonLockError(f"{context} wheel platforms do not match policy")


def validate_lock(lock: dict[str, object]) -> None:
    require_keys(
        lock,
        {"version", "revision", "requires-python", "resolution-markers", "package"},
        "uv.lock",
    )
    require_equal(lock["version"], 1, "uv.lock version")
    require_equal(lock["revision"], 3, "uv.lock revision")
    require_equal(lock["requires-python"], "==3.14.*", "uv.lock Python range")
    require_equal(
        lock["resolution-markers"],
        ["sys_platform != 'darwin'", "sys_platform == 'darwin'"],
        "uv.lock resolution markers",
    )
    package_values = require_list(lock["package"], "uv.lock packages")
    packages = [
        require_table(value, f"uv.lock package {index}")
        for index, value in enumerate(package_values)
    ]
    package_versions = [(package.get("name"), package.get("version")) for package in packages]
    require_equal(package_versions, EXPECTED_PACKAGES, "uv.lock package versions")
    for package, (name, version) in zip(packages, EXPECTED_PACKAGES):
        if name == "rusttorch-tooling":
            validate_tooling_package(package)
        else:
            validate_registry_package(package, name, version)


def validate_python_lock(pyproject_path: Path, lock_path: Path) -> None:
    uv_config = pyproject_path.parent / "uv.toml"
    if uv_config.exists():
        raise PythonLockError(f"alternate UV configuration is forbidden: {uv_config}")
    validate_manifest(load_toml(pyproject_path))
    validate_lock(load_toml(lock_path))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        pyproject_path, lock_path = select_inputs(args)
        validate_python_lock(pyproject_path, lock_path)
    except (OSError, PythonLockError, tomllib.TOMLDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
