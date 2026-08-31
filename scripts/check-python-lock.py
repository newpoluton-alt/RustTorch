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
EXPECTED_PACKAGES = [
    ("filelock", "3.32.4"),
    ("fsspec", "2026.7.0"),
    ("jinja2", "3.1.6"),
    ("markupsafe", "3.0.3"),
    ("mpmath", "1.3.0"),
    ("networkx", "3.6.1"),
    ("numpy", "2.5.2"),
    ("rusttorch-tooling", "0.0.0"),
    ("safetensors", "0.8.0"),
    ("setuptools", "84.0.0"),
    ("sympy", "1.14.0"),
    ("torch", "2.13.0"),
    ("torch", "2.13.0+cpu"),
    ("typing-extensions", "4.16.0"),
]
EXPECTED_DEPENDENCIES = {
    "jinja2": ["markupsafe"],
    "sympy": ["mpmath"],
    "torch": [
        "filelock",
        "fsspec",
        "jinja2",
        "networkx",
        "setuptools",
        "sympy",
        "typing-extensions",
    ],
}
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
    require_keys(pyproject, {"project", "tool"}, "pyproject")
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
        not isinstance(artifact["size"], int) or artifact["size"] <= 0
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
    normalized = []
    for index, dependency_value in enumerate(dependencies):
        dependency = require_table(
            dependency_value, f"{context} dependency {index}"
        )
        require_keys(dependency, {"name"}, f"{context} dependency {index}")
        normalized.append(dependency["name"])
    require_equal(normalized, expected_names, f"{context} dependencies")


def validate_tooling_package(package: dict[str, Any]) -> None:
    context = "rusttorch-tooling package"
    require_keys(
        package,
        {"name", "version", "source", "dependencies", "metadata"},
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
    require_equal(
        package["metadata"],
        {
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
    expected_artifact_count = len(wheels) if is_torch else len(wheels) + 1
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
