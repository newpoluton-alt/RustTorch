#!/usr/bin/env python3
"""Validate a RustTorch release and create its exact provenance subjects."""

from __future__ import annotations

import argparse
import base64
from datetime import date
import hashlib
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import tarfile
import tomllib


TAG_PATTERN = re.compile(r"v(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)")
PACKAGE_MANIFESTS = (
    ("rusttorch", Path("Cargo.toml")),
    ("rusttorch-cli", Path("crates/rusttorch-cli/Cargo.toml")),
)
FORBIDDEN_ARCHIVE_PARTS = {".venv", "target", "__pycache__", "libtorch"}
FORBIDDEN_ARCHIVE_FILES = {"pyproject.toml", "uv.lock"}
FORBIDDEN_ARCHIVE_SUFFIXES = {".dll", ".dylib", ".pyc", ".pyo", ".so"}


class ReleaseError(ValueError):
    """A release input does not satisfy the fail-closed release policy."""


def parse_tag(tag: str) -> str:
    """Return the stable SemVer from an exact ``vX.Y.Z`` tag."""

    match = TAG_PATTERN.fullmatch(tag)
    if match is None:
        raise ReleaseError(f"release tag {tag!r} must use stable vX.Y.Z without leading zeros")
    return ".".join(match.groups())


def _load_toml(path: Path, label: str) -> dict:
    try:
        with path.open("rb") as source:
            value = tomllib.load(source)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ReleaseError(f"cannot read {label}: {error}") from error
    if not isinstance(value, dict):
        raise ReleaseError(f"{label} must contain a TOML table")
    return value


def _package_metadata(root: Path, version: str) -> None:
    versions: list[str] = []
    for expected_name, relative_manifest in PACKAGE_MANIFESTS:
        manifest = _load_toml(root / relative_manifest, relative_manifest.as_posix())
        package = manifest.get("package")
        if not isinstance(package, dict):
            raise ReleaseError(f"{relative_manifest.as_posix()} has no package table")
        actual_name = package.get("name")
        actual_version = package.get("version")
        if actual_name != expected_name:
            raise ReleaseError(
                f"{relative_manifest.as_posix()} package name must be {expected_name!r}, "
                f"found {actual_name!r}"
            )
        if not isinstance(actual_version, str):
            raise ReleaseError(
                f"{relative_manifest.as_posix()} package version must be a string"
            )
        versions.append(actual_version)

    if len(set(versions)) != 1:
        raise ReleaseError(f"workspace package versions do not match: {versions}")
    if versions[0] != version:
        raise ReleaseError(
            f"release tag version {version!r} does not match package version {versions[0]!r}"
        )


def _lockfile_metadata(root: Path, version: str) -> None:
    lockfile = _load_toml(root / "Cargo.lock", "Cargo.lock")
    packages = lockfile.get("package")
    if not isinstance(packages, list):
        raise ReleaseError("Cargo.lock must contain package entries")
    for package_name, _ in PACKAGE_MANIFESTS:
        matches = [
            package
            for package in packages
            if isinstance(package, dict) and package.get("name") == package_name
        ]
        if len(matches) != 1:
            raise ReleaseError(
                f"lockfile must contain exactly one {package_name!r} package entry"
            )
        if matches[0].get("version") != version:
            raise ReleaseError(
                f"lockfile {package_name!r} version must be {version!r}, "
                f"found {matches[0].get('version')!r}"
            )


def _changelog_metadata(root: Path, version: str) -> None:
    try:
        changelog = (root / "CHANGELOG.md").read_text(encoding="utf-8")
    except OSError as error:
        raise ReleaseError(f"cannot read changelog: {error}") from error
    headings = re.findall(
        rf"^## {re.escape(version)} - ([0-9]{{4}}-[0-9]{{2}}-[0-9]{{2}})$",
        changelog,
        re.MULTILINE,
    )
    if len(headings) != 1:
        raise ReleaseError(
            f"changelog must contain exactly one '## {version} - YYYY-MM-DD' heading"
        )
    try:
        date.fromisoformat(headings[0])
    except ValueError as error:
        raise ReleaseError(f"changelog release date {headings[0]!r} is invalid") from error


def _compatibility_metadata(root: Path) -> None:
    checker = root / "scripts/check-compatibility.py"
    if not checker.is_file():
        raise ReleaseError("compatibility checker scripts/check-compatibility.py is missing")
    result = subprocess.run(
        [sys.executable, str(checker), "--check"],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "checker failed"
        raise ReleaseError(f"compatibility validation failed: {detail}")


def validate_release(root: Path, tag: str) -> str:
    """Validate release metadata under *root* and return its package version."""

    version = parse_tag(tag)
    _package_metadata(root, version)
    _lockfile_metadata(root, version)
    _changelog_metadata(root, version)
    _compatibility_metadata(root)
    return version


def _validate_archive(path: Path, package_root: str) -> None:
    try:
        with tarfile.open(path, mode="r:gz") as archive:
            members = archive.getmembers()
    except (OSError, tarfile.TarError) as error:
        raise ReleaseError(f"cannot read package archive {path.name!r}: {error}") from error

    regular_files = 0
    for member in members:
        name = member.name.rstrip("/")
        parts = name.split("/") if name else []
        if (
            not parts
            or "\\" in member.name
            or PurePosixPath(member.name).is_absolute()
            or any(part in {"", ".", ".."} for part in parts)
            or parts[0] != package_root
            or not (member.isfile() or member.isdir())
        ):
            raise ReleaseError(f"unsafe archive member {member.name!r} in {path.name!r}")
        relative_parts = parts[1:]
        relative_name = "/".join(relative_parts)
        if (
            any(part in FORBIDDEN_ARCHIVE_PARTS for part in relative_parts)
            or relative_name in FORBIDDEN_ARCHIVE_FILES
            or PurePosixPath(relative_name).suffix.lower() in FORBIDDEN_ARCHIVE_SUFFIXES
        ):
            raise ReleaseError(f"unsafe archive member {member.name!r} in {path.name!r}")
        regular_files += member.isfile()
    if regular_files == 0:
        raise ReleaseError(f"package archive {path.name!r} contains no regular files")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise ReleaseError(f"cannot hash package archive {path.name!r}: {error}") from error
    return digest.hexdigest()


def write_subjects(dist: Path, version: str, output: Path) -> str:
    """Validate two release archives, write GNU SHA-256 subjects, and return base64."""

    archive_names = (
        f"rusttorch-{version}.crate",
        f"rusttorch-cli-{version}.crate",
    )
    try:
        entries = sorted(path.name for path in dist.iterdir())
    except OSError as error:
        raise ReleaseError(f"cannot inspect release directory: {error}") from error
    if entries != sorted(archive_names):
        raise ReleaseError(
            f"release directory must contain exactly {list(archive_names)!r}, found {entries!r}"
        )

    lines: list[str] = []
    for archive_name in archive_names:
        archive_path = dist / archive_name
        if not archive_path.is_file() or archive_path.is_symlink():
            raise ReleaseError(f"release archive {archive_name!r} must be a regular file")
        _validate_archive(archive_path, archive_name.removesuffix(".crate"))
        lines.append(f"{_sha256(archive_path)}  {archive_name}\n")

    subjects = "".join(lines).encode("ascii")
    if not subjects:
        raise ReleaseError("release subjects must not be empty")
    try:
        output.write_bytes(subjects)
    except OSError as error:
        raise ReleaseError(f"cannot write release subjects: {error}") from error
    return base64.b64encode(subjects).decode("ascii")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True, help="exact release tag, such as v0.2.0")
    parser.add_argument("--dist", type=Path, help="directory containing both package archives")
    parser.add_argument("--subjects-output", type=Path, help="path for GNU SHA-256 subjects")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the release preflight or archive-subject validation command."""

    arguments = _parser().parse_args(argv)
    if (arguments.dist is None) != (arguments.subjects_output is None):
        _parser().error("--dist and --subjects-output must be used together")
    root = Path(__file__).resolve().parents[1]
    try:
        version = validate_release(root, arguments.tag)
        if arguments.dist is None:
            print(version)
        else:
            print(write_subjects(arguments.dist, version, arguments.subjects_output))
    except ReleaseError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
