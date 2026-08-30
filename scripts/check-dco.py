#!/usr/bin/env python3
"""Check DCO sign-offs for every commit in a Git revision range."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import re
import shlex
import subprocess
import sys


DEPENDABOT_NAME = "dependabot[bot]"
DEPENDABOT_EMAIL = "49699333+dependabot[bot]@users.noreply.github.com"
OBJECT_ID = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
TRAILER_IDENTITY = re.compile(r"^\s*(?P<name>[^<>]+?)\s*<(?P<email>[^<>\s]+)>\s*$")


class GitError(RuntimeError):
    """A Git command failed or returned malformed data."""


@dataclass(frozen=True)
class Identity:
    """One displayed and normalized Git identity."""

    name: str
    email: str

    @property
    def normalized(self) -> tuple[str, str]:
        return (" ".join(self.name.split()).casefold(), self.email.strip().casefold())

    @property
    def display(self) -> str:
        return f"{self.name} <{self.email}>"


@dataclass(frozen=True)
class CommitFailure:
    """DCO failures and missing identities for one commit."""

    commit: str
    errors: tuple[str, ...]
    missing: tuple[Identity, ...]


def _git(arguments: list[str], *, input_text: str | None = None) -> str:
    try:
        result = subprocess.run(
            ["git", *arguments],
            input=input_text,
            text=True,
            capture_output=True,
        )
    except OSError as error:
        raise GitError(f"cannot run Git: {error}") from error
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "Git command failed"
        raise GitError(detail)
    return result.stdout


def _resolve_revision(label: str, revision: str) -> str:
    if not revision or revision.startswith("-") or any(char in revision for char in "\0\r\n"):
        raise GitError(f"{label} revision {revision!r} does not resolve to a commit")
    try:
        resolved = _git(
            ["rev-parse", "--verify", "--end-of-options", f"{revision}^{{commit}}"]
        ).strip()
    except GitError as error:
        raise GitError(
            f"{label} revision {revision!r} does not resolve to a commit: {error}"
        ) from error
    if OBJECT_ID.fullmatch(resolved) is None:
        raise GitError(f"{label} revision {revision!r} returned an invalid object ID")
    return resolved


def _parse_identity(value: str) -> Identity | None:
    match = TRAILER_IDENTITY.fullmatch(value)
    if match is None:
        return None
    return Identity(" ".join(match.group("name").split()), match.group("email").strip())


def _commit_author_and_message(commit: str) -> tuple[Identity, str]:
    output = _git(
        ["show", "--quiet", "--no-show-signature", "--format=%an%x00%ae%x00%B", commit]
    )
    fields = output.split("\0", 2)
    if len(fields) != 3:
        raise GitError(f"commit {commit[:12]} has malformed author metadata")
    author = _parse_identity(f"{fields[0]} <{fields[1]}>")
    if author is None:
        raise GitError(f"commit {commit[:12]} has malformed author metadata")
    return author, fields[2]


def _trailers(message: str) -> tuple[list[Identity], list[Identity], list[str]]:
    parsed = _git(["interpret-trailers", "--parse"], input_text=message)
    signoffs: list[Identity] = []
    coauthors: list[Identity] = []
    errors: list[str] = []
    for line in parsed.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        normalized_key = key.strip().casefold()
        if normalized_key not in {"signed-off-by", "co-authored-by"}:
            continue
        identity = _parse_identity(value)
        if identity is None:
            label = "Signed-off-by" if normalized_key == "signed-off-by" else "Co-authored-by"
            errors.append(f"malformed {label} trailer: {value.strip()!r}")
        elif normalized_key == "signed-off-by":
            signoffs.append(identity)
        else:
            coauthors.append(identity)
    return signoffs, coauthors, errors


def _is_verified_dependabot(author: Identity, trusted_actor: bool) -> bool:
    return (
        trusted_actor
        and author.name == DEPENDABOT_NAME
        and author.email == DEPENDABOT_EMAIL
    )


def _check_commit(commit: str, *, trusted_dependabot_actor: bool) -> CommitFailure | None:
    author, message = _commit_author_and_message(commit)
    signoffs, coauthors, errors = _trailers(message)
    signed = {identity.normalized for identity in signoffs}
    required = list(coauthors)
    if not _is_verified_dependabot(author, trusted_dependabot_actor):
        required.insert(0, author)

    missing: list[Identity] = []
    seen: set[tuple[str, str]] = set()
    for identity in required:
        normalized = identity.normalized
        if normalized not in signed and normalized not in seen:
            missing.append(identity)
            seen.add(normalized)
    if not errors and not missing:
        return None
    return CommitFailure(commit, tuple(errors), tuple(missing))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="base revision excluded from the check")
    parser.add_argument("--head", required=True, help="head revision included in the check")
    parser.add_argument(
        "--trusted-dependabot-actor",
        action="store_true",
        help="trust the workflow actor as Dependabot; exact commit identity is still required",
    )
    return parser


def _print_failures(base: str, failures: list[CommitFailure]) -> None:
    for failure in failures:
        short = failure.commit[:12]
        for error in failure.errors:
            print(f"error: {short}: {error}", file=sys.stderr)
        for identity in failure.missing:
            print(
                f"error: {short}: missing Signed-off-by: {identity.display}",
                file=sys.stderr,
            )

    print("\nRepair each listed commit with:", file=sys.stderr)
    print(
        f"  git rebase --rebase-merges --interactive {shlex.quote(base)}",
        file=sys.stderr,
    )
    print(
        "Mark each listed pick as edit, or add a break line after a listed merge "
        "command. At each stop, run only that commit's commands:",
        file=sys.stderr,
    )
    for failure in failures:
        print(f"\nWhile stopped at {failure.commit[:12]}:", file=sys.stderr)
        for identity in failure.missing:
            trailer = shlex.quote(f"Signed-off-by: {identity.display}")
            print(f"  git commit --amend --no-edit --trailer {trailer}", file=sys.stderr)
        if failure.errors:
            print("  git commit --amend  # remove or correct malformed trailers", file=sys.stderr)
        print("  git rebase --continue", file=sys.stderr)
    print("  git push --force-with-lease", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    """Validate the requested Git range and print actionable failures."""

    arguments = _parser().parse_args(argv)
    try:
        base = _resolve_revision("base", arguments.base)
        head = _resolve_revision("head", arguments.head)
        commits = [
            line
            for line in _git(["rev-list", "--reverse", f"{base}..{head}"]).splitlines()
            if line
        ]
        failures = [
            failure
            for commit in commits
            if (
                failure := _check_commit(
                    commit,
                    trusted_dependabot_actor=arguments.trusted_dependabot_actor,
                )
            )
            is not None
        ]
    except GitError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    if failures:
        _print_failures(base, failures)
        return 1
    suffix = "" if len(commits) == 1 else "s"
    print(f"DCO check passed for {len(commits)} commit{suffix}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
