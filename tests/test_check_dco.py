import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CHECKER = ROOT / "scripts/check-dco.py"
AUTHOR = "Alice Example <alice@example.test>"
COAUTHOR = "Bob Example <bob@example.test>"
SECOND_AUTHOR = "Carol Example <carol@example.test>"
DEPENDABOT = (
    "dependabot[bot] "
    "<49699333+dependabot[bot]@users.noreply.github.com>"
)


class GitRepository:
    def __init__(self, testcase: unittest.TestCase) -> None:
        temporary = tempfile.TemporaryDirectory()
        testcase.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name)
        self.git("init", "--quiet", "--initial-branch=main")
        self.git("config", "user.name", "Alice Example")
        self.git("config", "user.email", "alice@example.test")
        self.commit("base without sign-off")
        self.base = self.revision("HEAD")

    def git(
        self,
        *arguments: str,
        input_text: str | None = None,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *arguments],
            cwd=self.path,
            input=input_text,
            text=True,
            capture_output=True,
            check=check,
        )

    def commit(self, message: str, *, author: str | None = None) -> str:
        command = ["commit", "--quiet", "--allow-empty", "--file=-"]
        if author is not None:
            command.extend(("--author", author))
        self.git(*command, input_text=message)
        return self.revision("HEAD")

    def revision(self, revision: str) -> str:
        return self.git("rev-parse", revision).stdout.strip()

    def check_dco(
        self,
        *,
        base: str | None = None,
        head: str = "HEAD",
        trusted_dependabot_actor: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        command = [
            sys.executable,
            str(CHECKER),
            "--base",
            self.base if base is None else base,
            "--head",
            head,
        ]
        if trusted_dependabot_actor:
            command.append("--trusted-dependabot-actor")
        return subprocess.run(command, cwd=self.path, text=True, capture_output=True)


def trailers(*lines: str, subject: str = "change") -> str:
    return f"{subject}\n\n" + "\n".join(lines) + "\n"


class DcoCheckerTests(unittest.TestCase):
    def repository(self) -> GitRepository:
        return GitRepository(self)

    def test_matching_author_signoff_passes_and_base_commit_is_excluded(self) -> None:
        repository = self.repository()
        repository.commit(trailers(f"Signed-off-by: {AUTHOR}"))

        result = repository.check_dco()

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("DCO check passed for 1 commit", result.stdout)

    def test_unsigned_commit_fails_with_commit_and_recovery_command(self) -> None:
        repository = self.repository()
        commit = repository.commit("unsigned change")

        result = repository.check_dco()

        self.assertNotEqual(result.returncode, 0)
        self.assertIn(commit[:12], result.stderr)
        self.assertIn(f"missing Signed-off-by: {AUTHOR}", result.stderr)
        self.assertIn("git rebase --rebase-merges --interactive", result.stderr)
        self.assertIn(
            "git commit --amend --no-edit --trailer "
            "'Signed-off-by: Alice Example <alice@example.test>'",
            result.stderr,
        )
        self.assertIn("git push --force-with-lease", result.stderr)

    def test_signoff_identity_is_case_insensitive_and_normalizes_name_space(self) -> None:
        repository = self.repository()
        repository.commit(
            trailers("Signed-off-by: alice   example <ALICE@EXAMPLE.TEST>")
        )

        result = repository.check_dco()

        self.assertEqual(result.returncode, 0, result.stderr)

    def test_mismatched_signatory_does_not_certify_the_author(self) -> None:
        repository = self.repository()
        repository.commit(
            trailers("Signed-off-by: Mallory Example <mallory@example.test>")
        )

        result = repository.check_dco()

        self.assertNotEqual(result.returncode, 0)
        self.assertIn(f"missing Signed-off-by: {AUTHOR}", result.stderr)

    def test_every_commit_in_range_is_checked(self) -> None:
        repository = self.repository()
        signed = repository.commit(trailers(f"Signed-off-by: {AUTHOR}", subject="signed"))
        unsigned = repository.commit("unsigned")

        result = repository.check_dco()

        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn(signed[:12], result.stderr)
        self.assertIn(unsigned[:12], result.stderr)

    def test_recovery_sequence_stops_and_continues_for_each_failing_commit(self) -> None:
        repository = self.repository()
        first = repository.commit("first unsigned change")
        second = repository.commit(
            trailers("Signed-off-by: Carol Example", subject="malformed sign-off"),
            author=SECOND_AUTHOR,
        )

        result = repository.check_dco()

        self.assertNotEqual(result.returncode, 0)
        self.assertIn(f"While stopped at {first[:12]}:", result.stderr)
        self.assertIn(f"While stopped at {second[:12]}:", result.stderr)
        first_stop = result.stderr.index(f"While stopped at {first[:12]}:")
        first_amend = result.stderr.index(
            "git commit --amend --no-edit --trailer "
            "'Signed-off-by: Alice Example <alice@example.test>'",
            first_stop,
        )
        first_continue = result.stderr.index("git rebase --continue", first_amend)
        second_stop = result.stderr.index(f"While stopped at {second[:12]}:")
        second_amend = result.stderr.index(
            "git commit --amend --no-edit --trailer "
            "'Signed-off-by: Carol Example <carol@example.test>'",
            second_stop,
        )
        malformed_repair = result.stderr.index(
            "git commit --amend  # remove or correct malformed trailers",
            second_amend,
        )
        second_continue = result.stderr.index("git rebase --continue", malformed_repair)
        push = result.stderr.index("git push --force-with-lease", second_continue)
        self.assertLess(first_stop, first_amend)
        self.assertLess(first_amend, first_continue)
        self.assertLess(first_continue, second_stop)
        self.assertLess(second_stop, second_amend)
        self.assertLess(second_amend, malformed_repair)
        self.assertLess(malformed_repair, second_continue)
        self.assertLess(second_continue, push)
        self.assertEqual(result.stderr.count("git rebase --continue"), 2)

    def test_body_line_cannot_spoof_a_trailer(self) -> None:
        repository = self.repository()
        repository.commit(
            f"change\n\nSigned-off-by: {AUTHOR}\n\nThis is still the commit body.\n"
        )

        result = repository.check_dco()

        self.assertNotEqual(result.returncode, 0)
        self.assertIn(f"missing Signed-off-by: {AUTHOR}", result.stderr)

    def test_malformed_signoff_does_not_certify_the_author(self) -> None:
        repository = self.repository()
        repository.commit(trailers("Signed-off-by: Alice Example"))

        result = repository.check_dco()

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("malformed Signed-off-by trailer", result.stderr)

    def test_multiple_signoffs_are_accepted_when_one_matches(self) -> None:
        repository = self.repository()
        repository.commit(
            trailers(
                "Signed-off-by: Reviewer <reviewer@example.test>",
                f"Signed-off-by: {AUTHOR}",
            )
        )

        result = repository.check_dco()

        self.assertEqual(result.returncode, 0, result.stderr)

    def test_every_coauthor_requires_a_matching_signoff(self) -> None:
        repository = self.repository()
        repository.commit(
            trailers(
                f"Co-authored-by: {COAUTHOR}",
                f"Signed-off-by: {AUTHOR}",
            )
        )

        result = repository.check_dco()

        self.assertNotEqual(result.returncode, 0)
        self.assertIn(f"missing Signed-off-by: {COAUTHOR}", result.stderr)

    def test_author_and_all_coauthors_can_certify_one_commit(self) -> None:
        repository = self.repository()
        repository.commit(
            trailers(
                f"Co-authored-by: {COAUTHOR}",
                f"Signed-off-by: {AUTHOR}",
                f"Signed-off-by: {COAUTHOR}",
            )
        )

        result = repository.check_dco()

        self.assertEqual(result.returncode, 0, result.stderr)

    def test_unsigned_merge_commit_is_included(self) -> None:
        repository = self.repository()
        repository.git("switch", "--quiet", "--create", "topic")
        repository.commit(trailers(f"Signed-off-by: {AUTHOR}", subject="topic"))
        repository.git("switch", "--quiet", "main")
        repository.commit(trailers(f"Signed-off-by: {AUTHOR}", subject="main"))
        repository.git("merge", "--quiet", "--no-ff", "topic", "--message", "merge topic")
        merge = repository.revision("HEAD")

        result = repository.check_dco()

        self.assertNotEqual(result.returncode, 0)
        self.assertIn(merge[:12], result.stderr)
        self.assertIn(f"missing Signed-off-by: {AUTHOR}", result.stderr)
        self.assertIn("add a break line after a listed merge command", result.stderr)

    def test_invalid_revisions_fail_closed(self) -> None:
        repository = self.repository()
        for base, head in (("missing-base", "HEAD"), (repository.base, "missing-head")):
            with self.subTest(base=base, head=head):
                result = repository.check_dco(base=base, head=head)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("does not resolve to a commit", result.stderr)

    def test_dependabot_exception_requires_trusted_actor_and_exact_identity(self) -> None:
        cases = (
            (DEPENDABOT, True, True),
            (DEPENDABOT, False, False),
            (
                "dependabot[bot] <dependabot[bot]@users.noreply.github.com>",
                True,
                False,
            ),
            (
                "renovate[bot] <29139614+renovate[bot]@users.noreply.github.com>",
                True,
                False,
            ),
        )
        for author, trusted, expected_pass in cases:
            with self.subTest(author=author, trusted=trusted):
                repository = self.repository()
                repository.commit("automated update", author=author)
                result = repository.check_dco(trusted_dependabot_actor=trusted)
                self.assertEqual(result.returncode == 0, expected_pass, result.stderr)


if __name__ == "__main__":
    unittest.main()
