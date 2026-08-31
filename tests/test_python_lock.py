import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "check-python-lock.py"
PYTORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"


class PythonLockCheckerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        shutil.copyfile(ROOT / "pyproject.toml", self.root / "pyproject.toml")
        shutil.copyfile(ROOT / "uv.lock", self.root / "uv.lock")

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def run_checker(self, *arguments: str, cwd: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(SCRIPT), *arguments],
            cwd=cwd,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_current_repository_passes_with_default_and_explicit_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            cwd = Path(temporary_directory)
            invocations = ((), ("--root", str(ROOT)))
            for arguments in invocations:
                with self.subTest(arguments=arguments):
                    result = self.run_checker(*arguments, cwd=cwd)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(result.stdout, "")
                    self.assertEqual(result.stderr, "")

    def assert_rejected(self, result: subprocess.CompletedProcess[str]) -> None:
        self.assertNotEqual(result.returncode, 0)
        self.assertTrue(result.stderr.startswith("error:"), result.stderr)
        self.assertNotIn("Traceback", result.stderr)
        self.assertEqual(result.stdout, "")

    def test_explicit_manifest_and_lock_paths_pass(self) -> None:
        policy_directory = self.root / "policy"
        policy_directory.mkdir()
        (self.root / "pyproject.toml").rename(policy_directory / "manifest.toml")
        (self.root / "uv.lock").rename(policy_directory / "environment.lock")
        result = self.run_checker(
            "--pyproject",
            str(policy_directory / "manifest.toml"),
            "--lock",
            str(policy_directory / "environment.lock"),
            cwd=self.root,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "")
        self.assertEqual(result.stderr, "")

    def test_root_and_explicit_path_modes_cannot_be_ambiguous(self) -> None:
        ambiguous = self.run_checker(
            "--root",
            str(self.root),
            "--pyproject",
            str(self.root / "pyproject.toml"),
            "--lock",
            str(self.root / "uv.lock"),
            cwd=self.root,
        )
        self.assert_rejected(ambiguous)
        for option, path in (
            ("--pyproject", self.root / "pyproject.toml"),
            ("--lock", self.root / "uv.lock"),
        ):
            with self.subTest(option=option):
                self.assert_rejected(
                    self.run_checker(option, str(path), cwd=self.root)
                )

    def test_missing_and_malformed_inputs_fail_closed(self) -> None:
        lock_path = self.root / "uv.lock"
        lock_path.unlink()
        self.assert_rejected(
            self.run_checker("--root", str(self.root), cwd=self.root)
        )

        lock_path.write_text("not = [valid", encoding="utf-8")
        self.assert_rejected(
            self.run_checker("--root", str(self.root), cwd=self.root)
        )

    def test_alternate_uv_configuration_is_rejected(self) -> None:
        (self.root / "uv.toml").write_text(
            'index-url = "https://evil.example/simple"\n', encoding="utf-8"
        )
        self.assert_rejected(
            self.run_checker("--root", str(self.root), cwd=self.root)
        )

    def test_manifest_contract_rejects_dependency_and_index_drift(self) -> None:
        path = self.root / "pyproject.toml"
        original = path.read_text(encoding="utf-8")
        mutations = {
            "dependency pin": original.replace("numpy==2.5.2", "numpy==2.5.3", 1),
            "Python range": original.replace(">=3.14,<3.15", ">=3.13,<3.15", 1),
            "alternate index": original.replace(
                PYTORCH_CPU_INDEX, "https://pypi.org/simple", 1
            ),
            "implicit index": original.replace(
                "explicit = true", "explicit = false", 1
            ),
            "integer index flag": original.replace(
                "explicit = true", "explicit = 1", 1
            ),
            "package mode": original.replace("package = false", "package = true", 1),
            "dependency group": original
            + '\n[dependency-groups]\ndev = ["requests==2.32.5"]\n',
            "optional dependencies": original
            + '\n[project.optional-dependencies]\ndev = ["requests==2.32.5"]\n',
            "extra tool policy": original + "\n[tool.attacker]\nenabled = true\n",
        }
        for name, mutation in mutations.items():
            with self.subTest(mutation=name):
                self.assertNotEqual(mutation, original)
                path.write_text(mutation, encoding="utf-8")
                self.assert_rejected(
                    self.run_checker("--root", str(self.root), cwd=self.root)
                )
                path.write_text(original, encoding="utf-8")

    def test_lock_contract_rejects_source_artifact_and_platform_drift(self) -> None:
        path = self.root / "uv.lock"
        original = path.read_text(encoding="utf-8")
        mutations = {
            "locked dependency pin": original.replace(
                'version = "2.13.0"', 'version = "2.13.1"', 1
            ),
            "locked torch index": original.replace(
                f'registry = "{PYTORCH_CPU_INDEX}"',
                'registry = "https://pypi.org/simple"',
                1,
            ),
            "transitive dependency drift": original.replace(
                'version = "3.32.4"', 'version = "3.32.5"', 1
            ),
            "transitive direct source": original.replace(
                'source = { registry = "https://pypi.org/simple" }',
                'source = { url = "https://evil.example/payload.whl" }',
                1,
            ),
            "transitive unknown source": original.replace(
                'source = { registry = "https://pypi.org/simple" }',
                'source = { git = "https://evil.example/repository.git" }',
                1,
            ),
            "unapproved transitive registry": original.replace(
                'source = { registry = "https://pypi.org/simple" }',
                'source = { registry = "https://evil.example/simple" }',
                1,
            ),
            "malicious torch artifact origin and digest": original.replace(
                "https://download-r2.pytorch.org/whl/cpu/"
                "torch-2.13.0%2Bcpu-cp314-cp314-manylinux_2_28_x86_64.whl\", "
                'hash = "sha256:d20fa53ee744502fa4c69818a720b05ca0d37abd055d4f6e66cae155114bc691"',
                "https://evil.example/whl/cpu/"
                "torch-2.13.0%2Bcpu-cp314-cp314-manylinux_2_28_x86_64.whl\", "
                f'hash = "sha256:{"0" * 64}"',
                1,
            ),
            "malicious PyPI artifact origin": original.replace(
                "https://files.pythonhosted.org/",
                "https://evil.example/",
                1,
            ),
            "non-HTTPS artifact origin": original.replace(
                "https://files.pythonhosted.org/",
                "http://files.pythonhosted.org/",
                1,
            ),
            "artifact hash algorithm": original.replace("sha256:", "sha512:", 1),
            "artifact hash uppercase": original.replace(
                "sha256:2bde2e4cf732e0153406d8a7bc80620ecf5e621fe0d25e41143c4e3b4733ff30",
                f'sha256:{"A" * 64}',
                1,
            ),
            "artifact missing hash": original.replace("hash =", "digest =", 1),
            "unknown nested artifact": original.replace(
                'sdist = { url = "https://files.pythonhosted.org/',
                'payload = { url = "https://evil.example/payload.whl", hash = "sha256:'
                + "0" * 64
                + '" }\nsdist = { url = "https://files.pythonhosted.org/',
                1,
            ),
            "torch sdist": original.replace(
                'name = "torch"\nversion = "2.13.0"\n',
                'name = "torch"\nversion = "2.13.0"\n'
                'sdist = { url = "https://download-r2.pytorch.org/whl/cpu/'
                'torch-2.13.0.tar.gz", hash = "sha256:'
                + "0" * 64
                + '" }\n',
                1,
            ),
            "altered torch wheel platform": original.replace(
                "torch-2.13.0%2Bcpu-cp314-cp314-manylinux_2_28_x86_64.whl",
                "torch-2.13.0%2Bcpu-cp314-cp314-manylinux_2_28_ppc64le.whl",
                1,
            ),
            "altered torch wheel CPU path": original.replace(
                "download-r2.pytorch.org/whl/cpu/torch-2.13.0-cp314",
                "download-r2.pytorch.org/whl/cu126/torch-2.13.0-cp314",
                1,
            ),
            "altered torch wheel version": original.replace(
                "torch-2.13.0-cp314-cp314-macosx_14_0_arm64.whl",
                "torch-2.13.1-cp314-cp314-macosx_14_0_arm64.whl",
                1,
            ),
            "extra locked package": original
            + '\n[[package]]\nname = "payload"\nversion = "1.0.0"\n'
            + 'source = { url = "https://evil.example/payload.whl" }\n',
            "extra lock field": original + "\nunsafe = true\n",
        }
        for name, mutation in mutations.items():
            with self.subTest(mutation=name):
                self.assertNotEqual(mutation, original)
                path.write_text(mutation, encoding="utf-8")
                self.assert_rejected(
                    self.run_checker("--root", str(self.root), cwd=self.root)
                )
                path.write_text(original, encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
