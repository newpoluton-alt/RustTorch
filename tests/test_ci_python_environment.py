"""Exercise the loader jobs' environment handoff to Cargo subprocesses."""
import os
from pathlib import Path
import re
import subprocess
import tempfile
import textwrap
import unittest

ROOT = Path(__file__).resolve().parents[1]


class LoaderPythonEnvironmentTests(unittest.TestCase):
    @unittest.skipIf(os.name == "nt", "Unix workflow shell regression")
    def test_loader_exports_select_the_locked_python_for_child_processes(self):
        workflow = (ROOT / ".github/workflows/ci.yml").read_text()
        for job, step, runner in (
            ("loader-platform", "Export Unix LibTorch library path", "Linux"),
            ("loader-platform", "Export Unix LibTorch library path", "macOS"),
            ("loader-stress", "Export LibTorch library path", "Linux"),
        ):
            with self.subTest(job=job, runner=runner), tempfile.TemporaryDirectory() as tmp:
                block = workflow.split(f"  {job}:\n", 1)[1]
                block = block.split(f"      - name: {step}\n", 1)[1]
                script = re.search(r"        run: \|\n((?:          .*\n)+)", block)
                self.assertIsNotNone(script)
                env_file, path_file = Path(tmp) / "env", Path(tmp) / "path"
                env_file.touch()
                path_file.touch()
                env = dict(os.environ, PATH="/usr/bin:/bin", RUNNER_OS=runner,
                           GITHUB_ENV=str(env_file), GITHUB_PATH=str(path_file))
                env.pop("VIRTUAL_ENV", None)
                subprocess.run(["/bin/bash", "-e", "-c", textwrap.dedent(script[1])],
                               cwd=ROOT, env=env, check=True, capture_output=True)
                env.update(line.split("=", 1) for line in env_file.read_text().splitlines())
                env["PATH"] = os.pathsep.join([*reversed(path_file.read_text().splitlines()), env["PATH"]])
                self.assertEqual(env.get("VIRTUAL_ENV"), str(ROOT / ".venv"))
                selected = subprocess.run(
                    ["python", "-c", "import sys, torch; print(sys.prefix)"],
                    cwd=ROOT, env=env, check=True, capture_output=True, text=True,
                )
                self.assertEqual(Path(selected.stdout.strip()).resolve(), (ROOT / ".venv").resolve())
