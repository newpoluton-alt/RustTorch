"""Launch two native executables on loopback and bound their joint runtime."""

from pathlib import Path
import subprocess
import sys
import tempfile
import time


def run(executable, scenario):
    with tempfile.TemporaryDirectory() as directory:
        store = Path(directory) / "rendezvous"
        workers = []
        try:
            for rank in range(2):
                workers.append(subprocess.Popen([executable, str(rank), store, scenario]))
            deadline = time.monotonic() + 40
            for worker in workers:
                if worker.wait(timeout=max(0, deadline - time.monotonic())):
                    raise RuntimeError(f"native rank exited with {worker.returncode}")
        finally:
            for worker in workers:
                if worker.poll() is None:
                    worker.kill()
                worker.wait()


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2])
    if sys.argv[2] == "rank_loss":
        run(sys.argv[1], "sum")
