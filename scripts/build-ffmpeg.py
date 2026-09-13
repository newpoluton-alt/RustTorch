#!/usr/bin/env python3
"""Build and audit the minimal dynamically linked FFmpeg profile used in CI.

The upstream source is unmodified. No native binaries enter crate archives.
See THIRD_PARTY_NOTICES.md and docs/domain-data.md for the linking contract.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile
import tempfile
import urllib.request

VERSION = "8.1.2"
SOURCE = f"https://ffmpeg.org/releases/ffmpeg-{VERSION}.tar.xz"
SHA256 = "464beb5e7bf0c311e68b45ae2f04e9cc2af88851abb4082231742a74d97b524c"
CONFIGURE = (
    "--disable-gpl", "--disable-nonfree", "--disable-version3",
    "--enable-shared", "--disable-static", "--disable-autodetect",
    "--disable-network", "--disable-doc", "--disable-programs",
    "--disable-everything", "--disable-x86asm",
    "--enable-decoder=ffv1,pcm_s16le", "--enable-demuxer=matroska,wav",
    "--enable-protocol=file",
)


def audit(prefix: Path) -> dict:
    """Reject a different ABI, static output or GPL/nonfree configuration."""
    suffix = "dylib" if sys.platform == "darwin" else "so"
    if list((prefix / "lib").glob("*.a")):
        raise ValueError("the CI FFmpeg profile must not contain static archives")
    records = {}
    for library, major in (("avcodec", 62), ("avformat", 62), ("avutil", 60),
                           ("avdevice", 62), ("avfilter", 11), ("swscale", 9), ("swresample", 6)):
        native = ctypes.CDLL(str(prefix / "lib" / f"lib{library}.{suffix}"))
        version = getattr(native, f"{library}_version")
        version.restype = ctypes.c_uint
        license_text = getattr(native, f"{library}_license")
        license_text.restype = ctypes.c_char_p
        configuration = getattr(native, f"{library}_configuration")
        configuration.restype = ctypes.c_char_p
        license_value = license_text().decode("utf-8")
        config = configuration().decode("utf-8")
        if version() >> 16 != major or license_value != "LGPL version 2.1 or later":
            raise ValueError(f"unexpected {library} ABI/license: {version()}, {license_value}")
        flags = shlex.split(config)
        if any(flag not in flags for flag in CONFIGURE):
            raise ValueError(f"unexpected {library} build configuration")
        if any(flag in flags for flag in ("--enable-gpl", "--enable-nonfree", "--enable-static")):
            raise ValueError(f"forbidden {library} build configuration")
        records[library] = {"version": version(), "license": license_value, "configuration": config}
    return {"source": SOURCE, "sha256": SHA256, "libraries": records}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument("--audit", action="store_true")
    args = parser.parse_args()
    prefix = args.prefix.resolve()
    if args.audit:
        print(json.dumps(audit(prefix), indent=2))
        return
    if prefix.exists():
        raise ValueError("use a fresh prefix; existing native builds are never overwritten")
    if sys.platform not in ("linux", "darwin"):
        raise ValueError("the reproducible source-build lane currently supports Linux and macOS")
    with tempfile.TemporaryDirectory(prefix="rusttorch-ffmpeg-") as directory:
        root = Path(directory)
        archive = root / "source.tar.xz"
        with urllib.request.urlopen(SOURCE, timeout=60) as response:
            payload = response.read(32 * 1024 * 1024 + 1)
        if len(payload) > 32 * 1024 * 1024 or hashlib.sha256(payload).hexdigest() != SHA256:
            raise ValueError("FFmpeg source size or SHA-256 mismatch")
        archive.write_bytes(payload)
        with tarfile.open(archive) as source:
            source.extractall(root, filter="data")
        source_root = root / f"ffmpeg-{VERSION}"
        subprocess.run([str(source_root / "configure"), f"--prefix={prefix}", *CONFIGURE], cwd=source_root, check=True)
        subprocess.run(["make", f"-j{min(os.cpu_count() or 1, 8)}"], cwd=source_root, check=True)
        subprocess.run(["make", "install"], cwd=source_root, check=True)
    environment = os.environ.copy()
    variable = "DYLD_LIBRARY_PATH" if sys.platform == "darwin" else "LD_LIBRARY_PATH"
    environment[variable] = str(prefix / "lib") + os.pathsep + environment.get(variable, "")
    result = subprocess.run([sys.executable, __file__, "--prefix", str(prefix), "--audit"],
                            env=environment, check=True, capture_output=True, text=True)
    (prefix / "rusttorch-build-audit.json").write_text(result.stdout)
    print(result.stdout)


if __name__ == "__main__":
    main()
