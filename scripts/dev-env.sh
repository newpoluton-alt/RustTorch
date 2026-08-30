#!/bin/sh
# Source from the repository root: . scripts/dev-env.sh

_rusttorch_root=${RUSTTORCH_ROOT:-"$(pwd)"}
_rusttorch_python="$_rusttorch_root/.venv/bin/python3"

if [ ! -f "$_rusttorch_root/Cargo.toml" ]; then
    echo "error: run this script from the RustTorch repository root" >&2
    return 1 2>/dev/null || exit 1
fi
if [ ! -x "$_rusttorch_python" ]; then
    echo "error: missing .venv; create the project-local Python environment first" >&2
    return 1 2>/dev/null || exit 1
fi

_rusttorch_torch_version=$("$_rusttorch_python" -c 'import torch; print(torch.__version__.split("+", 1)[0])') || {
    echo "error: project Python cannot import torch" >&2
    return 1 2>/dev/null || exit 1
}
_rusttorch_python_version=$("$_rusttorch_python" -c 'import platform; print(platform.python_version())')
if [ "$_rusttorch_torch_version" != "2.13.0" ]; then
    echo "error: tch 0.26.0 requires project torch 2.13.0; found $_rusttorch_torch_version" >&2
    return 1 2>/dev/null || exit 1
fi

_rusttorch_torch_lib=$("$_rusttorch_python" -c 'from pathlib import Path; import torch; print(Path(torch.__file__).resolve().parent / "lib")')
if [ ! -d "$_rusttorch_torch_lib" ]; then
    echo "error: torch library directory not found: $_rusttorch_torch_lib" >&2
    return 1 2>/dev/null || exit 1
fi

export RUSTTORCH_ROOT="$_rusttorch_root"
export VIRTUAL_ENV="$_rusttorch_root/.venv"
export PATH="$VIRTUAL_ENV/bin:$PATH"
export LIBTORCH_USE_PYTORCH=1

case $(uname -s) in
    Darwin)
        case :${DYLD_LIBRARY_PATH:-}: in
            *:"$_rusttorch_torch_lib":*) ;;
            *) export DYLD_LIBRARY_PATH="$_rusttorch_torch_lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}" ;;
        esac
        ;;
    Linux)
        case :${LD_LIBRARY_PATH:-}: in
            *:"$_rusttorch_torch_lib":*) ;;
            *) export LD_LIBRARY_PATH="$_rusttorch_torch_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
        esac
        ;;
esac

echo "RustTorch environment: Python $_rusttorch_python_version with PyTorch $_rusttorch_torch_version from $VIRTUAL_ENV"
unset _rusttorch_root _rusttorch_python _rusttorch_python_version _rusttorch_torch_lib _rusttorch_torch_version
