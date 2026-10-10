#!/usr/bin/env bash
# Install the GroundAnything `serve` profile into .venv-serve.
#
# Why this exists instead of just `python3 run.py setup serve`:
#
#   The CPFS/NFS project mount has pathological small-file metadata cost
#   (measured: 300 file creates = 9477 ms, 300 deletes = 10682 ms, vs 4 ms on
#   tmpfs). pip unpacks ~26k small files per install, so a NAS-backed venv ran
#   at ~7 MB/min with an estimated 11+ hours remaining.
#
#   The fix is to place the venv on /dev/shm (tmpfs, 100 GB) and bind-mount it
#   at the project-relative path the tooling expects. That requires the
#   directory to exist before the mount, but scripts/setup_environment.py:30
#   refuses any existing venv directory. This script therefore replays the
#   exact documented command sequence (see scripts/setup_environment.py) minus
#   that single existence guard.
#
# Usage:  bash scripts/install_serve_env_tmpfs.sh
# Result: .venv-serve -> bind mount of /dev/shm/ga-serve-venv, fully installed.
#
# NOTE: /dev/shm is memory-backed and does not survive a container restart.
#       Re-run this script after a restart to rebuild the environment.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$ROOT/.venv-serve"
SHM=/dev/shm/ga-serve-venv
TMPDIR_SHM=/dev/shm/ga-tmp
CACHE_SHM=/dev/shm/ga-pip-cache
CONSTRAINTS="$ROOT/requirements/serve.txt"

echo "== GroundAnything serve profile installer =="
echo "repo      : $ROOT"
echo "venv path : $VENV  (tmpfs via bind mount)"
echo "constraints: $CONSTRAINTS"

# --- 1. tmpfs-backed venv location, bind-mounted at the project-relative path --
mkdir -p "$SHM" "$TMPDIR_SHM" "$CACHE_SHM"
if mountpoint -q "$VENV"; then
  echo "bind mount already active"
else
  mkdir -p "$VENV"
  mount --bind "$SHM" "$VENV"
  echo "bind mounted $SHM -> $VENV"
fi
export TMPDIR="$TMPDIR_SHM"
export PIP_CACHE_DIR="$CACHE_SHM"

# --- 1b. package index --------------------------------------------------------
# pypi.org measured 16.2 s for a simple index page and ~0.9 MB/s for wheels from
# this host, while the Aliyun mirror answered in 0.14 s and carries every pinned
# version this profile needs (verified: torch 2.9.1, triton 3.5.1, sgl-kernel
# 0.3.20, flashinfer 0.5.3, transformers 5.5.4 — all x86_64/cp312). Use the
# mirror as primary and keep pypi.org as the fallback for anything it lacks.
# Override with: PIP_INDEX_URL=... bash scripts/install_serve_env_tmpfs.sh
MIRROR=https://mirrors.aliyun.com/pypi/simple
PYPI=https://pypi.org/simple
PIP_FLAGS=(-i "$MIRROR" --extra-index-url "$PYPI")
echo "index     : $MIRROR (fallback $PYPI)"

# --- 2. create the virtual environment (written to tmpfs) --------------------
if [ ! -x "$VENV/bin/python" ]; then
  python3 -m venv "$VENV"
  echo "created venv: $("$VENV/bin/python" -V)"
fi
PY="$VENV/bin/python"

# --- 3. base build tooling ---------------------------------------------------
"$PY" -m pip install "${PIP_FLAGS[@]}" -c "$CONSTRAINTS" setuptools\>=68 wheel packaging ninja psutil einops

# --- 4. extract the vendored custom SGLang engine ----------------------------
python3 "$ROOT/scripts/prepare_dependencies.py" --apply --name sglang

# --- 5. the serve requirement set + the vendored SGLang (editable) -----------
"$PY" -m pip install "${PIP_FLAGS[@]}" -c "$CONSTRAINTS" -r "$CONSTRAINTS" -e "$ROOT/vendor/sglang"

# --- 6. cuDNN override required by SGLang for Torch 2.9.1 Conv3d -------------
"$PY" -m pip install "${PIP_FLAGS[@]}" --no-deps nvidia-cudnn-cu12==9.16.0.29

# --- 7. the project itself, then the documented environment health check -----
"$PY" -m pip install "${PIP_FLAGS[@]}" --no-deps -e "$ROOT"
"$PY" "$ROOT/scripts/check_serve_environment.py"

"$PY" -m pip freeze --all > "$VENV/installed.freeze.txt"
echo "== serve environment ready =="
