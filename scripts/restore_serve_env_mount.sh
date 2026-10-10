#!/usr/bin/env bash
# Re-attach the GroundAnything serve environment after a container restart.
#
# Background
# ----------
# The serve venv lives on a NAS-backed volume:
#
#   /workspace/cpfs-data/ga-serve-venv        <- persistent storage (NFS)
#            │ bind mount
#            └──> <repo>/.venv-serve           <- the path all tooling expects
#
# The *data* survives a container restart, but the bind mount does not, so
# `.venv-serve` comes back as a plain empty directory and python is missing.
# This script re-creates the mount and verifies the environment. It does NOT
# reinstall anything, and it is safe to re-run.
#
# If the environment itself is gone (first run, or the NAS copy was removed),
# rebuild it with scripts/install_serve_env_tmpfs.sh instead — but note that
# script installs into /dev/shm (fast, volatile). To rebuild persistently,
# install there and then copy the result onto the NAS volume.
#
# Usage:  bash scripts/restore_serve_env_mount.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$ROOT/.venv-serve"
STORE=/workspace/cpfs-data/ga-serve-venv

echo "== GroundAnything serve environment: re-attach =="
echo "repo : $ROOT"
echo "store: $STORE"

if [ ! -x "$STORE/bin/python" ]; then
  echo "ERROR: no environment found at $STORE" >&2
  echo "Rebuild it first (see scripts/install_serve_env_tmpfs.sh)." >&2
  exit 1
fi

# Already attached?
if mountpoint -q "$VENV" && [ -x "$VENV/bin/python" ]; then
  echo "already attached; nothing to do"
else
  # A stale/empty directory may be sitting on the mountpoint. Only remove it
  # when it is genuinely empty, so a real environment is never destroyed.
  if [ -e "$VENV" ] && ! mountpoint -q "$VENV"; then
    if [ -n "$(ls -A "$VENV" 2>/dev/null)" ]; then
      echo "ERROR: $VENV exists and is not empty and is not a mountpoint." >&2
      echo "Refusing to touch it. Inspect manually." >&2
      exit 1
    fi
    rmdir "$VENV"
  fi
  mkdir -p "$VENV"
  mount --bind "$STORE" "$VENV"
  echo "attached: $STORE -> $VENV"
fi

# --- verify -----------------------------------------------------------------
echo
echo "--- verification ---"
"$VENV/bin/python" - <<'PY'
import sys, torch, sglang, transformers, importlib.metadata as md
print("sys.prefix  :", sys.prefix)
print("torch       :", torch.__version__, "| cuda", torch.cuda.is_available(),
      "| devices", torch.cuda.device_count())
print("cudnn       :", torch.backends.cudnn.version())
print("transformers:", transformers.__version__)
print("sglang      :", sglang.__version__)
print("sgl-kernel  :", md.version("sgl-kernel"))
print("flashinfer  :", md.version("flashinfer-python"))
PY

echo
"$VENV/bin/python" "$ROOT/scripts/check_serve_environment.py" | tail -12
echo
echo "== ready: python3 run.py serve --decoder denoise =="
