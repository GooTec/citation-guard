#!/usr/bin/env bash
# One-command install for citation-guard.
#   curl -fsSL .../install.sh | bash      (from a published repo)
#   ./install.sh                          (from a local clone)
# Prefers pipx (isolated, no venv hassle); falls back to pip --user.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET="$HERE"                      # install from this local checkout
[ -f "$HERE/pyproject.toml" ] || TARGET="citation-guard"   # fall back to PyPI name

echo "[citation-guard] installing $TARGET ..."
if command -v pipx >/dev/null 2>&1; then
  pipx install --force "$TARGET"
elif command -v pip >/dev/null 2>&1; then
  pip install --user "$TARGET"
elif command -v pip3 >/dev/null 2>&1; then
  pip3 install --user "$TARGET"
else
  echo "ERROR: need pipx or pip. Install Python 3.9+ first." >&2; exit 1
fi

echo "[citation-guard] verifying install (first run downloads the ~3 GB attribution model once) ..."
if command -v citation-guard >/dev/null 2>&1; then
  citation-guard --selftest
else
  python3 -m citation_guard.cli --selftest
fi

cat <<'EOF'

[citation-guard] ready ✓
  Usage:
    echo '{"answer":"...[1]...","ctxs":[{"text":"..."}]}' | citation-guard
    citation-guard --input answer.json --mode flag      # flag (default-safe), reattribute, or remove
  CPU works; a GPU just makes it faster (set CUDA_VISIBLE_DEVICES).
EOF
