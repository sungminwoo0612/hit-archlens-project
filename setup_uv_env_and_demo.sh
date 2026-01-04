#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./setup_uv_env_and_demo.sh
#
# Prereq:
#   - uv installed: https://docs.astral.sh/uv/

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

echo "[1/4] Sync dependencies with uv..."
uv sync

echo "[2/4] Run unit tests (if exists)..."
if [ -d "tests" ] || compgen -G "test_*.py" > /dev/null || compgen -G "**/test_*.py" > /dev/null; then
  uv run pytest -q || echo "  - Tests failed or not found. Continuing..."
else
  echo "  - No tests detected. Skipping."
fi

echo "[3/4] Demo command example..."
OUTDIR="runs/demo"
mkdir -p "$OUTDIR"

echo "  To run analysis, use:"
echo "    uv run archlens analyze <IMAGE_PATH> --output $OUTDIR"
echo "  Example:"
echo "    uv run archlens analyze data/images/your_diagram.png --output $OUTDIR"

echo "[4/4] Done. Outputs in: $OUTDIR"

