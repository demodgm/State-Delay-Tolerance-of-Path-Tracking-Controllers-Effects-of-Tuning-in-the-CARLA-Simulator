#!/usr/bin/env bash
# Reproduces the headline numbers. Requires a CARLA server on :2000.
#
# Works from any directory and whether or not the package is installed:
#   bash scripts/run_experiment.sh
#   ./scripts/run_experiment.sh
#   bash run_experiment.sh          (from inside scripts/)
set -euo pipefail

# resolve the repo root from this script's own location, then work from there
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# src/ on the path so `python -m latency...` works without `pip install -e .`
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

# `python` is not always present on Ubuntu; override with PYTHON=... if needed
PYTHON="${PYTHON:-$(command -v python3 || command -v python)}"
if [ -z "$PYTHON" ]; then
    echo "error: no python3 on PATH. Set PYTHON=/path/to/python" >&2
    exit 1
fi

if ! "$PYTHON" -c "import latency" 2>/dev/null; then
    echo "error: cannot import the 'latency' package from $ROOT/src" >&2
    echo "       check that src/latency/__init__.py exists" >&2
    exit 1
fi

# check the third-party deps up front rather than failing mid-tune
MISSING=$("$PYTHON" - <<'PYCHECK'
mods = {"numpy": "numpy", "pandas": "pandas",
        "matplotlib": "matplotlib", "yaml": "PyYAML"}
missing = []
for mod, pkg in mods.items():
    try:
        __import__(mod)
    except ImportError:
        missing.append(pkg)
print(" ".join(missing))
PYCHECK
)
if [ -n "$MISSING" ]; then
    echo "error: missing Python packages: $MISSING" >&2
    echo "       pip install $MISSING" >&2
    echo "   or: pip install -e .          (note the 'pip' - plain 'install'" >&2
    echo "                                  is a different, unrelated command)" >&2
    exit 1
fi

# pass --selftest through to run the pipeline with no CARLA server
EXTRA=""
RUNS="${RUNS:-results/runs.csv}"
GAINS="${GAINS:-results/gains.json}"
if [ "${1:-}" = "--selftest" ]; then
    EXTRA="--selftest"
    CONFIG="${CONFIG:-configs/selftest.yaml}"
    # separate files: selftest rows mixed into results/runs.csv make the real
    # sweep unanalysable, and the two must never be averaged together
    RUNS="results/selftest_runs.csv"
    GAINS="results/selftest_gains.json"
    echo "*** --selftest: kinematic stand-in, output is NOT real data ***"
fi

CONFIG="${CONFIG:-configs/sweep.yaml}"
THRESHOLD="${THRESHOLD:-0.50}"
mkdir -p results figures

echo "[1/3] accuracy-matched tuning at zero delay"
"$PYTHON" -m latency.sweep tune  --config "$CONFIG" --gains "$GAINS" $EXTRA

echo "[2/3] frozen-gain delay sweep"
"$PYTHON" -m latency.sweep sweep --config "$CONFIG" --gains "$GAINS" \
                                 --out "$RUNS" $EXTRA

echo "[3/3] tables and figures"
"$PYTHON" -m latency.analyze --runs "$RUNS" --cte-threshold "$THRESHOLD" \
        ${EXTRA:+--allow-selftest}
