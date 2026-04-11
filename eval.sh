#!/bin/bash
# eval.sh — MLSYS Phase 1 evaluation entry point
#
# Usage:
#   bash eval.sh                          # uses target_spec.json in current dir
#   bash eval.sh /path/to/target_spec.json
#
# Output:
#   results.json       — final answers, keys match spec targets
#   logs/              — per-target reasoning and raw profiling data

set -e

SPEC=${1:-target_spec.json}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$SCRIPT_DIR"

# Install dependencies if needed
if ! python3 -c "import openai, scipy, numpy" 2>/dev/null; then
    pip install -r requirements.txt -q
fi

# Run the pipeline
python3 main.py "$SPEC"
