#!/bin/bash
# Run evaluation against CTI-Nexus gold annotations
# Usage: ./scripts/run_eval.sh [annotations_dir] [max_docs]

set -e

ANNOTATIONS="${1:-data/annotations/ctinexus}"
MAX_DOCS="${2:-5}"

echo "Running evaluation..."
echo "  Annotations: $ANNOTATIONS"
echo "  Max docs:    $MAX_DOCS"

python -m src.cli evaluate "$ANNOTATIONS" \
    --output-dir output/eval \
    --max-docs "$MAX_DOCS"
