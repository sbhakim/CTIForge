#!/bin/bash
# Run ablation study with different pipeline configurations
# Usage: ./scripts/run_ablation.sh [annotations_dir] [max_docs]

set -e

ANNOTATIONS="${1:-data/annotations/ctinexus}"
MAX_DOCS="${2:-5}"

echo "Running ablation study..."
echo "  Annotations: $ANNOTATIONS"
echo "  Max docs:    $MAX_DOCS"

# B-only: extraction only
echo "=== Config: B-only ==="
python -m src.cli evaluate "$ANNOTATIONS" \
    --output-dir output/ablation/b_only \
    --max-docs "$MAX_DOCS" \
    --no-validation --no-canonicalization

# B+C: extraction + validation
echo "=== Config: B+C ==="
python -m src.cli evaluate "$ANNOTATIONS" \
    --output-dir output/ablation/b_c \
    --max-docs "$MAX_DOCS" \
    --no-canonicalization

# B+C+D: extraction + validation + canonicalization
echo "=== Config: B+C+D ==="
python -m src.cli evaluate "$ANNOTATIONS" \
    --output-dir output/ablation/b_c_d \
    --max-docs "$MAX_DOCS"

echo "Ablation complete. Results in output/ablation/"
