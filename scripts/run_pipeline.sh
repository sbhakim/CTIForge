#!/bin/bash
# Run CTIForge pipeline on a single document
# Usage: ./scripts/run_pipeline.sh <input_file> [output_dir]

set -e

INPUT="${1:?Usage: $0 <input_file> [output_dir]}"
OUTPUT="${2:-output}"

echo "Running CTIForge pipeline..."
echo "  Input:  $INPUT"
echo "  Output: $OUTPUT"

python -m src.cli extract "$INPUT" --output-dir "$OUTPUT"
