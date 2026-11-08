#!/usr/bin/env bash
set -euo pipefail
# Q1 compares RAG, graph content, full graph context, and rewired graph context.
source .venvs/grb-min/bin/activate
export OLLAMA_MODEL="${OLLAMA_MODEL:-qwen2.5:3b}"
for split in novel medical; do
  echo "=== Q1 $split $(date -Is) ==="
  # Resume enabled so a stopped run continues from saved question-level outputs.
  python scripts/run_experiment.py \
    --split "$split" \
    --conditions q1 \
    --budgets 1024,4096 \
    --model "$OLLAMA_MODEL" \
    --resume
done
echo "=== Q1 complete $(date -Is) ==="
