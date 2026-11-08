#!/usr/bin/env bash
set -euo pipefail
# Q2 compares full graph retrieval with greedy/adaptive retrieval and the RAG baseline.
source .venvs/grb-min/bin/activate
export OLLAMA_MODEL="${OLLAMA_MODEL:-qwen2.5:3b}"
for split in novel medical; do
  echo "=== Q2 $split $(date -Is) ==="
  # The shared RAG and graph_full predictions can be reused by the pipeline.
  python scripts/run_experiment.py \
    --split "$split" \
    --conditions q2 \
    --budgets 1024,4096 \
    --model "$OLLAMA_MODEL" \
    --resume
done
echo "=== Q2 complete $(date -Is) ==="
