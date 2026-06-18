#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/mnt/s/LLMs/Qwen3VL-children/Paper2_T-SNE_pipeline}"
PYTHON="${PYTHON:-/home/rk/miniconda3/envs/llm/bin/python}"
export HF_HOME="${HF_HOME:-/home/rk/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false

cd "$PROJECT_DIR"
"$PYTHON" scripts/08_affective_alignment.py "$@"
"$PYTHON" scripts/09_semantic_coverage_analysis.py
"$PYTHON" scripts/07_build_full_semantic_territory_report.py
