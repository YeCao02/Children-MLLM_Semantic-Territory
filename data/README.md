# Data Directory

This repository stores derived figures and non-sensitive summary metrics only.

Raw or large local files are intentionally excluded from git:

- `model_texts.pkl`: local MLLM output table.
- `corpus_children.json`: extracted child interview corpus.
- `corpus_ai.json`: extracted AI reason/suggestion corpus.
- `embeddings.pkl`: Qwen3-Embedding-8B vectors.
- `affective_predictions.pkl`: per-text probabilities from the two affect classifiers.

Expected `model_texts.pkl` structure:

- a Python dictionary keyed by model variant, such as `32B_Det`;
- each value is a pandas DataFrame;
- required columns: `image`, `model`, `lang`, `prompt`, `run_id`, `reason`, `suggestion`.

The public repository includes only placeholders, scripts, derived figures and aggregate
summary metrics. Per-text affect predictions are intentionally ignored.

Public derived semantic-coverage outputs:

- `semantic_coverage_metrics.json`: aggregate high-dimensional nearest-neighbor
  coverage rates and confidence intervals.
- `semantic_coverage_points.csv`: per-point group labels, t-SNE coordinates and
  thresholded coverage status. It does not include raw text.
