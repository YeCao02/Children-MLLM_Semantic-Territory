# Implementation Plan

## Objective

Build a reproducible semantic-territory pipeline for comparing children's real environmental perception with MLLM-generated child-perspective evaluations.

The goal is not only to visualize a semantic gap, but to distinguish:

1. shared semantic ground;
2. child-specific semantic territories;
3. AI-specific semantic territories.

## Methodological Reference

The pipeline adapts the logic of Hao et al. (2026), especially Fig. 3:

- embed texts into a high-dimensional semantic space;
- compute semantic extent in the original high-dimensional space;
- use t-SNE only for explanatory visualization;
- use balanced sampling to avoid sample-size artifacts;
- compare distributional concentration through entropy or related spread metrics.

## Pipeline Stages

1. `01_extract_student_corpus.py`
   - Extract child interview utterances from cleaned transcripts.
   - Extract AI `reason` and `suggestion` texts from `model_texts.pkl`.
   - Preserve AI text type labels.

2. `02_compute_embeddings_8b.py`
   - Compute Qwen3-Embedding-8B vectors through `llama-cpp-python`.
   - Save local-only `data/embeddings.pkl`.

3. `03_semantic_space_analysis.py`
   - Legacy t-SNE, cosine similarity and keyword figures.
   - Kept for continuity and comparison.

4. `05_method_audit_v2.py`
   - Separate child direct perception from child meta-evaluation of AI/expert outputs.
   - Reconstruct AI image-level combined units.
   - Compute high-dimensional semantic extent, centroid distance, nearest-neighbor overlap and diagnostic figures.

5. `06_make_hao_style_visual.py`
   - Build a Hao-style semantic territory map with shared centroid, KE circles and semantic anchors.

6. `07_build_full_semantic_territory_report.py`
   - Compile all figures, key metrics and methodological interpretation into a single HTML report.

## Reporting Principle

Use traditional t-SNE as a diagnostic figure and the Hao-style territory map as the core explanatory figure.

Formal claims should be based on high-dimensional Qwen3 embedding metrics rather than 2D t-SNE area.

