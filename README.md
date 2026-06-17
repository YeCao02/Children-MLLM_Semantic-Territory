# Children-MLLM Semantic Territory

This repository contains a reproducible analysis pipeline for comparing children's real environmental perception with MLLM-generated child-perspective evaluations in a shared semantic embedding space.

The method is inspired by Hao et al. (2026), *Artificial intelligence tools expand scientists' impact but contract science's focus*. Their Fig. 3 separates high-dimensional semantic extent measurement from t-SNE visualization. This project adapts the same logic to AI-vs-children text comparison:

- use Qwen3-Embedding-8B vectors as the semantic space;
- compute semantic extent, centroid distance, entropy and nearest-neighbor overlap in the original high-dimensional space;
- use traditional t-SNE and a Hao-style semantic territory map only as explanatory visualizations;
- distinguish shared semantic ground from group-specific semantic territories.

## Repository Contents

- `scripts/01_extract_student_corpus.py`  
  Extracts child interview segments and AI outputs into local JSON corpora.
- `scripts/02_compute_embeddings_8b.py`  
  Computes Qwen3-Embedding-8B GGUF embeddings via `llama-cpp-python`.
- `scripts/03_semantic_space_analysis.py`  
  Legacy t-SNE, cosine similarity and keyword analysis.
- `scripts/05_method_audit_v2.py`  
  Main method audit: text-type diagnosis, high-dimensional metrics, centroid matrix and keyword contrast.
- `scripts/06_make_hao_style_visual.py`  
  Generates a Hao-style AI-children semantic territory map.
- `scripts/07_build_full_semantic_territory_report.py`  
  Builds the comprehensive HTML report.
- `docs/`  
  Method notes and implementation documentation.
- `data/figures/` and `data/figures_v2/`  
  Derived figures only. Raw corpora and embedding files are intentionally excluded.

## Data Policy

Raw interview transcripts, model output pickle files and embedding pickle files are not committed.

Expected local-only files:

- `txt_cleaned/meeting_*.txt`
- `data/model_texts.pkl`
- `data/corpus_children.json`
- `data/corpus_ai.json`
- `data/embeddings.pkl`

See `txt_cleaned/README.md` and `data/README.md` for placeholders and expected schemas.

## Quick Start

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Extract corpora:

```bash
python scripts/01_extract_student_corpus.py
```

Compute embeddings in an environment where the GGUF model is available:

```bash
export QWEN3_EMBEDDING_GGUF=/home/rk/models/Qwen3-Embedding-8B-GGUF/Qwen3-Embedding-8B-Q8_0.gguf
python scripts/02_compute_embeddings_8b.py
```

Run the method audit and build reports:

```bash
python scripts/03_semantic_space_analysis.py
python scripts/05_method_audit_v2.py
python scripts/06_make_hao_style_visual.py
python scripts/07_build_full_semantic_territory_report.py
```

Primary report:

- `Children_MLLM_Semantic_Territory_Full_Report.html`

## Method Summary

The recommended paper framing is not simply "AI differs from children." A stronger framing is:

> AI and children share a semantic core around abstract child-friendly evaluation dimensions, but they expand into different semantic territories. Children emphasize situated, embodied and everyday urban experience; AI emphasizes standardized planning vocabulary and generic child-friendly design templates.

The Hao-style visualization is explanatory. Formal claims should rely on high-dimensional Qwen3 embedding metrics rather than t-SNE area.

