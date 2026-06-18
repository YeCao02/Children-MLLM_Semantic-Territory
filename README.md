# Children-MLLM Semantic Territory

This repository contains a reproducible analysis pipeline for comparing children's real environmental perception with MLLM-generated child-perspective evaluations in a shared semantic embedding space.

The method is inspired by Hao et al. (2026), *Artificial intelligence tools expand scientists' impact but contract science's focus*. Their Fig. 3 separates high-dimensional semantic extent measurement from t-SNE visualization. This project adapts the same logic to AI-vs-children text comparison:

- use Qwen3-Embedding-8B vectors as the semantic space;
- compute semantic extent, centroid distance, entropy and directional nearest-neighbor coverage in the original high-dimensional space;
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
- `scripts/08_affective_alignment.py`
  Runs dual-model Chinese affect classification, robustness checks and semantic-affective alignment.
- `scripts/09_semantic_coverage_analysis.py`  
  Computes the main high-dimensional directional coverage curves and t-SNE coverage-status diagnostic.
- `scripts/run_affective_alignment_wsl.sh`
  Runs the affective module and rebuilds the report in the existing WSL `llm` environment.
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
python scripts/09_semantic_coverage_analysis.py
python scripts/07_build_full_semantic_territory_report.py
```

Run the semantic-affective module in WSL:

```bash
cd /mnt/s/LLMs/Qwen3VL-children/Paper2_T-SNE_pipeline
bash scripts/run_affective_alignment_wsl.sh
```

The script uses `/home/rk/miniconda3/envs/llm/bin/python` by default. Hugging Face
weights are cached under `/home/rk/.cache/huggingface` and are not committed.

Primary report:

- `Children_MLLM_Semantic_Territory_Full_Report.html`

## Method Summary

The recommended paper framing is not simply "AI differs from children." A stronger framing is:

> AI outputs are easier to match into children's semantic territory than the reverse, while children occupy a broader and more situated semantic extent. Children emphasize embodied everyday urban experience; AI emphasizes standardized planning vocabulary and generic child-friendly design templates.

The high-dimensional directional coverage curves are the main evidence. The Hao-style visualization and t-SNE plots are explanatory. Formal claims should rely on Qwen3 embedding metrics rather than t-SNE area.

## Affective Analysis

The affective module uses `Johnson8187/Chinese-Emotion-Small` as a Chinese-specific
classifier and `tabularisai/multilingual-emotion-classification` as a robustness model.
Their different labels are mapped to four shared dimensions: positive engagement,
concern/distress, inquiry/surprise and neutral expression.

Predictions represent affective cues expressed in text. They are not measurements of a
child's internal emotion, mental health or stable disposition. Fine-grained claims require
human-coded validation.

## Related Methods

See `docs/related_work_visualization_references.md` for references on high-dimensional
knowledge extent, generated-human text distributional gaps, precision/recall-style
coverage, embedding-space visualization and AI-human urban perception comparisons.
