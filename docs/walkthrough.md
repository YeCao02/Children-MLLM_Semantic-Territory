# Results Walkthrough

## Main Report

Open:

```text
Children_MLLM_Semantic_Territory_Full_Report.html
```

This report contains:

- key data counts and high-dimensional metrics;
- Hao-style semantic territory visualization;
- traditional t-SNE;
- diagnostic t-SNE separating child direct perception, child meta-evaluation, AI reason and AI suggestion;
- semantic extent / radius distribution;
- exploratory semantic entropy;
- centroid distance matrix;
- domain-term contrast;
- pairwise cosine similarity distribution;
- recommended follow-up analyses.

## Interpretation

The current evidence supports an "overlap plus divergence" framing.

Shared ground:

- safety;
- facilities;
- interaction;
- attractiveness;
- general child-friendly evaluation.

Children-specific territory:

- practical use;
- school and commercial street context;
- bicycle / mobility;
- danger and fear;
- daily-life constraints.

AI-specific territory:

- smart facilities;
- creativity;
- fresh air;
- exploration;
- generic interaction devices;
- planning-style vocabulary.

## Method Caveat

The t-SNE maps are explanatory figures. The main claims should cite high-dimensional metrics from Qwen3-Embedding-8B vectors:

- semantic extent;
- p95 radius;
- centroid distance;
- nearest-neighbor overlap;
- semantic entropy;
- keyword/domain-term contrast.

## Current Limitation

The strongest next improvement is image-level pairing:

```text
image_id -> child comments -> AI reason/suggestion
```

Without this alignment, the current analysis is a distribution-level comparison rather than a per-image semantic error analysis.

