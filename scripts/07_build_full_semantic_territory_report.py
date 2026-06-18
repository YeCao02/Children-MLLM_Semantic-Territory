from __future__ import annotations

import base64
import importlib.util
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
FIG_DIR = DATA_DIR / "figures_v2"


def load_method_module():
    path = Path(__file__).with_name("05_method_audit_v2.py")
    spec = importlib.util.spec_from_file_location("method_audit_v2", path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    spec.loader.exec_module(module)
    return module


def image_uri(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("utf-8")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def semantic_entropy(points: np.ndarray, grid_size: int = 2) -> float:
    if len(points) == 0:
        return float("nan")
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    steps = (maxs - mins) / grid_size
    steps[steps == 0] = 1.0
    grid = np.floor((points - mins) / steps).astype(int)
    grid = np.clip(grid, 0, grid_size - 1)
    _, counts = np.unique(grid, axis=0, return_counts=True)
    probs = counts / counts.sum()
    return float(-np.sum(probs * np.log(probs)))


def bootstrap_entropy(child_embs: np.ndarray, ai_embs: np.ndarray, repeats: int = 500, sample_n: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    sample_n = min(sample_n, len(child_embs), len(ai_embs))
    rows = []
    for i in range(repeats):
        c_idx = rng.choice(len(child_embs), sample_n, replace=False)
        a_idx = rng.choice(len(ai_embs), sample_n, replace=False)
        stacked = np.vstack([child_embs[c_idx], ai_embs[a_idx]])
        pca_n = min(10, stacked.shape[0] - 1, stacked.shape[1])
        pca = PCA(n_components=pca_n, random_state=42)
        pca_points = pca.fit_transform(stacked)
        c_points = pca_points[:sample_n]
        a_points = pca_points[sample_n:]
        rows.append({"group": "Children direct", "entropy": semantic_entropy(c_points), "iteration": i})
        rows.append({"group": "AI combined", "entropy": semantic_entropy(a_points), "iteration": i})
    return pd.DataFrame(rows)


def make_entropy_figure() -> tuple[Path, dict]:
    method = load_method_module()
    method.configure_plotting()
    emb_data = method.load_embedding_data()
    child_records = method.extract_children_segments_with_metadata()
    child_mask = np.array([method.is_direct_environment_text(r["text"]) for r in child_records])
    child_embs = emb_data["children"]["embeddings"][child_mask]
    ai_meta = method.reconstruct_ai_metadata(emb_data["ai"]["texts"], emb_data["ai"]["embeddings"])
    ai_embs = ai_meta["combined"]["embeddings"]

    df = bootstrap_entropy(child_embs, ai_embs)
    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    sns.violinplot(
        data=df,
        x="group",
        y="entropy",
        hue="group",
        palette={"Children direct": "#4C78D8", "AI combined": "#F25A5A"},
        inner="quartile",
        legend=False,
        ax=ax,
    )
    ax.set_title("Exploratory semantic entropy\n(PCA-10D grid entropy, balanced resampling)")
    ax.set_xlabel("")
    ax.set_ylabel("Shannon entropy")
    fig.tight_layout()
    out = FIG_DIR / "Fig_v2_semantic_entropy.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)

    summary = {}
    for group, sub in df.groupby("group"):
        summary[group] = {
            "median": float(sub["entropy"].median()),
            "mean": float(sub["entropy"].mean()),
            "ci95": [
                float(np.percentile(sub["entropy"], 2.5)),
                float(np.percentile(sub["entropy"], 97.5)),
            ],
        }
    return out, summary


def figure_card(title: str, path: Path, caption: str) -> str:
    return f"""<section class="card figure-card">
      <h2>{title}</h2>
      <img src="{image_uri(path)}" alt="{title}">
      <p class="caption">{caption}</p>
    </section>"""


def metric(value: float | int | str, digits: int = 3) -> str:
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def build_html() -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    entropy_fig, entropy_summary = make_entropy_figure()

    v2 = read_json(DATA_DIR / "analysis_results_v2.json")
    hao = read_json(DATA_DIR / "hao_style_metrics.json")
    old = read_json(DATA_DIR / "analysis_results.json")
    affect = read_json(DATA_DIR / "affective_alignment_metrics.json")
    coverage = read_json(DATA_DIR / "semantic_coverage_metrics.json")

    figs = {
        "coverage": FIG_DIR / "Fig_v2_semantic_coverage_curves.png",
        "frontier": FIG_DIR / "Fig_v2_semantic_pr_frontier.png",
        "coverage_tsne": FIG_DIR / "Fig_v2_tsne_coverage_status.png",
        "hao": FIG_DIR / "Fig_v2_hao_style_ai_children.png",
        "traditional": DATA_DIR / "figures" / "Fig3_tsne.png",
        "diagnostic": FIG_DIR / "Fig_v2_tsne_diagnostic.png",
        "extent": FIG_DIR / "Fig_v2_highdim_extent.png",
        "heatmap": FIG_DIR / "Fig_v2_centroid_heatmap.png",
        "keywords": FIG_DIR / "Fig_v2_keyword_contrast.png",
        "entropy": entropy_fig,
        "violin": DATA_DIR / "figures" / "Fig3_violin.png",
        "affect_profiles": DATA_DIR / "figures_affect" / "Fig_affect_main_profiles.png",
        "affect_robustness": DATA_DIR / "figures_affect" / "Fig_affect_model_robustness.png",
        "semantic_affective": DATA_DIR / "figures_affect" / "Fig_semantic_affective_alignment.png",
    }

    transferred_methods = [
        ("High-dimensional embedding space", "Hao et al. use SPECTER 2.0 to represent papers in a 768D scientific-text space; here, Qwen3-Embedding-8B represents children and AI texts in a 4096D semantic space."),
        ("Knowledge extent", "The central indicator is not the t-SNE footprint. KE is computed in the original embedding space as the maximum distance from the group centroid."),
        ("Observed-sample comparison", "Hao repeatedly samples equal-sized AI and non-AI paper batches. Here, the main comparison keeps the observed units fixed: 331 direct child utterances and 220 AI image-level outputs; bootstrap intervals summarize uncertainty without changing the sample definition."),
        ("Directional coverage", "Nearest-neighbor retrieval is directional. AI-covered-by-children and child-covered-by-AI answer different questions and should not be inferred from 2D t-SNE overlap."),
        ("Visual t-SNE only", "The Hao-style map is kept as an explanatory figure. Statistical claims should refer to 4096D metrics, not 2D t-SNE area."),
        ("Entropy / concentration", "Hao also compares knowledge entropy. Here, exploratory semantic entropy tests whether AI wording is more concentrated than children’s direct environmental perception."),
        ("Sub-domain logic", "Hao tests the pattern across fields/subfields. The next equivalent step is to compare image types, space types, or child-friendly dimensions once labels are added."),
    ]

    method_rows = "\n".join(
        f"<tr><td>{name}</td><td>{desc}</td></tr>" for name, desc in transferred_methods
    )

    related_work = [
        (
            "Hao et al. (2026), Nature",
            "Knowledge extent in high-dimensional scientific-text space plus t-SNE explanation.",
            "Direct methodological template: compute extent in the original embedding space and treat t-SNE as interpretive visualization.",
            "https://www.nature.com/articles/s41586-025-09922-y",
        ),
        (
            "Pillutla et al. (2021), NeurIPS, MAUVE",
            "Distributional gap between neural text and human text using embedding-space divergence frontiers.",
            "Useful framing for generated-human text gaps: quality/diversity cannot be reduced to one mean similarity.",
            "https://proceedings.neurips.cc/paper/2021/hash/260c2432a0eecc28ce03c10dadc078a4-Abstract.html",
        ),
        (
            "Le Bronnec et al. (2024), ACL",
            "Precision and recall for assessing LLM quality and diversity.",
            "Closest conceptual support for directional coverage: precision asks whether AI falls in the human domain; recall asks how much human territory AI covers.",
            "https://aclanthology.org/2024.acl-long.616/",
        ),
        (
            "Guo et al. (2025), TACL",
            "Benchmarking linguistic diversity of LLMs across lexical, syntactic and semantic dimensions.",
            "Supports semantic diversity and dispersion analysis rather than relying only on t-SNE clusters.",
            "https://aclanthology.org/2025.tacl-1.69/",
        ),
        (
            "Boggust et al. (2022), ACM IUI",
            "Embedding Comparator visualizes global structure and local neighborhoods.",
            "Supports the report design that pairs global t-SNE/UMAP views with local nearest-neighbor evidence.",
            "https://dl.acm.org/doi/10.1145/3490099.3511122",
        ),
        (
            "Zhong et al. (2024), ACM ICMI",
            "Multimodal language models and human perception alignment in tactile descriptions.",
            "Relevant precedent for comparing MLLM descriptions with human perception using semantic similarity and t-SNE.",
            "https://dl.acm.org/doi/10.1145/3678957.3685756",
        ),
        (
            "Woloszyn and Gagl (2025), arXiv",
            "Whether LLMs describe pictures like children.",
            "Direct child-language precedent: LLMs may approximate some lexical properties while missing child-specific semantic patterns.",
            "https://arxiv.org/abs/2508.13769",
        ),
        (
            "Wedyan et al. (2025), PLOS ONE",
            "GPT-4o and human urban walkability perception.",
            "Urban-perception precedent for reporting AI-human alignment on some dimensions but divergence in thematic diversity and lived context.",
            "https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0322078",
        ),
        (
            "Malekzadeh et al. (2025), Computers, Environment and Urban Systems",
            "Urban attractiveness according to ChatGPT versus human insights.",
            "Useful for positioning AI-human differences in street-view based urban perception and contextual nuance.",
            "https://doi.org/10.1016/j.compenvurbsys.2024.102243",
        ),
    ]
    related_rows = "\n".join(
        f'<tr><td><a href="{url}">{title}</a></td><td>{topic}</td><td>{use}</td></tr>'
        for title, topic, use, url in related_work
    )

    audit = v2["audit"]
    primary = v2["primary_comparison"]
    child_env = v2["metrics"]["child_env_scene"]
    ai_combined = v2["metrics"]["ai_combined"]
    cov_nn = coverage["nearest_neighbor_summary"]
    pr_primary = coverage["precision_recall_frontier"]["primary_tau_065"]
    pr_best = coverage["precision_recall_frontier"]["best_f1_like"]
    local_nn = coverage["local_neighborhood"]
    tau_065 = next(row for row in coverage["coverage_by_threshold"] if abs(row["threshold"] - 0.65) < 1e-9)
    tau_626 = next(row for row in coverage["coverage_by_threshold"] if abs(row["threshold"] - 0.626) < 1e-9)
    affect_main = affect["main_model"]
    affect_robust = affect["robustness_model"]
    main_compare = affect_main["comparisons"]["ai_combined"]
    robust_compare = affect_robust["comparisons"]["ai_combined"]
    child_concern = np.mean(
        [
            affect_main["groups"]["children_direct"]["mean_common_profile"]["concern_distress"],
            affect_robust["groups"]["children_direct"]["mean_common_profile"]["concern_distress"],
        ]
    )
    ai_concern = np.mean(
        [
            affect_main["groups"]["ai_combined"]["mean_common_profile"]["concern_distress"],
            affect_robust["groups"]["ai_combined"]["mean_common_profile"]["concern_distress"],
        ]
    )
    child_neutral = np.mean(
        [
            affect_main["groups"]["children_direct"]["mean_common_profile"]["neutral"],
            affect_robust["groups"]["children_direct"]["mean_common_profile"]["neutral"],
        ]
    )
    ai_neutral = np.mean(
        [
            affect_main["groups"]["ai_combined"]["mean_common_profile"]["neutral"],
            affect_robust["groups"]["ai_combined"]["mean_common_profile"]["neutral"],
        ]
    )
    agreement = affect["cross_model_directional_agreement"]["ai_combined"]
    alignment = affect["semantic_affective_alignment"]

    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Children-MLLM Semantic Territory Report</title>
  <style>
    :root {{
      --bg: #f5f7fb;
      --card: #ffffff;
      --ink: #15202b;
      --muted: #5f6b7a;
      --line: #dce3ea;
      --blue: #4C78D8;
      --red: #F25A5A;
      --green: #2a9d8f;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: var(--bg); color: var(--ink); font-family: "Noto Sans SC", "Microsoft YaHei", Arial, sans-serif; }}
    main {{ max-width: 1240px; margin: 0 auto; padding: 28px 22px 48px; }}
    h1 {{ margin: 0 0 8px; font-size: 31px; letter-spacing: -0.01em; }}
    h2 {{ margin: 0 0 12px; font-size: 21px; }}
    h3 {{ margin: 18px 0 8px; font-size: 16px; }}
    p, li {{ line-height: 1.72; }}
    .lead {{ color: var(--muted); font-size: 16px; margin-bottom: 20px; }}
    .card {{ background: var(--card); border: 1px solid var(--line); border-radius: 8px; padding: 18px; margin: 16px 0; box-shadow: 0 2px 6px rgba(15, 23, 42, 0.04); }}
    .grid {{ display: grid; gap: 14px; }}
    .grid.metrics {{ grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); }}
    .grid.two {{ grid-template-columns: repeat(auto-fit, minmax(430px, 1fr)); align-items: start; }}
    .metric {{ background: #eef4f9; border-radius: 7px; padding: 14px; }}
    .metric .label {{ color: var(--muted); font-size: 13px; }}
    .metric .value {{ font-size: 27px; font-weight: 700; margin-top: 4px; }}
    img {{ width: 100%; height: auto; display: block; border: 1px solid #e5eaf0; border-radius: 6px; background: white; }}
    .caption {{ color: var(--muted); font-size: 14px; margin: 10px 0 0; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 14px; }}
    th, td {{ border-bottom: 1px solid #e8edf2; padding: 10px; vertical-align: top; text-align: left; }}
    th {{ background: #eef4f9; }}
    .pill {{ display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 12px; font-weight: 600; }}
    .blue {{ background: rgba(76,120,216,.14); color: #274a9b; }}
    .red {{ background: rgba(242,90,90,.14); color: #a83232; }}
    .green {{ background: rgba(42,157,143,.14); color: #176f64; }}
    .note {{ border-left: 4px solid var(--green); padding-left: 12px; }}
    code {{ background: #eef2f7; padding: 1px 5px; border-radius: 4px; }}
  </style>
</head>
<body>
<main>
  <header>
    <h1>Children-MLLM Semantic Territory Report</h1>
    <p class="lead">A Hao et al. inspired pipeline for comparing children’s real environmental perception with MLLM-generated child-perspective evaluations. Embedding model: Qwen3-Embedding-8B.</p>
  </header>

  <section class="card">
    <h2>1. Core Interpretation</h2>
    <p class="note">主发现来自 Qwen3-Embedding-8B 的 4096 维语义空间，而不是来自 t-SNE 的二维肉眼判断。t-SNE 只负责把点投影出来帮助读者理解；“覆盖”“广度”“最近邻相似度”均在原始高维空间中计算。</p>
    <ul>
      <li><span class="pill green">Directional coverage</span> AI outputs are easier to retrieve from the child corpus than child utterances are from the AI corpus: mean nearest-neighbor similarity is {cov_nn['ai_covered_by_children']['mean']:.3f} for AI-to-child retrieval versus {cov_nn['children_covered_by_ai']['mean']:.3f} for child-to-AI retrieval.</li>
      <li>At tau = 0.65, {tau_065['ai_covered_by_children']:.1%} of AI outputs are covered by at least one child utterance, but only {tau_065['children_covered_by_ai']:.1%} of child utterances are covered by AI. This supports the cautious wording: <strong>AI is more easily matched into children's semantic territory than the reverse</strong>.</li>
      <li><span class="pill blue">Children</span> still occupy the broader semantic extent around the shared centroid: KE {hao['children_ke_joint_centroid']:.3f} versus AI KE {hao['ai_ke_joint_centroid']:.3f}. Substantively, child language expands toward situated use, danger, mobility, school/commercial street contexts and everyday constraints.</li>
      <li><span class="pill red">AI</span> expands toward standardized planning vocabulary: smart facilities, creativity, fresh air, exploration, interaction devices, greenery and generic child-friendly design templates.</li>
      <li><strong>Exploratory affective module:</strong> BERT-style emotion classifiers are kept as supporting exploration only. They suggest that AI is more neutral and less concern/inquiry-oriented, but they should not be treated as the main evidence until manually validated.</li>
    </ul>
  </section>

  <section class="card">
    <h2>2. Key Data and High-dimensional Metrics</h2>
    <div class="grid metrics">
      <div class="metric"><div class="label">Children raw segments</div><div class="value">{audit['children_segments']}</div></div>
      <div class="metric"><div class="label">Children direct perception</div><div class="value">{audit['children_direct_segments']}</div></div>
      <div class="metric"><div class="label">Children meta-evaluation</div><div class="value">{audit['children_meta_segments']}</div></div>
      <div class="metric"><div class="label">AI image units</div><div class="value">{audit['ai_image_units']}</div></div>
      <div class="metric"><div class="label">Hao-style Children KE</div><div class="value">{metric(hao['children_ke_joint_centroid'])}</div></div>
      <div class="metric"><div class="label">Hao-style AI KE</div><div class="value">{metric(hao['ai_ke_joint_centroid'])}</div></div>
      <div class="metric"><div class="label">Centroid distance</div><div class="value">{metric(hao['centroid_distance'])}</div></div>
      <div class="metric"><div class="label">Primary permutation P</div><div class="value">{metric(primary['permutation_p_value'], 4)}</div></div>
      <div class="metric"><div class="label">Child-to-AI NN mean</div><div class="value">{cov_nn['children_covered_by_ai']['mean']:.3f}</div></div>
      <div class="metric"><div class="label">AI-to-child NN mean</div><div class="value">{cov_nn['ai_covered_by_children']['mean']:.3f}</div></div>
      <div class="metric"><div class="label">Child coverage at tau=.65</div><div class="value">{tau_065['children_covered_by_ai']:.1%}</div></div>
      <div class="metric"><div class="label">AI coverage at tau=.65</div><div class="value">{tau_065['ai_covered_by_children']:.1%}</div></div>
      <div class="metric"><div class="label">Children concern/distress</div><div class="value">{metric(child_concern)}</div></div>
      <div class="metric"><div class="label">AI concern/distress</div><div class="value">{metric(ai_concern)}</div></div>
      <div class="metric"><div class="label">Children neutral expression</div><div class="value">{metric(child_neutral)}</div></div>
      <div class="metric"><div class="label">AI neutral expression</div><div class="value">{metric(ai_neutral)}</div></div>
    </div>
    <table>
      <thead><tr><th>Metric view</th><th>Children</th><th>AI</th><th>Interpretation</th></tr></thead>
      <tbody>
        <tr><td>Fixed observed text view</td><td>KE {hao['children_ke_joint_centroid']:.3f}; n={hao['children_n']}</td><td>KE {hao['ai_ke_joint_centroid']:.3f}; n={hao['ai_n']}</td><td>All 331 direct child utterances and all 220 AI image-level outputs are retained; no balanced subsampling is used for the main comparison.</td></tr>
        <tr><td>Conservative scene-level audit</td><td>KE {child_env['ke']:.3f}; p95 {child_env['p95_radius']:.3f}</td><td>KE {ai_combined['ke']:.3f}; p95 {ai_combined['p95_radius']:.3f}</td><td>Scene aggregation reduces child-side variance; use as a sensitivity check rather than the only result.</td></tr>
        <tr><td>Nearest-neighbor overlap</td><td>child-to-AI mean {cov_nn['children_covered_by_ai']['mean']:.3f}; median {cov_nn['children_covered_by_ai']['median']:.3f}</td><td>AI-to-child mean {cov_nn['ai_covered_by_children']['mean']:.3f}; median {cov_nn['ai_covered_by_children']['median']:.3f}</td><td>AI outputs often find a close child-side analogue, but many child utterances remain hard to match.</td></tr>
        <tr><td>Coverage at tau = 0.65</td><td>{tau_065['children_covered_by_ai']:.1%} covered by AI</td><td>{tau_065['ai_covered_by_children']:.1%} covered by children</td><td>This is the clearest numerical basis for the directional coverage statement.</td></tr>
        <tr><td>Coverage at median threshold tau = 0.626</td><td>{tau_626['children_covered_by_ai']:.1%} covered by AI</td><td>{tau_626['ai_covered_by_children']:.1%} covered by children</td><td>The direction remains the same under the median-derived threshold used by the semantic-affective map.</td></tr>
      </tbody>
    </table>
  </section>

  <section class="card">
    <h2>3. What Was Learned from Hao et al. (2026)</h2>
    <p>Hao et al. embed papers into a high-dimensional scientific text space, compute knowledge extent in that original space, and use t-SNE only to visualize the relationship. The same logic is adapted here for children and AI texts.</p>
    <table>
      <thead><tr><th>Transferable element</th><th>Adaptation in this pipeline</th></tr></thead>
      <tbody>{method_rows}</tbody>
    </table>
  </section>

  {figure_card(
      "4. High-dimensional Directional Coverage",
      figs["coverage"],
      f"This is the main evidence for the coverage claim. Nearest-neighbor retrieval is computed in the original 4096D Qwen3 space. AI-to-child mean similarity exceeds child-to-AI by {cov_nn['ai_minus_children_mean_similarity']:.3f} (95% CI {cov_nn['ai_minus_children_mean_similarity_ci95'][0]:.3f} to {cov_nn['ai_minus_children_mean_similarity_ci95'][1]:.3f}). At tau = 0.65, AI coverage is {tau_065['ai_covered_by_children']:.1%}, while child coverage is {tau_065['children_covered_by_ai']:.1%}."
  )}

  {figure_card(
      "5. Precision/Recall-style Semantic Frontier",
      figs["frontier"],
      f"Inspired by precision/recall and neural-human text distribution work, AI semantic precision is defined as AI outputs covered by children, while child semantic recall is defined as child utterances covered by AI. At tau = 0.65, precision is {pr_primary['ai_semantic_precision']:.1%} and recall is {pr_primary['child_semantic_recall']:.1%}; the max F1-like balance in the scanned range occurs at tau = {pr_best['threshold']:.2f}, a permissive threshold kept only as a sensitivity reference. Local-neighborhood diagnostics show hubness: {local_nn['children_as_receivers_for_ai']['zero_receiver_share']:.1%} of child texts receive no AI nearest-neighbor assignments, while the top 10% receive {local_nn['children_as_receivers_for_ai']['top10_assignment_share']:.1%} of AI assignments."
  )}

  {figure_card(
      "6. t-SNE Annotated by 4096D Coverage Status",
      figs["coverage_tsne"],
      "This plot keeps the familiar t-SNE layout, but the color labels are computed before projection. It shows why t-SNE can be useful for inspection while still being weaker than high-dimensional coverage curves for inference."
  )}

  {figure_card(
      "7. Hao-style Semantic Territory Map",
      figs["hao"],
      "This is the Hao et al. Fig. 3b-style explanatory figure. Circles and KE values summarize high-dimensional semantic extent; t-SNE is used only to place points and labels in a visually interpretable shared space."
  )}

  <div class="grid two">
    {figure_card(
        "8. Traditional t-SNE View",
        figs["traditional"],
        "This preserves the conventional t-SNE plot for comparison. It is useful for seeing coarse separation, but it does not explain what the separation means."
    )}
    {figure_card(
        "9. Diagnostic t-SNE: Text Type Effect",
        figs["diagnostic"],
        "This diagnostic plot shows that AI reason and suggestion form different textual clusters. This is why reason and suggestion should be treated as separate or explicitly combined image-level units."
    )}
  </div>

  <div class="grid two">
    {figure_card(
        "10. High-dimensional Semantic Extent",
        figs["extent"],
        "This legacy sensitivity view shows how KE, p95 radius and mean radius change across children, AI combined, AI reason and AI suggestion. The main coverage analysis above keeps the observed 331/220 samples fixed."
    )}
    {figure_card(
        "11. Exploratory Semantic Entropy",
        figs["entropy"],
        f"Following Hao's entropy idea, this exploratory check estimates concentration in a PCA-10D grid. Median entropy: children {entropy_summary['Children direct']['median']:.3f}, AI {entropy_summary['AI combined']['median']:.3f}."
    )}
  </div>

  <div class="grid two">
    {figure_card(
        "12. Centroid Distance Matrix",
        figs["heatmap"],
        "This matrix separates children’s direct environment perception, children’s meta-evaluation of AI/expert text, and AI reason/suggestion/combined outputs."
    )}
    {figure_card(
        "13. Distinctive Semantic Anchors",
        figs["keywords"],
        "Domain-term contrast helps interpret what the embedding-space difference means substantively."
    )}
  </div>

  {figure_card(
      "14. Pairwise Similarity Distribution",
      figs["violin"],
      "Within-group and between-group cosine similarities provide a complementary semantic-gap check. Treat it as supportive evidence, not as the only finding."
  )}

  <section class="card">
    <h2>15. Exploratory Semantic-affective Module</h2>
    <p class="note">This module is exploratory and should not be treated as the main finding. It asks a different question from Qwen3 semantic coverage: even when AI and children mention similar spatial features, do they express those features with similar concern, engagement, inquiry and neutrality?</p>
    <table>
      <thead><tr><th>Evidence</th><th>Result</th><th>Interpretation</th></tr></thead>
      <tbody>
        <tr><td>Chinese-specific emotion model</td><td>centroid distance {main_compare['centroid_distance']:.3f}; permutation P {main_compare['permutation_p']:.4g}; JS divergence {main_compare['js_divergence']:.3f}</td><td>AI combined text is more neutral and contains less concern/distress and inquiry/surprise.</td></tr>
        <tr><td>Multilingual robustness model</td><td>centroid distance {robust_compare['centroid_distance']:.3f}; permutation P {robust_compare['permutation_p']:.4g}; JS divergence {robust_compare['js_divergence']:.3f}</td><td>The principal direction is replicated despite a different label system and multi-label objective.</td></tr>
        <tr><td>Cross-model direction</td><td>sign agreement {agreement['sign_agreement']:.0%}; delta cosine {agreement['delta_cosine_similarity']:.3f}</td><td>The two models agree strongly on the overall AI-versus-child affective shift.</td></tr>
        <tr><td>Alignment quadrants</td><td>AI semantic-only {alignment['quadrant_proportions']['ai_combined']['semantic_only']:.1%}; child distinctive-both {alignment['quadrant_proportions']['children_direct']['distinctive_both']:.1%}</td><td>These quadrants are useful for hypothesis generation, but the stronger coverage claim should be supported by Section 4's high-dimensional nearest-neighbor results. Median-defined quadrants are descriptive, not latent types.</td></tr>
        <tr><td>Face-validity diagnostic</td><td>Chinese model: concern 25%, inquiry 75%; multilingual model: concern 100%, inquiry 25%</td><td>Weaknesses are complementary. No fine-grained emotion label should be treated as ground truth without manual validation.</td></tr>
      </tbody>
    </table>
    <p class="caption">Interpretation boundary: these are probabilistic affective cues in written utterances. The analysis does not infer a child's internal emotion, mental health, or stable disposition.</p>
  </section>

  {figure_card(
      "16. Exploratory Affective Profiles by Text Type",
      figs["affect_profiles"],
      "Children's direct perception and meta-evaluation contain more varied affective cues. AI suggestion is almost entirely classified as neutral, consistent with standardized recommendation language."
  )}

  <div class="grid two">
    {figure_card(
        "17. Dual-model Robustness and Face Validity",
        figs["affect_robustness"],
        "The left panel compares the direction of the AI-minus-child difference in four common dimensions. The right panel exposes complementary model weaknesses using transparent Chinese anchor sentences."
    )}
    {figure_card(
        "18. Semantic-affective Alignment Map",
        figs["semantic_affective"],
        f"Semantic overlap and affective overlap are measured separately. Mean semantic overlap is {alignment['group_means']['children_direct']['semantic_overlap']:.3f} for child-to-AI retrieval and {alignment['group_means']['ai_combined']['semantic_overlap']:.3f} for AI-to-child retrieval. Treat this figure as an exploratory bridge between semantic and affective cues; the main directional-coverage evidence is Section 4."
    )}
  </div>

  <section class="card">
    <h2>19. Related Work to Cite and Borrow From</h2>
    <p class="note">These references are most useful for writing the method and visualization rationale. The strongest connection is not simply t-SNE, but the combination of high-dimensional distributional metrics, directional precision/recall or coverage, and low-dimensional explanatory visualization.</p>
    <table>
      <thead><tr><th>Reference</th><th>Relevant method or finding</th><th>How to borrow it here</th></tr></thead>
      <tbody>{related_rows}</tbody>
    </table>
  </section>

  <section class="card">
    <h2>20. Additional Analyses to Add Next</h2>
    <ol>
      <li><strong>Image-paired semantic error:</strong> create an <code>image_id -> children_comments -> AI_output</code> table, then compute per-image centroid distance and nearest-neighbor mismatch.</li>
      <li><strong>Dimension-level coverage:</strong> manually or semi-automatically code safety, practicality, accessibility, play, greenery, digital facilities, social supervision and everyday constraints, then compare coverage rates.</li>
      <li><strong>Sub-domain robustness:</strong> follow Hao's field/subfield logic by grouping images into school, commercial street, park, street, waterfront, transit and residual public spaces.</li>
      <li><strong>Template concentration:</strong> measure repeated AI motifs such as smart facilities, interactive installations, greenery, fresh air and creativity, then compare with children’s situated motifs.</li>
      <li><strong>t-SNE robustness:</strong> keep one polished Hao-style map, but run multiple seeds/batches in appendix to show the conclusion is not a single projection artifact.</li>
      <li><strong>Human affect validation:</strong> double-code a stratified sample for concern/distress, positive engagement, inquiry/surprise and neutral description; report inter-rater reliability and model precision/recall before making label-level claims.</li>
    </ol>
  </section>
</main>
</body>
</html>
"""

    out = BASE_DIR / "Children_MLLM_Semantic_Territory_Full_Report.html"
    out.write_text(html, encoding="utf-8")

    metrics_out = {
        "entropy_summary": entropy_summary,
        "source_metrics": {
            "analysis_results_v2": "data/analysis_results_v2.json",
            "hao_style_metrics": "data/hao_style_metrics.json",
            "legacy_analysis_results": "data/analysis_results.json",
            "affective_alignment_metrics": "data/affective_alignment_metrics.json",
            "semantic_coverage_metrics": "data/semantic_coverage_metrics.json",
        },
    }
    (DATA_DIR / "full_report_metrics.json").write_text(
        json.dumps(metrics_out, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return out


def main() -> None:
    out = build_html()
    print(f"Saved full semantic territory report: {out}")


if __name__ == "__main__":
    main()
