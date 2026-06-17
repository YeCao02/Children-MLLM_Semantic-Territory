from __future__ import annotations

import importlib.util
import json
import math
import base64
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
FIG_DIR = DATA_DIR / "figures_v2"
RANDOM_SEED = 42


def load_method_module():
    path = Path(__file__).with_name("05_method_audit_v2.py")
    spec = importlib.util.spec_from_file_location("method_audit_v2", path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    spec.loader.exec_module(module)
    return module


def scaled_circle_radii(coords: np.ndarray, embeddings: np.ndarray, child_n: int) -> dict:
    center_2d = coords.mean(axis=0)
    center_hd = embeddings.mean(axis=0)

    child_hd_dist = np.linalg.norm(embeddings[:child_n] - center_hd, axis=1)
    ai_hd_dist = np.linalg.norm(embeddings[child_n:] - center_hd, axis=1)
    all_hd_dist = np.linalg.norm(embeddings - center_hd, axis=1)
    all_2d_dist = np.linalg.norm(coords - center_2d, axis=1)

    scale = np.percentile(all_2d_dist, 95) / np.percentile(all_hd_dist, 95)
    return {
        "center_2d": center_2d,
        "child_ke": float(child_hd_dist.max()),
        "ai_ke": float(ai_hd_dist.max()),
        "child_p95": float(np.percentile(child_hd_dist, 95)),
        "ai_p95": float(np.percentile(ai_hd_dist, 95)),
        "child_radius_2d": float(child_hd_dist.max() * scale),
        "ai_radius_2d": float(ai_hd_dist.max() * scale),
        "scale": float(scale),
    }


def choose_anchor(texts: list[str], coords: np.ndarray, center: np.ndarray, term: str) -> int | None:
    candidates = [i for i, text in enumerate(texts) if term in text]
    if not candidates:
        return None
    distances = np.linalg.norm(coords[candidates] - center, axis=1)
    return candidates[int(np.argmax(distances))]


def draw_extent_arrow(ax, center: np.ndarray, radius: float, angle_deg: float, label: str) -> None:
    angle = math.radians(angle_deg)
    end = center + np.array([math.cos(angle), math.sin(angle)]) * radius
    ax.annotate(
        "",
        xy=end,
        xytext=center,
        arrowprops=dict(arrowstyle="->", lw=1.9, color="black", shrinkA=0, shrinkB=0),
        zorder=6,
    )
    label_pos = center + np.array([math.cos(angle), math.sin(angle)]) * radius * 0.58
    ax.text(
        label_pos[0],
        label_pos[1],
        label,
        fontsize=10,
        ha="left",
        va="bottom",
        color="black",
        bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none", alpha=0.72),
    )


def plot_hao_style() -> tuple[Path, dict]:
    method = load_method_module()
    method.configure_plotting()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    emb_data = method.load_embedding_data()
    child_records = method.extract_children_segments_with_metadata()
    child_texts = [str(x) for x in emb_data["children"]["texts"]]
    child_embs = emb_data["children"]["embeddings"]
    if len(child_records) != len(child_texts) or any(
        r["text"] != t for r, t in zip(child_records, child_texts)
    ):
        child_records = [{"text": text} for text in child_texts]

    child_direct_mask = np.array([method.is_direct_environment_text(r["text"]) for r in child_records])
    child_direct_embs = child_embs[child_direct_mask]
    child_direct_texts = [r["text"] for r, keep in zip(child_records, child_direct_mask) if keep]

    ai_meta = method.reconstruct_ai_metadata(emb_data["ai"]["texts"], emb_data["ai"]["embeddings"])
    ai_combined_embs = ai_meta["combined"]["embeddings"]
    ai_combined_texts = ai_meta["combined"]["texts"]

    # Keep all AI image-level units and a deterministic, size-balanced sample of child direct utterances.
    rng = np.random.default_rng(RANDOM_SEED)
    sample_n = min(len(child_direct_embs), len(ai_combined_embs))
    child_idx = rng.choice(len(child_direct_embs), size=sample_n, replace=False)
    child_plot_embs = child_direct_embs[child_idx]
    child_plot_texts = [child_direct_texts[i] for i in child_idx]

    plot_embs = np.vstack([child_plot_embs, ai_combined_embs])
    coords = TSNE(
        n_components=2,
        perplexity=30,
        random_state=RANDOM_SEED,
        init="pca",
        learning_rate="auto",
    ).fit_transform(plot_embs)
    child_coords = coords[:sample_n]
    ai_coords = coords[sample_n:]

    radii = scaled_circle_radii(coords, plot_embs, sample_n)
    center = radii["center_2d"]

    blue = "#4C78D8"
    red = "#F25A5A"
    fig, ax = plt.subplots(figsize=(9.2, 8.2))

    ax.scatter(
        child_coords[:, 0],
        child_coords[:, 1],
        s=30,
        facecolors="none",
        edgecolors=blue,
        alpha=0.38,
        linewidths=0.95,
        label="Children real data",
    )
    ax.scatter(
        ai_coords[:, 0],
        ai_coords[:, 1],
        s=30,
        facecolors="none",
        edgecolors=red,
        alpha=0.38,
        linewidths=0.95,
        label="AI output",
    )

    child_circle = Circle(center, radii["child_radius_2d"], fill=False, lw=2.4, color=blue, alpha=0.95)
    ai_circle = Circle(center, radii["ai_radius_2d"], fill=False, lw=2.4, color=red, alpha=0.95)
    ax.add_patch(child_circle)
    ax.add_patch(ai_circle)

    ax.scatter([center[0]], [center[1]], s=410, color="black", zorder=8)
    ax.scatter([center[0]], [center[1]], s=190, color="#FFD400", zorder=9)
    ax.text(
        center[0],
        center[1] + 2.8,
        "Shared centroid\n(c)",
        ha="center",
        va="bottom",
        fontsize=11,
        color="black",
    )

    larger_group = "Children" if radii["child_radius_2d"] >= radii["ai_radius_2d"] else "AI"
    draw_extent_arrow(
        ax,
        center,
        radii["child_radius_2d"],
        2,
        "Children semantic extent\n(KE)",
    )
    draw_extent_arrow(
        ax,
        center,
        radii["ai_radius_2d"],
        35,
        "AI semantic extent\n(KE)",
    )

    anchors = [
        ("child", "商业街", "Commercial street"),
        ("child", "学校", "School context"),
        ("child", "危险", "Situated danger"),
        ("child", "自行车", "Bicycle / mobility"),
        ("child", "实用", "Practical use"),
        ("ai", "智能", "Smart facilities"),
        ("ai", "创意", "Creativity"),
        ("ai", "空气清新", "Fresh air"),
        ("ai", "互动", "Interaction"),
        ("ai", "探索", "Exploration"),
    ]
    used: set[tuple[str, int]] = set()
    for group, term, label in anchors:
        if group == "child":
            idx = choose_anchor(child_plot_texts, child_coords, center, term)
            if idx is None or ("child", idx) in used:
                continue
            used.add(("child", idx))
            xy = child_coords[idx]
            color = blue
            marker = "o"
        else:
            idx = choose_anchor(ai_combined_texts, ai_coords, center, term)
            if idx is None or ("ai", idx) in used:
                continue
            used.add(("ai", idx))
            xy = ai_coords[idx]
            color = red
            marker = "o"
        ax.scatter([xy[0]], [xy[1]], s=115, color=color, edgecolor="white", linewidth=1.2, marker=marker, zorder=7)
        ax.annotate(
            label,
            xy=xy,
            xytext=(xy[0] + 1.5, xy[1] + 1.5),
            fontsize=10,
            color="black",
            arrowprops=dict(arrowstyle="-", lw=0.8, color="#555555", alpha=0.65),
            bbox=dict(boxstyle="round,pad=0.16", facecolor="white", edgecolor="none", alpha=0.72),
            zorder=8,
        )

    centroid_distance = float(np.linalg.norm(child_plot_embs.mean(axis=0) - ai_combined_embs.mean(axis=0)))
    nn_sim = cosine_similarity(child_plot_embs, ai_combined_embs)
    child_to_ai_nn = float(np.median(nn_sim.max(axis=1)))
    ai_to_child_nn = float(np.median(nn_sim.max(axis=0)))

    ax.text(
        0.02,
        0.02,
        (
            f"4096D metrics | Children KE={radii['child_ke']:.3f}, AI KE={radii['ai_ke']:.3f}; "
            f"centroid distance={centroid_distance:.3f}; median nearest-neighbor sim={child_to_ai_nn:.3f}/{ai_to_child_nn:.3f}"
        ),
        transform=ax.transAxes,
        fontsize=9,
        color="#333333",
        bbox=dict(facecolor="white", edgecolor="#cccccc", alpha=0.86),
    )

    ax.legend(loc="upper right", frameon=False, fontsize=11)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("AI vs Children Semantic Territory (Hao-style Visualization)", fontsize=15, pad=14)
    ax.text(0.0, 1.02, "b", transform=ax.transAxes, fontsize=24, fontweight="bold", ha="left", va="bottom")
    fig.tight_layout()

    out = FIG_DIR / "Fig_v2_hao_style_ai_children.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)

    metrics = {
        "visualization_unit": "balanced child direct environment utterances vs AI image-level combined outputs",
        "children_n": int(sample_n),
        "ai_n": int(len(ai_combined_embs)),
        "children_ke_joint_centroid": radii["child_ke"],
        "ai_ke_joint_centroid": radii["ai_ke"],
        "children_p95_joint_centroid": radii["child_p95"],
        "ai_p95_joint_centroid": radii["ai_p95"],
        "larger_extent_group": larger_group,
        "centroid_distance": centroid_distance,
        "child_to_ai_nn_median": child_to_ai_nn,
        "ai_to_child_nn_median": ai_to_child_nn,
        "figure": str(out),
    }
    return out, metrics


def write_method_note(fig_path: Path, metrics: dict) -> Path:
    out = BASE_DIR / "docs" / "hao_style_visual_method_note.md"
    rel_fig = fig_path.relative_to(BASE_DIR)
    text = f"""# Hao-style AI-Children 语义领地图：方法说明

## 是否可以用“异同并存”的方式分析 AI-children？

可以，而且这比只做“semantic gap”更适合第二篇论文。建议把问题拆成三层：

1. **共同语义核心**：AI 与儿童都谈到的维度，例如安全、互动、设施、绿化、可达性等。可用最近邻相似度、关键词共同出现、局部重叠区来描述。
2. **儿童特异语义领地**：儿童更强调具体生活场景、实用性、危险来源、店铺/学校/商业街/自行车等日常经验。
3. **AI 特异语义领地**：AI 更强调标准化规划话语，例如智能设施、创意空间、空气清新、探索、互动装置等。

这样论文叙事可以从“AI 是否像儿童”推进到“AI 与儿童共享了哪些抽象评价维度，又在哪些具身经验上系统错位”。

## 新图的读法

输出图：`{rel_fig}`

这张图效仿 Hao et al. 2026 Fig. 3b：

- 蓝色空心点：儿童真实语料中的直接环境感知片段。
- 红色空心点：AI 对每张图片的 reason + suggestion 合并文本。
- 黄色中心点：儿童与 AI 共同语义空间的 shared centroid。
- 蓝/红圆：按 4096 维 Qwen3 embedding 空间中的 semantic extent 映射到二维图上，用于直观表达两个语义领地的覆盖范围。
- 黑色箭头：从 shared centroid 指向对应语义边界，类比 Hao 图中的 KE 箭头。
- 标注点：两侧更突出的语义锚点，用来解释“差异在哪里”。

## 需要在正文里明确的边界

这张图是说明性可视化，不是正式统计推断。正式结论仍应基于 4096 维归一化嵌入空间：

- Children KE around shared centroid: {metrics['children_ke_joint_centroid']:.4f}
- AI KE around shared centroid: {metrics['ai_ke_joint_centroid']:.4f}
- Centroid distance: {metrics['centroid_distance']:.4f}
- Median nearest-neighbor similarity, child-to-AI / AI-to-child: {metrics['child_to_ai_nn_median']:.4f} / {metrics['ai_to_child_nn_median']:.4f}

## 推荐论文写法

> To visualize shared and distinctive semantic territories, we projected child interview utterances and AI-generated image evaluations into a joint two-dimensional t-SNE space. Following the visual logic of Hao et al. (2026), we overlaid semantic-extent boundaries centered on the shared centroid. The circles are visual summaries of high-dimensional semantic extent computed from Qwen3-Embedding-8B representations, while the t-SNE projection is used only for visual interpretation. This design allows us to distinguish common evaluative ground from group-specific semantic territories.

## 为什么不能只用传统 t-SNE

传统 t-SNE 能展示聚类，但很难回答“异同在哪里”。Hao-style 图增加了三个信息层：

1. 共同中心：表示 AI 与儿童被放在同一个语义空间里比较。
2. 语义边界：表示哪一组覆盖的语义领地更大。
3. 代表性锚点：说明边界扩展到哪些具体主题。

因此，传统 t-SNE 可以保留为诊断图；Hao-style 图更适合作为论文中的核心解释图。
"""
    out.write_text(text, encoding="utf-8")
    return out


def write_html(fig_path: Path, metrics: dict) -> Path:
    encoded = base64.b64encode(fig_path.read_bytes()).decode("utf-8")
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Hao-style AI-Children Semantic Territory</title>
  <style>
    body {{ margin: 0; background: #f7f9fc; color: #17202a; font-family: "Noto Sans SC", "Microsoft YaHei", Arial, sans-serif; }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 28px 22px 46px; }}
    h1 {{ margin: 0 0 8px; font-size: 28px; }}
    h2 {{ margin-top: 26px; font-size: 20px; }}
    p, li {{ line-height: 1.75; }}
    .card {{ background: white; border: 1px solid #dce3ea; border-radius: 8px; padding: 18px; margin: 16px 0; }}
    img {{ width: 100%; height: auto; display: block; border: 1px solid #e5eaf0; border-radius: 6px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 12px; margin-top: 14px; }}
    .metric {{ background: #eef4f9; border-radius: 6px; padding: 12px; }}
    .label {{ color: #5f6b7a; font-size: 13px; }}
    .value {{ font-size: 24px; font-weight: 700; margin-top: 4px; }}
    code {{ background: #eef2f7; padding: 1px 5px; border-radius: 4px; }}
  </style>
</head>
<body>
<main>
  <h1>Hao-style AI-Children Semantic Territory</h1>
  <p>这页用于论文方法讨论：保留传统 t-SNE 的直观性，同时增加 shared centroid、semantic extent 和代表性语义锚点，用“异同并存”的方式解释 AI 与儿童真实数据。</p>

  <div class="card">
    <img src="data:image/png;base64,{encoded}" alt="Hao-style AI-Children semantic territory">
  </div>

  <div class="grid">
    <div class="metric"><div class="label">Children KE</div><div class="value">{metrics['children_ke_joint_centroid']:.3f}</div></div>
    <div class="metric"><div class="label">AI KE</div><div class="value">{metrics['ai_ke_joint_centroid']:.3f}</div></div>
    <div class="metric"><div class="label">Centroid Distance</div><div class="value">{metrics['centroid_distance']:.3f}</div></div>
    <div class="metric"><div class="label">Median NN Similarity</div><div class="value">{metrics['child_to_ai_nn_median']:.3f} / {metrics['ai_to_child_nn_median']:.3f}</div></div>
  </div>

  <div class="card">
    <h2>方法解释</h2>
    <p>这张图把 AI 与儿童放入共同语义空间。蓝色圆表示儿童真实语料的 semantic extent，红色圆表示 AI 输出的 semantic extent。空心点是 t-SNE 投影，圆和指标则对应 4096 维 Qwen3-Embedding-8B 归一化向量空间中的高维计算。</p>
    <p>图中不只展示 gap，也展示 common ground：两组都围绕安全、设施、互动、空间吸引力等共同维度展开；差异在于儿童更向日常使用、具体危险、学校/商业街/自行车等具身经验扩展，AI 更向智能设施、创意、空气清新、探索等标准化规划话语扩展。</p>
  </div>
</main>
</body>
</html>
"""
    out = BASE_DIR / "AI_vs_Children_HaoStyle_Method.html"
    out.write_text(html, encoding="utf-8")
    return out


def main() -> None:
    fig_path, metrics = plot_hao_style()
    metrics_path = DATA_DIR / "hao_style_metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    note_path = write_method_note(fig_path, metrics)
    html_path = write_html(fig_path, metrics)
    print(f"Saved Hao-style figure: {fig_path}")
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved method note: {note_path}")
    print(f"Saved HTML: {html_path}")


if __name__ == "__main__":
    main()
