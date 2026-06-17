from __future__ import annotations

import base64
import json
import math
import os
import pickle
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Ellipse
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
TXT_DIR = BASE_DIR / "txt_cleaned"
FIG_DIR = DATA_DIR / "figures_v2"

AI_MODEL_KEY = "32B_Det"
AI_PROMPT = "P1_Child"
AI_LANG = "zh"
RANDOM_SEED = 42
BOOTSTRAP_N = 1000


def l2_normalize(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


def configure_plotting() -> None:
    sns.set_theme(style="whitegrid")
    font_candidates = [
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("C:/Windows/Fonts/NotoSansSC-VF.ttf"),
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    ]
    font_list = ["DejaVu Sans"]
    for font_path in font_candidates:
        if font_path.exists():
            import matplotlib.font_manager as fm

            fm.fontManager.addfont(str(font_path))
            font_list.insert(0, fm.FontProperties(fname=str(font_path)).get_name())
            break

    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "font.family": "sans-serif",
            "font.sans-serif": font_list,
            "axes.unicode_minus": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelsize": 10,
            "axes.titlesize": 12,
            "legend.fontsize": 9,
        }
    )


def load_embedding_data() -> dict:
    with open(DATA_DIR / "embeddings.pkl", "rb") as f:
        emb_data = pickle.load(f)
    emb_data["children"]["embeddings"] = l2_normalize(emb_data["children"]["embeddings"])
    emb_data["ai"]["embeddings"] = l2_normalize(emb_data["ai"]["embeddings"])
    return emb_data


def is_header(line: str) -> bool:
    return "原文" in line or bool(re.match(r"^\d{4}年\d{2}月\d{2}日", line))


def get_speaker_type(speaker_name: str) -> str | None:
    cleaned = re.sub(r"[\s_]+", "", speaker_name).lower()
    if cleaned.startswith("student") or "student" in cleaned or cleaned.startswith("sudden"):
        return "student"
    researcher_aliases = {
        "researcher",
        "yuxingao",
        "明明",
        "刘明明",
        "老师",
        "host",
    }
    if cleaned in researcher_aliases or cleaned.startswith("发言人"):
        return "researcher"
    return None


NEXT_SCENE_RE = re.compile(
    r"下一张|下一个|往下|往后|新的|再看|看下|看一下这张|这张图片|这幅图|这个照片|我们看"
)


def is_next_scene_cue(text: str) -> bool:
    return bool(NEXT_SCENE_RE.search(text))


def extract_children_segments_with_metadata() -> list[dict]:
    """Replicate the current extractor closely, while attaching file/scene metadata."""
    records: list[dict] = []
    for filepath in sorted(TXT_DIR.glob("meeting_*.txt")):
        lines = filepath.read_text(encoding="utf-8").splitlines()
        is_meeting_01 = filepath.name == "meeting_01.txt"
        current_speaker_type: str | None = None
        current_speaker_name: str | None = None
        current_text_buffer: list[str] = []
        current_start_line: int | None = None
        scene_idx = 1

        def flush(end_line: int) -> None:
            nonlocal current_text_buffer, current_start_line
            if current_speaker_type == "student":
                joined = " ".join(current_text_buffer).strip()
                if joined:
                    records.append(
                        {
                            "text": joined,
                            "meeting": filepath.stem,
                            "scene": scene_idx,
                            "speaker": current_speaker_name or "",
                            "start_line": current_start_line,
                            "end_line": end_line,
                        }
                    )
            current_text_buffer = []
            current_start_line = None

        for line_no, raw_line in enumerate(lines, 1):
            line = raw_line.strip()
            if not line or is_header(line):
                continue

            if is_meeting_01:
                same_line_match = re.match(
                    r"^([^:：()（）\s]+(?:\s+[^:：()（）\s]+)*)(?:\s*[(（][^）)][)）])?\s*[:：]\s*(.*)",
                    line,
                )
                if same_line_match:
                    speaker = same_line_match.group(1).strip()
                    text = same_line_match.group(2).strip()
                    spk_type = get_speaker_type(speaker)
                    if spk_type:
                        flush(line_no - 1)
                        current_speaker_type = spk_type
                        current_speaker_name = speaker
                        current_text_buffer = [text] if spk_type == "student" else []
                        current_start_line = line_no if spk_type == "student" else None
                        if spk_type == "researcher" and is_next_scene_cue(text):
                            scene_idx += 1
                continue

            cleaned = re.sub(r"[\s_]+", "", line).lower()
            spk_type = get_speaker_type(cleaned)
            if spk_type:
                flush(line_no - 1)
                current_speaker_type = spk_type
                current_speaker_name = line
                current_text_buffer = []
                current_start_line = None
                continue

            if current_speaker_type == "student":
                if current_start_line is None:
                    current_start_line = line_no
                current_text_buffer.append(line)
            elif current_speaker_type == "researcher" and is_next_scene_cue(line):
                scene_idx += 1

        flush(len(lines))

    return records


META_PATTERNS = [
    "AI",
    "ai",
    "专家",
    "评价",
    "评论",
    "生成",
    "儿童视角",
    "专家视角",
    "打分",
    "分点",
    "标准答案",
    "书面",
    "官方",
]


def is_direct_environment_text(text: str) -> bool:
    if len(text.strip()) < 6:
        return False
    return not any(p in text for p in META_PATTERNS)


def aggregate_embeddings(
    embs: np.ndarray, records: list[dict], keep_mask: np.ndarray | None = None
) -> tuple[np.ndarray, list[dict]]:
    if keep_mask is None:
        keep_mask = np.ones(len(records), dtype=bool)

    groups: dict[tuple[str, int], list[int]] = defaultdict(list)
    for idx, record in enumerate(records):
        if keep_mask[idx]:
            groups[(record["meeting"], int(record["scene"]))].append(idx)

    out_embs: list[np.ndarray] = []
    out_meta: list[dict] = []
    for (meeting, scene), indices in sorted(groups.items()):
        if not indices:
            continue
        avg = embs[indices].mean(axis=0)
        avg = l2_normalize(avg)[0]
        out_embs.append(avg)
        out_meta.append(
            {
                "unit": f"{meeting}_scene_{scene:02d}",
                "meeting": meeting,
                "scene": scene,
                "segment_count": len(indices),
                "text": " ".join(records[i]["text"] for i in indices),
            }
        )
    if not out_embs:
        return np.zeros((0, embs.shape[1]), dtype=np.float32), []
    return np.vstack(out_embs).astype(np.float32), out_meta


def reconstruct_ai_metadata(ai_texts: list, ai_embs: np.ndarray) -> dict:
    model_texts_path = DATA_DIR / "model_texts.pkl"
    data = pd.read_pickle(model_texts_path)
    if AI_MODEL_KEY not in data:
        raise KeyError(f"{AI_MODEL_KEY} not found in {model_texts_path}")

    df = data[AI_MODEL_KEY].copy()
    df = df[(df["lang"] == AI_LANG) & (df["prompt"] == AI_PROMPT)].reset_index(drop=True)
    df["reason"] = df["reason"].fillna("").astype(str).str.strip()
    df["suggestion"] = df["suggestion"].fillna("").astype(str).str.strip()
    n = len(df)
    if len(ai_embs) != 2 * n:
        raise ValueError(f"Expected {2 * n} AI embeddings from {n} rows, found {len(ai_embs)}")

    validation_mismatches = 0
    for i, row in df.iterrows():
        reason_text = ai_texts[i]["text"] if isinstance(ai_texts[i], dict) else str(ai_texts[i])
        suggestion_text = (
            ai_texts[i + n]["text"] if isinstance(ai_texts[i + n], dict) else str(ai_texts[i + n])
        )
        if reason_text.strip() != row["reason"].strip():
            validation_mismatches += 1
        if suggestion_text.strip() != row["suggestion"].strip():
            validation_mismatches += 1

    reason_embs = ai_embs[:n]
    suggestion_embs = ai_embs[n : 2 * n]
    combined_embs = l2_normalize(reason_embs + suggestion_embs)
    combined_texts = [
        f"{row.reason} {row.suggestion}".strip() for row in df.itertuples(index=False)
    ]

    return {
        "df": df,
        "validation_mismatches": validation_mismatches,
        "reason": {
            "texts": df["reason"].tolist(),
            "embeddings": reason_embs,
        },
        "suggestion": {
            "texts": df["suggestion"].tolist(),
            "embeddings": suggestion_embs,
        },
        "combined": {
            "texts": combined_texts,
            "embeddings": combined_embs,
        },
    }


def compute_group_stats(embeddings: np.ndarray) -> dict:
    if len(embeddings) == 0:
        return {"n": 0, "ke": None, "mean_radius": None, "median_radius": None, "p95_radius": None}
    centroid = embeddings.mean(axis=0)
    distances = np.linalg.norm(embeddings - centroid, axis=1)
    return {
        "n": int(len(embeddings)),
        "centroid": centroid,
        "ke": float(np.max(distances)),
        "mean_radius": float(np.mean(distances)),
        "median_radius": float(np.median(distances)),
        "p95_radius": float(np.percentile(distances, 95)),
    }


def bootstrap_extent(
    embeddings: np.ndarray,
    sample_n: int,
    repeats: int = BOOTSTRAP_N,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    replace = sample_n > len(embeddings)
    for i in range(repeats):
        idx = rng.choice(len(embeddings), size=sample_n, replace=replace)
        stats = compute_group_stats(embeddings[idx])
        rows.append(
            {
                "iteration": i,
                "ke": stats["ke"],
                "mean_radius": stats["mean_radius"],
                "p95_radius": stats["p95_radius"],
            }
        )
    return pd.DataFrame(rows)


def percentile_ci(values: Iterable[float]) -> list[float]:
    arr = np.asarray(list(values), dtype=float)
    return [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))]


def centroid_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a.mean(axis=0) - b.mean(axis=0)))


def permutation_centroid_test(
    a: np.ndarray, b: np.ndarray, repeats: int = 1000, seed: int = RANDOM_SEED
) -> dict:
    rng = np.random.default_rng(seed)
    observed = centroid_distance(a, b)
    combined = np.vstack([a, b])
    n_a = len(a)
    permuted = np.empty(repeats, dtype=float)
    for i in range(repeats):
        idx = rng.permutation(len(combined))
        permuted[i] = centroid_distance(combined[idx[:n_a]], combined[idx[n_a:]])
    p_value = float((np.sum(permuted >= observed) + 1) / (repeats + 1))
    return {
        "observed": observed,
        "p_value": p_value,
        "null_ci": percentile_ci(permuted),
    }


def nearest_neighbor_summary(a: np.ndarray, b: np.ndarray) -> dict:
    sim = cosine_similarity(a, b)
    a_to_b = sim.max(axis=1)
    b_to_a = sim.max(axis=0)
    return {
        "a_to_b_mean": float(a_to_b.mean()),
        "a_to_b_median": float(np.median(a_to_b)),
        "b_to_a_mean": float(b_to_a.mean()),
        "b_to_a_median": float(np.median(b_to_a)),
    }


def add_cov_ellipse(ax, coords: np.ndarray, color: str, label: str | None = None) -> None:
    if len(coords) < 3:
        return
    cov = np.cov(coords.T)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    # 80% chi-square radius for 2 dimensions.
    chi2_80 = 3.2188758248682006
    width, height = 2 * np.sqrt(vals * chi2_80)
    ellipse = Ellipse(
        xy=coords.mean(axis=0),
        width=width,
        height=height,
        angle=angle,
        fill=False,
        lw=1.8,
        ls="--",
        color=color,
        alpha=0.85,
        label=label,
    )
    ax.add_patch(ellipse)


def plot_tsne(
    child_records: list[dict],
    child_embs: np.ndarray,
    child_direct_mask: np.ndarray,
    ai_reason: np.ndarray,
    ai_suggestion: np.ndarray,
) -> Path:
    labels = []
    plot_embs = []
    for idx, emb in enumerate(child_embs):
        labels.append("儿童直接环境感知" if child_direct_mask[idx] else "儿童对AI/专家的元评价")
        plot_embs.append(emb)
    for emb in ai_reason:
        labels.append("AI Reason")
        plot_embs.append(emb)
    for emb in ai_suggestion:
        labels.append("AI Suggestion")
        plot_embs.append(emb)

    plot_embs_arr = np.vstack(plot_embs)
    perplexity = max(5, min(30, (len(plot_embs_arr) - 1) // 3))
    coords = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=RANDOM_SEED,
        init="pca",
        learning_rate="auto",
    ).fit_transform(plot_embs_arr)

    df = pd.DataFrame(coords, columns=["x", "y"])
    df["group"] = labels

    palette = {
        "儿童直接环境感知": "#2c7fb8",
        "儿童对AI/专家的元评价": "#7fcdbb",
        "AI Reason": "#d7301f",
        "AI Suggestion": "#fdae61",
    }
    markers = {
        "儿童直接环境感知": "o",
        "儿童对AI/专家的元评价": "s",
        "AI Reason": "o",
        "AI Suggestion": "^",
    }

    fig, ax = plt.subplots(figsize=(9, 7.2))
    for group in palette:
        sub = df[df["group"] == group]
        ax.scatter(
            sub["x"],
            sub["y"],
            s=26,
            alpha=0.68,
            c=palette[group],
            marker=markers[group],
            label=f"{group} (n={len(sub)})",
            edgecolors="none",
        )
        add_cov_ellipse(ax, sub[["x", "y"]].to_numpy(), palette[group])

    ax.set_title("t-SNE 语义空间诊断图：体裁/任务类型正在主导分群")
    ax.set_xlabel("t-SNE dimension 1")
    ax.set_ylabel("t-SNE dimension 2")
    ax.legend(loc="best", frameon=True)
    ax.text(
        0.01,
        -0.13,
        "注：t-SNE 仅用于二维可视化；knowledge extent、质心距离等正式指标均在 4096 维归一化嵌入空间中计算。",
        transform=ax.transAxes,
        fontsize=9,
        color="#444444",
    )
    fig.tight_layout()
    out = FIG_DIR / "Fig_v2_tsne_diagnostic.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_highdim_extent(boot_tables: dict[str, pd.DataFrame]) -> Path:
    rows = []
    for group, table in boot_tables.items():
        for metric in ["ke", "p95_radius", "mean_radius"]:
            tmp = table[["iteration", metric]].copy()
            tmp["group"] = group
            tmp["metric"] = metric
            tmp = tmp.rename(columns={metric: "value"})
            rows.append(tmp)
    df = pd.concat(rows, ignore_index=True)
    metric_names = {
        "ke": "Knowledge extent (max radius)",
        "p95_radius": "p95 radius",
        "mean_radius": "Mean radius",
    }
    df["metric_label"] = df["metric"].map(metric_names)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6), sharey=False)
    palette = {
        "儿童环境场景": "#2c7fb8",
        "AI图像合并": "#8856a7",
        "AI Reason": "#d7301f",
        "AI Suggestion": "#fdae61",
    }
    for ax, metric_label in zip(axes, metric_names.values()):
        sub = df[df["metric_label"] == metric_label]
        sns.boxplot(
            data=sub,
            x="group",
            y="value",
            ax=ax,
            palette=palette,
            showfliers=False,
            width=0.62,
        )
        ax.set_title(metric_label)
        ax.set_xlabel("")
        ax.set_ylabel("4096D Euclidean distance")
        ax.tick_params(axis="x", rotation=25)

    fig.suptitle("Hao-style 高维知识广度指标：按相同样本量重复抽样", y=1.03, fontsize=13)
    fig.tight_layout()
    out = FIG_DIR / "Fig_v2_highdim_extent.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_centroid_heatmap(group_embs: dict[str, np.ndarray]) -> Path:
    names = list(group_embs)
    matrix = np.zeros((len(names), len(names)), dtype=float)
    for i, n1 in enumerate(names):
        for j, n2 in enumerate(names):
            matrix[i, j] = centroid_distance(group_embs[n1], group_embs[n2])
    df = pd.DataFrame(matrix, index=names, columns=names)

    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    sns.heatmap(df, annot=True, fmt=".3f", cmap="mako_r", square=True, cbar_kws={"label": "Centroid distance"}, ax=ax)
    ax.set_title("4096D 质心距离矩阵")
    fig.tight_layout()
    out = FIG_DIR / "Fig_v2_centroid_heatmap.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


DOMAIN_TERMS = [
    "安全",
    "危险",
    "实用",
    "观赏",
    "绿化",
    "互动",
    "涂鸦",
    "数字",
    "设施",
    "厕所",
    "店铺",
    "手机",
    "充电",
    "自行车",
    "电动车",
    "停车",
    "落叶",
    "护栏",
    "栏杆",
    "监控",
    "干净",
    "阴暗",
    "恐怖",
    "偏僻",
    "宽敞",
    "拥挤",
    "现实",
    "死板",
    "官方",
    "书面",
    "好玩",
    "友好",
    "吸引",
    "探索",
    "创意",
    "智能",
    "导览",
    "喷泉",
    "座椅",
    "长椅",
    "遮阳",
    "商场",
    "商业街",
    "图书馆",
    "学校",
    "马路",
    "交通",
    "生活",
    "日常",
    "游戏",
    "玩具",
    "空气清新",
]


def count_domain_terms(texts: list[str]) -> Counter:
    counter: Counter = Counter()
    for text in texts:
        for term in DOMAIN_TERMS:
            c = str(text).count(term)
            if c:
                counter[term] += c
    return counter


def plot_keyword_contrast(child_texts: list[str], ai_texts: list[str]) -> tuple[Path, list[dict]]:
    child_counts = count_domain_terms(child_texts)
    ai_counts = count_domain_terms(ai_texts)
    vocab = sorted(set(child_counts) | set(ai_counts))
    rows = []
    child_total = sum(child_counts.values()) or 1
    ai_total = sum(ai_counts.values()) or 1
    for term in vocab:
        c = child_counts[term]
        a = ai_counts[term]
        # Smoothed log ratio. Positive means child-skewed.
        score = math.log((c + 0.5) / (child_total + 0.5 * len(vocab))) - math.log(
            (a + 0.5) / (ai_total + 0.5 * len(vocab))
        )
        rows.append({"term": term, "child_count": c, "ai_count": a, "log_ratio": score})
    rows = sorted(rows, key=lambda r: r["log_ratio"])
    top_ai = rows[:12]
    top_child = rows[-12:]
    plot_rows = top_ai + top_child
    df = pd.DataFrame(plot_rows)
    df["side"] = np.where(df["log_ratio"] >= 0, "儿童更突出", "AI更突出")

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.barplot(
        data=df,
        y="term",
        x="log_ratio",
        hue="side",
        dodge=False,
        palette={"儿童更突出": "#2c7fb8", "AI更突出": "#d7301f"},
        ax=ax,
    )
    ax.axvline(0, color="#333333", lw=1)
    ax.set_title("领域词的差异性：儿童真实语料 vs AI合并文本")
    ax.set_xlabel("Smoothed log ratio（正值=儿童更突出）")
    ax.set_ylabel("")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    out = FIG_DIR / "Fig_v2_keyword_contrast.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out, rows


def image_to_data_uri(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def write_html_report(summary: dict, figures: dict[str, Path]) -> Path:
    def metric_table_rows(metrics: dict[str, dict]) -> str:
        labels = {
            "child_env_scene": "儿童环境场景",
            "ai_combined": "AI图像合并",
            "ai_reason": "AI Reason",
            "ai_suggestion": "AI Suggestion",
        }
        rows = []
        for key, label in labels.items():
            m = metrics[key]
            rows.append(
                f"<tr><td>{label}</td><td>{m['n']}</td><td>{m['ke']:.4f}</td>"
                f"<td>{m['p95_radius']:.4f}</td><td>{m['mean_radius']:.4f}</td></tr>"
            )
        return "\n".join(rows)

    fig_html = "\n".join(
        f"""
        <section>
          <h2>{title}</h2>
          <img src="{image_to_data_uri(path)}" alt="{title}">
        </section>
        """
        for title, path in figures.items()
    )

    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AI vs Children Semantic Gap - v2 Method Audit</title>
  <style>
    body {{ margin: 0; font-family: "Noto Sans SC", "Microsoft YaHei", Arial, sans-serif; color: #17202a; background: #f6f8fb; }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 28px 20px 48px; }}
    header {{ margin-bottom: 24px; }}
    h1 {{ margin: 0 0 8px; font-size: 30px; }}
    h2 {{ margin: 0 0 14px; font-size: 20px; }}
    p {{ line-height: 1.72; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 14px; margin: 18px 0 24px; }}
    .stat {{ background: white; border: 1px solid #dce3ea; border-radius: 8px; padding: 16px; }}
    .label {{ color: #637083; font-size: 13px; }}
    .value {{ font-size: 28px; font-weight: 700; margin-top: 5px; }}
    section {{ background: white; border: 1px solid #dce3ea; border-radius: 8px; padding: 18px; margin: 18px 0; }}
    img {{ display: block; width: 100%; height: auto; border: 1px solid #edf0f4; border-radius: 6px; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 12px; font-size: 14px; }}
    th, td {{ border-bottom: 1px solid #e8edf2; padding: 10px; text-align: left; }}
    th {{ background: #eef4f9; }}
    code {{ background: #eef2f7; padding: 1px 5px; border-radius: 4px; }}
    .warn {{ border-left: 4px solid #d7301f; padding-left: 12px; }}
  </style>
</head>
<body>
<main>
  <header>
    <h1>AI vs Children Semantic Gap - v2 Method Audit</h1>
    <p>本报告使用现有 <code>Qwen3-Embedding-8B</code> 向量重新组织分析，不重算 embedding。重点是区分正式高维指标和 t-SNE 可视化，并诊断儿童语料中的“直接环境感知”和“对 AI/专家的元评价”。</p>
  </header>

  <div class="grid">
    <div class="stat"><div class="label">儿童原始片段</div><div class="value">{summary['audit']['children_segments']}</div></div>
    <div class="stat"><div class="label">直接环境感知片段</div><div class="value">{summary['audit']['children_direct_segments']}</div></div>
    <div class="stat"><div class="label">儿童环境场景单元</div><div class="value">{summary['audit']['child_env_scene_units']}</div></div>
    <div class="stat"><div class="label">AI图片单元</div><div class="value">{summary['audit']['ai_image_units']}</div></div>
  </div>

  <section>
    <h2>方法审计结论</h2>
    <p class="warn">当前原始图的主要风险不是 embedding 模型本身，而是比较单位混杂：儿童侧包含大量对 AI/专家文本的评价；AI 侧把同一图片的 <code>reason</code> 与 <code>suggestion</code> 当成独立样本，t-SNE 因此很容易显示为两个 AI 团块。</p>
    <p>v2 的主比较采用儿童“直接环境感知”场景聚合单元 vs AI 图像级合并单元。正式指标仍在 4096 维归一化嵌入空间中计算，t-SNE 只作为二维诊断图。</p>
  </section>

  <section>
    <h2>高维指标摘要</h2>
    <table>
      <thead><tr><th>组别</th><th>n</th><th>Knowledge extent</th><th>p95 radius</th><th>Mean radius</th></tr></thead>
      <tbody>
        {metric_table_rows(summary['metrics'])}
      </tbody>
    </table>
    <p>主比较质心距离：<strong>{summary['primary_comparison']['centroid_distance']:.4f}</strong>；置换检验 P = <strong>{summary['primary_comparison']['permutation_p_value']:.4g}</strong>；最近邻相似度（儿童到AI中位数）= <strong>{summary['primary_comparison']['child_to_ai_nn_median']:.4f}</strong>。</p>
  </section>

  {fig_html}
</main>
</body>
</html>
"""
    out = BASE_DIR / "AI_vs_Children_Semantic_Gap_v2.html"
    out.write_text(html, encoding="utf-8")
    return out


def write_markdown_report(summary: dict, figures: dict[str, Path], html_path: Path) -> Path:
    out = BASE_DIR / "docs" / "method_audit_v2.md"
    rel_figs = {k: str(v.relative_to(BASE_DIR)).replace("\\", "/") for k, v in figures.items()}
    text = f"""# Paper2 T-SNE Pipeline: v2 方法审计与改进建议

## 核心诊断

1. 原 pipeline 的最大问题不是 Qwen3-Embedding-8B 本身，而是比较单位混杂：儿童侧是访谈片段，且包含大量对 AI/专家输出的元评价；AI 侧是同一图片的 `reason` 与 `suggestion` 两类文本。
2. t-SNE 上 AI 分成两个团，主要对应 `reason` 和 `suggestion` 的体裁/任务差异，不能直接解释为“AI 和儿童真实感知的知识广度差异”。
3. Hao et al. 的关键做法是：正式指标在高维 embedding 空间中计算，t-SNE 只负责说明性可视化。本项目应保留这一边界。

## v2 已完成的改动

- 自动使用脚本所在目录定位 pipeline，不再依赖 `/home/rk/...` 硬编码路径。
- 保留现有 `embeddings.pkl`，不重算 Qwen3 embedding。
- 重建 AI 的 image-level 结构：`reason`、`suggestion`、`reason+suggestion` 合并单元。
- 重建儿童片段元数据，并把儿童文本拆分为“直接环境感知”和“对 AI/专家的元评价”。
- 对儿童直接环境感知片段按访谈场景做平均向量聚合，降低短句和轮次切分造成的噪声。
- 正式输出 4096 维空间中的 `knowledge extent`、`p95 radius`、`mean radius`、质心距离、置换检验和最近邻相似度。

## 当前数据审计

- 儿童原始片段：{summary['audit']['children_segments']}
- 儿童直接环境感知片段：{summary['audit']['children_direct_segments']}
- 儿童环境场景单元：{summary['audit']['child_env_scene_units']}
- AI 图片单元：{summary['audit']['ai_image_units']}
- AI 文本校验错配数：{summary['audit']['ai_validation_mismatches']}

## 主比较结果

- 主比较：儿童环境场景 vs AI 图像合并文本
- 质心距离：{summary['primary_comparison']['centroid_distance']:.4f}
- 置换检验 P 值：{summary['primary_comparison']['permutation_p_value']:.4g}
- 儿童到 AI 最近邻相似度中位数：{summary['primary_comparison']['child_to_ai_nn_median']:.4f}
- AI 到儿童最近邻相似度中位数：{summary['primary_comparison']['ai_to_child_nn_median']:.4f}

## 输出文件

- HTML 报告：`{html_path.name}`
- t-SNE 诊断图：`{rel_figs['t-SNE 诊断图']}`
- 高维广度指标：`{rel_figs['高维知识广度']}`
- 质心距离矩阵：`{rel_figs['质心距离矩阵']}`
- 关键词差异：`{rel_figs['关键词差异']}`

## 下一步建议

1. 把儿童访谈重新标注到图片/场景 ID，形成 `image -> children_comments` 的配对表。没有这个配对层，实验只能做分布比较，不能做逐图片误差分析。
2. 不要把 `reason` 和 `suggestion` 当作同一分布里的独立样本。建议三条线并行报告：AI reason、AI suggestion、AI combined。
3. 重新在 WSL 中按 scene/image 单元计算 embedding，而不是对短句先 embedding 后平均。当前 v2 的场景向量是过渡方案。
4. 论文图中保留 t-SNE，但正文结论基于高维指标、置换检验、最近邻检索和关键词差异，不基于 t-SNE 圆圈面积。
5. 建议增加一个人工编码表，把儿童真实关注点分成安全、实用、可达、日常使用、趣味、数字设施、绿化等维度，再与 AI 的维度覆盖率做交叉验证。
"""
    out.write_text(text, encoding="utf-8")
    return out


def main() -> None:
    configure_plotting()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    emb_data = load_embedding_data()
    child_texts = [str(x) for x in emb_data["children"]["texts"]]
    child_embs = emb_data["children"]["embeddings"]
    ai_texts = emb_data["ai"]["texts"]
    ai_embs = emb_data["ai"]["embeddings"]

    child_records = extract_children_segments_with_metadata()
    if len(child_records) != len(child_texts) or any(
        r["text"] != t for r, t in zip(child_records, child_texts)
    ):
        print("Warning: reconstructed child metadata does not perfectly match corpus_children.json.")
        child_records = [
            {
                "text": text,
                "meeting": "unknown",
                "scene": i + 1,
                "speaker": "",
                "start_line": None,
                "end_line": None,
            }
            for i, text in enumerate(child_texts)
        ]

    child_direct_mask = np.array([is_direct_environment_text(r["text"]) for r in child_records])
    child_meta_mask = ~child_direct_mask

    child_scene_all_embs, child_scene_all_meta = aggregate_embeddings(child_embs, child_records)
    child_scene_env_embs, child_scene_env_meta = aggregate_embeddings(
        child_embs, child_records, keep_mask=child_direct_mask
    )

    ai_meta = reconstruct_ai_metadata(ai_texts, ai_embs)
    ai_reason_embs = ai_meta["reason"]["embeddings"]
    ai_suggestion_embs = ai_meta["suggestion"]["embeddings"]
    ai_combined_embs = ai_meta["combined"]["embeddings"]

    groups = {
        "child_env_scene": child_scene_env_embs,
        "child_scene_all": child_scene_all_embs,
        "child_direct_segments": child_embs[child_direct_mask],
        "child_meta_segments": child_embs[child_meta_mask],
        "ai_combined": ai_combined_embs,
        "ai_reason": ai_reason_embs,
        "ai_suggestion": ai_suggestion_embs,
    }
    metrics = {name: compute_group_stats(arr) for name, arr in groups.items()}

    primary_a = child_scene_env_embs
    primary_b = ai_combined_embs
    sample_n = max(5, min(len(primary_a), len(primary_b), 100))
    boot_tables = {
        "儿童环境场景": bootstrap_extent(primary_a, sample_n=sample_n, seed=RANDOM_SEED),
        "AI图像合并": bootstrap_extent(primary_b, sample_n=sample_n, seed=RANDOM_SEED + 1),
        "AI Reason": bootstrap_extent(ai_reason_embs, sample_n=sample_n, seed=RANDOM_SEED + 2),
        "AI Suggestion": bootstrap_extent(ai_suggestion_embs, sample_n=sample_n, seed=RANDOM_SEED + 3),
    }
    boot_summary = {
        name: {
            metric: percentile_ci(table[metric])
            for metric in ["ke", "p95_radius", "mean_radius"]
        }
        for name, table in boot_tables.items()
    }

    perm = permutation_centroid_test(primary_a, primary_b, repeats=1000)
    nn = nearest_neighbor_summary(primary_a, primary_b)

    figures = {
        "t-SNE 诊断图": plot_tsne(child_records, child_embs, child_direct_mask, ai_reason_embs, ai_suggestion_embs),
        "高维知识广度": plot_highdim_extent(boot_tables),
        "质心距离矩阵": plot_centroid_heatmap(
            {
                "儿童环境场景": primary_a,
                "儿童AI元评价": child_embs[child_meta_mask],
                "AI图像合并": primary_b,
                "AI Reason": ai_reason_embs,
                "AI Suggestion": ai_suggestion_embs,
            }
        ),
    }
    keyword_fig, keyword_rows = plot_keyword_contrast(
        [r["text"] for r, keep in zip(child_records, child_direct_mask) if keep],
        ai_meta["combined"]["texts"],
    )
    figures["关键词差异"] = keyword_fig

    clean_metrics = {}
    for name, metric in metrics.items():
        clean_metrics[name] = {
            k: v
            for k, v in metric.items()
            if k != "centroid"
        }

    summary = {
        "base_dir": str(BASE_DIR),
        "audit": {
            "children_segments": int(len(child_records)),
            "children_direct_segments": int(child_direct_mask.sum()),
            "children_meta_segments": int(child_meta_mask.sum()),
            "child_scene_units": int(len(child_scene_all_meta)),
            "child_env_scene_units": int(len(child_scene_env_meta)),
            "ai_image_units": int(len(ai_combined_embs)),
            "ai_reason_units": int(len(ai_reason_embs)),
            "ai_suggestion_units": int(len(ai_suggestion_embs)),
            "ai_validation_mismatches": int(ai_meta["validation_mismatches"]),
            "bootstrap_sample_n": int(sample_n),
        },
        "metrics": clean_metrics,
        "bootstrap_ci": boot_summary,
        "primary_comparison": {
            "group_a": "child_env_scene",
            "group_b": "ai_combined",
            "centroid_distance": perm["observed"],
            "permutation_p_value": perm["p_value"],
            "permutation_null_ci": perm["null_ci"],
            "child_to_ai_nn_mean": nn["a_to_b_mean"],
            "child_to_ai_nn_median": nn["a_to_b_median"],
            "ai_to_child_nn_mean": nn["b_to_a_mean"],
            "ai_to_child_nn_median": nn["b_to_a_median"],
        },
        "keyword_contrast": keyword_rows,
        "figures": {k: str(v) for k, v in figures.items()},
    }

    json_path = DATA_DIR / "analysis_results_v2.json"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    html_path = write_html_report(summary, figures)
    md_path = write_markdown_report(summary, figures, html_path)
    print(f"Saved v2 JSON: {json_path}")
    print(f"Saved v2 HTML: {html_path}")
    print(f"Saved v2 Markdown: {md_path}")


if __name__ == "__main__":
    main()
