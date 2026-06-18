from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import ks_2samp, mannwhitneyu
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
FIG_DIR = DATA_DIR / "figures_v2"
RANDOM_SEED = 42
BOOTSTRAP_N = 2000
THRESHOLDS = [0.50, 0.55, 0.60, 0.626, 0.65, 0.70, 0.75]
FRONTIER_THRESHOLDS = [round(x, 3) for x in np.arange(0.45, 0.781, 0.01)]
PRIMARY_THRESHOLD = 0.65


def load_method_module():
    path = Path(__file__).with_name("05_method_audit_v2.py")
    spec = importlib.util.spec_from_file_location("method_audit_v2", path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    spec.loader.exec_module(module)
    return module


def percentile_ci(values: np.ndarray | list[float]) -> list[float]:
    arr = np.asarray(values, dtype=float)
    return [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))]


def summarize_scores(scores: np.ndarray) -> dict:
    return {
        "n": int(len(scores)),
        "mean": float(np.mean(scores)),
        "median": float(np.median(scores)),
        "q25": float(np.percentile(scores, 25)),
        "q75": float(np.percentile(scores, 75)),
        "p10": float(np.percentile(scores, 10)),
        "p90": float(np.percentile(scores, 90)),
        "min": float(np.min(scores)),
        "max": float(np.max(scores)),
    }


def gini(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    if np.all(arr == 0):
        return 0.0
    arr = np.sort(arr)
    n = len(arr)
    weighted = np.sum((np.arange(1, n + 1) * arr))
    return float((2 * weighted) / (n * arr.sum()) - (n + 1) / n)


def top_share(values: np.ndarray, proportion: float = 0.10) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.sum() == 0:
        return 0.0
    top_n = max(1, int(np.ceil(len(arr) * proportion)))
    return float(np.sort(arr)[-top_n:].sum() / arr.sum())


def lorenz_curve(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    arr = np.sort(np.asarray(values, dtype=float))
    if arr.sum() == 0:
        x = np.linspace(0, 1, len(arr) + 1)
        return x, np.zeros_like(x)
    cum = np.concatenate([[0.0], np.cumsum(arr) / arr.sum()])
    x = np.linspace(0, 1, len(cum))
    return x, cum


def bootstrap_rate(scores: np.ndarray, threshold: float, repeats: int = BOOTSTRAP_N) -> list[float]:
    rng = np.random.default_rng(RANDOM_SEED + int(threshold * 10000))
    boot = np.empty(repeats, dtype=float)
    n = len(scores)
    for i in range(repeats):
        sample = scores[rng.integers(0, n, size=n)]
        boot[i] = np.mean(sample >= threshold)
    return percentile_ci(boot)


def bootstrap_mean_difference(
    ai_scores: np.ndarray,
    child_scores: np.ndarray,
    repeats: int = BOOTSTRAP_N,
) -> tuple[float, list[float]]:
    rng = np.random.default_rng(RANDOM_SEED + 1001)
    boot = np.empty(repeats, dtype=float)
    for i in range(repeats):
        ai_sample = ai_scores[rng.integers(0, len(ai_scores), size=len(ai_scores))]
        child_sample = child_scores[rng.integers(0, len(child_scores), size=len(child_scores))]
        boot[i] = ai_sample.mean() - child_sample.mean()
    return float(ai_scores.mean() - child_scores.mean()), percentile_ci(boot)


def bootstrap_rate_difference(
    ai_scores: np.ndarray,
    child_scores: np.ndarray,
    threshold: float,
    repeats: int = BOOTSTRAP_N,
) -> tuple[float, list[float]]:
    rng = np.random.default_rng(RANDOM_SEED + int(threshold * 10000) + 2001)
    boot = np.empty(repeats, dtype=float)
    for i in range(repeats):
        ai_sample = ai_scores[rng.integers(0, len(ai_scores), size=len(ai_scores))]
        child_sample = child_scores[rng.integers(0, len(child_scores), size=len(child_scores))]
        boot[i] = np.mean(ai_sample >= threshold) - np.mean(child_sample >= threshold)
    observed = float(np.mean(ai_scores >= threshold) - np.mean(child_scores >= threshold))
    return observed, percentile_ci(boot)


def load_fixed_samples() -> tuple[list[str], np.ndarray, list[str], np.ndarray]:
    method = load_method_module()
    emb_data = method.load_embedding_data()
    child_records = method.extract_children_segments_with_metadata()
    child_texts = [str(text) for text in emb_data["children"]["texts"]]
    if len(child_records) != len(child_texts):
        raise ValueError("Child metadata no longer aligns with embeddings.pkl")

    direct_mask = np.array(
        [method.is_direct_environment_text(record["text"]) for record in child_records],
        dtype=bool,
    )
    ai_meta = method.reconstruct_ai_metadata(
        emb_data["ai"]["texts"], emb_data["ai"]["embeddings"]
    )
    return (
        [text for text, keep in zip(child_texts, direct_mask) if keep],
        emb_data["children"]["embeddings"][direct_mask],
        ai_meta["combined"]["texts"],
        ai_meta["combined"]["embeddings"],
    )


def compute_coverage(child_embs: np.ndarray, ai_embs: np.ndarray) -> tuple[pd.DataFrame, dict]:
    sim = cosine_similarity(child_embs, ai_embs)
    child_to_ai = sim.max(axis=1)
    child_to_ai_arg = sim.argmax(axis=1)
    ai_to_child = sim.max(axis=0)
    ai_to_child_arg = sim.argmax(axis=0)
    child_receiver_counts = np.bincount(ai_to_child_arg, minlength=len(child_embs))
    ai_receiver_counts = np.bincount(child_to_ai_arg, minlength=len(ai_embs))
    mutual_pairs = [
        (int(child_idx), int(ai_idx))
        for child_idx, ai_idx in enumerate(child_to_ai_arg)
        if ai_to_child_arg[ai_idx] == child_idx
    ]

    mean_diff, mean_diff_ci = bootstrap_mean_difference(ai_to_child, child_to_ai)
    try:
        mw = mannwhitneyu(ai_to_child, child_to_ai, alternative="greater")
        mw_two_sided = mannwhitneyu(ai_to_child, child_to_ai, alternative="two-sided")
        ks = ks_2samp(ai_to_child, child_to_ai, alternative="two-sided")
        tests = {
            "mann_whitney_u_greater_p": float(mw.pvalue),
            "mann_whitney_u_two_sided_p": float(mw_two_sided.pvalue),
            "ks_two_sided_p": float(ks.pvalue),
            "ks_statistic": float(ks.statistic),
        }
    except Exception as exc:  # pragma: no cover - defensive for minimal scipy installs
        tests = {"error": str(exc)}

    rows = []
    for threshold in THRESHOLDS:
        child_rate = float(np.mean(child_to_ai >= threshold))
        ai_rate = float(np.mean(ai_to_child >= threshold))
        diff, diff_ci = bootstrap_rate_difference(ai_to_child, child_to_ai, threshold)
        rows.append(
            {
                "threshold": float(threshold),
                "children_covered_by_ai": child_rate,
                "children_covered_by_ai_ci95": bootstrap_rate(child_to_ai, threshold),
                "ai_covered_by_children": ai_rate,
                "ai_covered_by_children_ci95": bootstrap_rate(ai_to_child, threshold),
                "ai_minus_children_coverage": diff,
                "ai_minus_children_coverage_ci95": diff_ci,
            }
        )

    frontier_rows = []
    for threshold in FRONTIER_THRESHOLDS:
        ai_precision = float(np.mean(ai_to_child >= threshold))
        child_recall = float(np.mean(child_to_ai >= threshold))
        f1 = (
            0.0
            if ai_precision + child_recall == 0
            else float(2 * ai_precision * child_recall / (ai_precision + child_recall))
        )
        frontier_rows.append(
            {
                "threshold": float(threshold),
                "ai_semantic_precision": ai_precision,
                "child_semantic_recall": child_recall,
                "f1_like_balance": f1,
                "precision_minus_recall": float(ai_precision - child_recall),
            }
        )

    point_rows = []
    for idx, score in enumerate(child_to_ai):
        point_rows.append(
            {
                "group": "children_direct",
                "nearest_neighbor_similarity": float(score),
                "nearest_neighbor_index": int(child_to_ai_arg[idx]),
                "covered_tau_065": bool(score >= PRIMARY_THRESHOLD),
            }
        )
    for idx, score in enumerate(ai_to_child):
        point_rows.append(
            {
                "group": "ai_combined",
                "nearest_neighbor_similarity": float(score),
                "nearest_neighbor_index": int(ai_to_child_arg[idx]),
                "covered_tau_065": bool(score >= PRIMARY_THRESHOLD),
            }
        )

    metrics = {
        "method": {
            "embedding_model": "Qwen/Qwen3-Embedding-8B",
            "embedding_dimensions": int(child_embs.shape[1]),
            "sample_policy": "Fixed observed samples; no balanced subsampling.",
            "children_direct_n": int(len(child_embs)),
            "ai_combined_n": int(len(ai_embs)),
            "definition": {
                "children_covered_by_ai": "For each child utterance, max cosine similarity to any AI combined image-level output.",
                "ai_covered_by_children": "For each AI combined output, max cosine similarity to any direct child utterance.",
            },
        },
        "nearest_neighbor_summary": {
            "children_covered_by_ai": summarize_scores(child_to_ai),
            "ai_covered_by_children": summarize_scores(ai_to_child),
            "ai_minus_children_mean_similarity": mean_diff,
            "ai_minus_children_mean_similarity_ci95": mean_diff_ci,
            "ai_minus_children_median_similarity": float(np.median(ai_to_child) - np.median(child_to_ai)),
            "tests": tests,
        },
        "coverage_by_threshold": rows,
        "precision_recall_frontier": {
            "definition": {
                "ai_semantic_precision": "Share of AI outputs that have a child nearest-neighbor above threshold.",
                "child_semantic_recall": "Share of child utterances that have an AI nearest-neighbor above threshold.",
                "f1_like_balance": "Harmonic mean of AI semantic precision and child semantic recall at the same threshold.",
            },
            "thresholds": frontier_rows,
            "best_f1_like": max(frontier_rows, key=lambda row: row["f1_like_balance"]),
            "primary_tau_065": next(
                row for row in frontier_rows if abs(row["threshold"] - PRIMARY_THRESHOLD) < 1e-9
            ),
        },
        "local_neighborhood": {
            "mutual_nearest_neighbor_pairs": int(len(mutual_pairs)),
            "mutual_pair_share_of_children": float(len(mutual_pairs) / len(child_embs)),
            "mutual_pair_share_of_ai": float(len(mutual_pairs) / len(ai_embs)),
            "children_as_receivers_for_ai": {
                "zero_receiver_share": float(np.mean(child_receiver_counts == 0)),
                "top10_assignment_share": top_share(child_receiver_counts),
                "gini": gini(child_receiver_counts),
                "max_assignments_to_one_child": int(child_receiver_counts.max()),
                "receiver_counts": child_receiver_counts.astype(int).tolist(),
            },
            "ai_as_receivers_for_children": {
                "zero_receiver_share": float(np.mean(ai_receiver_counts == 0)),
                "top10_assignment_share": top_share(ai_receiver_counts),
                "gini": gini(ai_receiver_counts),
                "max_assignments_to_one_ai": int(ai_receiver_counts.max()),
                "receiver_counts": ai_receiver_counts.astype(int).tolist(),
            },
        },
    }
    return pd.DataFrame(point_rows), metrics


def configure_plotting(method) -> None:
    method.configure_plotting()
    sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelsize": 10,
            "axes.titlesize": 12,
            "legend.fontsize": 9,
        }
    )


def plot_coverage(metrics: dict, points: pd.DataFrame) -> Path:
    rows = metrics["coverage_by_threshold"]
    curve = pd.DataFrame(rows)
    dist = points.copy()
    dist["direction"] = dist["group"].map(
        {
            "children_direct": "Child utterances covered by AI",
            "ai_combined": "AI outputs covered by children",
        }
    )
    palette = {
        "Child utterances covered by AI": "#4C78D8",
        "AI outputs covered by children": "#F25A5A",
    }

    fig = plt.figure(figsize=(12, 8.3))
    gs = fig.add_gridspec(2, 2, hspace=0.34, wspace=0.27)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    sns.histplot(
        data=dist,
        x="nearest_neighbor_similarity",
        hue="direction",
        palette=palette,
        stat="density",
        common_norm=False,
        bins=26,
        alpha=0.25,
        element="step",
        ax=ax1,
    )
    sns.kdeplot(
        data=dist,
        x="nearest_neighbor_similarity",
        hue="direction",
        palette=palette,
        common_norm=False,
        lw=2.2,
        ax=ax1,
        legend=False,
    )
    ax1.axvline(PRIMARY_THRESHOLD, color="#333333", ls="--", lw=1.2)
    ax1.set_title("A. 4096D nearest-neighbor similarity")
    ax1.set_xlabel("Max cosine similarity to the other group")
    ax1.set_ylabel("Density")

    x = curve["threshold"].to_numpy()
    child = curve["children_covered_by_ai"].to_numpy()
    child_ci = np.array(curve["children_covered_by_ai_ci95"].tolist(), dtype=float)
    ai = curve["ai_covered_by_children"].to_numpy()
    ai_ci = np.array(curve["ai_covered_by_children_ci95"].tolist(), dtype=float)
    ax2.plot(x, child, marker="o", color="#4C78D8", label="Child utterances covered by AI")
    ax2.fill_between(x, child_ci[:, 0], child_ci[:, 1], color="#4C78D8", alpha=0.15)
    ax2.plot(x, ai, marker="o", color="#F25A5A", label="AI outputs covered by children")
    ax2.fill_between(x, ai_ci[:, 0], ai_ci[:, 1], color="#F25A5A", alpha=0.15)
    ax2.axvline(PRIMARY_THRESHOLD, color="#333333", ls="--", lw=1.2)
    ax2.set_ylim(-0.02, 1.02)
    ax2.set_title("B. Directional semantic coverage curve")
    ax2.set_xlabel("Similarity threshold (tau)")
    ax2.set_ylabel("Coverage rate")
    ax2.legend(frameon=True, loc="lower left")

    selected = curve[curve["threshold"].isin([0.60, 0.626, 0.65, 0.70])].copy()
    selected["threshold_label"] = selected["threshold"].map(lambda v: f"{v:.3f}")
    bars = pd.melt(
        selected,
        id_vars=["threshold_label"],
        value_vars=["children_covered_by_ai", "ai_covered_by_children"],
        var_name="direction",
        value_name="coverage_rate",
    )
    bars["direction"] = bars["direction"].map(
        {
            "children_covered_by_ai": "Child utterances\ncovered by AI",
            "ai_covered_by_children": "AI outputs\ncovered by children",
        }
    )
    sns.barplot(
        data=bars,
        x="threshold_label",
        y="coverage_rate",
        hue="direction",
        palette=["#4C78D8", "#F25A5A"],
        ax=ax3,
    )
    ax3.set_ylim(0, 1.0)
    ax3.set_title("C. Coverage at interpretable thresholds")
    ax3.set_xlabel("Threshold")
    ax3.set_ylabel("Coverage rate")
    ax3.legend(frameon=False, loc="upper right")

    diff = curve["ai_minus_children_coverage"].to_numpy()
    diff_ci = np.array(curve["ai_minus_children_coverage_ci95"].tolist(), dtype=float)
    ax4.plot(x, diff, marker="o", color="#2A9D8F")
    ax4.fill_between(x, diff_ci[:, 0], diff_ci[:, 1], color="#2A9D8F", alpha=0.18)
    ax4.axhline(0, color="#555555", ls="--", lw=1.0)
    ax4.axvline(PRIMARY_THRESHOLD, color="#333333", ls="--", lw=1.2)
    ax4.set_title("D. Directional asymmetry")
    ax4.set_xlabel("Similarity threshold (tau)")
    ax4.set_ylabel("AI coverage rate minus child coverage rate")

    fig.suptitle(
        "High-dimensional semantic coverage, computed before t-SNE projection",
        fontsize=14,
        y=0.985,
    )
    out = FIG_DIR / "Fig_v2_semantic_coverage_curves.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_tsne_coverage(
    child_embs: np.ndarray,
    ai_embs: np.ndarray,
    points: pd.DataFrame,
) -> Path:
    plot_embs = np.vstack([child_embs, ai_embs])
    perplexity = max(5, min(30, (len(plot_embs) - 1) // 3))
    coords = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=RANDOM_SEED,
        init="pca",
        learning_rate="auto",
    ).fit_transform(plot_embs)

    df = points.copy()
    df["x"] = coords[:, 0]
    df["y"] = coords[:, 1]
    df["status"] = np.where(
        (df["group"] == "children_direct") & df["covered_tau_065"],
        "Child covered by AI",
        np.where(
            (df["group"] == "children_direct") & ~df["covered_tau_065"],
            "Child distinctive at tau=0.65",
            np.where(
                (df["group"] == "ai_combined") & df["covered_tau_065"],
                "AI covered by children",
                "AI distinctive at tau=0.65",
            ),
        ),
    )

    palette = {
        "Child covered by AI": "#9ECAE1",
        "Child distinctive at tau=0.65": "#08519C",
        "AI covered by children": "#FCAE91",
        "AI distinctive at tau=0.65": "#A50F15",
    }
    order = [
        "Child covered by AI",
        "Child distinctive at tau=0.65",
        "AI covered by children",
        "AI distinctive at tau=0.65",
    ]

    fig, ax = plt.subplots(figsize=(9, 7.4))
    for status in order:
        sub = df[df["status"] == status]
        marker = "o" if status.startswith("Child") else "^"
        ax.scatter(
            sub["x"],
            sub["y"],
            s=32,
            c=palette[status],
            alpha=0.72,
            marker=marker,
            edgecolors="white",
            linewidths=0.3,
            label=f"{status} (n={len(sub)})",
        )

    ax.set_title("t-SNE annotated by 4096D semantic coverage")
    ax.set_xlabel("t-SNE dimension 1")
    ax.set_ylabel("t-SNE dimension 2")
    ax.legend(loc="best", frameon=True)
    ax.text(
        0.01,
        -0.13,
        "Coverage status is computed in the original 4096D Qwen3 embedding space; "
        "t-SNE only places the points for visual inspection.",
        transform=ax.transAxes,
        fontsize=9,
        color="#444444",
    )
    fig.tight_layout()
    out = FIG_DIR / "Fig_v2_tsne_coverage_status.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)

    df[["group", "nearest_neighbor_similarity", "covered_tau_065", "x", "y"]].to_csv(
        DATA_DIR / "semantic_coverage_points.csv",
        index=False,
        encoding="utf-8",
    )
    return out


def plot_precision_recall_frontier(metrics: dict) -> Path:
    frontier = pd.DataFrame(metrics["precision_recall_frontier"]["thresholds"])
    local = metrics["local_neighborhood"]
    child_receivers = np.asarray(local["children_as_receivers_for_ai"]["receiver_counts"])
    ai_receivers = np.asarray(local["ai_as_receivers_for_children"]["receiver_counts"])

    fig = plt.figure(figsize=(12, 8.2))
    gs = fig.add_gridspec(2, 2, hspace=0.34, wspace=0.30)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    sc = ax1.scatter(
        frontier["child_semantic_recall"],
        frontier["ai_semantic_precision"],
        c=frontier["threshold"],
        cmap="viridis",
        s=48,
        edgecolors="white",
        linewidths=0.4,
    )
    ax1.plot(frontier["child_semantic_recall"], frontier["ai_semantic_precision"], color="#555555", lw=1)
    primary = metrics["precision_recall_frontier"]["primary_tau_065"]
    best = metrics["precision_recall_frontier"]["best_f1_like"]
    ax1.scatter(
        [primary["child_semantic_recall"]],
        [primary["ai_semantic_precision"]],
        s=140,
        marker="*",
        color="#D55E00",
        edgecolors="white",
        linewidths=0.7,
        label="tau = 0.65",
        zorder=4,
    )
    ax1.scatter(
        [best["child_semantic_recall"]],
        [best["ai_semantic_precision"]],
        s=110,
        marker="D",
        color="#0072B2",
        edgecolors="white",
        linewidths=0.7,
        label=f"best balance tau = {best['threshold']:.2f}",
        zorder=4,
    )
    ax1.set_xlim(-0.02, 1.02)
    ax1.set_ylim(-0.02, 1.02)
    ax1.set_title("A. Semantic precision-recall frontier")
    ax1.set_xlabel("Child semantic recall\n(child utterances covered by AI)")
    ax1.set_ylabel("AI semantic precision\n(AI outputs covered by children)")
    ax1.legend(frameon=True, loc="lower left")
    cb = fig.colorbar(sc, ax=ax1, fraction=0.046, pad=0.04)
    cb.set_label("Similarity threshold")

    ax2.plot(frontier["threshold"], frontier["f1_like_balance"], color="#009E73", marker="o", ms=4, label="F1-like balance")
    ax2.plot(frontier["threshold"], frontier["precision_minus_recall"], color="#CC79A7", marker="s", ms=4, label="Precision - recall")
    ax2.axvline(PRIMARY_THRESHOLD, color="#333333", ls="--", lw=1.1)
    ax2.axhline(0, color="#777777", ls=":", lw=1.0)
    ax2.set_title("B. Threshold sensitivity")
    ax2.set_xlabel("Similarity threshold")
    ax2.set_ylabel("Score")
    ax2.legend(frameon=True, loc="best")

    for counts, label, color in [
        (child_receivers, "Child texts receiving AI assignments", "#F25A5A"),
        (ai_receivers, "AI texts receiving child assignments", "#4C78D8"),
    ]:
        x, y = lorenz_curve(counts)
        ax3.plot(x, y, lw=2.2, color=color, label=label)
    ax3.plot([0, 1], [0, 1], color="#777777", ls="--", lw=1)
    ax3.set_title("C. Local-neighborhood concentration")
    ax3.set_xlabel("Cumulative share of receiver texts")
    ax3.set_ylabel("Cumulative share of NN assignments")
    ax3.legend(frameon=True, loc="upper left")

    summary_rows = pd.DataFrame(
        [
            {
                "receiver_set": "Children\nreceiving AI",
                "Top 10% share": local["children_as_receivers_for_ai"]["top10_assignment_share"],
                "Zero-receiver share": local["children_as_receivers_for_ai"]["zero_receiver_share"],
            },
            {
                "receiver_set": "AI\nreceiving children",
                "Top 10% share": local["ai_as_receivers_for_children"]["top10_assignment_share"],
                "Zero-receiver share": local["ai_as_receivers_for_children"]["zero_receiver_share"],
            },
        ]
    )
    summary_long = pd.melt(
        summary_rows,
        id_vars=["receiver_set"],
        value_vars=["Top 10% share", "Zero-receiver share"],
        var_name="metric",
        value_name="share",
    )
    sns.barplot(
        data=summary_long,
        x="receiver_set",
        y="share",
        hue="metric",
        palette=["#E69F00", "#56B4E9"],
        ax=ax4,
    )
    ax4.set_ylim(0, 1)
    ax4.set_title("D. Receiver hubness summary")
    ax4.set_xlabel("")
    ax4.set_ylabel("Share")
    ax4.legend(frameon=False, loc="upper right")
    ax4.text(
        0.02,
        0.04,
        (
            f"Mutual NN pairs: {local['mutual_nearest_neighbor_pairs']} "
            f"({local['mutual_pair_share_of_ai']:.1%} of AI outputs)"
        ),
        transform=ax4.transAxes,
        fontsize=9,
        color="#333333",
        bbox=dict(facecolor="white", edgecolor="#cccccc", alpha=0.85),
    )

    fig.suptitle(
        "Precision/recall-style semantic coverage and local-neighborhood diagnostics",
        fontsize=14,
        y=0.985,
    )
    out = FIG_DIR / "Fig_v2_semantic_pr_frontier.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    method = load_method_module()
    configure_plotting(method)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    child_texts, child_embs, ai_texts, ai_embs = load_fixed_samples()
    points, metrics = compute_coverage(child_embs, ai_embs)
    coverage_fig = plot_coverage(metrics, points)
    tsne_fig = plot_tsne_coverage(child_embs, ai_embs, points)
    frontier_fig = plot_precision_recall_frontier(metrics)
    metrics["figures"] = {
        "coverage_curves": str(coverage_fig.relative_to(BASE_DIR)),
        "tsne_coverage_status": str(tsne_fig.relative_to(BASE_DIR)),
        "precision_recall_frontier": str(frontier_fig.relative_to(BASE_DIR)),
    }
    metrics["derived_files"] = {
        "coverage_points": "data/semantic_coverage_points.csv",
    }

    out = DATA_DIR / "semantic_coverage_metrics.json"
    out.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Children direct n={len(child_texts)}; AI combined n={len(ai_texts)}")
    print(f"Saved semantic coverage metrics: {out}")
    print(f"Saved figures: {coverage_fig}; {tsne_fig}; {frontier_fig}")


if __name__ == "__main__":
    main()
