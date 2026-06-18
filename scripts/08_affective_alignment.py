from __future__ import annotations

import argparse
import importlib.util
import json
import pickle
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib.font_manager import FontProperties, fontManager
from scipy.spatial.distance import jensenshannon
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForSequenceClassification, AutoTokenizer


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
FIG_DIR = DATA_DIR / "figures_affect"
PREDICTION_CACHE = DATA_DIR / "affective_predictions.pkl"
METRICS_PATH = DATA_DIR / "affective_alignment_metrics.json"

RANDOM_SEED = 42
MAIN_MODEL = "Johnson8187/Chinese-Emotion-Small"
ROBUSTNESS_MODEL = "tabularisai/multilingual-emotion-classification"

MAIN_LABELS = [
    "neutral",
    "concerned",
    "happy",
    "angry",
    "sad",
    "questioning",
    "surprised",
    "disgusted",
]
ROBUSTNESS_LABELS = [
    "anger",
    "contempt",
    "disgust",
    "fear",
    "frustration",
    "gratitude",
    "joy",
    "love",
    "neutral",
    "sadness",
    "surprise",
]
COMMON_LABELS = ["positive_engagement", "concern_distress", "inquiry_surprise", "neutral"]

DISPLAY_LABELS = {
    "neutral": "中性",
    "concerned": "关切",
    "happy": "开心",
    "angry": "愤怒",
    "sad": "悲伤",
    "questioning": "疑问",
    "surprised": "惊奇",
    "disgusted": "厌恶",
    "positive_engagement": "积极参与",
    "concern_distress": "忧虑/困扰",
    "inquiry_surprise": "探询/惊奇",
}

GROUP_DISPLAY = {
    "children_direct": "儿童直接感知",
    "children_meta": "儿童元评价",
    "ai_reason": "AI reason",
    "ai_suggestion": "AI suggestion",
    "ai_combined": "AI combined",
}

COLORS = {
    "children_direct": "#0072B2",
    "children_meta": "#56B4E9",
    "ai_reason": "#D55E00",
    "ai_suggestion": "#E69F00",
    "ai_combined": "#CC79A7",
}

ANCHOR_CASES = [
    ("这里有很多好玩的地方，我特别开心，还想再来。", "positive_engagement"),
    ("树很多，空气也很好，我喜欢在这里玩。", "positive_engagement"),
    ("这个滑梯太有趣了，我玩得很高兴。", "positive_engagement"),
    ("这里很漂亮，我很喜欢。", "positive_engagement"),
    ("这里很危险，我有点害怕，不敢一个人走。", "concern_distress"),
    ("路上车太多了，我担心会被撞到。", "concern_distress"),
    ("这里又脏又乱，让人很不舒服。", "concern_distress"),
    ("没有护栏，我觉得不安全。", "concern_distress"),
    ("为什么这里没有儿童座椅？", "inquiry_surprise"),
    ("这个东西是做什么用的？", "inquiry_surprise"),
    ("居然还有会说话的路灯，太奇怪了。", "inquiry_surprise"),
    ("为什么不能从这里进去？", "inquiry_surprise"),
    ("这里有一条路，旁边有几棵树。", "neutral"),
    ("图中可以看到长椅和路灯。", "neutral"),
    ("这个空间位于道路旁边。", "neutral"),
    ("前面是一栋建筑。", "neutral"),
]


def load_method_module():
    path = Path(__file__).with_name("05_method_audit_v2.py")
    spec = importlib.util.spec_from_file_location("method_audit_v2", path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    spec.loader.exec_module(module)
    return module


def configure_plotting() -> None:
    candidates = [
        Path("/mnt/c/Windows/Fonts/NotoSansSC-VF.ttf"),
        Path("/mnt/c/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/NotoSansSC-VF.ttf"),
        Path("C:/Windows/Fonts/msyh.ttc"),
    ]
    fonts = ["DejaVu Sans"]
    for path in candidates:
        if path.exists():
            fontManager.addfont(str(path))
            fonts.insert(0, FontProperties(fname=str(path)).get_name())
            break
    sns.set_theme(style="ticks", context="paper")
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": fonts,
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.dpi": 300,
            "figure.dpi": 150,
        }
    )


def load_inputs() -> tuple[dict[str, list[str]], dict[str, np.ndarray]]:
    method = load_method_module()
    emb_data = method.load_embedding_data()
    child_texts = [str(text) for text in emb_data["children"]["texts"]]
    child_records = method.extract_children_segments_with_metadata()
    if len(child_records) == len(child_texts):
        direct_mask = np.array(
            [method.is_direct_environment_text(record["text"]) for record in child_records],
            dtype=bool,
        )
    else:
        raise ValueError("Child metadata no longer aligns with embeddings.pkl")

    ai_meta = method.reconstruct_ai_metadata(
        emb_data["ai"]["texts"], emb_data["ai"]["embeddings"]
    )
    texts = {
        "children_direct": [text for text, keep in zip(child_texts, direct_mask) if keep],
        "children_meta": [text for text, keep in zip(child_texts, direct_mask) if not keep],
        "ai_reason": ai_meta["reason"]["texts"],
        "ai_suggestion": ai_meta["suggestion"]["texts"],
        "ai_combined": ai_meta["combined"]["texts"],
    }
    embeddings = {
        "children_direct": emb_data["children"]["embeddings"][direct_mask],
        "ai_combined": ai_meta["combined"]["embeddings"],
    }
    return texts, embeddings


def infer_model(
    model_id: str,
    texts_by_group: dict[str, list[str]],
    multilabel: bool,
    batch_size: int,
    max_length: int,
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id, dtype=dtype)
    model.to(device).eval()

    output: dict[str, np.ndarray] = {}
    with torch.inference_mode():
        for group, texts in texts_by_group.items():
            batches = []
            for start in range(0, len(texts), batch_size):
                inputs = tokenizer(
                    texts[start : start + batch_size],
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                ).to(device)
                logits = model(**inputs).logits.float()
                probs = torch.sigmoid(logits) if multilabel else torch.softmax(logits, dim=-1)
                batches.append(probs.cpu().numpy())
            output[group] = np.vstack(batches).astype(np.float32)

    commit_hash = getattr(model.config, "_commit_hash", None)
    del model
    torch.cuda.empty_cache()
    return {"model_id": model_id, "revision": commit_hash, "probs": output}


def to_common_profiles(probs: np.ndarray, model_key: str) -> np.ndarray:
    out = np.zeros((len(probs), len(COMMON_LABELS)), dtype=np.float32)
    if model_key == "main":
        index = {label: idx for idx, label in enumerate(MAIN_LABELS)}
        out[:, 0] = probs[:, index["happy"]]
        out[:, 1] = probs[:, [index[x] for x in ["concerned", "angry", "sad", "disgusted"]]].sum(axis=1)
        out[:, 2] = probs[:, [index["questioning"], index["surprised"]]].sum(axis=1)
        out[:, 3] = probs[:, index["neutral"]]
    else:
        index = {label: idx for idx, label in enumerate(ROBUSTNESS_LABELS)}
        out[:, 0] = probs[:, [index[x] for x in ["gratitude", "joy", "love"]]].sum(axis=1)
        out[:, 1] = probs[
            :,
            [
                index[x]
                for x in [
                    "anger",
                    "contempt",
                    "disgust",
                    "fear",
                    "frustration",
                    "sadness",
                ]
            ],
        ].sum(axis=1)
        out[:, 2] = probs[:, index["surprise"]]
        out[:, 3] = probs[:, index["neutral"]]
    row_sums = out.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return out / row_sums


def entropy_rows(probs: np.ndarray) -> np.ndarray:
    safe = np.clip(probs, 1e-12, 1.0)
    return -(safe * np.log2(safe)).sum(axis=1)


def permutation_centroid_test(
    a: np.ndarray, b: np.ndarray, repeats: int = 2000, seed: int = RANDOM_SEED
) -> dict:
    rng = np.random.default_rng(seed)
    observed = float(np.linalg.norm(a.mean(axis=0) - b.mean(axis=0)))
    combined = np.vstack([a, b])
    n_a = len(a)
    null = np.empty(repeats, dtype=float)
    for idx in range(repeats):
        order = rng.permutation(len(combined))
        null[idx] = np.linalg.norm(
            combined[order[:n_a]].mean(axis=0) - combined[order[n_a:]].mean(axis=0)
        )
    return {
        "centroid_distance": observed,
        "permutation_p": float((np.count_nonzero(null >= observed) + 1) / (repeats + 1)),
        "null_ci95": [float(np.percentile(null, 2.5)), float(np.percentile(null, 97.5))],
    }


def bootstrap_delta(
    child: np.ndarray, ai: np.ndarray, repeats: int = 2000, seed: int = RANDOM_SEED
) -> dict:
    rng = np.random.default_rng(seed)
    values = np.empty((repeats, child.shape[1]), dtype=float)
    for idx in range(repeats):
        child_sample = child[rng.integers(0, len(child), len(child))]
        ai_sample = ai[rng.integers(0, len(ai), len(ai))]
        values[idx] = ai_sample.mean(axis=0) - child_sample.mean(axis=0)
    return {
        label: {
            "delta_ai_minus_children": float(ai[:, col].mean() - child[:, col].mean()),
            "ci95": [
                float(np.percentile(values[:, col], 2.5)),
                float(np.percentile(values[:, col], 97.5)),
            ],
        }
        for col, label in enumerate(COMMON_LABELS)
    }


def compare_groups(child: np.ndarray, ai: np.ndarray, seed: int) -> dict:
    child_mean = child.mean(axis=0)
    ai_mean = ai.mean(axis=0)
    result = permutation_centroid_test(child, ai, seed=seed)
    result.update(
        {
            "js_divergence": float(jensenshannon(child_mean, ai_mean, base=2.0) ** 2),
            "dimension_deltas": bootstrap_delta(child, ai, seed=seed),
        }
    )
    return result


def summarize_model(
    probs_by_group: dict[str, np.ndarray], labels: list[str], model_key: str
) -> tuple[dict, dict[str, np.ndarray]]:
    common = {
        group: to_common_profiles(probs, model_key)
        for group, probs in probs_by_group.items()
    }
    groups = {}
    for group, probs in probs_by_group.items():
        dominant = np.argmax(probs, axis=1)
        counts = Counter(labels[idx] for idx in dominant)
        groups[group] = {
            "n": int(len(probs)),
            "mean_raw_probabilities": {
                label: float(value) for label, value in zip(labels, probs.mean(axis=0))
            },
            "dominant_label_proportions": {
                label: float(counts[label] / len(probs)) for label in labels
            },
            "mean_common_profile": {
                label: float(value)
                for label, value in zip(COMMON_LABELS, common[group].mean(axis=0))
            },
            "mean_common_entropy": float(entropy_rows(common[group]).mean()),
        }
    comparisons = {
        target: compare_groups(common["children_direct"], common[target], RANDOM_SEED + idx)
        for idx, target in enumerate(["ai_reason", "ai_suggestion", "ai_combined"])
    }
    return {"groups": groups, "comparisons": comparisons}, common


def anchor_diagnostic(
    main_result: dict, robustness_result: dict
) -> dict:
    expected = [label for _, label in ANCHOR_CASES]
    output = {}
    for model_key, result in [("main", main_result), ("robustness", robustness_result)]:
        common = to_common_profiles(result["probs"]["anchors"], model_key)
        predicted = [COMMON_LABELS[idx] for idx in np.argmax(common, axis=1)]
        correct = [p == e for p, e in zip(predicted, expected)]
        per_class = {}
        for label in COMMON_LABELS:
            mask = np.array([item == label for item in expected])
            per_class[label] = float(np.mean(np.array(correct)[mask]))
        output[model_key] = {
            "overall_match_rate": float(np.mean(correct)),
            "per_class_match_rate": per_class,
            "predicted_labels": predicted,
        }
    return output


def semantic_affective_alignment(
    embeddings: dict[str, np.ndarray],
    common: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, dict]:
    child_emb = embeddings["children_direct"]
    ai_emb = embeddings["ai_combined"]
    child_affect = common["children_direct"]
    ai_affect = common["ai_combined"]

    semantic = cosine_similarity(child_emb, ai_emb)
    child_semantic = semantic.max(axis=1)
    ai_semantic = semantic.max(axis=0)
    child_target = ai_affect.mean(axis=0)
    ai_target = child_affect.mean(axis=0)
    child_affective = np.array(
        [1.0 - jensenshannon(row, child_target, base=2.0) for row in child_affect]
    )
    ai_affective = np.array(
        [1.0 - jensenshannon(row, ai_target, base=2.0) for row in ai_affect]
    )

    df = pd.DataFrame(
        {
            "group": ["children_direct"] * len(child_semantic)
            + ["ai_combined"] * len(ai_semantic),
            "semantic_overlap": np.concatenate([child_semantic, ai_semantic]),
            "affective_overlap": np.concatenate([child_affective, ai_affective]),
        }
    )
    semantic_threshold = float(df["semantic_overlap"].median())
    affective_threshold = float(df["affective_overlap"].median())

    def quadrant(row) -> str:
        high_s = row.semantic_overlap >= semantic_threshold
        high_a = row.affective_overlap >= affective_threshold
        if high_s and high_a:
            return "shared_semantic_and_affective"
        if high_s:
            return "semantic_only"
        if high_a:
            return "affective_only"
        return "distinctive_both"

    df["quadrant"] = df.apply(quadrant, axis=1)
    quadrant_counts = df.groupby(["group", "quadrant"]).size()
    summary = {
        "semantic_overlap_threshold": semantic_threshold,
        "affective_overlap_threshold": affective_threshold,
        "group_means": {
            group: {
                "semantic_overlap": float(sub["semantic_overlap"].mean()),
                "affective_overlap": float(sub["affective_overlap"].mean()),
            }
            for group, sub in df.groupby("group")
        },
        "quadrant_proportions": {
            group: {
                quadrant: float(
                    quadrant_counts.get((group, quadrant), 0) / (df["group"] == group).sum()
                )
                for quadrant in [
                    "shared_semantic_and_affective",
                    "semantic_only",
                    "affective_only",
                    "distinctive_both",
                ]
            }
            for group in ["children_direct", "ai_combined"]
        },
    }
    return df, summary


def plot_main_profiles(main_summary: dict) -> Path:
    groups = ["children_direct", "children_meta", "ai_reason", "ai_suggestion", "ai_combined"]
    rows = []
    for group in groups:
        values = main_summary["groups"][group]["mean_raw_probabilities"]
        for label in MAIN_LABELS:
            rows.append(
                {
                    "group": GROUP_DISPLAY[group],
                    "emotion": DISPLAY_LABELS[label],
                    "probability": values[label],
                }
            )
    df = pd.DataFrame(rows)
    pivot = df.pivot(index="group", columns="emotion", values="probability")
    pivot = pivot[[DISPLAY_LABELS[label] for label in MAIN_LABELS]]

    palette = [
        "#999999",
        "#D55E00",
        "#009E73",
        "#CC3311",
        "#0072B2",
        "#E69F00",
        "#56B4E9",
        "#882255",
    ]
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    pivot.plot(kind="barh", stacked=True, color=palette, width=0.72, ax=ax)
    ax.set_title("中文情感模型：各类文本的平均情感概率")
    ax.set_xlabel("平均预测概率")
    ax.set_ylabel("")
    ax.set_xlim(0, 1)
    ax.legend(title="", ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.34), frameon=False)
    fig.tight_layout()
    out = FIG_DIR / "Fig_affect_main_profiles.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_model_robustness(
    main_common: dict[str, np.ndarray],
    robustness_common: dict[str, np.ndarray],
    anchors: dict,
) -> Path:
    targets = ["ai_reason", "ai_suggestion", "ai_combined"]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4))

    rows = []
    for model_key, profiles in [("中文模型", main_common), ("多语模型", robustness_common)]:
        child_mean = profiles["children_direct"].mean(axis=0)
        for target in targets:
            delta = profiles[target].mean(axis=0) - child_mean
            for label, value in zip(COMMON_LABELS, delta):
                rows.append(
                    {
                        "model": model_key,
                        "target": GROUP_DISPLAY[target],
                        "dimension": DISPLAY_LABELS[label],
                        "delta": value,
                    }
                )
    df = pd.DataFrame(rows)
    combined = df[df["target"] == GROUP_DISPLAY["ai_combined"]]
    sns.barplot(
        data=combined,
        x="dimension",
        y="delta",
        hue="model",
        palette=["#0072B2", "#D55E00"],
        ax=axes[0],
    )
    axes[0].axhline(0, color="#333333", lw=0.8)
    axes[0].set_title("AI combined - 儿童直接感知")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("共同情感维度概率差")
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].legend(title="", frameon=False)

    anchor_rows = []
    for model_key, label in [("main", "中文模型"), ("robustness", "多语模型")]:
        for dimension, value in anchors[model_key]["per_class_match_rate"].items():
            anchor_rows.append(
                {
                    "model": label,
                    "dimension": DISPLAY_LABELS[dimension],
                    "match": value,
                }
            )
    anchor_df = pd.DataFrame(anchor_rows)
    sns.barplot(
        data=anchor_df,
        x="dimension",
        y="match",
        hue="model",
        palette=["#0072B2", "#D55E00"],
        ax=axes[1],
    )
    axes[1].set_ylim(0, 1)
    axes[1].set_title("中文锚点句面效度诊断")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("预期维度匹配率")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].legend(title="", frameon=False)

    fig.suptitle("双模型稳健性：共同维度方向与面效度", y=1.02)
    fig.tight_layout()
    out = FIG_DIR / "Fig_affect_model_robustness.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_alignment(df: pd.DataFrame, summary: dict) -> Path:
    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    for group in ["children_direct", "ai_combined"]:
        sub = df[df["group"] == group]
        ax.scatter(
            sub["semantic_overlap"],
            sub["affective_overlap"],
            s=20,
            alpha=0.45,
            color=COLORS[group],
            label=f"{GROUP_DISPLAY[group]} (n={len(sub)})",
            edgecolors="none",
        )
    ax.axvline(summary["semantic_overlap_threshold"], color="#555555", ls="--", lw=1)
    ax.axhline(summary["affective_overlap_threshold"], color="#555555", ls="--", lw=1)
    ax.set_xlabel("跨群体语义重叠（最近邻余弦相似度）")
    ax.set_ylabel("跨群体情感重叠（1 - Jensen-Shannon 距离）")
    ax.set_title("Semantic-affective alignment：语义相似不等于情感一致")
    ax.legend(frameon=False)
    ax.text(0.99, 0.98, "语义与情感均共享", transform=ax.transAxes, ha="right", va="top")
    ax.text(0.01, 0.02, "语义与情感均独特", transform=ax.transAxes, ha="left", va="bottom")
    fig.tight_layout()
    out = FIG_DIR / "Fig_semantic_affective_alignment.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def directional_agreement(
    main_common: dict[str, np.ndarray],
    robustness_common: dict[str, np.ndarray],
) -> dict:
    output = {}
    for target in ["ai_reason", "ai_suggestion", "ai_combined"]:
        main_delta = (
            main_common[target].mean(axis=0) - main_common["children_direct"].mean(axis=0)
        )
        robust_delta = (
            robustness_common[target].mean(axis=0)
            - robustness_common["children_direct"].mean(axis=0)
        )
        denom = np.linalg.norm(main_delta) * np.linalg.norm(robust_delta)
        output[target] = {
            "sign_agreement": float(np.mean(np.sign(main_delta) == np.sign(robust_delta))),
            "delta_cosine_similarity": float(
                np.dot(main_delta, robust_delta) / denom if denom else 0.0
            ),
            "main_delta": {
                label: float(value) for label, value in zip(COMMON_LABELS, main_delta)
            },
            "robustness_delta": {
                label: float(value) for label, value in zip(COMMON_LABELS, robust_delta)
            },
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Dual-model semantic-affective alignment")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--force", action="store_true", help="Ignore local prediction cache")
    args = parser.parse_args()

    np.random.seed(RANDOM_SEED)
    configure_plotting()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    texts, embeddings = load_inputs()
    texts_with_anchors = dict(texts)
    texts_with_anchors["anchors"] = [text for text, _ in ANCHOR_CASES]

    if PREDICTION_CACHE.exists() and not args.force:
        with PREDICTION_CACHE.open("rb") as handle:
            predictions = pickle.load(handle)
    else:
        main_result = infer_model(
            MAIN_MODEL, texts_with_anchors, False, args.batch_size, args.max_length
        )
        robustness_result = infer_model(
            ROBUSTNESS_MODEL, texts_with_anchors, True, args.batch_size, args.max_length
        )
        predictions = {"main": main_result, "robustness": robustness_result}
        with PREDICTION_CACHE.open("wb") as handle:
            pickle.dump(predictions, handle)

    main_summary, main_common = summarize_model(
        {key: value for key, value in predictions["main"]["probs"].items() if key != "anchors"},
        MAIN_LABELS,
        "main",
    )
    robustness_summary, robustness_common = summarize_model(
        {
            key: value
            for key, value in predictions["robustness"]["probs"].items()
            if key != "anchors"
        },
        ROBUSTNESS_LABELS,
        "robustness",
    )
    anchors = anchor_diagnostic(predictions["main"], predictions["robustness"])
    consensus_common = {
        group: (main_common[group] + robustness_common[group]) / 2.0
        for group in main_common
    }
    alignment_df, alignment_summary = semantic_affective_alignment(
        embeddings, consensus_common
    )
    alignment_summary["affective_profile_source"] = (
        "Mean of the Chinese-specific and multilingual models in the four common dimensions."
    )
    agreement = directional_agreement(main_common, robustness_common)

    figures = {
        "main_profiles": plot_main_profiles(main_summary),
        "model_robustness": plot_model_robustness(
            main_common, robustness_common, anchors
        ),
        "semantic_affective_alignment": plot_alignment(
            alignment_df, alignment_summary
        ),
    }
    metrics = {
        "method": {
            "primary_text_unit": "utterance; AI combined is reason + suggestion per image",
            "interpretation_boundary": (
                "Predictions describe affective cues expressed in text, not participants' latent emotions."
            ),
            "common_dimensions": COMMON_LABELS,
            "main_model": predictions["main"]["model_id"],
            "main_model_revision": predictions["main"]["revision"],
            "robustness_model": predictions["robustness"]["model_id"],
            "robustness_model_revision": predictions["robustness"]["revision"],
            "robustness_model_license": "CC-BY-NC-4.0",
        },
        "main_model": main_summary,
        "robustness_model": robustness_summary,
        "anchor_diagnostic": anchors,
        "cross_model_directional_agreement": agreement,
        "semantic_affective_alignment": alignment_summary,
        "figures": {
            key: str(path.relative_to(BASE_DIR)).replace("\\", "/")
            for key, path in figures.items()
        },
    }
    METRICS_PATH.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Saved aggregate metrics: {METRICS_PATH}")
    for path in figures.values():
        print(f"Saved figure: {path}")
    print(
        "Anchor match rates:",
        f"Chinese={anchors['main']['overall_match_rate']:.3f}",
        f"multilingual={anchors['robustness']['overall_match_rate']:.3f}",
    )


if __name__ == "__main__":
    main()
