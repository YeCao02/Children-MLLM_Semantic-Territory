import os
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import mannwhitneyu
from matplotlib.patches import Circle
from collections import Counter
import re
from pathlib import Path

def compute_group_stats(embeddings: np.ndarray) -> dict:
    centroid = embeddings.mean(axis=0)
    distances = np.linalg.norm(embeddings - centroid, axis=1)
    return {
        "centroid": centroid,
        "distances": distances,
        "ke": float(np.max(distances)) if len(distances) else np.nan,
        "mean_radius": float(np.mean(distances)) if len(distances) else np.nan,
        "p95_radius": float(np.percentile(distances, 95)) if len(distances) else np.nan,
    }

def bootstrap_metrics(children_embs: np.ndarray, ai_embs: np.ndarray, sample_n: int, bootstrap_n: int = 100, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    
    boot_children_ke = []
    boot_children_mean = []
    boot_children_p95 = []
    
    boot_ai_ke = []
    boot_ai_mean = []
    boot_ai_p95 = []
    
    boot_centroid_dist = []
    
    for i in range(bootstrap_n):
        # Sample with replacement
        idx_c = rng.choice(len(children_embs), size=sample_n, replace=True)
        idx_a = rng.choice(len(ai_embs), size=sample_n, replace=True)
        
        c_samp = children_embs[idx_c]
        a_samp = ai_embs[idx_a]
        
        c_stats = compute_group_stats(c_samp)
        a_stats = compute_group_stats(a_samp)
        
        boot_children_ke.append(c_stats["ke"])
        boot_children_mean.append(c_stats["mean_radius"])
        boot_children_p95.append(c_stats["p95_radius"])
        
        boot_ai_ke.append(a_stats["ke"])
        boot_ai_mean.append(a_stats["mean_radius"])
        boot_ai_p95.append(a_stats["p95_radius"])
        
        dist = np.linalg.norm(c_stats["centroid"] - a_stats["centroid"])
        boot_centroid_dist.append(dist)
        
    def get_ci(vals):
        return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]
        
    return {
        "children": {
            "ke_ci": get_ci(boot_children_ke),
            "mean_radius_ci": get_ci(boot_children_mean),
            "p95_radius_ci": get_ci(boot_children_p95),
        },
        "ai": {
            "ke_ci": get_ci(boot_ai_ke),
            "mean_radius_ci": get_ci(boot_ai_mean),
            "p95_radius_ci": get_ci(boot_ai_p95),
        },
        "centroid_distance_ci": get_ci(boot_centroid_dist)
    }

def extract_keywords_zh(texts: list, top_n: int = 15) -> list:
    try:
        import jieba
    except ImportError:
        domain_terms = [
            "安全", "危险", "实用", "观赏", "绿化", "互动", "涂鸦", "数字", "设施", "厕所",
            "店铺", "手机", "充电", "自行车", "电动车", "停车", "落叶", "护栏", "栏杆",
            "监控", "干净", "阴暗", "恐怖", "偏僻", "宽敞", "拥挤", "现实", "死板",
            "官方", "书面", "好玩", "友好", "吸引", "探索", "创意", "智能", "导览",
            "喷泉", "座椅", "长椅", "遮阳", "商场", "商业街", "图书馆", "学校", "马路",
            "交通", "生活", "日常", "游戏", "玩具", "空气清新",
        ]
        counter = Counter()
        for text in texts:
            text = str(text)
            for term in domain_terms:
                count = text.count(term)
                if count:
                    counter[term] += count
        return counter.most_common(top_n)
    
    # Aggressive stop words list based on user requests and common filler words
    zh_stops = {
        # Standard grammatical words / pronouns / adverbs
        "的", "了", "和", "是", "在", "我", "有", "也", "就", "不", "人", "都", "一", "一个", "上", "很", "能", "到", "说", "要", "去", "你", "会", "着",
        "没有", "看", "好", "自己", "这", "可以", "提供", "增加", "对于", "一些", "需要", "应该", "建议", "感觉", "觉得", "非常", "比较", "虽然",
        "这里", "同时", "中", "让", "可能", "具有", "作为", "更加", "考虑", "能够", "以及", "良好", "这个", "那个", "就是", "因为", "所以",
        "好像", "其实", "而且", "还是", "因此", "如果", "然后", "什么", "的话", "我们", "那种", "但是", "评价", "不是", "你们", "有点",
        "不会", "一个", "一般", "整体", "总体", "地方", "区域", "空间", "环境", "有些", "也是", "一样", "这样", "那样", "怎么", "怎么样",
        "哪些", "哪些地方", "整个", "许多", "目前", "已经", "进行", "进行城市", "辅助", "进行城市规划", "城市规划", "进行城市规划辅助",
        # Pronouns, demonstratives, locatives, and conversational fillers
        "他们", "我们", "你们", "大家", "自己", "那个", "这个", "这些", "那些", "这里", "那里", "那边", "这么", "那么", "怎么", "什么",
        "还有", "上面", "下面", "旁边", "旁边儿", "东西", "一点", "一块", "有点", "有些", "一样", "这样", "那样", "怎么说", "觉得",
        "感觉", "好像", "其实", "也就是说", "比如", "比如说", "或者", "还是", "但是", "然后", "然后的话", "的话", "那种", "不是",
        "不会", "不能", "不要", "不用", "不管", "以及", "而且", "所以", "因为", "虽然", "因此", "比如", "很多", "许多", "有些", "一点点",
        "一下", "看到", "一个", "一般", "整体", "总体",
        # Researcher Names / References to roles / context artifacts
        "yuxin", "gao", "yuxingao", "明明", "刘明明", "老师", "学生", "儿童", "小孩", "小朋友", "发言人", "研究者", "专家", "图片",
        "图中", "图像", "照片", "这张图", "这个图", "这个图片", "日常", "日常环境", "日常环境图片", "日常环境图片汇总", "日常环境图片汇总得分",
        "日常环境图片汇总得分统计表", "日常环境图片汇总得分统计表.csv", "日常环境图片汇总得分统计表.csv已", "日常环境图片汇总得分统计表.csv已生成",
        "日常环境图片汇总得分统计表.csv已生成图", "日常环境图片汇总得分统计表.csv已生成图3b", "日常环境图片汇总得分统计表.csv已生成图3b-style",
        "日常环境图片汇总得分统计表.csv已生成图3b-styleTSNE", "日常环境图片汇总得分统计表.csv已生成图3b-styleTSNE和",
        "日常环境图片汇总得分统计表.csv已生成图3b-styleTSNE和相似度", "日常环境图片汇总得分统计表.csv已生成图3b-styleTSNE和相似度小提琴图"
    }
    
    tokens = []
    for t in texts:
        words = list(jieba.cut(str(t)))
        for w in words:
            w = w.strip()
            # Lowercase the English words like 'yuxin' or 'gao' to catch them
            w_lower = w.lower()
            if len(w) > 1 and w not in zh_stops and w_lower not in zh_stops and not re.match(r'^\d+$', w):
                tokens.append(w)
                
    return Counter(tokens).most_common(top_n)

def main():
    base_dir = Path(__file__).resolve().parents[1]
    data_dir = os.path.join(base_dir, "data")
    fig_dir = os.path.join(data_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    
    # Configure publication style plots
    font_candidates = [
        base_dir / "SimHei.ttf",
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("C:/Windows/Fonts/NotoSansSC-VF.ttf"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    ]
    font_path = next((p for p in font_candidates if p.exists()), None)
    if font_path:
        import matplotlib.font_manager as fm
        fm.fontManager.addfont(str(font_path))
        font_name = fm.FontProperties(fname=str(font_path)).get_name()
        font_list = [font_name, "Arial", "Helvetica", "DejaVu Sans"]
    else:
        font_list = ["Arial", "Helvetica", "DejaVu Sans"]
        
    plt.rcParams.update({
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "font.family": "sans-serif",
        "font.sans-serif": font_list,
        "axes.labelsize": 11,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "axes.edgecolor": "black",
        "axes.linewidth": 1.1,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.unicode_minus": False
    })
    
    print("Loading computed embeddings...")
    with open(os.path.join(data_dir, "embeddings.pkl"), "rb") as f:
        # Note: We need to load from the latest generated embeddings
        # Wait, since the embeddings file contains "children" and "ai" embeddings, and we re-extracted the corpora,
        # we need to recompute embeddings before running this analysis!
        # Yes, we will re-run the embedding computation script after editing this analysis script!
        pass
        
    with open(os.path.join(data_dir, "embeddings.pkl"), "rb") as f:
        emb_data = pickle.load(f)
        
    children_texts = emb_data["children"]["texts"]
    children_embs = emb_data["children"]["embeddings"]
    ai_texts = emb_data["ai"]["texts"]
    ai_embs = emb_data["ai"]["embeddings"]
    
    print(f"Children embeddings shape: {children_embs.shape}")
    print(f"AI embeddings shape: {ai_embs.shape}")
    
    # Balanced sampling
    sample_n = min(len(children_embs), len(ai_embs))
    print(f"\nPerforming analysis with balanced sample size: {sample_n}")
    
    # Deterministic sampling for AI
    rng = np.random.default_rng(42)
    ai_indices = rng.choice(len(ai_embs), size=sample_n, replace=False)
    ai_embs_sampled = ai_embs[ai_indices]
    ai_texts_sampled = [ai_texts[i] for i in ai_indices]
    
    # 1. High-Dimensional Centroids & Metrics
    print("Computing high-dimensional statistics...")
    children_stats = compute_group_stats(children_embs)
    ai_stats = compute_group_stats(ai_embs_sampled)
    
    centroid_distance = float(np.linalg.norm(children_stats["centroid"] - ai_stats["centroid"]))
    print(f"High-Dim Centroid Distance: {centroid_distance:.4f}")
    print(f"Children Knowledge Extent (KE): {children_stats['ke']:.4f}")
    print(f"AI Knowledge Extent (KE): {ai_stats['ke']:.4f}")
    
    # Bootstrap CIs (100 iterations)
    print("Running bootstrap confidence intervals...")
    ci_results = bootstrap_metrics(children_embs, ai_embs_sampled, sample_n, bootstrap_n=100, seed=42)
    
    # 2. t-SNE Projection (2D)
    print("Computing t-SNE (2D)...")
    combined_embs = np.vstack([children_embs, ai_embs_sampled])
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, init="pca", learning_rate="auto")
    coords = tsne.fit_transform(combined_embs)
    
    children_coords = coords[:len(children_embs)]
    ai_coords = coords[len(children_embs):]
    
    ai_reason_coords = []
    ai_suggestion_coords = []
    for i, t in enumerate(ai_texts_sampled):
        if isinstance(t, dict) and t.get("type") == "reason":
            ai_reason_coords.append(ai_coords[i])
        elif isinstance(t, dict) and t.get("type") == "suggestion":
            ai_suggestion_coords.append(ai_coords[i])
            
    ai_reason_coords = np.array(ai_reason_coords) if ai_reason_coords else np.empty((0, 2))
    ai_suggestion_coords = np.array(ai_suggestion_coords) if ai_suggestion_coords else np.empty((0, 2))
    
    # Compute 2D centers and scaling
    center_c_2d = children_coords.mean(axis=0)
    center_a_2d = ai_coords.mean(axis=0)
    
    d2_c = np.linalg.norm(children_coords - center_c_2d, axis=1)
    d2_a = np.linalg.norm(ai_coords - center_a_2d, axis=1)
    
    p95_2d_c = np.percentile(d2_c, 95)
    p95_2d_a = np.percentile(d2_a, 95)
    
    scale_c = p95_2d_c / children_stats["p95_radius"]
    scale_a = p95_2d_a / ai_stats["p95_radius"]
    radius_scale = np.median([scale_c, scale_a])
    
    radius_c_2d = children_stats["ke"] * radius_scale
    radius_a_2d = ai_stats["ke"] * radius_scale
    
    # Plot Fig 3b-style t-SNE
    print("Plotting t-SNE layout (Hao et al. 2026 style)...")
    fig, ax = plt.subplots(figsize=(8, 7.5))
    
    # Plot points
    ax.scatter(children_coords[:, 0], children_coords[:, 1], s=20, alpha=0.6, color="#3498db", label=f"Child Interview (n={len(children_embs)})", edgecolors="none")
    if len(ai_reason_coords) > 0 and len(ai_suggestion_coords) > 0:
        ax.scatter(ai_reason_coords[:, 0], ai_reason_coords[:, 1], s=25, marker="o", alpha=0.6, color="#e74c3c", label=f"AI Reason (n={len(ai_reason_coords)})", edgecolors="none")
        ax.scatter(ai_suggestion_coords[:, 0], ai_suggestion_coords[:, 1], s=25, marker="^", alpha=0.6, color="#f39c12", label=f"AI Suggestion (n={len(ai_suggestion_coords)})", edgecolors="none")
    else:
        ax.scatter(ai_coords[:, 0], ai_coords[:, 1], s=20, alpha=0.6, color="#e74c3c", label=f"AI Baseline (n={len(ai_embs_sampled)})", edgecolors="none")
    
    # Plot centroids
    ax.scatter([center_c_2d[0]], [center_c_2d[1]], s=120, color="#1f77b4", edgecolors="black", linewidths=1.2, zorder=5)
    ax.scatter([center_a_2d[0]], [center_a_2d[1]], s=120, color="#d62728", edgecolors="black", linewidths=1.2, zorder=5)
    
    # Global center & arrows
    global_center_2d = coords.mean(axis=0)
    ax.scatter([global_center_2d[0]], [global_center_2d[1]], s=150, marker="X", color="black", label="Global Centroid", zorder=6)
    
    ax.annotate("", xy=(center_c_2d[0], center_c_2d[1]), xytext=(global_center_2d[0], global_center_2d[1]),
                arrowprops=dict(arrowstyle="->", lw=2, color="#1f77b4", alpha=0.8))
    ax.annotate("", xy=(center_a_2d[0], center_a_2d[1]), xytext=(global_center_2d[0], global_center_2d[1]),
                arrowprops=dict(arrowstyle="->", lw=2, color="#d62728", alpha=0.8))
    
    # Draw circles
    circle_c = Circle(center_c_2d, radius=radius_c_2d, fill=False, lw=1.5, ls="--", color="#3498db", alpha=0.8)
    circle_a = Circle(center_a_2d, radius=radius_a_2d, fill=False, lw=1.5, ls="--", color="#e74c3c", alpha=0.8)
    ax.add_patch(circle_c)
    ax.add_patch(circle_a)
    
    # Labels and layout
    ax.set_title("t-SNE Semantic Space Overlap\n(Qwen3-Embedding-8B)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("t-SNE Dimension 1")
    ax.set_ylabel("t-SNE Dimension 2")
    ax.grid(True, alpha=0.15)
    
    summary_text = (
        f"Centroid Distance: {centroid_distance:.3f} (95% CI: [{ci_results['centroid_distance_ci'][0]:.3f}, {ci_results['centroid_distance_ci'][1]:.3f}])\n"
        f"Child Interview KE: {children_stats['ke']:.3f} (95% CI: [{ci_results['children']['ke_ci'][0]:.3f}, {ci_results['children']['ke_ci'][1]:.3f}])\n"
        f"AI Baseline KE: {ai_stats['ke']:.3f} (95% CI: [{ci_results['ai']['ke_ci'][0]:.3f}, {ci_results['ai']['ke_ci'][1]:.3f}])"
    )
    fig.text(0.05, 0.02, summary_text, fontsize=9, family="monospace", bbox=dict(facecolor="white", alpha=0.8, edgecolor="#cccccc"))
    
    ax.legend(loc="upper right", frameon=False)
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    
    fig_tsne_path = os.path.join(fig_dir, "Fig3_tsne.png")
    fig.savefig(fig_tsne_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved t-SNE plot to {fig_tsne_path}")
    
    # 3. Cosine Similarity distributions
    print("Computing Cosine Similarities...")
    sim_cc = cosine_similarity(children_embs)
    sim_cc_vals = sim_cc[np.triu_indices(len(children_embs), k=1)]
    
    sim_aa = cosine_similarity(ai_embs_sampled)
    sim_aa_vals = sim_aa[np.triu_indices(len(ai_embs_sampled), k=1)]
    
    sim_ca_vals = cosine_similarity(children_embs, ai_embs_sampled).flatten()
    
    # Statistics
    print("Running Mann-Whitney U tests...")
    mw_cc_vs_ca = mannwhitneyu(sim_cc_vals, sim_ca_vals, alternative="greater")
    mw_aa_vs_ca = mannwhitneyu(sim_aa_vals, sim_ca_vals, alternative="greater")
    
    print(f"  Child-Child vs Between P-value: {mw_cc_vs_ca.pvalue}")
    print(f"  AI-AI vs Between P-value: {mw_aa_vs_ca.pvalue}")
    
    # Plot violin plot
    print("Plotting Violin distributions...")
    fig, ax = plt.subplots(figsize=(7, 5.5))
    
    sim_data = [sim_cc_vals, sim_aa_vals, sim_ca_vals]
    sim_labels = ["Child-Child\n(Within Group)", "AI-AI\n(Within Group)", "Child-AI\n(Between Group)"]
    
    sns.violinplot(data=sim_data, ax=ax, palette=["#3498db", "#e74c3c", "#9b59b6"], inner="quartile")
    ax.set_xticks(range(len(sim_labels)))
    ax.set_xticklabels(sim_labels)
    ax.set_ylabel("Cosine Similarity")
    ax.set_title("Cosine Similarity Distributions & Semantic Gap", fontsize=13, fontweight="bold", pad=12)
    
    sig_text = (
        f"Mann-Whitney U Test:\n"
        f"  Within Child vs Between: P {format_p_value(mw_cc_vs_ca.pvalue)}\n"
        f"  Within AI vs Between: P {format_p_value(mw_aa_vs_ca.pvalue)}"
    )
    ax.text(0.05, 0.05, sig_text, transform=ax.transAxes, fontsize=8.5, family="monospace",
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="#cccccc"))
    
    plt.tight_layout()
    fig_violin_path = os.path.join(fig_dir, "Fig3_violin.png")
    fig.savefig(fig_violin_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Violin plot to {fig_violin_path}")
    
    # 4. Top keywords & frequency bar chart
    print("Extracting keywords...")
    top_kws_c = extract_keywords_zh(children_texts, top_n=15)
    ai_strings = [t["text"] if isinstance(t, dict) else str(t) for t in ai_texts_sampled]
    top_kws_a = extract_keywords_zh(ai_strings, top_n=15)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Children keywords
    words_c, freq_c = zip(*top_kws_c)
    y_pos_c = np.arange(len(words_c))
    ax1.barh(y_pos_c, freq_c, align='center', color='#3498db', alpha=0.8)
    ax1.set_yticks(y_pos_c)
    ax1.set_yticklabels(words_c, fontsize=10)
    ax1.invert_yaxis()  # top-down
    ax1.set_xlabel('Frequency')
    ax1.set_title('Top Children Interview Keywords\n(Chinese Corpus)', fontsize=11, fontweight='bold')
    
    # AI keywords
    words_a, freq_a = zip(*top_kws_a)
    y_pos_a = np.arange(len(words_a))
    ax2.barh(y_pos_a, freq_a, align='center', color='#e74c3c', alpha=0.8)
    ax2.set_yticks(y_pos_a)
    ax2.set_yticklabels(words_a, fontsize=10)
    ax2.invert_yaxis()  # top-down
    ax2.set_xlabel('Frequency')
    ax2.set_title('Top AI Evaluation Keywords\n(Chinese Corpus)', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    fig_kws_path = os.path.join(fig_dir, "Fig3_keywords.png")
    fig.savefig(fig_kws_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Keywords plot to {fig_kws_path}")
    
    # 5. Save quantitative summary JSON
    summary_data = {
        "sample_size": sample_n,
        "centroid_distance": centroid_distance,
        "centroid_distance_ci": ci_results["centroid_distance_ci"],
        "children": {
            "ke": children_stats["ke"],
            "ke_ci": ci_results["children"]["ke_ci"],
            "mean_radius": children_stats["mean_radius"],
            "mean_radius_ci": ci_results["children"]["mean_radius_ci"],
            "p95_radius": children_stats["p95_radius"],
            "p95_radius_ci": ci_results["children"]["p95_radius_ci"],
            "top_keywords": [{"word": w, "count": c} for w, c in top_kws_c]
        },
        "ai": {
            "ke": ai_stats["ke"],
            "ke_ci": ci_results["ai"]["ke_ci"],
            "mean_radius": ai_stats["mean_radius"],
            "mean_radius_ci": ci_results["ai"]["mean_radius_ci"],
            "p95_radius": ai_stats["p95_radius"],
            "p95_radius_ci": ci_results["ai"]["p95_radius_ci"],
            "top_keywords": [{"word": w, "count": c} for w, c in top_kws_a]
        },
        "statistical_tests": {
            "within_child_vs_between": {
                "u_statistic": float(mw_cc_vs_ca.statistic),
                "p_value": float(mw_cc_vs_ca.pvalue)
            },
            "within_ai_vs_between": {
                "u_statistic": float(mw_aa_vs_ca.statistic),
                "p_value": float(mw_aa_vs_ca.pvalue)
            }
        }
    }
    
    json_path = os.path.join(data_dir, "analysis_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    print(f"Saved quantitative summary to {json_path}")
    print("\nStage 3: Semantic Analysis complete.")

def format_p_value(p: float) -> str:
    if p < 0.001:
        return "< 0.001"
    elif p < 0.01:
        return "< 0.01"
    else:
        return f"= {p:.3f}"

if __name__ == "__main__":
    main()
