from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
FIG_DIR = DATA_DIR / "figures_v2"


def image_uri(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("utf-8")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def metric(value: float | int | str, digits: int = 3) -> str:
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def directional_coverage_explanation(cov_nn: dict, tau_065: dict) -> str:
    return f"""
      <h3>分析问题与方向定义</h3>
      <p>该图检验两组语料在原始 4096 维 Qwen3 嵌入空间中是否能够对称地覆盖彼此。对每条儿童文本，<strong>儿童文本被 AI 覆盖</strong>是它与 220 条 AI 图像级文本中最相似一条的余弦相似度；对每条 AI 文本，<strong>AI 文本被儿童覆盖</strong>是它与 331 条儿童文本中最相似一条的余弦相似度。两者分母不同，回答的是两个方向不同的问题。</p>
      <h3>A-D 四个子图如何阅读</h3>
      <ul>
        <li><strong>A，最近邻相似度分布：</strong>AI-to-child 分布更集中且整体右移，均值为 {cov_nn['ai_covered_by_children']['mean']:.3f}；child-to-AI 均值为 {cov_nn['children_covered_by_ai']['mean']:.3f}。均值差为 {cov_nn['ai_minus_children_mean_similarity']:.3f}（95% CI：{cov_nn['ai_minus_children_mean_similarity_ci95'][0]:.3f}-{cov_nn['ai_minus_children_mean_similarity_ci95'][1]:.3f}）。这表示典型 AI 文本在儿童语料中找到的近邻，比典型儿童文本在 AI 语料中找到的近邻更相似。</li>
        <li><strong>B，阈值覆盖曲线：</strong>只有当最佳跨组匹配达到阈值 tau 时才记为“被覆盖”。在有解释价值的阈值区间内，红线持续高于蓝线，说明方向差异并非由单一阈值偶然造成。</li>
        <li><strong>C，关键阈值比较：</strong>在 tau = 0.65 时，{tau_065['ai_covered_by_children']:.1%} 的 AI 文本能找到合格的儿童近邻，但仅 {tau_065['children_covered_by_ai']:.1%} 的儿童文本能找到合格的 AI 近邻。主分析保留全部 331 条儿童文本和 220 条 AI 文本，没有进行平衡抽样。</li>
        <li><strong>D，方向不对称：</strong>纵轴是“AI 文本被儿童覆盖率”减去“儿童文本被 AI 覆盖率”。正值表示儿童语料接纳 AI 文本的能力高于 AI 语料接纳儿童文本的能力；tau = 0.65 时差值为 {tau_065['ai_minus_children_coverage']:.1%}。</li>
      </ul>
      <h3>实质性结论</h3>
      <p>当前证据支持的表述是：<strong>AI 输出常落入儿童已经表达过的共享语义核心，但儿童具有大量未被 AI 重现的情境化语义变化。</strong>简言之，AI 覆盖了儿童经验中的部分共享核心，而儿童语料还保留了显著的 AI 未匹配语义长尾。</p>
      <p class="boundary"><strong>推断边界：</strong>这是方向性最近邻覆盖，不等于证明一个语义集合在几何上完全包含另一个集合，也不表示儿童理解或认同 AI 文本。结果还可能受到诱发任务、文本长度、AI 基于图片回答而儿童基于访谈回答等形式差异的影响。</p>
    """


def semantic_frontier_explanation(pr_primary: dict, pr_best: dict, local_nn: dict) -> str:
    return f"""
      <h3>该图比第 4 节多回答了什么</h3>
      <p>第 4 节确认方向不对称；本图进一步检验这种重叠是<strong>广泛分散</strong>在许多文本上，还是由少量高度通用的“接收文本”造成。这里借用生成文本评价中的 precision/recall 思路：<strong>AI semantic precision</strong>是至少接近一条儿童文本的 AI 输出比例；<strong>child semantic recall</strong>是至少被一条 AI 输出找回的儿童文本比例。</p>
      <h3>A-D 四个子图如何阅读</h3>
      <ul>
        <li><strong>A，语义前沿：</strong>每个点对应一个相似度阈值。tau = 0.65 时，AI semantic precision 为 {pr_primary['ai_semantic_precision']:.1%}，child semantic recall 仅为 {pr_primary['child_semantic_recall']:.1%}。这是“高 precision、低 recall”结构：AI 通常没有离开儿童可识别的语义区域，但只重现了儿童语料中有限的一部分。</li>
        <li><strong>B，阈值敏感性：</strong>F1-like 调和均值和 precision-minus-recall 差值用于观察结论如何随 tau 改变。扫描范围中的表面最佳平衡位于 tau = {pr_best['threshold']:.2f}，precision 为 {pr_best['ai_semantic_precision']:.1%}、recall 为 {pr_best['child_semantic_recall']:.1%}；但该阈值较宽松，只能作为敏感性参考，不能取代主阈值证据。</li>
        <li><strong>C，最近邻分配集中度：</strong>曲线表示最近邻分配如何累积到接收文本上。曲线长期贴近零、到最右端才陡升，意味着大多数匹配被极少数文本吸收，而不是均匀分布在整组语料中。</li>
        <li><strong>D，hubness 摘要：</strong>{local_nn['children_as_receivers_for_ai']['zero_receiver_share']:.1%} 的儿童文本没有接收到任何 AI 最近邻分配；儿童接收文本中的前 10% 吸收了 {local_nn['children_as_receivers_for_ai']['top10_assignment_share']:.1%} 的 AI 分配（Gini = {local_nn['children_as_receivers_for_ai']['gini']:.3f}）。反方向上，{local_nn['ai_as_receivers_for_children']['zero_receiver_share']:.1%} 的 AI 文本没有接收到儿童分配，前 10% 吸收了 {local_nn['ai_as_receivers_for_children']['top10_assignment_share']:.1%}。互为最近邻的配对仅有 {local_nn['mutual_nearest_neighbor_pairs']} 对，占 AI 输出的 {local_nn['mutual_pair_share_of_ai']:.1%}。</li>
      </ul>
      <h3>与第 4 节合并解释</h3>
      <p>“70.0% 的 AI 文本被儿童覆盖”不能解读成“AI 表达了儿童 70.0% 的语义领地”。它只表示 70.0% 的 AI 文本能找到至少一个合格的儿童近邻。由于这些匹配高度集中在少数儿童文本上，更准确的结构是：<strong>双方存在狭窄的共享语义核心，但儿童一侧具有更宽的独特语义长尾</strong>，而不是两组文本广泛、均匀地双向一致。</p>
      <p class="boundary"><strong>术语与方法边界：</strong>semantic precision/recall 是描述性类比，不是监督分类中的 precision/recall；F1-like 只是报告辅助量，不是经验证的分类器 F1。hubness 可能被高维嵌入几何、重复模板和文本粒度差异放大，因此在把“共享核心”解释成具体儿童经验之前，还应人工审阅代表性的 hub 文本及其最近邻。</p>
    """


def figure_card(title: str, path: Path, caption: str) -> str:
    return f"""<section class="card figure-card">
      <h2>{title}</h2>
      <img src="{image_uri(path)}" alt="{title}">
      <div class="caption">{caption}</div>
    </section>"""


def related_rows() -> str:
    rows = [
        (
            "Hao et al. (2026), Nature",
            "高维科学文本空间中的 knowledge extent 与 t-SNE 解释图。",
            "直接支撑本文做法：正式指标在原始嵌入空间计算，t-SNE 只作解释性可视化。",
            "https://www.nature.com/articles/s41586-025-09922-y",
        ),
        (
            "Pillutla et al. (2021), NeurIPS, MAUVE",
            "用嵌入空间中的 divergence frontier 衡量机器文本与人类文本的分布差异。",
            "支撑“生成文本与人类文本的差距不能只看一个平均相似度”，应同时讨论质量、覆盖和多样性。",
            "https://proceedings.neurips.cc/paper/2021/hash/260c2432a0eecc28ce03c10dadc078a4-Abstract.html",
        ),
        (
            "Le Bronnec et al. (2024), ACL",
            "用 precision / recall 思路评估 LLM 文本质量与多样性。",
            "直接启发本文的 semantic precision / child semantic recall 前沿图。",
            "https://aclanthology.org/2024.acl-long.616/",
        ),
        (
            "Guo et al. (2025), TACL",
            "从词汇、句法和语义层面评估 LLM 的语言多样性。",
            "提示后续可把当前语义覆盖扩展为多层语言多样性分析。",
            "https://aclanthology.org/2025.tacl-1.69/",
        ),
        (
            "Boggust et al. (2022), ACM IUI",
            "Embedding Comparator 同时展示全局投影和局部近邻结构。",
            "启发本文新增局部近邻集中度与 hubness 诊断，避免只看全局散点图。",
            "https://dl.acm.org/doi/10.1145/3490099.3511122",
        ),
        (
            "Zhong et al. (2024), ACM ICMI",
            "比较多模态语言模型与人类感知描述的对齐关系。",
            "为 MLLM 输出与人类感知文本的语义相似度和 t-SNE 表达提供相近先例。",
            "https://dl.acm.org/doi/10.1145/3678957.3685756",
        ),
        (
            "Woloszyn and Gagl (2025), arXiv",
            "比较 LLM 图片描述与儿童图片描述。",
            "支撑“儿童表达不是成人或 AI 语言的简化版”，AI 可能错过儿童特有语义模式。",
            "https://arxiv.org/abs/2508.13769",
        ),
        (
            "Wedyan et al. (2025), PLOS ONE",
            "比较 GPT-4o 与人类的城市步行性感知。",
            "可借鉴“部分维度对齐，但在人类情境、舒适性和主题多样性上偏离”的表达。",
            "https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0322078",
        ),
        (
            "Malekzadeh et al. (2025), CEUS",
            "比较 ChatGPT 与人类对街景城市吸引力的判断。",
            "可用于定位 AI-人类城市视觉感知差异及其情境性不足。",
            "https://doi.org/10.1016/j.compenvurbsys.2024.102243",
        ),
    ]
    return "\n".join(
        f'<tr><td><a href="{url}">{title}</a></td><td>{method}</td><td>{use}</td></tr>'
        for title, method, use, url in rows
    )


def build_html() -> Path:
    v2 = read_json(DATA_DIR / "analysis_results_v2.json")
    hao = read_json(DATA_DIR / "hao_style_metrics.json")
    affect = read_json(DATA_DIR / "affective_alignment_metrics.json")
    coverage = read_json(DATA_DIR / "semantic_coverage_metrics.json")
    full = read_json(DATA_DIR / "full_report_metrics.json")

    figs = {
        "coverage": FIG_DIR / "Fig_v2_semantic_coverage_curves.png",
        "frontier": FIG_DIR / "Fig_v2_semantic_pr_frontier.png",
        "coverage_tsne": FIG_DIR / "Fig_v2_tsne_coverage_status.png",
        "hao": FIG_DIR / "Fig_v2_hao_style_ai_children.png",
        "traditional": DATA_DIR / "figures" / "Fig3_tsne.png",
        "diagnostic": FIG_DIR / "Fig_v2_tsne_diagnostic.png",
        "extent": FIG_DIR / "Fig_v2_highdim_extent.png",
        "entropy": FIG_DIR / "Fig_v2_semantic_entropy.png",
        "heatmap": FIG_DIR / "Fig_v2_centroid_heatmap.png",
        "keywords": FIG_DIR / "Fig_v2_keyword_contrast.png",
        "violin": DATA_DIR / "figures" / "Fig3_violin.png",
        "affect_profiles": DATA_DIR / "figures_affect" / "Fig_affect_main_profiles.png",
        "affect_robustness": DATA_DIR / "figures_affect" / "Fig_affect_model_robustness.png",
        "semantic_affective": DATA_DIR / "figures_affect" / "Fig_semantic_affective_alignment.png",
    }

    audit = v2["audit"]
    child_env = v2["metrics"]["child_env_scene"]
    ai_combined = v2["metrics"]["ai_combined"]
    cov_nn = coverage["nearest_neighbor_summary"]
    tau_065 = next(row for row in coverage["coverage_by_threshold"] if abs(row["threshold"] - 0.65) < 1e-9)
    tau_626 = next(row for row in coverage["coverage_by_threshold"] if abs(row["threshold"] - 0.626) < 1e-9)
    pr_primary = coverage["precision_recall_frontier"]["primary_tau_065"]
    pr_best = coverage["precision_recall_frontier"]["best_f1_like"]
    local_nn = coverage["local_neighborhood"]
    affect_main = affect["main_model"]
    affect_robust = affect["robustness_model"]
    main_compare = affect_main["comparisons"]["ai_combined"]
    robust_compare = affect_robust["comparisons"]["ai_combined"]
    agreement = affect["cross_model_directional_agreement"]["ai_combined"]
    alignment = affect["semantic_affective_alignment"]
    entropy = full["entropy_summary"]
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

    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>儿童-MLLM 语义领地分析报告（中文版）</title>
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
    h1 {{ margin: 0 0 8px; font-size: 31px; }}
    h2 {{ margin: 0 0 12px; font-size: 21px; }}
    p, li {{ line-height: 1.75; }}
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
    .caption h3 {{ color: var(--ink); font-size: 16px; margin: 16px 0 6px; }}
    .caption p {{ margin: 6px 0; }}
    .caption ul {{ margin: 6px 0 8px; padding-left: 22px; }}
    .caption li {{ margin: 4px 0; }}
    .caption .boundary {{ color: var(--ink); background: #f1f5f8; border-left: 4px solid #7b8794; padding: 9px 11px; margin-top: 12px; }}
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
    <h1>儿童-MLLM 语义领地分析报告</h1>
    <p class="lead">本报告以 Hao 等（2026）的知识广度可视化逻辑为参照，比较儿童真实环境感知文本与 MLLM 生成的儿童视角评价文本。嵌入模型为 Qwen3-Embedding-8B。</p>
  </header>

  <section class="card">
    <h2>1. 核心结论</h2>
    <p class="note">主发现来自 Qwen3-Embedding-8B 的 4096 维原始语义空间，而不是来自 t-SNE 的二维肉眼判断。t-SNE 只用于解释性展示；语义覆盖、语义广度和最近邻相似度均在高维空间中计算。</p>
    <ul>
      <li><span class="pill green">方向性覆盖</span> AI 文本更容易在儿童真实表达中找到语义近邻：AI-to-child 最近邻均值为 {cov_nn['ai_covered_by_children']['mean']:.3f}，child-to-AI 最近邻均值为 {cov_nn['children_covered_by_ai']['mean']:.3f}。</li>
      <li>在阈值 tau = 0.65 时，{tau_065['ai_covered_by_children']:.1%} 的 AI 文本能被至少一条儿童文本覆盖，但只有 {tau_065['children_covered_by_ai']:.1%} 的儿童文本能被 AI 覆盖。因此更稳妥的表述是：<strong>AI 更容易被匹配进儿童语义领地，但 AI 不能反向覆盖儿童表达的广度</strong>。</li>
      <li><span class="pill blue">儿童</span> 在 shared centroid 周围拥有更大的语义广度：儿童 KE 为 {hao['children_ke_joint_centroid']:.3f}，AI KE 为 {hao['ai_ke_joint_centroid']:.3f}。儿童表达更偏向具体使用、危险感、出行、学校/商业街语境和日常约束。</li>
      <li><span class="pill red">AI</span> 更偏向标准化规划话语，例如智能设施、创意空间、空气清新、探索、互动装置、绿化和泛化的儿童友好设计模板。</li>
      <li><strong>情感一致性模块</strong> 目前仅作为探索性补充。它提示 AI 更中性、较少表达担忧/追问，但在人工验证之前，不应作为主发现。</li>
    </ul>
  </section>

  <section class="card">
    <h2>2. 关键数据与高维指标</h2>
    <div class="grid metrics">
      <div class="metric"><div class="label">儿童原始片段</div><div class="value">{audit['children_segments']}</div></div>
      <div class="metric"><div class="label">儿童直接感知片段</div><div class="value">{audit['children_direct_segments']}</div></div>
      <div class="metric"><div class="label">儿童元评价片段</div><div class="value">{audit['children_meta_segments']}</div></div>
      <div class="metric"><div class="label">AI 图像级输出</div><div class="value">{audit['ai_image_units']}</div></div>
      <div class="metric"><div class="label">儿童 KE</div><div class="value">{metric(hao['children_ke_joint_centroid'])}</div></div>
      <div class="metric"><div class="label">AI KE</div><div class="value">{metric(hao['ai_ke_joint_centroid'])}</div></div>
      <div class="metric"><div class="label">质心距离</div><div class="value">{metric(hao['centroid_distance'])}</div></div>
      <div class="metric"><div class="label">child-to-AI NN 均值</div><div class="value">{cov_nn['children_covered_by_ai']['mean']:.3f}</div></div>
      <div class="metric"><div class="label">AI-to-child NN 均值</div><div class="value">{cov_nn['ai_covered_by_children']['mean']:.3f}</div></div>
      <div class="metric"><div class="label">儿童覆盖率 tau=.65</div><div class="value">{tau_065['children_covered_by_ai']:.1%}</div></div>
      <div class="metric"><div class="label">AI 覆盖率 tau=.65</div><div class="value">{tau_065['ai_covered_by_children']:.1%}</div></div>
      <div class="metric"><div class="label">儿童担忧/困扰线索</div><div class="value">{metric(child_concern)}</div></div>
      <div class="metric"><div class="label">AI 担忧/困扰线索</div><div class="value">{metric(ai_concern)}</div></div>
      <div class="metric"><div class="label">儿童中性表达</div><div class="value">{metric(child_neutral)}</div></div>
      <div class="metric"><div class="label">AI 中性表达</div><div class="value">{metric(ai_neutral)}</div></div>
    </div>
    <table>
      <thead><tr><th>指标视角</th><th>儿童</th><th>AI</th><th>解释</th></tr></thead>
      <tbody>
        <tr><td>固定样本主分析</td><td>KE {hao['children_ke_joint_centroid']:.3f}; n={hao['children_n']}</td><td>KE {hao['ai_ke_joint_centroid']:.3f}; n={hao['ai_n']}</td><td>主分析保留 331 条儿童直接感知文本和 220 条 AI 图像级合并输出，不进行平衡抽样。</td></tr>
        <tr><td>保守场景级审计</td><td>KE {child_env['ke']:.3f}; p95 {child_env['p95_radius']:.3f}</td><td>KE {ai_combined['ke']:.3f}; p95 {ai_combined['p95_radius']:.3f}</td><td>场景聚合会压缩儿童侧差异，适合作为敏感性检验，而不是唯一结论。</td></tr>
        <tr><td>最近邻重叠</td><td>child-to-AI 均值 {cov_nn['children_covered_by_ai']['mean']:.3f}; 中位数 {cov_nn['children_covered_by_ai']['median']:.3f}</td><td>AI-to-child 均值 {cov_nn['ai_covered_by_children']['mean']:.3f}; 中位数 {cov_nn['ai_covered_by_children']['median']:.3f}</td><td>AI 文本通常能找到儿童侧近邻，但很多儿童表达难以找到同等相似的 AI 对应物。</td></tr>
        <tr><td>tau = 0.65 覆盖率</td><td>{tau_065['children_covered_by_ai']:.1%} 被 AI 覆盖</td><td>{tau_065['ai_covered_by_children']:.1%} 被儿童覆盖</td><td>这是方向性覆盖结论最直接的数值依据。</td></tr>
        <tr><td>tau = 0.626 覆盖率</td><td>{tau_626['children_covered_by_ai']:.1%} 被 AI 覆盖</td><td>{tau_626['ai_covered_by_children']:.1%} 被儿童覆盖</td><td>在语义-情感图使用的中位阈值下，方向性结论仍保持一致。</td></tr>
      </tbody>
    </table>
  </section>

  <section class="card">
    <h2>3. 从 Hao 等（2026）借鉴的方法逻辑</h2>
    <p>Hao 等把论文嵌入到高维科学文本空间，在原始空间计算 knowledge extent，再用 t-SNE 进行解释性展示。本文将同一逻辑迁移到儿童真实表达与 AI 生成文本的比较中。</p>
    <table>
      <thead><tr><th>可迁移元素</th><th>本文中的对应做法</th></tr></thead>
      <tbody>
        <tr><td>高维嵌入空间</td><td>Hao 等使用 SPECTER 2.0 的 768 维科学文本向量；本文使用 Qwen3-Embedding-8B 的 4096 维中文语义向量。</td></tr>
        <tr><td>知识/语义广度</td><td>语义广度不由 t-SNE 面积决定，而是在原始嵌入空间中围绕质心计算。</td></tr>
        <tr><td>固定样本主比较</td><td>由于儿童文本与 AI 图像级输出并非同构样本，主分析保留 331/220 的实际样本规模，不做平衡抽样。</td></tr>
        <tr><td>方向性覆盖</td><td>AI-to-child 和 child-to-AI 是不同问题，必须分别报告。</td></tr>
        <tr><td>t-SNE 的定位</td><td>t-SNE 只作解释性图示，不能作为覆盖和广度结论的正式证据。</td></tr>
      </tbody>
    </table>
  </section>

  {figure_card(
      "4. 高维方向性语义覆盖",
      figs["coverage"],
      directional_coverage_explanation(cov_nn, tau_065)
  )}

  {figure_card(
      "5. Precision/Recall 风格的语义覆盖前沿",
      figs["frontier"],
      semantic_frontier_explanation(pr_primary, pr_best, local_nn)
  )}

  {figure_card(
      "6. 用 4096D 覆盖状态标注的 t-SNE",
      figs["coverage_tsne"],
      "颜色标签来自 4096 维原始空间的覆盖状态，而不是 t-SNE 坐标本身。该图说明 t-SNE 适合辅助检查，但不应作为正式推断依据。"
  )}

  {figure_card(
      "7. 仿 Hao Fig. 3b 的语义领地图",
      figs["hao"],
      "该图保留 Hao 等 Fig. 3b 的直观表达方式：点位由 t-SNE 放置，圆形边界和 KE 指标来自高维语义空间。"
  )}

  <div class="grid two">
    {figure_card("8. 传统 t-SNE 图", figs["traditional"], "保留传统 t-SNE 图用于对照。它能展示粗略聚类，但不能解释覆盖方向和高维广度。")}
    {figure_card("9. 文本类型诊断 t-SNE", figs["diagnostic"], "AI reason 与 suggestion 会形成不同文本簇，说明 AI 输出内部也存在任务/文本类型效应。")}
  </div>

  <div class="grid two">
    {figure_card("10. 高维语义广度敏感性图", figs["extent"], "该图展示 KE、p95 半径和平均半径在不同文本类型间的变化。主覆盖分析仍以固定 331/220 样本为准。")}
    {figure_card("11. 探索性语义熵", figs["entropy"], f"借鉴 Hao 等的 entropy 思路，在 PCA-10D 网格中估计语义集中度。中位熵：儿童 {entropy['Children direct']['median']:.3f}，AI {entropy['AI combined']['median']:.3f}。")}
  </div>

  <div class="grid two">
    {figure_card("12. 质心距离矩阵", figs["heatmap"], "用于区分儿童直接环境感知、儿童对 AI/专家文本的元评价，以及 AI reason/suggestion/combined 输出。")}
    {figure_card("13. 差异性语义锚点", figs["keywords"], "关键词对比帮助解释嵌入空间差异具体对应哪些主题。")}
  </div>

  {figure_card("14. 成对相似度分布", figs["violin"], "组内和组间余弦相似度可作为辅助证据，但不应替代高维方向性覆盖。")}

  <section class="card">
    <h2>15. 探索性语义-情感模块</h2>
    <p class="note">该模块只作为探索性补充。它回答的是另一个问题：即便 AI 和儿童提到相似空间特征，它们是否以类似的担忧、参与、追问和中性描述方式表达？</p>
    <table>
      <thead><tr><th>证据</th><th>结果</th><th>解释边界</th></tr></thead>
      <tbody>
        <tr><td>中文情感模型</td><td>质心距离 {main_compare['centroid_distance']:.3f}; 置换检验 P {main_compare['permutation_p']:.4g}; JS divergence {main_compare['js_divergence']:.3f}</td><td>AI combined 文本更中性，担忧/困扰和追问/惊讶线索更少。</td></tr>
        <tr><td>多语鲁棒模型</td><td>质心距离 {robust_compare['centroid_distance']:.3f}; 置换检验 P {robust_compare['permutation_p']:.4g}; JS divergence {robust_compare['js_divergence']:.3f}</td><td>尽管标签体系不同，AI-儿童差异方向基本复现。</td></tr>
        <tr><td>跨模型方向一致性</td><td>符号一致率 {agreement['sign_agreement']:.0%}; delta cosine {agreement['delta_cosine_similarity']:.3f}</td><td>两个模型对总体方向较一致，但细粒度情绪标签仍需人工验证。</td></tr>
        <tr><td>语义-情感象限</td><td>AI semantic-only {alignment['quadrant_proportions']['ai_combined']['semantic_only']:.1%}; child distinctive-both {alignment['quadrant_proportions']['children_direct']['distinctive_both']:.1%}</td><td>该图适合生成假设，正式覆盖结论仍应回到第 4 节高维最近邻结果。</td></tr>
      </tbody>
    </table>
    <p class="caption">这些结果描述的是文本中的情感线索，不是儿童内在情绪、心理健康或稳定人格特质。</p>
  </section>

  {figure_card("16. 探索性情感轮廓", figs["affect_profiles"], "儿童直接感知和元评价包含更多样的情感线索；AI suggestion 基本呈现为标准化、中性建议语言。")}

  <div class="grid two">
    {figure_card("17. 双模型鲁棒性与面向效度", figs["affect_robustness"], "左图比较四个共同维度上的 AI-minus-child 差异方向；右图用透明中文锚句暴露模型互补弱点。")}
    {figure_card("18. 语义-情感对齐图", figs["semantic_affective"], f"语义重叠和情感重叠分开计算。child-to-AI 平均语义重叠为 {alignment['group_means']['children_direct']['semantic_overlap']:.3f}，AI-to-child 为 {alignment['group_means']['ai_combined']['semantic_overlap']:.3f}。该图应视为探索性桥接图。")}
  </div>

  <section class="card">
    <h2>19. 从相关文献进一步优化了什么</h2>
    <p class="note">第 18 节参考文献最重要的启发不是“多画 t-SNE”，而是把高维分布指标、方向性 precision/recall、局部近邻结构和低维解释图组合起来。</p>
    <table>
      <thead><tr><th>文献来源</th><th>对现有分析的启发</th><th>本次已落实或后续可做</th></tr></thead>
      <tbody>
        <tr><td>MAUVE / 机器文本-人类文本分布差异</td><td>不能只看平均相似度，应同时讨论分布差异、质量和多样性。</td><td>已新增 precision/recall 风格的语义覆盖前沿；后续可加入真正的 divergence frontier 或 MAUVE-like 指标。</td></tr>
        <tr><td>LLM precision / recall 评估</td><td>AI 是否落入儿童语义域，与 AI 是否覆盖儿童语义域，是两个方向不同的问题。</td><td>已将 AI semantic precision 与 child semantic recall 分开报告。</td></tr>
        <tr><td>Embedding Comparator</td><td>全局投影不足以说明局部近邻关系。</td><td>已新增局部近邻集中度和 hubness 诊断。</td></tr>
        <tr><td>LLM 语言多样性研究</td><td>语义空间只是多样性的一层。</td><td>已有语义熵和成对相似度；后续可加入词汇、句法、叙事结构多样性。</td></tr>
        <tr><td>AI-人类城市感知研究</td><td>AI 可能在抽象维度对齐，但在人类经验、情境和地方性上偏离。</td><td>后续应加入图像级配对、空间类型分组和人工维度编码。</td></tr>
      </tbody>
    </table>
  </section>

  <section class="card">
    <h2>20. 可引用参考文献与借鉴点</h2>
    <table>
      <thead><tr><th>参考文献</th><th>相关方法或发现</th><th>本文如何借鉴</th></tr></thead>
      <tbody>{related_rows()}</tbody>
    </table>
  </section>

  <section class="card">
    <h2>21. 下一步优先分析</h2>
    <ol>
      <li><strong>图像级配对误差：</strong>建立 <code>image_id -> children_comments -> AI_output</code> 表，计算每张图的质心距离和最近邻错位。</li>
      <li><strong>维度级覆盖：</strong>人工或半自动标注安全、实用性、可达性、游戏、绿化、数字设施、社会监督和日常约束，比较每个维度的覆盖率。</li>
      <li><strong>空间类型鲁棒性：</strong>按学校、商业街、公园、街道、滨水、交通节点等类型分组，复刻 Hao 的 subfield 思路。</li>
      <li><strong>模板集中度：</strong>定量分析 AI 对“智能设施、互动装置、绿化、空气清新、创意”等模板化表达的重复依赖。</li>
      <li><strong>情感人工验证：</strong>抽样双人编码担忧/困扰、积极参与、追问/惊讶和中性描述，报告一致性后再提升情感模块的证据等级。</li>
    </ol>
  </section>
</main>
</body>
</html>
"""

    out = BASE_DIR / "Children_MLLM_Semantic_Territory_Full_Report_ZH.html"
    out.write_text(html, encoding="utf-8")
    return out


def main() -> None:
    out = build_html()
    print(f"Saved Chinese semantic territory report: {out}")


if __name__ == "__main__":
    main()
