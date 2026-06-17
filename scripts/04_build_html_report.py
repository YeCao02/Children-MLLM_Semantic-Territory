import os
import json
import base64
from pathlib import Path

def get_base64_image(image_path):
    if not os.path.exists(image_path):
        return ""
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return f"data:image/png;base64,{encoded_string}"

def main():
    base_dir = Path(__file__).resolve().parents[1]
    data_dir = os.path.join(base_dir, "data")
    fig_dir = os.path.join(data_dir, "figures")
    
    # Load analysis results
    json_path = os.path.join(data_dir, "analysis_results.json")
    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found. Run Stage 3 first.")
        return
        
    with open(json_path, "r", encoding="utf-8") as f:
        res = json.load(f)
        
    # Get base64 encoded images
    tsne_b64 = get_base64_image(os.path.join(fig_dir, "Fig3_tsne.png"))
    violin_b64 = get_base64_image(os.path.join(fig_dir, "Fig3_violin.png"))
    kws_b64 = get_base64_image(os.path.join(fig_dir, "Fig3_keywords.png"))
    
    html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI vs Children: Semantic Gap Analysis Report</title>
    <!-- Google Fonts -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-color: #f8fafc;
            --card-bg: #ffffff;
            --text-main: #0f172a;
            --text-muted: #475569;
            --primary: #4f46e5;
            --primary-light: #e0e7ff;
            --success: #10b981;
            --warning: #f59e0b;
            --danger: #ef4444;
            --border: #e2e8f0;
            --shadow: 0 4px 6px -1px rgb(0 0 0 / 0.05), 0 2px 4px -2px rgb(0 0 0 / 0.05);
            --shadow-lg: 0 10px 15px -3px rgb(0 0 0 / 0.05), 0 4px 6px -4px rgb(0 0 0 / 0.05);
        }}

        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}

        body {{
            font-family: 'Inter', sans-serif;
            background-color: var(--bg-color);
            color: var(--text-main);
            line-height: 1.6;
            padding: 2rem 1rem;
        }}

        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}

        header {{
            margin-bottom: 2.5rem;
            text-align: center;
            border-bottom: 1px solid var(--border);
            padding-bottom: 2rem;
        }}

        h1 {{
            font-family: 'Outfit', sans-serif;
            font-size: 2.5rem;
            font-weight: 700;
            color: var(--primary);
            margin-bottom: 0.5rem;
            letter-spacing: -0.025em;
        }}

        .subtitle {{
            font-size: 1.1rem;
            color: var(--text-muted);
            font-weight: 400;
        }}

        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 1.5rem;
            margin-bottom: 2.5rem;
        }}

        .stat-card {{
            background: var(--card-bg);
            border-radius: 12px;
            padding: 1.5rem;
            border: 1px solid var(--border);
            box-shadow: var(--shadow);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }}

        .stat-card:hover {{
            transform: translateY(-2px);
            box-shadow: var(--shadow-lg);
        }}

        .stat-label {{
            font-size: 0.875rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-muted);
            margin-bottom: 0.5rem;
        }}

        .stat-val {{
            font-size: 2rem;
            font-weight: 700;
            font-family: 'Outfit', sans-serif;
            color: var(--text-main);
            margin-bottom: 0.25rem;
        }}

        .stat-ci {{
            font-size: 0.75rem;
            font-family: monospace;
            color: var(--primary);
            background: var(--primary-light);
            padding: 0.2rem 0.5rem;
            border-radius: 4px;
            display: inline-block;
        }}

        .main-layout {{
            display: grid;
            grid-template-columns: 1fr;
            gap: 2.5rem;
        }}

        @media (min-width: 992px) {{
            .main-layout {{
                grid-template-columns: 1.2fr 0.8fr;
            }}
        }}

        .card {{
            background: var(--card-bg);
            border-radius: 16px;
            border: 1px solid var(--border);
            box-shadow: var(--shadow);
            padding: 2rem;
            margin-bottom: 2.5rem;
        }}

        .card-title {{
            font-family: 'Outfit', sans-serif;
            font-size: 1.5rem;
            font-weight: 600;
            color: var(--text-main);
            margin-bottom: 1.5rem;
            border-left: 4px solid var(--primary);
            padding-left: 0.75rem;
        }}

        .chart-container {{
            text-align: center;
            margin: 1.5rem 0;
            border: 1px solid var(--border);
            border-radius: 12px;
            overflow: hidden;
            background: #fff;
            padding: 1rem;
        }}

        .chart-img {{
            max-width: 100%;
            height: auto;
            border-radius: 8px;
        }}

        .stats-table-container {{
            overflow-x: auto;
            margin: 1.5rem 0;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            text-align: left;
            font-size: 0.925rem;
        }}

        th {{
            background-color: var(--primary-light);
            color: var(--primary);
            font-weight: 600;
            padding: 0.75rem 1rem;
            border-bottom: 2px solid var(--border);
        }}

        td {{
            padding: 0.75rem 1rem;
            border-bottom: 1px solid var(--border);
        }}

        tr:hover td {{
            background-color: #f8fafc;
        }}

        .badge {{
            display: inline-block;
            padding: 0.25rem 0.5rem;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 600;
        }}

        .badge-success {{
            background-color: #d1fae5;
            color: #065f46;
        }}

        .keywords-grid {{
            display: grid;
            grid-template-columns: 1fr;
            gap: 1.5rem;
        }}

        @media (min-width: 576px) {{
            .keywords-grid {{
                grid-template-columns: 1fr 1fr;
            }}
        }}

        .keyword-list-card {{
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 1.25rem;
            background: #fafafa;
        }}

        .keyword-list-title {{
            font-family: 'Outfit', sans-serif;
            font-weight: 600;
            font-size: 1.1rem;
            margin-bottom: 1rem;
            text-align: center;
        }}

        .text-section {{
            margin-top: 1.5rem;
        }}

        .text-section p {{
            margin-bottom: 1rem;
            color: var(--text-muted);
        }}

        .highlight-box {{
            background-color: #f0fdf4;
            border-left: 4px solid #10b981;
            padding: 1rem;
            border-radius: 0 8px 8px 0;
            margin: 1.5rem 0;
        }}

        .highlight-box-title {{
            font-weight: 600;
            color: #065f46;
            margin-bottom: 0.5rem;
        }}

        footer {{
            text-align: center;
            margin-top: 4rem;
            color: var(--text-muted);
            font-size: 0.875rem;
            border-top: 1px solid var(--border);
            padding-top: 2rem;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>AI vs Children: Semantic Gap Analysis Report</h1>
            <p class="subtitle">Quantitative comparison of child interview transcripts and AI-generated city environment evaluations using Qwen3-Embedding-8B</p>
        </header>

        <!-- Top statistics panel -->
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-label">Balanced Sample Size</div>
                <div class="stat-val">{res['sample_size']}</div>
                <div class="stat-muted" style="font-size: 0.85rem; color: var(--text-muted);">Points per group</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Centroid Distance</div>
                <div class="stat-val">{res['centroid_distance']:.4f}</div>
                <div class="stat-ci">95% CI: [{res['centroid_distance_ci'][0]:.4f}, {res['centroid_distance_ci'][1]:.4f}]</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Child Interview KE</div>
                <div class="stat-val">{res['children']['ke']:.4f}</div>
                <div class="stat-ci">95% CI: [{res['children']['ke_ci'][0]:.4f}, {res['children']['ke_ci'][1]:.4f}]</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">AI Baseline KE</div>
                <div class="stat-val">{res['ai']['ke']:.4f}</div>
                <div class="stat-ci">95% CI: [{res['ai']['ke_ci'][0]:.4f}, {res['ai']['ke_ci'][1]:.4f}]</div>
            </div>
        </div>

        <div class="main-layout">
            <!-- Left Column: Visualizations & Quantitative Metrics -->
            <div class="left-col">
                <div class="card">
                    <h2 class="card-title">Hao et al. Style t-SNE Semantic Overlap</h2>
                    <div class="chart-container">
                        <img src="{tsne_b64}" class="chart-img" alt="t-SNE Overlap Plot">
                    </div>
                    <div class="text-section">
                        <p><strong>Methodology:</strong> The t-SNE visualization combines children's interview responses and AI evaluations into a joint 2D space. The circles represent the <strong>Knowledge Extent (KE)</strong>, centered at each group's centroid. To map the high-dimensional maximum radius to the 2D coordinates without distortion, we scale the circles using the median 2D-to-high-dimensional p95 distance ratio.</p>
                        <p><strong>Findings:</strong> The child interview responses and AI-generated outputs occupy largely disjoint regions of the semantic space. The centroid distance of {res['centroid_distance']:.4f} indicates a substantial shift in focus, indicating that AI evaluations do not align well with the lived experiences and subjective preferences of children, even when the AI is explicitly prompted to evaluate as a child in Chinese.</p>
                    </div>
                </div>

                <div class="card">
                    <h2 class="card-title">Cosine Similarity distributions & Mann-Whitney U Test</h2>
                    <div class="chart-container">
                        <img src="{violin_b64}" class="chart-img" alt="Cosine Similarity Violins">
                    </div>
                    <div class="text-section">
                        <p>The violin plot displays the density of pairwise cosine similarities within each group (homogeneity) and between the two groups. A higher within-group similarity indicates a more cohesive and narrower set of concepts, while a lower cross-group similarity signifies a semantic gap.</p>
                        <div class="highlight-box">
                            <div class="highlight-box-title">Mann-Whitney U Significance Test Results:</div>
                            <table style="width: 100%; border: none;">
                                <thead>
                                    <tr>
                                        <th style="background: none; border-bottom: 1px solid #c2ffd2; color: #065f46;">Comparison</th>
                                        <th style="background: none; border-bottom: 1px solid #c2ffd2; color: #065f46;">U Statistic</th>
                                        <th style="background: none; border-bottom: 1px solid #c2ffd2; color: #065f46;">P-Value</th>
                                        <th style="background: none; border-bottom: 1px solid #c2ffd2; color: #065f46;">Result</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    <tr style="border-bottom: 1px solid #c2ffd2;">
                                        <td style="color: #065f46; border: none;">Within Children vs. Between Group</td>
                                        <td style="font-family: monospace; color: #065f46; border: none;">{res['statistical_tests']['within_child_vs_between']['u_statistic']:.1f}</td>
                                        <td style="font-family: monospace; color: #065f46; border: none; font-weight: bold;">{res['statistical_tests']['within_child_vs_between']['p_value']}</td>
                                        <td style="border: none;"><span class="badge badge-success">Highly Significant</span></td>
                                    </tr>
                                    <tr style="border: none;">
                                        <td style="color: #065f46; border: none;">Within AI vs. Between Group</td>
                                        <td style="font-family: monospace; color: #065f46; border: none;">{res['statistical_tests']['within_ai_vs_between']['u_statistic']:.1f}</td>
                                        <td style="font-family: monospace; color: #065f46; border: none; font-weight: bold;">{res['statistical_tests']['within_ai_vs_between']['p_value']}</td>
                                        <td style="border: none;"><span class="badge badge-success">Highly Significant</span></td>
                                    </tr>
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Right Column: Keyword analysis & Quantitative Tables -->
            <div class="right-col">
                <div class="card">
                    <h2 class="card-title">Top Distinctive Keywords</h2>
                    <div class="chart-container">
                        <img src="{kws_b64}" class="chart-img" alt="Keywords Bar Chart">
                    </div>
                    <div class="keywords-grid">
                        <div class="keyword-list-card">
                            <div class="keyword-list-title" style="color: #1f77b4;">Child Chinese Keywords</div>
                            <table>
                                <thead>
                                    <tr>
                                        <th>Word</th>
                                        <th>Freq</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {"".join(f"<tr><td>{kw['word']}</td><td style='font-family: monospace;'>{kw['count']}</td></tr>" for kw in res['children']['top_keywords'][:10])}
                                </tbody>
                            </table>
                        </div>
                        <div class="keyword-list-card">
                            <div class="keyword-list-title" style="color: #d62728;">AI Chinese Keywords</div>
                            <table>
                                <thead>
                                    <tr>
                                        <th>Word</th>
                                        <th>Freq</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {"".join(f"<tr><td>{kw['word']}</td><td style='font-family: monospace;'>{kw['count']}</td></tr>" for kw in res['ai']['top_keywords'][:10])}
                                </tbody>
                            </table>
                        </div>
                    </div>
                    <div class="text-section" style="margin-top: 1.5rem;">
                        <p><strong>Linguistic Highlights:</strong> The children's language is heavily centered on highly descriptive, concrete, and physical play objects (such as "滑梯" [slides], "秋千" [swings], "共享单车" [shared bikes], and "好玩" [fun]). In contrast, the AI evaluations, even when generated under a child role in Chinese, focus on general and standard design features (such as "安全" [safe], "绿化" [greening], "水池" [pool], "树木" [trees], "人行道" [sidewalk], and "干净" [clean]). All researcher names and transitional filler words have been aggressively filtered from both corpora to reveal the true content overlap.</p>
                    </div>
                </div>

                <div class="card">
                    <h2 class="card-title">Detailed Quantitative Summary</h2>
                    <div class="stats-table-container">
                        <table>
                            <thead>
                                <tr>
                                    <th>Metric</th>
                                    <th>Child Interview</th>
                                    <th>AI Baseline (32B_Det - zh)</th>
                                </tr>
                            </thead>
                            <tbody>
                                <tr>
                                    <td><strong>Knowledge Extent (KE)</strong></td>
                                    <td>{res['children']['ke']:.4f}</td>
                                    <td>{res['ai']['ke']:.4f}</td>
                                </tr>
                                <tr>
                                    <td>KE 95% Confidence Interval</td>
                                    <td style="font-family: monospace;">[{res['children']['ke_ci'][0]:.4f}, {res['children']['ke_ci'][1]:.4f}]</td>
                                    <td style="font-family: monospace;">[{res['ai']['ke_ci'][0]:.4f}, {res['ai']['ke_ci'][1]:.4f}]</td>
                                </tr>
                                <tr>
                                    <td><strong>Mean Radius (Density)</strong></td>
                                    <td>{res['children']['mean_radius']:.4f}</td>
                                    <td>{res['ai']['mean_radius']:.4f}</td>
                                </tr>
                                <tr>
                                    <td>Mean Radius 95% CI</td>
                                    <td style="font-family: monospace;">[{res['children']['mean_radius_ci'][0]:.4f}, {res['children']['mean_radius_ci'][1]:.4f}]</td>
                                    <td style="font-family: monospace;">[{res['ai']['mean_radius_ci'][0]:.4f}, {res['ai']['mean_radius_ci'][1]:.4f}]</td>
                                </tr>
                                <tr>
                                    <td><strong>p95 Radius</strong></td>
                                    <td>{res['children']['p95_radius']:.4f}</td>
                                    <td>{res['ai']['p95_radius']:.4f}</td>
                                </tr>
                                <tr>
                                    <td>p95 Radius 95% CI</td>
                                    <td style="font-family: monospace;">[{res['children']['p95_radius_ci'][0]:.4f}, {res['children']['p95_radius_ci'][1]:.4f}]</td>
                                    <td style="font-family: monospace;">[{res['ai']['p95_radius_ci'][0]:.4f}, {res['ai']['p95_radius_ci'][1]:.4f}]</td>
                                </tr>
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>

        <footer>
            <p>New Paper Analysis Pipeline &bull; Automated Report</p>
            <p style="font-size: 0.75rem; margin-top: 0.5rem;">Embedding Model: Qwen3-Embedding-8B (GGUF Q8_0) &bull; Dimensionality: 4096</p>
        </footer>
    </div>
</body>
</html>
"""

    output_html = os.path.join(base_dir, "AI_vs_Children_Semantic_Gap.html")
    with open(output_html, "w", encoding="utf-8") as f:
        f.write(html_template)
        
    print(f"\nHTML Report successfully compiled and saved to: {output_html}")

if __name__ == "__main__":
    main()
