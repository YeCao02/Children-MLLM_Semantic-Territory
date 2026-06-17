# Paper2 T-SNE Pipeline: v2 方法审计与改进建议

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

- 儿童原始片段：463
- 儿童直接环境感知片段：331
- 儿童环境场景单元：22
- AI 图片单元：220
- AI 文本校验错配数：0

## 主比较结果

- 主比较：儿童环境场景 vs AI 图像合并文本
- 质心距离：0.7003
- 置换检验 P 值：0.000999
- 儿童到 AI 最近邻相似度中位数：0.7006
- AI 到儿童最近邻相似度中位数：0.6454

## 输出文件

- HTML 报告：`AI_vs_Children_Semantic_Gap_v2.html`
- t-SNE 诊断图：`data/figures_v2/Fig_v2_tsne_diagnostic.png`
- 高维广度指标：`data/figures_v2/Fig_v2_highdim_extent.png`
- 质心距离矩阵：`data/figures_v2/Fig_v2_centroid_heatmap.png`
- 关键词差异：`data/figures_v2/Fig_v2_keyword_contrast.png`

## 下一步建议

1. 把儿童访谈重新标注到图片/场景 ID，形成 `image -> children_comments` 的配对表。没有这个配对层，实验只能做分布比较，不能做逐图片误差分析。
2. 不要把 `reason` 和 `suggestion` 当作同一分布里的独立样本。建议三条线并行报告：AI reason、AI suggestion、AI combined。
3. 重新在 WSL 中按 scene/image 单元计算 embedding，而不是对短句先 embedding 后平均。当前 v2 的场景向量是过渡方案。
4. 论文图中保留 t-SNE，但正文结论基于高维指标、置换检验、最近邻检索和关键词差异，不基于 t-SNE 圆圈面积。
5. 建议增加一个人工编码表，把儿童真实关注点分成安全、实用、可达、日常使用、趣味、数字设施、绿化等维度，再与 AI 的维度覆盖率做交叉验证。
