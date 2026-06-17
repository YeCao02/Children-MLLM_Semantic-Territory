# Hao-style AI-Children 语义领地图：方法说明

## 是否可以用“异同并存”的方式分析 AI-children？

可以，而且这比只做“semantic gap”更适合第二篇论文。建议把问题拆成三层：

1. **共同语义核心**：AI 与儿童都谈到的维度，例如安全、互动、设施、绿化、可达性等。可用最近邻相似度、关键词共同出现、局部重叠区来描述。
2. **儿童特异语义领地**：儿童更强调具体生活场景、实用性、危险来源、店铺/学校/商业街/自行车等日常经验。
3. **AI 特异语义领地**：AI 更强调标准化规划话语，例如智能设施、创意空间、空气清新、探索、互动装置等。

这样论文叙事可以从“AI 是否像儿童”推进到“AI 与儿童共享了哪些抽象评价维度，又在哪些具身经验上系统错位”。

## 新图的读法

输出图：`data\figures_v2\Fig_v2_hao_style_ai_children.png`

这张图效仿 Hao et al. 2026 Fig. 3b：

- 蓝色空心点：儿童真实语料中的直接环境感知片段。
- 红色空心点：AI 对每张图片的 reason + suggestion 合并文本。
- 黄色中心点：儿童与 AI 共同语义空间的 shared centroid。
- 蓝/红圆：按 4096 维 Qwen3 embedding 空间中的 semantic extent 映射到二维图上，用于直观表达两个语义领地的覆盖范围。
- 黑色箭头：从 shared centroid 指向对应语义边界，类比 Hao 图中的 KE 箭头。
- 标注点：两侧更突出的语义锚点，用来解释“差异在哪里”。

## 需要在正文里明确的边界

这张图是说明性可视化，不是正式统计推断。正式结论仍应基于 4096 维归一化嵌入空间：

- Children KE around shared centroid: 0.9255
- AI KE around shared centroid: 0.7360
- Centroid distance: 0.6199
- Median nearest-neighbor similarity, child-to-AI / AI-to-child: 0.5426 / 0.6700

## 推荐论文写法

> To visualize shared and distinctive semantic territories, we projected child interview utterances and AI-generated image evaluations into a joint two-dimensional t-SNE space. Following the visual logic of Hao et al. (2026), we overlaid semantic-extent boundaries centered on the shared centroid. The circles are visual summaries of high-dimensional semantic extent computed from Qwen3-Embedding-8B representations, while the t-SNE projection is used only for visual interpretation. This design allows us to distinguish common evaluative ground from group-specific semantic territories.

## 为什么不能只用传统 t-SNE

传统 t-SNE 能展示聚类，但很难回答“异同在哪里”。Hao-style 图增加了三个信息层：

1. 共同中心：表示 AI 与儿童被放在同一个语义空间里比较。
2. 语义边界：表示哪一组覆盖的语义领地更大。
3. 代表性锚点：说明边界扩展到哪些具体主题。

因此，传统 t-SNE 可以保留为诊断图；Hao-style 图更适合作为论文中的核心解释图。
