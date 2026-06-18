# Related Work for Semantic Territory and Coverage Visualization

This note collects studies that are useful for framing the Children-MLLM semantic
territory analysis. The most relevant methodological pattern is:

1. compute distributional or coverage metrics in the original embedding space;
2. use t-SNE, UMAP or small multiples only for interpretation;
3. report directionality, because AI-to-human and human-to-AI retrieval are not the
   same question.

## Closest Methodological References

| Reference | Why it matters | How to borrow it |
| --- | --- | --- |
| [Hao et al. (2026), Nature](https://www.nature.com/articles/s41586-025-09922-y) | Defines knowledge extent in a high-dimensional scientific-paper embedding space and visualizes it with t-SNE. | Direct template for separating high-dimensional metrics from Fig. 3b-style explanatory visualization. |
| [Pillutla et al. (2021), NeurIPS, MAUVE](https://proceedings.neurips.cc/paper/2021/hash/260c2432a0eecc28ce03c10dadc078a4-Abstract.html) | Measures the distributional gap between neural and human text in an embedding space. | Use its language of generated-human distributional gaps and quality/diversity tradeoffs. |
| [Le Bronnec et al. (2024), ACL](https://aclanthology.org/2024.acl-long.616/) | Applies precision and recall concepts to LLM quality and diversity. | Supports directional semantic coverage: AI precision in the child domain is different from AI recall of the child domain. |
| [Guo et al. (2025), TACL](https://aclanthology.org/2025.tacl-1.69/) | Benchmarks LLM linguistic diversity across lexical, syntactic and semantic levels. | Supports reporting semantic dispersion and diversity instead of relying only on clusters. |
| [Boggust et al. (2022), ACM IUI](https://dl.acm.org/doi/10.1145/3490099.3511122) | Compares embedding spaces through global projections and local neighborhoods. | Useful precedent for combining t-SNE/UMAP views with nearest-neighbor evidence. |

## Human, Child and Urban Perception References

| Reference | Why it matters | How to borrow it |
| --- | --- | --- |
| [Zhong et al. (2024), ACM ICMI](https://dl.acm.org/doi/10.1145/3678957.3685756) | Compares multimodal language models with human perception descriptions using semantic similarity and t-SNE. | Closely matches the MLLM-output versus human-perception setup. |
| [Woloszyn and Gagl (2025), arXiv](https://arxiv.org/abs/2508.13769) | Compares LLM picture descriptions with children's picture descriptions. | Supports the idea that child expression is not merely simplified adult/AI language. |
| [Liu and Fourtassi (2024), arXiv](https://arxiv.org/abs/2412.09318) | Evaluates whether LLMs can mimic child-caregiver interaction. | Supports the hypothesis that LLMs can approximate surface patterns while missing discourse diversity. |
| [Wedyan et al. (2025), PLOS ONE](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0322078) | Compares GPT-4o and human urban walkability perception. | Useful for discussing partial alignment plus divergence in comfort, liveliness and thematic diversity. |
| [Malekzadeh et al. (2025), Computers, Environment and Urban Systems](https://doi.org/10.1016/j.compenvurbsys.2024.102243) | Contrasts ChatGPT and human urban attractiveness judgments from street-view images. | Useful for positioning AI-human differences in urban visual perception and contextual nuance. |
| [Zhao et al. (2026), arXiv](https://arxiv.org/abs/2604.20048) | Studies culturally uneven LLM baselines in city perception. | Relevant for arguing that LLM city perception is not neutral and should be benchmarked against human groups. |

## Suggested Manuscript Wording

> Following recent work on high-dimensional knowledge extent and neural-human text
> distribution gaps, we distinguish two forms of alignment. First, semantic extent
> asks how broadly each corpus spreads around a shared centroid. Second, directional
> coverage asks whether texts from one group can retrieve close semantic analogues
> in the other group. This distinction matters because AI-to-child retrieval and
> child-to-AI retrieval operationalize different questions: whether AI outputs lie
> inside children's semantic territory, and whether AI covers the breadth of
> children's situated environmental expressions.

## Figure Logic to Borrow

- Use a main high-dimensional coverage curve rather than making t-SNE the main
  statistical evidence.
- Show nearest-neighbor distributions and thresholded coverage rates in both
  directions.
- Add a precision/recall-style frontier: AI semantic precision measures whether AI
  outputs fall inside children's semantic territory, while child semantic recall
  measures how much of children's territory AI covers.
- Pair global t-SNE maps with local-neighborhood diagnostics such as mutual nearest
  neighbors, receiver concentration and hubness.
- Keep the Hao-style t-SNE figure as an explanatory map, explicitly noting that
  circles and coverage labels come from the original embedding space.
- Place BERT-style affective alignment in an exploratory or supplementary section
  until manual validation is added.
