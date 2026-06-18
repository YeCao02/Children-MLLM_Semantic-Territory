# Semantic-affective alignment module

## Research question

The semantic embedding pipeline asks whether AI and children discuss similar spatial
features and occupy similar semantic territory. The affective module asks a distinct
question: do semantically similar texts express comparable concern, engagement, inquiry
and neutrality?

This distinction matters in child-friendly geography. A child may mention the same object
as an AI system while framing it through fear, practical constraint, curiosity or embodied
experience. Semantic agreement alone can therefore overstate experiential agreement.

## Models

1. `Johnson8187/Chinese-Emotion-Small`
   - mDeBERTa-v3-based Chinese emotion classifier.
   - Eight softmax labels: neutral, concerned, happy, angry, sad, questioning, surprised
     and disgusted.
   - Used as the Chinese-specific model.

2. `tabularisai/multilingual-emotion-classification`
   - XLM-R-based multilingual multi-label classifier.
   - Eleven sigmoid labels including fear, frustration, joy, neutral and surprise.
   - Used as a robustness model.
   - License: CC-BY-NC-4.0.

The models have different objectives and label systems, so raw probabilities are not
pooled directly. Both are mapped to four normalized shared dimensions:

- positive engagement;
- concern/distress;
- inquiry/surprise;
- neutral expression.

## Units and comparisons

- Children: direct environmental-perception utterances and meta-evaluations are reported
  separately.
- AI: `reason`, `suggestion` and image-level `reason + suggestion` are reported separately.
- Primary comparison: child direct-perception utterances versus AI combined text.

Utterances remain the affective unit because averaging language across a whole scene can
erase localized concern or curiosity.

## Statistical outputs

- Mean probability profiles and dominant-label proportions.
- Jensen-Shannon divergence between group mean profiles.
- Euclidean distance between emotion-profile centroids.
- Permutation test for centroid separation.
- Bootstrap 95% intervals for AI-minus-child differences in each shared dimension.
- Directional agreement between the two models' group-difference vectors.

## Semantic-affective alignment

Each child utterance and AI combined text receives two cross-group overlap scores:

1. Semantic overlap: maximum Qwen3 embedding cosine similarity to the opposite group.
2. Affective overlap: one minus Jensen-Shannon distance from the opposite group's mean
   four-dimensional affective profile.

The affective profile is the mean of the two models in their shared dimensions. Median
thresholds create four descriptive quadrants: shared on both axes, semantic-only,
affective-only and distinctive on both axes.

## Validation and interpretation limits

A transparent 16-sentence Chinese anchor set checks face validity. Both models currently
match 68.8% overall, but their weaknesses differ:

- the Chinese-specific model detects inquiry better but misses many explicit danger/fear
  statements;
- the multilingual model detects concern/distress better but often treats Chinese
  questions as neutral.

The robust current claim is distributional: AI outputs are more neutral and less
concern/distress-oriented than children's direct perception. Fine-grained claims about
fear, anger, sadness or curiosity require a manually coded validation sample.

Model output describes affective cues in text, not children's internal emotional states.
