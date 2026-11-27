# Model cards

Optional disclosure documents (`model_cards/<model_id>.md`) for reviewers. They are not
parsed by the leaderboard; scores and provenance come from `submissions/<model_id>.json`
and `benchmark/model_meta.json`.

Suggested sections:

```markdown
# <model_id>

- **Display name:**
- **Parameters:**
- **Architecture:**
- **Weights / URL:**
- **Tokenizer / input:**
- **Pooling:** (e.g. mean over tokens)

## Training data
<pretraining corpora and filtering>

## Disclosure
- **Zero-shot relative to benchmark tasks:** 
- **Known train/test overlap with benchmark data:** 
- **Harness version:** GENEB-0.1.1
- **Submitted by:**
```
