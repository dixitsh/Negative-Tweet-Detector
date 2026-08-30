# Dataset preparation

Place a licensed Hindi-English/code-mixed sentiment CSV here as `hinglish_sentiment.csv`.

Minimum schema:

```csv
text,label
"bahut acchi movie hai",positive
"theek hai but not great",neutral
"service bilkul bakwas hai",negative
```

Accepted text columns: `text`, `tweet`, `sentence`.

Accepted label columns: `label`, `sentiment`, `polarity`.

Accepted labels: `negative`, `neutral`, `positive` or `0`, `1`, `2`.

## Recommended research protocol

1. Keep a fixed held-out test set.
2. Do not leak users or near-duplicate posts between train and test.
3. Preserve code-mixed text; do not translate all Hindi into English before the transformer sees it.
4. Report class counts and language/code-mixing statistics.
5. Create client partitions only from the training split.
6. Keep validation/test data centralized and untouched by clients for the global comparison.
7. If using a public dataset, record its citation, license, original split and preprocessing decisions in your dissertation.
