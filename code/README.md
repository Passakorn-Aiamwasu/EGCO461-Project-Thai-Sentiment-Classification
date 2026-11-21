# Code

Run every command from the repository root.

```bash
pip install -r code/requirements.txt
```

## Run order

1. `baseline_naive_bayes.ipynb` (repo root): the Naive Bayes baseline from the course.
2. `python code/approach_a_tfidf.py`: Approach A, TF-IDF + linear models (about 3–4 minutes on a laptop CPU).
3. `python code/approach_b_normalize_q.py`: Approach B, normalization + q-class handling + label noise (about 7–8 minutes). It imports Approach A's code but does not change it, so the scripts can run in any order.

Before submitting, set `GROUP` in `code/common.py` to your group number.

## Approach A: what it does

An ablation where each step changes one thing from the step before:

| Step | Change |
|---|---|
| A0 | baseline: word count + MultinomialNB |
| A1 | TF-IDF (word unigram) + Logistic Regression |
| A2 | A1 + `class_weight='balanced'` |
| A3 | A2 + word n-gram (1,2) |
| A4 | A3 + char n-gram (`char_wb`, 2–4) |
| A5 | A4 features + LinearSVC (balanced) |
| + | small C grid for A4 (LR) and A5 (SVC) |

The config with the best **valid** macro-F1 is retrained on train+valid and used to predict `test.csv`.

Outputs:
- `results/approach_a_results.csv`: macro-F1 and per-class F1 on valid and sample_labeled for every step
- `results/approach_a_valid_predictions.csv`: valid predictions of every step, for error analysis
- `predictions/<GROUP>_run2_tfidf_lr.csv`: test predictions, checked against `sample_submission.csv`

## Approach B: what it does

Starts from A4 and changes one thing per step:

| Step | Change |
|---|---|
| B0 | A4 rerun (reference) |
| B1 | + text normalization (repeated chars, 555, URLs, numbers, emoji, PyThaiNLP normalize) |
| B2 | + hand-made question features (particles, `?`, info words such as price/branch) |
| B3 | + weight on P(q), tuned with 3-fold CV on train |
| B4 | + relabel train rows the CV model disagrees with at ≥ 0.9 confidence (label noise) |

Tuning in B3/B4 uses only out-of-fold predictions on train, so valid stays held out. The best step on valid (excluding B0) is retrained on train+valid.

Outputs: `results/approach_b_results.csv`, `results/approach_b_valid_predictions.csv`, `predictions/<GROUP>_run3_<step>_norm_q.csv`.

## Shared code

`common.py` holds shared helpers (`load_data`, `thai_tokenize`, `evaluate`, `save_submission`). Use them in later approaches so every result is measured the same way.
