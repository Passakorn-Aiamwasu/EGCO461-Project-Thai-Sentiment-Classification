# Code

Run every command from the repository root.

```bash
pip install -r code/requirements.txt
```

## Run order

1. `baseline_naive_bayes.ipynb` (repo root): the Naive Bayes baseline from the course.
2. `python code/approach_a_tfidf.py`: Approach A, TF-IDF + linear models (about 3–4 minutes on a laptop CPU).
3. `python code/approach_b_normalize_q.py`: Approach B, normalization + q-class handling + label noise (about 7–8 minutes). It imports Approach A's code but does not change it, so the scripts can run in any order.
4. `code/approach_c_wangchanberta.ipynb`: Approach C, WangchanBERTa fine-tuning. Run in Google Colab with a GPU (see below).
5. `python code/ensemble.py`: ensembles of WangchanBERTa and A4 (about 1–2 minutes). Needs step 4's outputs: unzip the Colab zip so `predictions/approach_c_*_test_proba.csv` exists.

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

## Approach C: WangchanBERTa (Google Colab, GPU)

`code/approach_c_wangchanberta.ipynb` fine-tunes `airesearch/wangchanberta-base-att-spm-uncased`.
Open it in Colab, set Runtime → T4 GPU, then Run all. The first cell clones this repo. If the repo is private,
upload a zip of the repo and unzip it to `/content/repo` instead. The last cell downloads `results/` and `predictions/` as a zip.

| Step | Change |
|---|---|
| C1 | fine-tune with plain cross-entropy |
| C2 | C1 + `balanced` class weights in the loss |

Best epoch per run is chosen on valid macro-F1. Outputs: `results/approach_c_results.csv`, `results/approach_c_history.csv`
(loss and valid F1 per epoch), `results/approach_c_valid_predictions.csv`, valid/sample probabilities
in `results/approach_c_*_proba.csv` (for ensembles), and `predictions/<GROUP>_run4_wangchanberta.csv`.
Set `RETRAIN_ON_TRAIN_VALID = True` to retrain the chosen run on train+valid (doubles the run time).
GPU training is not bit-for-bit deterministic, so a rerun can differ slightly.

## Ensemble: what it does

Averages class probabilities of WangchanBERTa (C) and TF-IDF LR (A4):

| Name | How |
|---|---|
| E1 | `w * C1 + (1 - w) * A4`, `w` chosen on valid macro-F1 (grid 0.0–1.0) |
| E2 | equal mean of C1, C2 and A4 (nothing tuned) |

Outputs: `results/ensemble_results.csv`, `results/ensemble_weight_sweep.csv`, `results/ensemble_valid_predictions.csv`,
`predictions/<GROUP>_run5_e1_bert_tfidf.csv`, `predictions/<GROUP>_run5_e2_bert2_tfidf.csv`.

## Shared code

`common.py` holds shared helpers (`load_data`, `thai_tokenize`, `evaluate`, `save_submission`). Use them in later approaches so every result is measured the same way.
