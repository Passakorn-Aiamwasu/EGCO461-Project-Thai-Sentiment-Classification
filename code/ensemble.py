"""Ensemble: average class probabilities of WangchanBERTa (Approach C) and TF-IDF LR A4 (Approach A).

Needs the Approach C outputs from Colab:
  results/approach_c_<run>_{valid,sample}_proba.csv     (committed)
  predictions/approach_c_<run>_test_proba.csv           (from the Colab zip, not committed)

  E1  w * C1 + (1 - w) * A4, with w chosen on valid macro-F1 (one parameter, grid 0.0..1.0)
  E2  equal average of C1, C2 and A4 (nothing tuned)

The A4 probabilities for valid/sample come from a model trained on train only; for test,
A4 is retrained on train+valid, the same way as predictions/<GROUP>_run4_tfidf_lr.csv.
The C runs were trained on train only (RETRAIN_ON_TRAIN_VALID = False in the notebook).

Run from the repo root:  python code/ensemble.py
"""
import numpy as np
import pandas as pd

from approach_a_tfidf import EXPERIMENTS as A_EXPERIMENTS, Features
from common import GROUP, LABELS, PRED_DIR, RESULTS_DIR, evaluate, load_data, save_submission, tokenize_all

A4 = "A4_tfidf12+char_lr_balanced"
C_RUNS = ["C1_wangchanberta", "C2_wangchanberta_balanced"]
W_GRID = np.round(np.arange(0.0, 1.01, 0.1), 2)


def read_proba(path, ids):
    df = pd.read_csv(path).set_index("id").loc[ids]  # align rows by id
    return df[LABELS].to_numpy()


def a4_proba(train_df, train_tok, splits):
    feat_cfg, make_model = A_EXPERIMENTS[A4]
    feats = Features(**feat_cfg)
    model = make_model().fit(feats.fit_transform(train_tok, train_df["text"]), train_df["category"])
    order = [list(model.classes_).index(l) for l in LABELS]  # columns in LABELS order
    return {name: model.predict_proba(feats.transform(tok, df["text"]))[:, order]
            for name, (df, tok) in splits.items()}


def to_labels(p):
    return np.array(LABELS)[p.argmax(axis=1)]


def main():
    train, valid, sample, test = load_data()
    tok = {k: tokenize_all(df["text"]) for k, df in
           [("train", train), ("valid", valid), ("sample", sample), ("test", test)]}

    # A4 probabilities: train-only model for valid/sample, train+valid model for test
    a4 = a4_proba(train, tok["train"], {"valid": (valid, tok["valid"]), "sample": (sample, tok["sample"])})
    full = pd.concat([train, valid], ignore_index=True)
    a4["test"] = a4_proba(full, tok["train"] + tok["valid"], {"test": (test, tok["test"])})["test"]

    c = {run: {"valid": read_proba(RESULTS_DIR / f"approach_c_{run}_valid_proba.csv", valid["id"]),
               "sample": read_proba(RESULTS_DIR / f"approach_c_{run}_sample_proba.csv", sample["id"]),
               "test": read_proba(PRED_DIR / f"approach_c_{run}_test_proba.csv", test["id"])}
         for run in C_RUNS}
    c1, c2 = c[C_RUNS[0]], c[C_RUNS[1]]
    y = {"valid": valid["category"], "sample": sample["category"]}

    def score(name, probs):
        return evaluate(name, {s: (y[s], to_labels(probs[s])) for s in ["valid", "sample"]})

    # E1: tune the BERT weight on valid
    sweep = []
    for w in W_GRID:
        probs = {s: w * c1[s] + (1 - w) * a4[s] for s in ["valid", "sample", "test"]}
        sweep.append({"w_bert": w, **score(f"E1_w{w}", probs)})
    sweep = pd.DataFrame(sweep)
    best_w = sweep.loc[sweep["valid_macroF1"].idxmax(), "w_bert"]
    print(sweep[["w_bert", "valid_macroF1", "sample_macroF1", "valid_F1_q", "sample_F1_q"]].round(3).to_string(index=False))
    print(f"best BERT weight on valid: {best_w}")

    e1 = {s: best_w * c1[s] + (1 - best_w) * a4[s] for s in ["valid", "sample", "test"]}
    e2 = {s: (c1[s] + c2[s] + a4[s]) / 3 for s in ["valid", "sample", "test"]}

    rows = [score(A4 + " (train only)", a4), score(C_RUNS[0], c1), score(C_RUNS[1], c2),
            score(f"E1_C1+A4_w{best_w}", e1), score("E2_C1+C2+A4_mean", e2)]
    results = pd.DataFrame(rows).set_index("experiment")
    results.round(4).to_csv(RESULTS_DIR / "ensemble_results.csv")
    sweep.round(4).to_csv(RESULTS_DIR / "ensemble_weight_sweep.csv", index=False)
    pd.set_option("display.width", 200)
    print("\n" + results.round(3).to_string())

    err = valid[["id", "text", "category"]].copy()
    err["E1"], err["E2"] = to_labels(e1["valid"]), to_labels(e2["valid"])
    err.to_csv(RESULTS_DIR / "ensemble_valid_predictions.csv", index=False)

    for run, tag, probs in [(1, "e2_bert2_tfidf", e2), (2, "e1_bert_tfidf", e1)]:
        pred = to_labels(probs["test"])
        path = save_submission(test["id"], pred, f"{GROUP}_run{run}_{tag}.csv")
        print(f"saved {path.name}: {pd.Series(pred).value_counts(normalize=True).round(3).to_dict()}")


if __name__ == "__main__":
    main()
