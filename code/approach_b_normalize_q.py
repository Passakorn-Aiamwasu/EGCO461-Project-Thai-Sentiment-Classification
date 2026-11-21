"""Approach B: text normalization + handling the small `q` class and label noise.

Starts from the best Approach A config (A4: word 1-2 gram + char 2-4 gram TF-IDF,
LogisticRegression, class_weight='balanced') and changes one thing per step:

  B0  A4 rerun                                       (reference)
  B1  B0 + text normalization                        (repeated chars, 555, URLs, emoji, numbers)
  B2  B1 + hand-made question features               (question particles, '?', info words: price/branch/...)
  B3  B2 + q-probability weight tuned with 3-fold CV on train   (decision rule for the rare class)
  B4  B3 + relabel train rows the CV model strongly disagrees with   (label noise)

Tuning in B3/B4 uses only out-of-fold predictions on TRAIN, so valid stays an
honest held-out set for model selection. The best config on valid macro-F1 is
retrained on train+valid and used to predict test.csv.

This file does not change Approach A: it imports A's Features / lr() and builds on them.

Run from the repo root:  python code/approach_b_normalize_q.py
"""
import re
import time

import numpy as np
import pandas as pd
from pythainlp.util import normalize as thai_normalize
from scipy.sparse import csr_matrix, hstack
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

from approach_a_tfidf import SEED, Features, lr
from common import GROUP, LABELS, RESULTS_DIR, evaluate, load_data, save_submission, tokenize_all

# ---------------------------------------------------------------- normalization
URL_RE = re.compile(r"https?://\S+|www\.\S+")
LAUGH_RE = re.compile(r"5{3,}")
REPEAT_RE = re.compile(r"(.)\1{2,}")           # มากกกก -> มาก, !!!! -> !
NUM_RE = re.compile(r"\d+([.,]\d+)*")
EMOJI_RE = re.compile("[\U0001F300-\U0001FAFF☀-➿]")


def normalize_text(text):
    text = str(text).lower()
    text = URL_RE.sub(" xurl ", text)
    text = LAUGH_RE.sub(" xlaugh ", text)      # before NUM_RE, otherwise 555 becomes a number
    text = REPEAT_RE.sub(r"\1", text)
    text = NUM_RE.sub(" xnum ", text)
    text = EMOJI_RE.sub(lambda m: f" {m.group(0)} ", text)  # emoji as its own token
    text = thai_normalize(text)                # PyThaiNLP: fix vowel/tone mark order etc.
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------- question features
# Small hand-made features that point at the class definition of q:
# a question that asks for information about a product or service.
Q_PATTERNS = {
    "q_particle": r"ไหม|มั้ย|มั๊ย|มั๋ย|รึเปล่า|หรือเปล่า|ป่าว|หรอ|เหรอ|รึยัง|หรือยัง|ใช่ไหม|ได้ไหม",
    "q_wh": r"เท่าไ|กี่|ยังไง|อย่างไร|ที่ไหน|ตรงไหน|อะไร|ไหนบ้าง|เมื่อไ|ทำไม|ใคร|แบบไหน|ตัวไหน|รุ่นไหน",
    "q_mark": r"\?",
    "q_info": r"ราคา|สาขา|โปร|ดาวน์|ผ่อน|ส่งฟรี|ค่าส่ง|สั่งซื้อ|สั่งได้|วิธีใช้|ใช้ได้|มีขาย|ขายที่|สนใจ|สอบถาม|ขอทราบ|อยากทราบ|อยากถาม|รบกวน",
    "q_polite_end": r"(คะ|ครับ|คับ|ค่ะ|ค่า)\s*\??\s*$",
}
Q_RES = {k: re.compile(v) for k, v in Q_PATTERNS.items()}


def question_features(texts, weight=1.0):
    rows = [[1.0 if r.search(str(t)) else 0.0 for r in Q_RES.values()] for t in texts]
    X = np.array(rows)
    # interaction: a question form AND an info word -> strong q signal
    asks = np.maximum.reduce([X[:, 0], X[:, 1], X[:, 2]])
    X = np.column_stack([X, asks * X[:, 3]])
    return csr_matrix(X * weight)


class FeaturesB(Features):
    """A4 features (+ optional question features). Texts passed in are already normalized or not."""

    def __init__(self, q_feats=False, q_weight=1.0, **kwargs):
        super().__init__(**kwargs)
        self.q_feats, self.q_weight = q_feats, q_weight

    def fit_transform(self, tokens, texts):
        X = super().fit_transform(tokens, texts)
        return hstack([X, question_features(texts, self.q_weight)]).tocsr() if self.q_feats else X

    def transform(self, tokens, texts):
        X = super().transform(tokens, texts)
        return hstack([X, question_features(texts, self.q_weight)]).tocsr() if self.q_feats else X


A4_FEATS = dict(ngram=(1, 2), use_char=True)
Q_IDX = None  # index of "q" in model.classes_, set after fitting


# ------------------------------------------------------------- experiment steps
# name -> settings
EXPERIMENTS = {
    "B0_A4_rerun": dict(norm=False, q_feats=False, tune_q=False, relabel=False),
    "B1_+normalize": dict(norm=True, q_feats=False, tune_q=False, relabel=False),
    "B2_+q_features": dict(norm=True, q_feats=True, tune_q=False, relabel=False),
    "B3_+q_weight_cv": dict(norm=True, q_feats=True, tune_q=True, relabel=False),
    "B4_+relabel_noise": dict(norm=True, q_feats=True, tune_q=True, relabel=True),
}
Q_WEIGHT_GRID = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]
RELABEL_THRESHOLD = 0.9  # relabel only when the out-of-fold model is this confident


def fit_model(cfg, tokens, texts, y):
    feats = FeaturesB(q_feats=cfg["q_feats"], **A4_FEATS)
    model = lr(balanced=True).fit(feats.fit_transform(tokens, texts), y)
    return feats, model


def predict(feats, model, tokens, texts, q_weight=1.0):
    proba = feats_proba(feats, model, tokens, texts)
    proba[:, list(model.classes_).index("q")] *= q_weight
    return model.classes_[proba.argmax(axis=1)]


def feats_proba(feats, model, tokens, texts):
    return model.predict_proba(feats.transform(tokens, texts))


def out_of_fold_proba(cfg, tokens, texts, y, n_splits=3):
    """Out-of-fold class probabilities on the training data (used for tuning only)."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    proba, classes = np.zeros((len(y), len(LABELS))), None
    for tr_idx, te_idx in skf.split(texts, y):
        feats, model = fit_model(cfg, [tokens[i] for i in tr_idx], texts.iloc[tr_idx], y.iloc[tr_idx])
        proba[te_idx] = feats_proba(feats, model, [tokens[i] for i in te_idx], texts.iloc[te_idx])
        classes = model.classes_
    return proba, classes


def tune_q_weight(proba, classes, y):
    q = list(classes).index("q")
    best_w, best_f1 = 1.0, -1
    for w in Q_WEIGHT_GRID:
        p = proba.copy()
        p[:, q] *= w
        f1 = f1_score(y, classes[p.argmax(axis=1)], labels=LABELS, average="macro")
        if f1 > best_f1:
            best_w, best_f1 = w, f1
    return best_w, best_f1


def train_with_cfg(cfg, tokens, texts, y, log=print):
    """Train one config. Returns (feats, model, q_weight). Tuning uses train CV only."""
    q_weight = 1.0
    if cfg["tune_q"] or cfg["relabel"]:
        proba, classes = out_of_fold_proba(cfg, tokens, texts, y)
        if cfg["relabel"]:
            pred, conf = classes[proba.argmax(axis=1)], proba.max(axis=1)
            flip = (pred != y.values) & (conf >= RELABEL_THRESHOLD)
            changes = pd.crosstab(pd.Series(y.values[flip], name="given"), pd.Series(pred[flip], name="new"))
            log(f"    relabel {flip.sum()} train rows (conf >= {RELABEL_THRESHOLD}):\n{changes.to_string()}")
            y = pd.Series(np.where(flip, pred, y.values), index=y.index)
            # rows that kept their label keep their OOF proba; tune on corrected labels
        if cfg["tune_q"]:
            q_weight, cv_f1 = tune_q_weight(proba, classes, y)
            log(f"    q weight = {q_weight} (train 3-fold CV macro-F1 {cv_f1:.3f})")
    feats, model = fit_model(cfg, tokens, texts, y)
    return feats, model, q_weight


def main():
    train, valid, sample, test = load_data()
    splits = {"train": train, "valid": valid, "sample": sample, "test": test}

    t0 = time.time()
    texts = {"raw": {k: df["text"].astype(str) for k, df in splits.items()}}
    texts["norm"] = {k: s.map(normalize_text) for k, s in texts["raw"].items()}
    tok = {kind: {k: tokenize_all(s) for k, s in d.items()} for kind, d in texts.items()}
    print(f"normalized + tokenized in {time.time() - t0:.0f}s")

    rows, valid_preds, trained_q_weight = [], {}, {}
    for name, cfg in EXPERIMENTS.items():
        t0 = time.time()
        kind = "norm" if cfg["norm"] else "raw"
        print(f"{name}")
        feats, model, q_w = train_with_cfg(cfg, tok[kind]["train"], texts[kind]["train"], train["category"])
        trained_q_weight[name] = q_w
        preds = {split: (splits[split]["category"],
                         predict(feats, model, tok[kind][split], texts[kind][split], q_w))
                 for split in ["valid", "sample"]}
        rows.append(evaluate(name, preds))
        valid_preds[name] = preds["valid"][1]
        print(f"    valid={rows[-1]['valid_macroF1']:.3f}  sample={rows[-1]['sample_macroF1']:.3f}  "
              f"({time.time() - t0:.0f}s)")

    results = pd.DataFrame(rows).set_index("experiment")
    RESULTS_DIR.mkdir(exist_ok=True)
    results.round(4).to_csv(RESULTS_DIR / "approach_b_results.csv")
    pd.set_option("display.width", 200)
    print("\n" + results.round(3).to_string())

    err = valid[["id", "text", "category"]].copy()
    for name, p in valid_preds.items():
        err[name] = p
    err.to_csv(RESULTS_DIR / "approach_b_valid_predictions.csv", index=False)

    # Select on valid macro-F1 only. B0 is a rerun of A4 (submitted as run4).
    best = results.drop(index="B0_A4_rerun")["valid_macroF1"].idxmax()
    print(f"\nselected by valid macro-F1 (excluding B0): {best}")

    # Retrain on train+valid. Tuning (q weight / relabel) is redone with CV on train+valid.
    cfg = EXPERIMENTS[best]
    kind = "norm" if cfg["norm"] else "raw"
    full_texts = pd.concat([texts[kind]["train"], texts[kind]["valid"]], ignore_index=True)
    full_y = pd.concat([train["category"], valid["category"]], ignore_index=True)
    feats, model, q_w = train_with_cfg(cfg, tok[kind]["train"] + tok[kind]["valid"], full_texts, full_y)
    test_pred = predict(feats, model, tok[kind]["test"], texts[kind]["test"], q_w)

    tag = best.split("_", 1)[0].lower()
    path = save_submission(test["id"], test_pred, f"{GROUP}_run5_{tag}_norm_q.csv")
    print(f"saved {path.relative_to(path.parent.parent)}")
    print("predicted class distribution on test:")
    print(pd.Series(test_pred).value_counts(normalize=True).round(3).to_string())


if __name__ == "__main__":
    main()
