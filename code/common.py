"""Shared helpers: data loading, Thai tokenization, evaluation and submission writing.

Every experiment should use `evaluate()` so all results are measured the same way
(macro-F1 + per-class F1 on valid.csv and sample_labeled.csv).
"""
from pathlib import Path

import pandas as pd
from pythainlp.tokenize import word_tokenize
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
PRED_DIR = ROOT / "predictions"
RESULTS_DIR = ROOT / "results"

LABELS = ["pos", "neu", "neg", "q"]
GROUP = "G00"  # TODO: change to your group number, e.g. "G05"


def load_data():
    train = pd.read_csv(DATA / "train.csv")
    valid = pd.read_csv(DATA / "valid.csv")
    sample = pd.read_csv(DATA / "sample_labeled.csv")
    test = pd.read_csv(DATA / "test.csv")
    return train, valid, sample, test


def thai_tokenize(text):
    """Same tokenizer as the baseline notebook (PyThaiNLP newmm)."""
    return word_tokenize(str(text), engine="newmm", keep_whitespace=False)


def tokenize_all(texts):
    """Tokenize once and reuse the token lists for every word-level vectorizer."""
    return [thai_tokenize(t) for t in texts]


def evaluate(name, model_preds):
    """model_preds: dict {split_name: (y_true, y_pred)} -> one result row (dict)."""
    row = {"experiment": name}
    for split, (y_true, y_pred) in model_preds.items():
        row[f"{split}_macroF1"] = f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)
        per_class = f1_score(y_true, y_pred, labels=LABELS, average=None, zero_division=0)
        for label, f1 in zip(LABELS, per_class):
            row[f"{split}_F1_{label}"] = f1
    return row


def save_submission(test_ids, preds, filename):
    """Write id,category and check it against sample_submission.csv before saving."""
    sub = pd.DataFrame({"id": test_ids, "category": preds})
    ref = pd.read_csv(DATA / "sample_submission.csv")
    assert list(sub.columns) == ["id", "category"], "wrong columns"
    assert len(sub) == len(ref) and set(sub["id"]) == set(ref["id"]), "ids do not match test.csv"
    assert sub["category"].isin(LABELS).all(), "unknown label in predictions"
    PRED_DIR.mkdir(exist_ok=True)
    path = PRED_DIR / filename
    sub.to_csv(path, index=False, encoding="utf-8")
    return path
