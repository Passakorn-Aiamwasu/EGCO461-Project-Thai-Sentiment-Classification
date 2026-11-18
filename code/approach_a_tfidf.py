"""Approach A: TF-IDF + linear models, run as a step-by-step ablation.

Each step changes ONE thing compared with the step before it, so the results table
shows what actually helps:

  A0  baseline: CountVectorizer (word unigram) + MultinomialNB        (reference)
  A1  TF-IDF (word unigram) + LogisticRegression                      (features + model)
  A2  A1 + class_weight='balanced'                                    (class imbalance)
  A3  A2 + word n-gram (1,2)                                          (context: negation, phrases)
  A4  A3 + char n-gram (char_wb 2-4)                                  (robust to tokenization / slang)
  A5  A4 features + LinearSVC (balanced)                              (model choice)
  + a small C grid for the A4/A5 models

Model selection uses valid macro-F1 only. The chosen config is retrained on
train+valid and used to predict test.csv.

Run from the repo root:  python code/approach_a_tfidf.py
"""
import time

import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from common import GROUP, RESULTS_DIR, evaluate, load_data, save_submission, tokenize_all

SEED = 42


def identity(x):
    return x


def word_vectorizer(kind="tfidf", ngram=(1, 1)):
    """Vectorizer over pre-tokenized lists (tokenize once, reuse everywhere)."""
    cls = TfidfVectorizer if kind == "tfidf" else CountVectorizer
    kwargs = dict(tokenizer=identity, preprocessor=identity, lowercase=False,
                  token_pattern=None, ngram_range=ngram)
    if kind == "tfidf":
        kwargs.update(sublinear_tf=True, min_df=2)
    return cls(**kwargs)


def char_vectorizer():
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True, min_df=2)


class Features:
    """Fit the chosen vectorizers on the training split and transform other splits."""

    def __init__(self, word_kind="tfidf", ngram=(1, 1), use_char=False):
        self.word = word_vectorizer(word_kind, ngram)
        self.char = char_vectorizer() if use_char else None

    def fit_transform(self, tokens, texts):
        X = self.word.fit_transform(tokens)
        if self.char is not None:
            X = hstack([X, self.char.fit_transform(texts)]).tocsr()
        return X

    def transform(self, tokens, texts):
        X = self.word.transform(tokens)
        if self.char is not None:
            X = hstack([X, self.char.transform(texts)]).tocsr()
        return X


def lr(C=1.0, balanced=False):
    return LogisticRegression(C=C, max_iter=2000, class_weight="balanced" if balanced else None,
                              random_state=SEED)


def svc(C=1.0):
    return LinearSVC(C=C, class_weight="balanced", random_state=SEED)


# name -> (feature settings, model factory)
EXPERIMENTS = {
    "A0_count_nb (baseline)": (dict(word_kind="count"), lambda: MultinomialNB(alpha=1.0)),
    "A1_tfidf_lr": (dict(), lambda: lr()),
    "A2_tfidf_lr_balanced": (dict(), lambda: lr(balanced=True)),
    "A3_tfidf12_lr_balanced": (dict(ngram=(1, 2)), lambda: lr(balanced=True)),
    "A4_tfidf12+char_lr_balanced": (dict(ngram=(1, 2), use_char=True), lambda: lr(balanced=True)),
    "A5_tfidf12+char_svc_balanced": (dict(ngram=(1, 2), use_char=True), lambda: svc()),
}
for C in [0.3, 3.0, 10.0]:
    EXPERIMENTS[f"A4_lr_balanced_C{C}"] = (dict(ngram=(1, 2), use_char=True), lambda C=C: lr(C=C, balanced=True))
for C in [0.1, 0.3]:
    EXPERIMENTS[f"A5_svc_balanced_C{C}"] = (dict(ngram=(1, 2), use_char=True), lambda C=C: svc(C=C))


def main():
    train, valid, sample, test = load_data()

    t0 = time.time()
    tok = {name: tokenize_all(df["text"]) for name, df in
           [("train", train), ("valid", valid), ("sample", sample), ("test", test)]}
    print(f"tokenized in {time.time() - t0:.0f}s")

    rows, valid_preds = [], {}
    for name, (feat_cfg, make_model) in EXPERIMENTS.items():
        t0 = time.time()
        feats = Features(**feat_cfg)
        X_train = feats.fit_transform(tok["train"], train["text"])
        model = make_model().fit(X_train, train["category"])
        preds = {split: (df["category"], model.predict(feats.transform(tok[split], df["text"])))
                 for split, df in [("valid", valid), ("sample", sample)]}
        rows.append(evaluate(name, preds))
        valid_preds[name] = preds["valid"][1]
        print(f"{name:32s} valid={rows[-1]['valid_macroF1']:.3f}  "
              f"sample={rows[-1]['sample_macroF1']:.3f}  ({time.time() - t0:.0f}s)")

    results = pd.DataFrame(rows).set_index("experiment")
    RESULTS_DIR.mkdir(exist_ok=True)
    results.round(4).to_csv(RESULTS_DIR / "approach_a_results.csv")
    pd.set_option("display.width", 200)
    print("\n" + results.round(3).to_string())

    # Keep valid predictions for error analysis (report section 6).
    err = valid[["id", "text", "category"]].copy()
    for name, p in valid_preds.items():
        err[name] = p
    err.to_csv(RESULTS_DIR / "approach_a_valid_predictions.csv", index=False)

    # Select on valid macro-F1 only (never on test), excluding the baseline.
    best = results.drop(index="A0_count_nb (baseline)")["valid_macroF1"].idxmax()
    print(f"\nselected by valid macro-F1: {best}")

    # Retrain the selected config on train+valid, then predict test.
    feat_cfg, make_model = EXPERIMENTS[best]
    full = pd.concat([train, valid], ignore_index=True)
    feats = Features(**feat_cfg)
    X_full = feats.fit_transform(tok["train"] + tok["valid"], full["text"])
    model = make_model().fit(X_full, full["category"])
    test_pred = model.predict(feats.transform(tok["test"], test["text"]))

    model_tag = "tfidf_lr" if "lr" in best else "tfidf_svc"
    path = save_submission(test["id"], test_pred, f"{GROUP}_run2_{model_tag}.csv")
    print(f"saved {path.relative_to(path.parent.parent)}")
    print("predicted class distribution on test:")
    print(pd.Series(test_pred).value_counts(normalize=True).round(3).to_string())


if __name__ == "__main__":
    main()
