"""Error analysis sheet for the main model (E2 = mean of WangchanBERTa C1, C2 and TF-IDF LR A4).

Step 1, build the sheet (only valid and sample_labeled are used, never test):
    python code/error_analysis.py
  -> results/error_analysis_sheet.csv
     * every E2 error on sample_labeled (clean labels: real model errors), with a draft cause
     * a random sample of 50 E2 errors on valid (seed 42), cause left blank
     * predictions of A4, C1 and E2 side by side, to see which model gets what right

  Fill in, for each row:
     label_correct  yes / no  (is the gold label right according to the class definitions in the brief?)
     better_label   if label_correct = no: the label it should have
     cause          one of CAUSES below (use label_noise when label_correct = no)
     note           free text, optional

Step 2, after filling it in, summarize:
    python code/error_analysis.py --summarize
  -> results/error_analysis_summary.csv  (count and share of each cause, per source)

The sheet is saved as UTF-8 with BOM so Excel / Google Sheets shows Thai correctly.
Running step 1 again overwrites the sheet: copy it before filling it in if you rerun.
"""
import sys

import numpy as np
import pandas as pd

from common import LABELS, RESULTS_DIR, load_data, tokenize_all
from ensemble import a4_proba, read_proba, to_labels

SHEET = RESULTS_DIR / "error_analysis_sheet.csv"
SUMMARY = RESULTS_DIR / "error_analysis_summary.csv"
N_VALID = 50
SEED = 42

CAUSES = {
    "label_noise": "gold label is wrong by the brief's class definitions; the model may be right",
    "sarcasm": "positive words about a bad situation (ประชด)",
    "keyword_trap": "sentiment word inside a factual / neutral sentence (ชอบ, สะดวก, ต้องรอ)",
    "mixed_or_comparison": "two-sided opinion or a comparison (แพงกว่า แต่เยอะกว่า, สวยกว่าในจอ)",
    "negation": "ไม่ / ไม่ได้ flips the meaning",
    "slang_idiom": "slang, idiom, typo or dialect the model does not know",
    "implicit_emotion": "emotion with no clear sentiment word (เหงา, น้อยใจ)",
    "question_no_keyword": "a product question without question words, or a question that is not about a product",
    "ambiguous": "even people could disagree on the label",
    "other": "anything else, explain in note",
}

# Draft causes for the sample_labeled errors (from reading them); the group should check these.
SAMPLE_DRAFT = {
    "sp001": ("keyword_trap", "ชอบ in a sentence that only describes preferences"),
    "sp016": ("slang_idiom", "หน้างอ is an idiom for people; here it describes the fish"),
    "sp017": ("sarcasm", "เยี่ยมเลย about forgetting an umbrella in the rain"),
    "sp046": ("mixed_or_comparison", "สวยกว่าในจอ is praise; all three models say neg"),
    "sp050": ("implicit_emotion", "loneliness without complaint words; A4 right, BERT wrong"),
    "sp057": ("mixed_or_comparison", "more expensive but bigger portion: two sides cancel"),
    "sp059": ("keyword_trap", "สะดวก in travel information"),
    "sp097": ("keyword_trap", "a wait time is a fact, not a complaint"),
}


def build():
    train, valid, sample, test = load_data()

    # sample_labeled: E2 needs A4 probabilities from the train-only model
    a4 = a4_proba(train, tokenize_all(train["text"]), {"sample": (sample, tokenize_all(sample["text"]))})["sample"]
    c1 = read_proba(RESULTS_DIR / "approach_c_C1_wangchanberta_sample_proba.csv", sample["id"])
    c2 = read_proba(RESULTS_DIR / "approach_c_C2_wangchanberta_balanced_sample_proba.csv", sample["id"])
    s = sample.assign(A4=to_labels(a4), C1=to_labels(c1), E2=to_labels((a4 + c1 + c2) / 3))
    s = s[s["category"] != s["E2"]].assign(source="sample")
    s["cause"] = s["id"].map(lambda i: SAMPLE_DRAFT.get(i, ("", ""))[0])
    s["note"] = s["id"].map(lambda i: SAMPLE_DRAFT.get(i, ("", ""))[1])
    s["label_correct"] = "yes"  # sample_labeled follows the class definitions

    # valid: predictions saved by the earlier scripts
    v = valid[["id", "text", "category"]].copy()
    v["A4"] = pd.read_csv(RESULTS_DIR / "approach_a_valid_predictions.csv")["A4_tfidf12+char_lr_balanced"].values
    v["C1"] = pd.read_csv(RESULTS_DIR / "approach_c_valid_predictions.csv")["C1_wangchanberta"].values
    v["E2"] = pd.read_csv(RESULTS_DIR / "ensemble_valid_predictions.csv")["E2"].values
    errors = v[v["category"] != v["E2"]]
    v = errors.sample(N_VALID, random_state=SEED).assign(source="valid", cause="", note="", label_correct="")
    print(f"valid: {len(errors)} E2 errors of {len(valid)}, sampled {N_VALID}")

    sheet = pd.concat([s, v], ignore_index=True)
    sheet["error"] = sheet["category"] + " -> " + sheet["E2"]
    sheet["better_label"] = ""
    cols = ["source", "id", "text", "category", "E2", "A4", "C1", "error",
            "label_correct", "better_label", "cause", "note"]
    sheet = sheet[cols].sort_values(["source", "error", "id"])  # sample rows first
    sheet.to_csv(SHEET, index=False, encoding="utf-8-sig")
    print(f"saved {SHEET.relative_to(SHEET.parent.parent)} ({len(sheet)} rows: "
          f"{(sheet.source == 'sample').sum()} sample, {(sheet.source == 'valid').sum()} valid)")
    print("\ncauses to use in the 'cause' column:")
    for k, d in CAUSES.items():
        print(f"  {k:20s} {d}")


def summarize():
    sheet = pd.read_csv(SHEET, encoding="utf-8-sig").fillna("")
    blank = (sheet["cause"] == "").sum()
    unknown = sorted(set(sheet["cause"]) - set(CAUSES) - {""})
    if blank:
        print(f"WARNING: {blank} rows have no cause yet")
    if unknown:
        print(f"WARNING: causes not in the list: {unknown}")
    filled = sheet[sheet["cause"] != ""]
    counts = filled.groupby(["source", "cause"]).size().rename("count").reset_index()
    counts["share"] = counts["count"] / counts.groupby("source")["count"].transform("sum")
    counts = counts.sort_values(["source", "count"], ascending=[False, False])
    counts.round(3).to_csv(SUMMARY, index=False)
    print(counts.round(3).to_string(index=False))

    v = filled[filled["source"] == "valid"]
    if len(v):
        noise = (v["label_correct"].str.lower() == "no").mean()
        print(f"\nvalid: {noise:.0%} of the checked errors have a wrong gold label")
    # which model gets the errors of E2 right?
    for model in ["A4", "C1"]:
        right = (filled[model] == filled["category"]).mean()
        print(f"{model} is right on {right:.0%} of the E2 errors in the sheet")
    print(f"saved {SUMMARY.relative_to(SUMMARY.parent.parent)}")


if __name__ == "__main__":
    summarize() if "--summarize" in sys.argv else build()
