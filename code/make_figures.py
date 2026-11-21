"""Figures for the report, built from results/*.csv.

  report/figures/fig1_macro_f1_valid_vs_sample.png   macro-F1 on valid vs sample_labeled for every model
  report/figures/fig2_per_class_f1_valid.png         per-class F1 on valid for 4 key models

Run from the repo root (after the experiment scripts):  python code/make_figures.py
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import LABELS, RESULTS_DIR, ROOT

OUT = ROOT / "report" / "figures"

# Validated categorical palette (light mode, 4 slots, fixed order) + neutral tokens
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
SURFACE, TEXT, TEXT_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": TEXT_2, "xtick.color": TEXT_2, "ytick.color": TEXT_2,
    "text.color": TEXT, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
})


def load_all():
    a = pd.read_csv(RESULTS_DIR / "approach_a_results.csv", index_col=0)
    b = pd.read_csv(RESULTS_DIR / "approach_b_results.csv", index_col=0)
    e = pd.read_csv(RESULTS_DIR / "ensemble_results.csv", index_col=0)
    rows = {
        "Baseline NB": a.loc["A0_count_nb (baseline)"],
        "A2 TF-IDF LR balanced": a.loc["A2_tfidf_lr_balanced"],
        "A4 + n-gram + char": a.loc["A4_tfidf12+char_lr_balanced"],
        "B4 + norm + q + relabel": b.loc["B4_+relabel_noise"],
        "C1 WangchanBERTa": e.loc["C1_wangchanberta"],
        "C2 WangchanBERTa balanced": e.loc["C2_wangchanberta_balanced"],
        "E1 0.4*C1 + 0.6*A4": e.loc["E1_C1+A4_w0.4"],
        "E2 mean(C1, C2, A4)": e.loc["E2_C1+C2+A4_mean"],
    }
    return pd.DataFrame(rows).T


def fig_valid_vs_sample(df):
    fig, ax = plt.subplots(figsize=(6.4, 3.4), dpi=200)
    y = np.arange(len(df))[::-1]
    for yi, (_, r) in zip(y, df.iterrows()):
        ax.plot([r.valid_macroF1, r.sample_macroF1], [yi, yi], color=GRID, lw=2, zorder=1)
    ax.scatter(df.valid_macroF1, y, s=42, color=SERIES[0], zorder=2, label="valid (noisy labels)",
               edgecolors=SURFACE, linewidths=2)
    ax.scatter(df.sample_macroF1, y, s=42, color=SERIES[1], zorder=3, label="sample_labeled (clean, n=100)",
               edgecolors=SURFACE, linewidths=2)
    for yi, (name, r) in zip(y, df.iterrows()):
        ax.text(max(r.valid_macroF1, r.sample_macroF1) + 0.012, yi, f"{r.valid_macroF1:.3f} / {r.sample_macroF1:.3f}",
                va="center", fontsize=7.5, color=TEXT_2)
    ax.set_yticks(y, df.index)
    ax.set_xlim(0.4, 1.02)
    ax.set_xlabel("macro-F1")
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_macro_f1_valid_vs_sample.png")
    plt.close(fig)


def fig_per_class(df):
    models = ["Baseline NB", "A4 + n-gram + char", "C1 WangchanBERTa", "E2 mean(C1, C2, A4)"]
    groups = LABELS + ["macro"]
    cols = [f"valid_F1_{l}" for l in LABELS] + ["valid_macroF1"]
    fig, ax = plt.subplots(figsize=(6.4, 3.0), dpi=200)
    x = np.arange(len(groups))
    width = 0.2
    for i, m in enumerate(models):
        vals = df.loc[m, cols].to_numpy(dtype=float)
        ax.bar(x + (i - 1.5) * width, vals, width, color=SERIES[i], label=m,
               edgecolor=SURFACE, linewidth=1.5)  # surface gap between adjacent bars
    # exact numbers are in the results table (table view); only mark the zero bar so it does not read as missing
    q_idx = groups.index("q")
    ax.text(x[q_idx] - 1.5 * width, 0.015, "0", ha="center", fontsize=7, color=TEXT_2)
    ax.set_xticks(x, ["pos", "neu", "neg", "q", "macro-F1"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("F1 on valid")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", length=0)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4, frameon=False, fontsize=7.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_per_class_f1_valid.png")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_all()
    fig_valid_vs_sample(df)
    fig_per_class(df)
    print(f"saved figures to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
