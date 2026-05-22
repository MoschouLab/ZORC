"""
09g_confusion_matrix.py — Publication-quality confusion matrix figure.

Input : results/09f_predictions_final_numt_clean.csv
Output: results/figures/confusion_matrix_numt_clean.{png,pdf}
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

PREDICTIONS_CSV = "results/09f_predictions_final_numt_clean.csv"
OUT_PNG = "results/figures/confusion_matrix_numt_clean.png"
OUT_PDF = "results/figures/confusion_matrix_numt_clean.pdf"
AUROC = 0.7695  # manuscript value (numt_clean, P9f)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/zorc_config.yaml")
    return p.parse_args()


def main():
    parse_args()  # accept --config for Snakemake compatibility

    df = pd.read_csv(PREDICTIONS_CSV)
    test = df[df["split"] == "test"].copy()
    n_total = len(test)

    y_true = test["class"].values
    y_pred = test["pred"].values

    cm = confusion_matrix(y_true, y_pred, labels=[1, 0])
    tn_pos = cm[0, 0]  # true positive (class=1 pred=1)
    fp_pos = cm[0, 1]  # false negative (class=1 pred=0)
    fn_pos = cm[1, 0]  # false positive (class=0 pred=1)
    tn_neg = cm[1, 1]  # true negative (class=0 pred=0)

    TP, FN, FP, TN = tn_pos, fp_pos, fn_pos, tn_neg

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall    = recall_score(y_true, y_pred, zero_division=0)
    f1        = f1_score(y_true, y_pred, zero_division=0)

    print(f"Test set: {n_total} genes")
    print(f"TP={TP}  FP={FP}")
    print(f"FN={FN}  TN={TN}")
    print(f"Precision={precision:.4f}  Recall={recall:.4f}  F1={f1:.4f}")

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 5))

    # Cell values: absolute count + % of total test set
    annot = np.array([
        [f"{TP}\n({100*TP/n_total:.1f}%)", f"{FN}\n({100*FN/n_total:.1f}%)"],
        [f"{FP}\n({100*FP/n_total:.1f}%)", f"{TN}\n({100*TN/n_total:.1f}%)"],
    ])

    # Normalised values for colour intensity (row-normalised avoids colour
    # dominance from class-size imbalance)
    cm_norm = cm.astype(float)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # avoid /0
    cm_norm = cm_norm / row_sums

    sns.heatmap(
        cm_norm,
        annot=annot,
        fmt="",
        cmap="Blues",
        linewidths=0.5,
        linecolor="white",
        cbar=False,
        ax=ax,
        annot_kws={"size": 14, "weight": "bold"},
        vmin=0,
        vmax=1,
    )

    ax.set_xlabel("Predicted label", fontsize=13, labelpad=8)
    ax.set_ylabel("True label", fontsize=13, labelpad=8)
    ax.set_xticklabels(["Enriched", "Not enriched"], fontsize=11)
    ax.set_yticklabels(["Enriched", "Not enriched"], fontsize=11, va="center")
    ax.tick_params(left=False, bottom=False)

    # Remove all spines
    for spine in ax.spines.values():
        spine.set_visible(False)

    metrics_line = (
        f"Precision = {precision:.3f}    "
        f"Recall = {recall:.3f}    "
        f"F1 = {f1:.3f}    "
        f"AUROC = {AUROC:.4f}"
    )
    fig.text(
        0.5, 0.01,
        metrics_line,
        ha="center", va="bottom",
        fontsize=9.5,
        color="#333333",
    )

    fig.suptitle(
        "ZORC — Test set confusion matrix (NUMT-clean, n = {})".format(n_total),
        fontsize=12,
        y=0.98,
    )

    plt.tight_layout(rect=[0, 0.06, 1, 0.96])

    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {OUT_PNG}")
    print(f"Saved: {OUT_PDF}")


if __name__ == "__main__":
    main()
