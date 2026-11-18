"""
Merge every model's results into one comparison table and the comparison charts.

  python -m models.compare
Reads results/models/<model>.json written by the train_*.py scripts.
"""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from models.common import RES

ORDER = ["DecisionTree", "RandomForest", "HistGradientBoosting", "LightGBM", "MLP"]


def main():
    rows = [json.load(open(f)) for f in glob.glob(os.path.join(RES, "*.json"))
            if not f.endswith("baseline_nearest_arc.json")]
    df = pd.DataFrame(rows)
    df["o"] = df.model.map({m: i for i, m in enumerate(ORDER)}).fillna(99)
    df = df.sort_values("o").drop(columns="o").reset_index(drop=True)
    df.to_csv(os.path.join(RES, "model_comparison.csv"), index=False)
    folds = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(RES, "cv_folds_*.csv"))])
    folds.to_csv(os.path.join(RES, "cv_folds.csv"), index=False)

    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    ax[0].bar(df.model, df.cv_macro_f1_mean, yerr=df.cv_macro_f1_std, capsize=4, color="tab:blue")
    ax[0].set(title="5-fold GroupKFold CV macro F1 (train trips)", ylim=(df.cv_macro_f1_mean.min() - .05, 1))
    x = np.arange(len(df)); w = .38
    ax[1].bar(x - w / 2, df.test_top1, w, label="Top-1 accuracy")
    ax[1].bar(x + w / 2, df.test_macro_f1, w, label="Macro F1")
    ax[1].set_xticks(x, df.model); ax[1].legend()
    ax[1].set(title="Student-test performance", ylim=(min(df.test_top1.min(), df.test_macro_f1.min()) - .05, 1))
    ax[2].bar(df.model, df.test_log_loss, color="tab:red")
    ax[2].set(title="Student-test log loss (lower is better)")
    for a in ax:
        a.tick_params(axis="x", rotation=30)
    plt.tight_layout(); plt.savefig(os.path.join(RES, "model_comparison.png"), dpi=120); plt.close()

    show = ["model", "cv_macro_f1_mean", "cv_macro_f1_std", "train_macro_f1", "test_top1",
            "test_macro_f1", "test_log_loss", "test_route_continuity", "train_time_s"]
    print(df[show].round(4).to_string(index=False))
    best = df.sort_values("cv_macro_f1_mean").iloc[-1].model
    print(f"\nSelected by CV macro F1: {best}")


if __name__ == "__main__":
    main()
