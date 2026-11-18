"""
Shared training / evaluation harness used by every model script in models/.

- Training data : 80 training trips only (reliable labels, every STRIDE-th GPS point,
                  because consecutive 1 Hz points are near-duplicates).
- Validation    : GroupKFold(5) grouped by trip_id -> no trip in both train and validation.
- Student-test  : all 10 student-test trips, every observation.
Each model's results -> results/models/<name>.json (+ cv folds, confusion matrix);
trained model -> artifacts/models/<name>.joblib.  models/compare.py merges them.
"""
import json
import os
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import GroupKFold

from data_processing.candidates import FEATURE_COLS
from data_processing.loader import load_trip
from evaluation.metrics import argmax_flags, candidate_metrics, predicted_arcs, route_continuity

SEED = 42
STRIDE = 3
N_FOLDS = 5
ROOT = os.path.join(os.path.dirname(__file__), "..")
RES = os.path.join(ROOT, "results", "models")
ART = os.path.join(ROOT, "artifacts", "models")
os.makedirs(RES, exist_ok=True)
os.makedirs(ART, exist_ok=True)


def load_data():
    tr = pd.read_parquet(os.path.join(ROOT, "data", "processed", "features_train.parquet"))
    te = pd.read_parquet(os.path.join(ROOT, "data", "processed", "features_test.parquet"))
    tr = tr[(tr.label_reliable == 1) & (tr.point_idx % STRIDE == 0)].reset_index(drop=True)
    te = te[te.label_reliable == 1].reset_index(drop=True)
    return tr, te


def normalise(df, raw):
    """Renormalise P(true) across the candidates of each GPS observation."""
    s = pd.Series(raw, index=df.index).groupby([df.trip_id, df.point_idx]).transform("sum")
    return raw / np.maximum(s.values, 1e-12)


def continuity_for(df, arcs_by_trip):
    pa = predicted_arcs(df)
    return float(np.mean([route_continuity(g.arc_id.values, arcs_by_trip[t])
                          for t, g in pa.groupby("trip_id")]))


def run_model(name, family, model, config, data=None):
    """Cross-validate, refit on all training trips, evaluate on student-test, save everything."""
    tr, te = data if data is not None else load_data()
    Xtr, ytr, gtr = tr[FEATURE_COLS].values, tr.label.values, tr.trip_id.values
    Xte = te[FEATURE_COLS].values
    arcs_by_trip = {t: load_trip(os.path.join(ROOT, "data", "raw", "student_test", t)).arcs
                    for t in te.trip_id.unique()}
    print(f"\n=== {name}  (train rows {len(tr)}, trips {tr.trip_id.nunique()} | test rows {len(te)})", flush=True)

    folds = []
    for k, (a, b) in enumerate(GroupKFold(n_splits=N_FOLDS).split(Xtr, ytr, gtr)):
        m = clone(model).fit(Xtr[a], ytr[a])
        vb = tr.iloc[b][["trip_id", "point_idx", "arc_id", "label"]].copy()
        vb["prob"] = normalise(vb, m.predict_proba(Xtr[b])[:, 1])
        r = candidate_metrics(vb)
        folds.append(dict(model=name, fold=k, **r))
        print(f"  fold {k}: macroF1={r['macro_f1']:.4f} top1={r['top1_acc']:.4f}", flush=True)

    t0 = time.time()
    m = clone(model).fit(Xtr, ytr)
    train_time = time.time() - t0
    t1 = time.time()
    raw_te = m.predict_proba(Xte)[:, 1]
    infer = time.time() - t1
    te_ = te[["trip_id", "point_idx", "arc_id", "label"]].copy()
    te_["prob"] = normalise(te_, raw_te)
    rt = candidate_metrics(te_)

    # training-fit check on a fixed subset of 20 training trips (over/underfitting evidence)
    sub = tr.trip_id.isin(sorted(tr.trip_id.unique())[::4])
    tr_ = tr.loc[sub, ["trip_id", "point_idx", "arc_id", "label"]].copy()
    tr_["prob"] = normalise(tr_, m.predict_proba(Xtr[sub.values])[:, 1])
    rtr = candidate_metrics(tr_)
    cont = continuity_for(te_, arcs_by_trip)

    f1s = [f["macro_f1"] for f in folds]
    result = dict(
        model=name, family=family, config=config,
        cv_macro_f1_mean=float(np.mean(f1s)), cv_macro_f1_std=float(np.std(f1s)),
        cv_top1_mean=float(np.mean([f["top1_acc"] for f in folds])),
        train_macro_f1=rtr["macro_f1"], train_top1=rtr["top1_acc"],
        test_top1=rt["top1_acc"], test_macro_f1=rt["macro_f1"],
        test_precision_macro=rt["precision_macro"], test_recall_macro=rt["recall_macro"],
        test_log_loss=rt["log_loss"], test_candidate_miss_rate=rt["candidate_miss_rate"],
        test_route_continuity=cont, train_time_s=train_time,
        infer_ms_per_1k_points=1e6 * infer / rt["n_points"],
    )
    print(f"  TEST top1={rt['top1_acc']:.4f} macroF1={rt['macro_f1']:.4f} logloss={rt['log_loss']:.4f} "
          f"cont={cont:.4f} | train top1={rtr['top1_acc']:.4f} | fit {train_time:.1f}s", flush=True)

    json.dump(result, open(os.path.join(RES, f"{name}.json"), "w"), indent=2)
    pd.DataFrame(folds).to_csv(os.path.join(RES, f"cv_folds_{name}.csv"), index=False)
    joblib.dump(m, os.path.join(ART, f"{name}.joblib"), compress=3)
    cm = confusion_matrix(te_.label, argmax_flags(te_))
    ConfusionMatrixDisplay(cm, display_labels=["other", "true arc"]).plot(values_format="d")
    plt.title(f"{name} - student-test confusion matrix"); plt.tight_layout()
    plt.savefig(os.path.join(RES, f"cm_{name}.png"), dpi=110); plt.close()
    return m, result


def nearest_arc_baseline(data=None):
    """Non-learned reference: always pick the closest candidate arc."""
    tr, te = data if data is not None else load_data()
    arcs_by_trip = {t: load_trip(os.path.join(ROOT, "data", "raw", "student_test", t)).arcs
                    for t in te.trip_id.unique()}
    te_ = te[["trip_id", "point_idx", "arc_id", "label"]].copy()
    te_["prob"] = np.exp(-te["dist_m"].values / 10)
    r = candidate_metrics(te_)
    out = {"test_top1": r["top1_acc"], "test_macro_f1": r["macro_f1"],
           "route_continuity": continuity_for(te_, arcs_by_trip)}
    json.dump(out, open(os.path.join(RES, "baseline_nearest_arc.json"), "w"), indent=2)
    return out
