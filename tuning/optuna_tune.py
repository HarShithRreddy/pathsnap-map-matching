"""
Bayesian hyperparameter optimisation (Optuna TPE) for the selected model (LightGBM).

  python -m tuning.optuna_tune [n_trials]
Objective : mean 3-fold GroupKFold (by trip) macro F1 on the 80 training trips.
Outputs   : results/tuning/optuna_trials.csv, tuned_vs_untuned.csv, plots,
            artifacts/final_model.joblib, artifacts/inference_config.json
"""
import json
import os
import sys
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import GroupKFold

from data_processing.candidates import FEATURE_COLS
from evaluation.metrics import candidate_metrics
from models.common import SEED, load_data, normalise

ROOT = os.path.join(os.path.dirname(__file__), "..")
RES = os.path.join(ROOT, "results", "tuning")
ART = os.path.join(ROOT, "artifacts")
os.makedirs(RES, exist_ok=True)

UNTUNED = dict(n_estimators=400, learning_rate=0.05, num_leaves=63, subsample=0.8,
               subsample_freq=1, colsample_bytree=0.8)


def make(params):
    return LGBMClassifier(**params, random_state=SEED, n_jobs=2, verbose=-1)


def cv_score(params, tr, folds=3):
    X, y, g = tr[FEATURE_COLS].values, tr.label.values, tr.trip_id.values
    f1s, top1s = [], []
    for a, b in GroupKFold(n_splits=folds).split(X, y, g):
        m = make(params).fit(X[a], y[a])
        vb = tr.iloc[b][["trip_id", "point_idx", "arc_id", "label"]].copy()
        vb["prob"] = normalise(vb, m.predict_proba(X[b])[:, 1])
        r = candidate_metrics(vb)
        f1s.append(r["macro_f1"]); top1s.append(r["top1_acc"])
    return float(np.mean(f1s)), float(np.mean(top1s))


def test_eval(params, tr, te):
    t0 = time.time()
    m = make(params).fit(tr[FEATURE_COLS].values, tr.label.values)
    fit_t = time.time() - t0
    t1 = time.time()
    raw = m.predict_proba(te[FEATURE_COLS].values)[:, 1]
    inf_t = time.time() - t1
    te_ = te[["trip_id", "point_idx", "arc_id", "label"]].copy()
    te_["prob"] = normalise(te_, raw)
    r = candidate_metrics(te_)
    r.update(train_time_s=fit_t, infer_ms_per_1k_points=1e6 * inf_t / r["n_points"])
    return m, r


def main(n_trials=30):
    tr, te = load_data()

    def objective(trial):
        p = dict(
            n_estimators=trial.suggest_int("n_estimators", 100, 800, step=50),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            num_leaves=trial.suggest_int("num_leaves", 15, 255, log=True),
            min_child_samples=trial.suggest_int("min_child_samples", 5, 200, log=True),
            subsample=trial.suggest_float("subsample", 0.5, 1.0), subsample_freq=1,
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
        )
        f1, top1 = cv_score(p, tr)
        trial.set_user_attr("cv_top1", top1)
        return f1

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    t0 = time.time()
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    opt_time = time.time() - t0
    trials = study.trials_dataframe()
    trials.to_csv(os.path.join(RES, "optuna_trials.csv"), index=False)

    best = dict(study.best_params, subsample_freq=1)
    rows = []
    for name, p in (("untuned", UNTUNED), ("tuned", best)):
        cvf1, cvtop1 = cv_score(p, tr)
        m, r = test_eval(p, tr, te)
        rows.append(dict(model=f"LightGBM_{name}", cv_macro_f1=cvf1, cv_top1=cvtop1,
                         test_top1=r["top1_acc"], test_macro_f1=r["macro_f1"],
                         test_log_loss=r["log_loss"], train_time_s=r["train_time_s"],
                         infer_ms_per_1k_points=r["infer_ms_per_1k_points"], params=json.dumps(p)))
        if name == "tuned":
            final_model = m
    cmp = pd.DataFrame(rows)
    cmp.to_csv(os.path.join(RES, "tuned_vs_untuned.csv"), index=False)
    print(cmp.drop(columns="params").round(4))

    winner = cmp.sort_values("cv_macro_f1").iloc[-1]
    if winner.model.endswith("untuned"):
        final_model = make(UNTUNED).fit(tr[FEATURE_COLS].values, tr.label.values)
    joblib.dump(final_model, os.path.join(ART, "final_model.joblib"), compress=3)
    cfg_path = os.path.join(ART, "inference_config.json")
    cfg = json.load(open(cfg_path)) if os.path.exists(cfg_path) else {}
    cfg.update(model=winner.model, params=json.loads(winner.params), feature_cols=FEATURE_COLS)
    cfg.setdefault("beta", 5.0); cfg.setdefault("emit_w", 1.0); cfg.setdefault("gap_s", 60.0)
    json.dump(cfg, open(cfg_path, "w"), indent=2)
    json.dump(dict(n_trials=n_trials, optimisation_time_s=opt_time, best_cv_macro_f1=study.best_value,
                   best_params=best, sampler="TPE", cv="GroupKFold(3) by trip"),
              open(os.path.join(RES, "optuna_summary.json"), "w"), indent=2)

    v = trials["value"].values
    plt.figure(figsize=(8, 4))
    plt.plot(v, "o", alpha=.5, label="trial CV macro F1")
    plt.plot(np.maximum.accumulate(v), "r-", label="best so far")
    plt.xlabel("trial"); plt.ylabel("CV macro F1"); plt.legend(); plt.title("Optuna TPE optimisation history")
    plt.tight_layout(); plt.savefig(os.path.join(RES, "optuna_history.png"), dpi=120); plt.close()
    try:
        imp = optuna.importance.get_param_importances(study)
        plt.figure(figsize=(7, 4)); plt.barh(list(imp)[::-1], list(imp.values())[::-1])
        plt.title("Hyperparameter importance (fANOVA)"); plt.tight_layout()
        plt.savefig(os.path.join(RES, "optuna_param_importance.png"), dpi=120); plt.close()
    except Exception as e:
        print("importance skipped:", e)
    cmp_plot = cmp.set_index("model")[["cv_macro_f1", "test_top1", "test_macro_f1"]]
    ax = cmp_plot.T.plot.bar(figsize=(8, 4)); ax.set_ylim(cmp_plot.values.min() - .02, 1)
    ax.set_title("Untuned vs tuned LightGBM"); plt.xticks(rotation=0); plt.tight_layout()
    plt.savefig(os.path.join(RES, "tuned_vs_untuned.png"), dpi=120); plt.close()


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 30)
