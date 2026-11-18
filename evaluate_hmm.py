"""
Classifier-only vs classifier + HMM/Viterbi, and decoder-parameter search.

  python -m evaluation.evaluate_hmm tune       # grid over beta / emit_w on 15 TRAIN trips
  python -m evaluation.evaluate_hmm            # final comparison on the 10 student-test trips
Outputs -> results/hmm/
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_processing.candidates import LABEL_MAX_DIST_M, align_labels
from data_processing.loader import list_trips, load_trip
from evaluation.metrics import route_continuity, route_level
from hmm.pipeline import load_artifacts, predict_trip

ROOT = os.path.join(os.path.dirname(__file__), "..")
RES = os.path.join(ROOT, "results", "hmm")
os.makedirs(RES, exist_ok=True)


def evaluate(trip_dirs, model, cfg):
    rows = []
    for d in trip_dirs:
        r = predict_trip(d, model, cfg)
        gt = load_trip(d)                       # ground truth used only here, for scoring
        true_arc, _, true_d = align_labels(gt)
        rel = true_d <= LABEL_MAX_DIST_M
        for name, arcs in (("classifier_only", r["clf_arcs"]), ("classifier_hmm", r["hmm_arcs"])):
            valid = arcs >= 0
            # classifier-only route = its raw arg-max sequence; HMM route = connected Viterbi path
            seq = np.array(r["route_arcs"]) if name == "classifier_hmm" else arcs[valid]
            rl = route_level(seq, gt.route, gt.arcs)
            rows.append(dict(trip_id=gt.trip_id, method=name,
                             top1_acc=float((arcs == true_arc)[rel].mean()),
                             obs_continuity=route_continuity(arcs[valid], gt.arcs),
                             route_continuity=rl["route_continuity"],
                             route_precision=rl["route_precision"], route_recall=rl["route_recall"],
                             route_f1=rl["route_f1"], n_points=int(rel.sum()),
                             **({k: v for k, v in r["stats"].items()} if name == "classifier_hmm" else {}),
                             **r["timing"]))
    return pd.DataFrame(rows)


def tune():
    """Grid over decoder parameters on 16 TRAIN trips (features/model scored once per trip)."""
    from data_processing.candidates import build_features
    from hmm.pipeline import score_candidates
    from hmm.viterbi import RoadGraph, viterbi
    model, cfg = load_artifacts()
    cache = []
    for d in list_trips("train")[::5][:16]:
        trip = load_trip(d)
        feats, info = build_features(trip, labels=True)
        feats["prob"] = score_candidates(model, feats, cfg["feature_cols"])
        rel = info["true_dist"] <= LABEL_MAX_DIST_M
        cache.append((trip, feats, RoadGraph(trip), info["true_arc"], rel))
        print("cached", trip.trip_id, flush=True)
    res = []
    for beta in (2, 5, 10, 20):
        for w in (0.5, 1.0, 2.0):
            correct = total = 0
            for trip, feats, G, true_arc, rel in cache:
                arcs, _ = viterbi(feats, trip, G, beta=beta, emit_w=w, gap_s=cfg["gap_s"])
                correct += int((arcs == true_arc)[rel].sum()); total += int(rel.sum())
            res.append(dict(beta=beta, emit_w=w, train_top1=correct / total, n_points=total))
            print(res[-1], flush=True)
    res = pd.DataFrame(res)
    res.to_csv(os.path.join(RES, "decoder_grid_train.csv"), index=False)
    best = res.sort_values("train_top1").iloc[-1]
    cfg_path = os.path.join(ROOT, "artifacts", "inference_config.json")
    c = json.load(open(cfg_path)) if os.path.exists(cfg_path) else {}
    c.update(beta=float(best.beta), emit_w=float(best.emit_w))
    json.dump(c, open(cfg_path, "w"), indent=2)
    print("best", dict(best))


def final():
    model, cfg = load_artifacts()
    df = evaluate(list_trips("student_test"), model, cfg)
    df.to_csv(os.path.join(RES, "hmm_vs_classifier_per_trip.csv"), index=False)
    summ = df.groupby("method").apply(lambda g: pd.Series(dict(
        top1_acc_weighted=np.average(g.top1_acc, weights=g.n_points),
        obs_continuity=g.obs_continuity.mean(), route_continuity=g.route_continuity.mean(), route_precision=g.route_precision.mean(),
        route_recall=g.route_recall.mean(), route_f1=g.route_f1.mean())))
    summ.to_csv(os.path.join(RES, "hmm_vs_classifier_summary.csv"))
    print(summ.round(4))
    piv = df.pivot(index="trip_id", columns="method", values="top1_acc")
    ax = piv[["classifier_only", "classifier_hmm"]].plot.bar(figsize=(10, 4))
    ax.set(title="Student-test Top-1 accuracy per trip: classifier only vs + HMM/Viterbi",
           ylim=(max(0, piv.values.min() - .05), 1), ylabel="Top-1 accuracy")
    plt.tight_layout(); plt.savefig(os.path.join(RES, "hmm_vs_classifier.png"), dpi=120); plt.close()


if __name__ == "__main__":
    tune() if sys.argv[1:] == ["tune"] else final()
