"""
End-to-end prediction for one trip folder (used by evaluation scripts and the API).

  predict_trip(trip_dir) -> dict(pred_arcs, clf_arcs, route_arcs, stats, ...)

Ground-truth (.route) is never read here: the trip is loaded with with_route=False.
"""
import json
import os
import time

import joblib
import numpy as np
import pandas as pd

from data_processing.candidates import FEATURE_COLS, build_features
from data_processing.loader import load_trip
from hmm.viterbi import RoadGraph, fill_route, viterbi, BETA, EMIT_W, GAP_S

ROOT = os.path.join(os.path.dirname(__file__), "..")
ART = os.path.join(ROOT, "artifacts")


def load_artifacts(model_path=None, cfg_path=None):
    model_path = model_path or os.path.join(ART, "final_model.joblib")
    cfg_path = cfg_path or os.path.join(ART, "inference_config.json")
    cfg = dict(feature_cols=FEATURE_COLS, beta=BETA, emit_w=EMIT_W, gap_s=GAP_S)
    if os.path.exists(cfg_path):
        cfg.update(json.load(open(cfg_path)))
    return joblib.load(model_path), cfg


def score_candidates(model, feats, cols):
    raw = model.predict_proba(feats[cols].values)[:, 1]
    s = pd.Series(raw, index=feats.index).groupby(feats.point_idx).transform("sum").values
    return raw / np.maximum(s, 1e-12)


def classifier_only(feats):
    idx = feats.groupby("point_idx")["prob"].idxmax()
    return idx.index.values, feats.loc[idx.values, "arc_id"].values


def predict_trip(trip_dir, model=None, cfg=None):
    if model is None:
        model, cfg = load_artifacts()
    t0 = time.time()
    trip = load_trip(trip_dir, with_route=False)        # no ground truth on the prediction side
    feats, _ = build_features(trip, labels=False)
    feats["prob"] = score_candidates(model, feats, cfg["feature_cols"])
    t1 = time.time()
    N = len(trip.track_xy)
    clf = np.full(N, -1, dtype=np.int64)
    p, a = classifier_only(feats)
    clf[p] = a
    G = RoadGraph(trip)
    hmm_arcs, stats = viterbi(feats, trip, G, beta=cfg["beta"], emit_w=cfg["emit_w"], gap_s=cfg["gap_s"])
    route, n_filled = fill_route(G, hmm_arcs)   # connected route (shortest-path gap filling)
    stats["gap_filled_arcs"] = n_filled
    t2 = time.time()
    return dict(trip=trip, clf_arcs=clf, hmm_arcs=hmm_arcs, route_arcs=route, stats=stats,
                timing=dict(features_and_model_s=round(t1 - t0, 3), viterbi_s=round(t2 - t1, 3)))
