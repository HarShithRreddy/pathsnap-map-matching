"""
PathSnap FastAPI backend (Laptop 1).

Run:
  PATHSNAP_TRIPS_DIR=data/raw/evaluator_test uvicorn api.app:app --host 0.0.0.0 --port 8000

Endpoints
  GET  /health                 -> status, model name
  GET  /trips                  -> trip ids available for evaluation
  POST /predict/{trip_id}      -> predicted route + server-side accuracy

Ground truth (.route) is read ONLY inside `server_side_score`, after prediction,
and is never returned to the client.
"""
import logging
import os
import sys
import time

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data_processing.candidates import LABEL_MAX_DIST_M, align_labels   # noqa: E402
from data_processing.loader import load_trip                             # noqa: E402
from evaluation.metrics import route_continuity, route_level             # noqa: E402
from hmm.pipeline import load_artifacts, predict_trip                    # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
TRIPS_DIR = os.environ.get("PATHSNAP_TRIPS_DIR", os.path.join(ROOT, "data", "raw", "student_test"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("pathsnap")

app = FastAPI(title="PathSnap Map-Matching API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

MODEL, CFG = load_artifacts()
log.info("Loaded model=%s beta=%s emit_w=%s | trips dir=%s",
         CFG.get("model", type(MODEL).__name__), CFG["beta"], CFG["emit_w"], TRIPS_DIR)


def available_trips():
    if not os.path.isdir(TRIPS_DIR):
        return []
    return sorted(d for d in os.listdir(TRIPS_DIR) if os.path.isdir(os.path.join(TRIPS_DIR, d)))


def server_side_score(trip_dir, clf_arcs, hmm_arcs, hmm_route):
    """Uses the reference route held on the server. Returns metrics only."""
    gt = load_trip(trip_dir, with_route=True)
    if gt.route is None:
        return None
    true_arc, _, true_d = align_labels(gt)
    rel = true_d <= LABEL_MAX_DIST_M
    out = {}
    for name, arcs in (("classifier_only", clf_arcs), ("classifier_hmm", hmm_arcs)):
        v = arcs >= 0
        seq = np.array(hmm_route) if name == "classifier_hmm" else arcs[v]
        rl = route_level(seq, gt.route, gt.arcs)
        out[name] = dict(top1_accuracy=round(float((arcs == true_arc)[rel].mean()), 4),
                         route_precision=round(rl["route_precision"], 4),
                         route_recall=round(rl["route_recall"], 4),
                         route_f1=round(rl["route_f1"], 4),
                         route_continuity=round(rl["route_continuity"], 4))
    out["evaluated_points"] = int(rel.sum())
    return out


@app.get("/")
def root():
    return {"service": "PathSnap map-matching API",
            "endpoints": ["GET /health", "GET /trips", "POST /predict/{trip_id}", "GET /docs"]}

@app.get("/health")
def health():
    return {"status": "ok", "model": CFG.get("model", type(MODEL).__name__), "trips_dir": TRIPS_DIR}


@app.get("/trips")
def trips():
    t = available_trips()
    log.info("GET /trips -> %d trips", len(t))
    return {"trips": t}


@app.post("/predict/{trip_id}")
def predict(trip_id: str):
    if trip_id not in available_trips():
        raise HTTPException(404, f"trip {trip_id} not found")
    trip_dir = os.path.join(TRIPS_DIR, trip_id)
    t0 = time.time()
    log.info(">>> POST /predict/%s received", trip_id)
    r = predict_trip(trip_dir, MODEL, CFG)
    pred_time = time.time() - t0
    metrics = server_side_score(trip_dir, r["clf_arcs"], r["hmm_arcs"], r["route_arcs"])
    trip = r["trip"]
    nl = trip.nodes_ll
    route_geo = [[[float(nl[a, 1]), float(nl[a, 0])], [float(nl[b, 1]), float(nl[b, 0])]]
                 for a, b in trip.arcs[r["route_arcs"]]]
    ll = trip.track_ll
    step = max(1, len(ll) // 2000)
    gps = [[float(la), float(lo)] for lo, la in ll[::step, :2]]
    if metrics:
        log.info("<<< trip %s | %d GPS pts | HMM top1=%.4f (clf-only %.4f) | route F1=%.4f | %.2fs",
                 trip_id, len(ll), metrics["classifier_hmm"]["top1_accuracy"],
                 metrics["classifier_only"]["top1_accuracy"], metrics["classifier_hmm"]["route_f1"], pred_time)
    return {
        "trip_id": trip_id,
        "n_gps_points": int(len(ll)),
        "n_predicted_arcs": len(r["route_arcs"]),
        "predicted_arc_ids": r["route_arcs"],
        "predicted_route_segments": route_geo,
        "gps_points": gps,
        "decoder_stats": r["stats"],
        "timing_s": dict(r["timing"], total=round(pred_time, 3)),
        "metrics": metrics,
    }
