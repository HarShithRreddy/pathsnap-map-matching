"""
Build candidate-level feature tables for the training and student-test partitions.

  python -m data_processing.build_dataset
Outputs:
  data/processed/features_train.parquet
  data/processed/features_test.parquet
  results/eda/label_alignment_stats.csv
"""
import os
import time

import numpy as np
import pandas as pd

from data_processing.loader import list_trips, load_trip
from data_processing.candidates import build_features, LABEL_MAX_DIST_M, CAND_RADIUS_M

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "data", "processed")
RES = os.path.join(ROOT, "results", "eda")


def run(partition):
    frames, stats = [], []
    for d in list_trips(partition):
        t0 = time.time()
        trip = load_trip(d)
        df, info = build_features(trip)
        N = len(trip.track_xy)
        rel = info["true_dist"] <= LABEL_MAX_DIST_M
        has = np.zeros(N, bool)
        has[df.point_idx.unique()] = True
        hit = np.zeros(N, bool)
        hit[df.loc[df.label == 1, "point_idx"].unique()] = True
        stats.append(dict(
            partition=partition, trip_id=trip.trip_id, n_points=N, n_rows=len(df),
            reliable_points=int(rel.sum()), unreliable_frac=round(1 - rel.mean(), 4),
            points_without_candidates=int((~has).sum()),
            candidate_miss_rate_reliable=round(1 - hit[rel].mean(), 4) if rel.any() else np.nan,
            median_label_dist_m=round(float(np.median(info["true_dist"][rel])), 2) if rel.any() else np.nan,
            avg_cands_per_point=round(len(df) / max(has.sum(), 1), 2),
            secs=round(time.time() - t0, 2)))
        frames.append(df)
        print(partition, trip.trip_id, N, len(df), f"{time.time()-t0:.1f}s", flush=True)
    feats = pd.concat(frames, ignore_index=True)
    return feats, pd.DataFrame(stats)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(RES, exist_ok=True)
    all_stats = []
    for part, name in (("train", "train"), ("student_test", "test")):
        f, s = run(part)
        f.to_parquet(os.path.join(OUT, f"features_{name}.parquet"), index=False)
        all_stats.append(s)
        print(name, f.shape)
    pd.concat(all_stats).to_csv(os.path.join(RES, "label_alignment_stats.csv"), index=False)
    print(f"radius={CAND_RADIUS_M}m label_max={LABEL_MAX_DIST_M}m")
