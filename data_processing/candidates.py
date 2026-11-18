"""
Candidate generation, per-observation label alignment and feature engineering.

One row = (GPS observation i, candidate arc a). Label = 1 if a is the reference
arc aligned to observation i.
"""
import numpy as np
import pandas as pd
import shapely
from shapely import STRtree

CAND_RADIUS_M = 50.0     # search radius around each GPS point
CAND_TOPK = 20           # keep at most K nearest arcs (directed, so ~K/2 roads)
HEADING_WIN = 2          # +-points used for the smoothed GPS bearing
NEIGH_K = 5              # look-ahead/look-behind offset for context distance features
LABEL_MAX_DIST_M = 100.0  # aligned label farther than this is treated as unreliable

FEATURE_COLS = [
    "dist_m", "dist_minus_min", "dist_ratio", "rank", "n_cands",
    "heading_diff", "speed", "heading_valid", "proj_frac", "arc_len",
    "from_in_deg", "to_out_deg", "is_bidir", "dist_prev", "dist_next",
    "head_diff_prev", "head_diff_next", "dt_prev",
]


# ----------------------------------------------------------------------------- geometry
def seg_arrays(trip):
    a = trip.arcs
    p0 = trip.nodes_xy[a[:, 0]]
    p1 = trip.nodes_xy[a[:, 1]]
    return p0, p1


def point_seg_dist(P, A, B):
    """Distance from points P to segments AB (all (n,2), broadcast). Returns dist, frac."""
    AB = B - A
    L2 = (AB ** 2).sum(-1)
    L2 = np.where(L2 == 0, 1e-9, L2)
    u = np.clip(((P - A) * AB).sum(-1) / L2, 0, 1)
    proj = A + u[..., None] * AB
    return np.sqrt(((P - proj) ** 2).sum(-1)), u


def bearing(dx, dy):
    return (np.degrees(np.arctan2(dx, dy)) + 360) % 360


def ang_diff(a, b):
    d = np.abs(a - b) % 360
    return np.minimum(d, 360 - d)


def gps_kinematics(trip, w=HEADING_WIN):
    xy, t = trip.track_xy, trip.t
    n = len(xy)
    lo = np.clip(np.arange(n) - w, 0, n - 1)
    hi = np.clip(np.arange(n) + w, 0, n - 1)
    d = xy[hi] - xy[lo]
    dt = np.maximum(t[hi] - t[lo], 1e-6)
    speed = np.hypot(d[:, 0], d[:, 1]) / dt
    head = bearing(d[:, 0], d[:, 1])
    dt_prev = np.r_[0.0, np.diff(t)]
    return head, speed, dt_prev


# ----------------------------------------------------------------------------- labels
def align_labels(trip):
    """
    The .route file lists arcs in order but gives NO per-observation label.
    We assign each GPS point to a route position j(i) such that j is
    non-decreasing in time (vehicle moves forward along the route) and the
    total point-to-arc distance is minimal (dynamic programming, O(N*R)).
    Returns arc_id per point and the aligned distance.
    """
    r = trip.route
    p0, p1 = seg_arrays(trip)
    A, B = p0[r], p1[r]
    P = trip.track_xy
    N, R = len(P), len(r)
    back = np.zeros((N, R), dtype=np.int32)
    prev = None
    for i in range(N):
        c, _ = point_seg_dist(P[i][None, :], A, B)
        if prev is None:
            cur = c
            back[i] = np.arange(R)
        else:
            # best predecessor k <= j : running minimum of prev
            idx = np.arange(R)
            run_min_idx = np.maximum.accumulate(np.where(prev == np.minimum.accumulate(prev), idx, 0))
            cur = c + prev[run_min_idx]
            back[i] = run_min_idx
        prev = cur
    j = np.empty(N, dtype=np.int64)
    j[-1] = int(np.argmin(prev))
    for i in range(N - 1, 0, -1):
        j[i - 1] = back[i, j[i]]
    arc = r[j]
    d, _ = point_seg_dist(P, p0[arc], p1[arc])
    return arc, j, d


# ----------------------------------------------------------------------------- candidates
class ArcIndex:
    def __init__(self, trip):
        self.p0, self.p1 = seg_arrays(trip)
        geoms = shapely.linestrings(np.stack([self.p0, self.p1], 1))
        self.tree = STRtree(geoms)
        a = trip.arcs
        V = len(trip.nodes_xy)
        self.in_deg = np.bincount(a[:, 1], minlength=V)
        self.out_deg = np.bincount(a[:, 0], minlength=V)
        key = a[:, 0] * (V + 1) + a[:, 1]
        rkey = a[:, 1] * (V + 1) + a[:, 0]
        self.is_bidir = np.isin(rkey, key)
        d = self.p1 - self.p0
        self.arc_len = np.hypot(d[:, 0], d[:, 1])
        self.arc_head = bearing(d[:, 0], d[:, 1])
        self.arcs = a


def make_candidates(trip, idx=None, radius=CAND_RADIUS_M, topk=CAND_TOPK):
    """Return DataFrame (point_idx, arc_id, dist_m, proj_frac) for every point."""
    idx = idx or ArcIndex(trip)
    pts = shapely.points(trip.track_xy)
    pi, ai = idx.tree.query(pts, predicate="dwithin", distance=radius)
    d, u = point_seg_dist(trip.track_xy[pi], idx.p0[ai], idx.p1[ai])
    df = pd.DataFrame({"point_idx": pi, "arc_id": ai, "dist_m": d, "proj_frac": u})
    df = df.sort_values(["point_idx", "dist_m"], kind="stable")
    df["rank"] = df.groupby("point_idx").cumcount()
    df = df[df["rank"] < topk].reset_index(drop=True)
    return df, idx


def build_features(trip, labels=True, point_stride=1):
    """Candidates + features (+ label) for one trip."""
    df, idx = make_candidates(trip)
    if point_stride > 1:
        df = df[df.point_idx % point_stride == 0].reset_index(drop=True)
    head, speed, dt_prev = gps_kinematics(trip)
    pi, ai = df.point_idx.values, df.arc_id.values
    N = len(trip.track_xy)

    g = df.groupby("point_idx")["dist_m"]
    df["n_cands"] = g.transform("size")
    dmin = g.transform("min")
    df["dist_minus_min"] = df.dist_m - dmin
    df["dist_ratio"] = df.dist_m / (dmin + 1.0)
    df["heading_diff"] = ang_diff(head[pi], idx.arc_head[ai])
    df["speed"] = speed[pi]
    df["heading_valid"] = (speed[pi] > 1.0).astype(np.int8)
    df["arc_len"] = idx.arc_len[ai]
    df["from_in_deg"] = idx.in_deg[idx.arcs[ai, 0]]
    df["to_out_deg"] = idx.out_deg[idx.arcs[ai, 1]]
    df["is_bidir"] = idx.is_bidir[ai].astype(np.int8)
    for name, off in (("prev", -NEIGH_K), ("next", NEIGH_K)):
        q = np.clip(pi + off, 0, N - 1)
        dd, _ = point_seg_dist(trip.track_xy[q], idx.p0[ai], idx.p1[ai])
        df[f"dist_{name}"] = np.minimum(dd, 200.0)
        df[f"head_diff_{name}"] = ang_diff(head[q], idx.arc_head[ai])
    df["dt_prev"] = np.minimum(dt_prev[pi], 600.0)
    df["t"] = trip.t[pi]
    df.insert(0, "trip_id", trip.trip_id)

    if labels and trip.route is not None:
        true_arc, _, true_d = align_labels(trip)
        df["true_arc"] = true_arc[pi]
        df["label"] = (df.arc_id.values == true_arc[pi]).astype(np.int8)
        df["label_reliable"] = (true_d[pi] <= LABEL_MAX_DIST_M).astype(np.int8)
        return df, dict(true_arc=true_arc, true_dist=true_d)
    return df, None
