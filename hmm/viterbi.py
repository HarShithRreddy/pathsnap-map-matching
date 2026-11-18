"""
HMM map matching with Viterbi decoding.

Hidden state at observation i : one candidate arc (from data_processing.candidates).
Emission      log P(z_i | arc)   = w_e * log(p_clf(arc | features_i))
                                    (normalised classifier score per observation)
Transition    log P(arc_b | arc_a) = -| d_route(a->b) - d_gc(i-1,i) | / beta
                d_route : distance along the road network from the projection of
                          z_{i-1} on arc a to the projection of z_i on arc b
                          (Dijkstra on arc lengths, cached, bounded search)
                d_gc    : straight-line distance between the two GPS points
              (Newson & Krumm 2009 style). Unreachable pairs get -inf.
Breaks        The chain is restarted (fresh Viterbi segment) when
                - the time gap dt > GAP_S, or
                - an observation has no candidates (it gets arc -1), or
                - every transition into observation i is -inf (disconnected).
"""
import heapq
from collections import defaultdict

import numpy as np

GAP_S = 60.0
BETA = 5.0
EMIT_W = 1.0
BACKWARD_TOL_M = 15.0     # GPS jitter: small backward moves on the same arc allowed
MAX_SPEED = 60.0          # m/s, bounds the Dijkstra search radius


class RoadGraph:
    def __init__(self, trip):
        self.arcs = trip.arcs
        p0 = trip.nodes_xy[trip.arcs[:, 0]]
        p1 = trip.nodes_xy[trip.arcs[:, 1]]
        self.arc_len = np.hypot(*(p1 - p0).T)
        self.adj = defaultdict(list)
        for e, (u, v) in enumerate(trip.arcs):
            self.adj[int(u)].append((int(v), float(self.arc_len[e])))
        self.cache = {}   # node -> (cutoff, {node: dist})

    def dists_from(self, src, cutoff):
        hit = self.cache.get(src)
        if hit is not None and hit[0] >= cutoff:
            return hit[1]
        dist = {src: 0.0}
        pq = [(0.0, src)]
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist.get(u, np.inf) or d > cutoff:
                continue
            for v, w in self.adj[u]:
                nd = d + w
                if nd <= cutoff and nd < dist.get(v, np.inf):
                    dist[v] = nd
                    heapq.heappush(pq, (nd, v))
        self.cache[src] = (cutoff, dist)
        return dist

    def arc_path(self, src, dst, cutoff=3000.0):
        """Arc ids of the shortest path src_node -> dst_node (None if not reachable)."""
        if not hasattr(self, "arc_of"):
            self.arc_of = {(int(u), int(v)): e for e, (u, v) in enumerate(self.arcs)}
        dist, prev = {src: 0.0}, {}
        pq = [(0.0, src)]
        while pq:
            d, u = heapq.heappop(pq)
            if u == dst:
                break
            if d > dist.get(u, np.inf) or d > cutoff:
                continue
            for v, w in self.adj[u]:
                nd = d + w
                if nd <= cutoff and nd < dist.get(v, np.inf):
                    dist[v], prev[v] = nd, u
                    heapq.heappush(pq, (nd, v))
        if dst not in dist:
            return None
        path, v = [], dst
        while v != src:
            path.append(self.arc_of[(prev[v], v)])
            v = prev[v]
        return path[::-1]


def fill_route(G, arc_seq):
    """
    Turn the per-observation decoded arcs into a connected route: consecutive distinct
    arcs that are not adjacent are joined by the shortest network path (a car can cross
    several short arcs between two 1 Hz fixes). Unreachable pairs are left as a break.
    """
    seq = [int(x) for i, x in enumerate(arc_seq) if x >= 0 and (i == 0 or x != arc_seq[i - 1])]
    if not seq:
        return [], 0
    route, filled = [seq[0]], 0
    for b in seq[1:]:
        a = route[-1]
        if a == b:
            continue
        if G.arcs[a, 1] != G.arcs[b, 0]:
            p = G.arc_path(int(G.arcs[a, 1]), int(G.arcs[b, 0]))
            if p:
                route.extend(p)
                filled += len(p)
        route.append(b)
    return route, filled


def transition_matrix(G, A, ua, B, ub, gc, dt, beta):
    """log-transition between candidate arcs A (prev) and B (cur)."""
    la, lb = G.arc_len[A], G.arc_len[B]
    cutoff = max(gc * 3 + 100, min(MAX_SPEED * dt, 5000) + 50)
    T = np.full((len(A), len(B)), -np.inf)
    to_nodes = G.arcs[A, 1]
    from_nodes = G.arcs[B, 0]
    for i in range(len(A)):
        rest = (1 - ua[i]) * la[i]
        dmap = G.dists_from(int(to_nodes[i]), cutoff)
        for j in range(len(B)):
            if A[i] == B[j]:
                step = (ub[j] - ua[i]) * la[i]
                if step < -BACKWARD_TOL_M:
                    continue
                d = max(step, 0.0)
            else:
                sp = dmap.get(int(from_nodes[j]))
                if sp is None:
                    continue
                d = rest + sp + ub[j] * lb[j]
            T[i, j] = -abs(d - gc) / beta
    return T


def viterbi(cands, trip, G=None, beta=BETA, emit_w=EMIT_W, gap_s=GAP_S):
    """
    cands : DataFrame with point_idx, arc_id, proj_frac, prob (normalised per point)
    Returns predicted arc per observation (len N, -1 if none) and decode stats.
    """
    G = G or RoadGraph(trip)
    N = len(trip.track_xy)
    xy, t = trip.track_xy, trip.t
    groups = {k: g for k, g in cands.groupby("point_idx", sort=True)}
    out = np.full(N, -1, dtype=np.int64)
    stats = dict(segments=0, gap_breaks=0, no_candidate_points=0, disconnected_breaks=0)

    seg_arcs, seg_u, seg_score, seg_back, seg_pts = [], [], [], [], []

    def flush():
        if not seg_pts:
            return
        k = int(np.argmax(seg_score[-1]))
        for s in range(len(seg_pts) - 1, -1, -1):
            out[seg_pts[s]] = seg_arcs[s][k]
            if s > 0:
                k = seg_back[s][k]
        stats["segments"] += 1
        seg_arcs.clear(); seg_u.clear(); seg_score.clear(); seg_back.clear(); seg_pts.clear()

    prev_i = None
    for i in range(N):
        g = groups.get(i)
        if g is None:
            stats["no_candidate_points"] += 1
            flush(); prev_i = None
            continue
        B = g.arc_id.values
        ub = g.proj_frac.values
        emit = emit_w * np.log(np.clip(g.prob.values, 1e-12, 1))
        if prev_i is not None and t[i] - t[prev_i] > gap_s:
            stats["gap_breaks"] += 1
            flush(); prev_i = None
        if prev_i is None:
            seg_arcs.append(B); seg_u.append(ub); seg_score.append(emit)
            seg_back.append(np.zeros(len(B), dtype=np.int64)); seg_pts.append(i)
            prev_i = i
            continue
        gc = float(np.hypot(*(xy[i] - xy[prev_i])))
        T = transition_matrix(G, seg_arcs[-1], seg_u[-1], B, ub, gc, t[i] - t[prev_i], beta)
        tot = seg_score[-1][:, None] + T
        best = np.argmax(tot, axis=0)
        score = tot[best, np.arange(len(B))] + emit
        if not np.isfinite(score).any():
            stats["disconnected_breaks"] += 1
            flush()
            seg_arcs.append(B); seg_u.append(ub); seg_score.append(emit)
            seg_back.append(np.zeros(len(B), dtype=np.int64)); seg_pts.append(i)
            prev_i = i
            continue
        seg_arcs.append(B); seg_u.append(ub); seg_score.append(score)
        seg_back.append(best); seg_pts.append(i)
        prev_i = i
    flush()
    return out, stats
