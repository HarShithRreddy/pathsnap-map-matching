"""
Exploratory data analysis for PathSnap (training + student-test partitions only).

  python -m eda.eda
Outputs -> results/eda/  (CSV tables, PNG plots, Folium HTML maps)
"""
import os

import folium
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from data_processing.loader import list_trips, load_trip, trip_files

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "results", "eda")
os.makedirs(OUT, exist_ok=True)


def haversine(lon1, lat1, lon2, lat2):
    R = 6371000.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi, dl = p2 - p1, np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def inventory():
    rows = []
    for part in ("train", "student_test"):
        for d in list_trips(part):
            tid, f = trip_files(d)
            raw = np.loadtxt(f["track"], ndmin=2)
            n_raw = len(raw)
            n_nan = int((~np.isfinite(raw)).any(1).sum())
            t = raw[:, 2]
            dt = np.diff(t)
            trip = load_trip(d)
            G = nx.DiGraph()
            G.add_nodes_from(range(len(trip.nodes_ll)))
            G.add_edges_from(map(tuple, trip.arcs))
            wcc = nx.number_weakly_connected_components(G)
            scc = [len(c) for c in nx.strongly_connected_components(G)]
            isolated = sum(1 for n in G.nodes if G.degree(n) == 0)
            step = haversine(trip.track_ll[:-1, 0], trip.track_ll[:-1, 1],
                             trip.track_ll[1:, 0], trip.track_ll[1:, 1])
            dt_c = np.diff(trip.t)
            spd = step / np.maximum(dt_c, 1e-6)
            r = trip.route
            cont = float(np.mean(trip.arcs[r[:-1], 1] == trip.arcs[r[1:], 0])) if r is not None and len(r) > 1 else np.nan
            rows.append(dict(
                partition=part, trip_id=tid,
                track_kb=round(os.path.getsize(f["track"]) / 1024, 1),
                n_points_raw=n_raw, n_points_clean=len(trip.track_ll),
                missing_or_nan=n_nan, non_increasing_t=int((dt <= 0).sum()),
                duration_s=float(t.max() - t.min()),
                pct_dt_1s=round(float(np.mean(dt == 1)) * 100, 2),
                n_gaps_gt_10s=int((dt > 10).sum()), max_gap_s=float(dt.max()) if len(dt) else 0,
                length_km=round(step.sum() / 1000, 2),
                median_speed_mps=round(float(np.median(spd)), 2),
                n_speed_gt_70mps=int((spd > 70).sum()),
                lon_min=trip.track_ll[:, 0].min(), lon_max=trip.track_ll[:, 0].max(),
                lat_min=trip.track_ll[:, 1].min(), lat_max=trip.track_ll[:, 1].max(),
                utm_epsg=trip.epsg,
                n_nodes=len(trip.nodes_ll), n_arcs=len(trip.arcs),
                weak_components=wcc, strong_components=len(scc),
                largest_scc_frac=round(max(scc) / len(trip.nodes_ll), 4),
                isolated_nodes=isolated,
                route_len_arcs=len(r) if r is not None else 0,
                route_connected_frac=round(cont, 4),
                route_repeated_arcs=int(len(r) - len(set(r))) if r is not None else 0,
            ))
            print(part, tid, flush=True)
    inv = pd.DataFrame(rows)
    inv.to_csv(os.path.join(OUT, "dataset_inventory.csv"), index=False)
    summ = inv.groupby("partition").agg(
        trips=("trip_id", "count"), total_points=("n_points_clean", "sum"),
        median_points=("n_points_clean", "median"), total_km=("length_km", "sum"),
        missing=("missing_or_nan", "sum"), gaps_gt_10s=("n_gaps_gt_10s", "sum"),
        median_pct_dt_1s=("pct_dt_1s", "median"), median_nodes=("n_nodes", "median"),
        median_arcs=("n_arcs", "median"), isolated_nodes=("isolated_nodes", "sum"),
        trips_multi_wcc=("weak_components", lambda s: int((s > 1).sum())),
        median_route_arcs=("route_len_arcs", "median"))
    summ.to_csv(os.path.join(OUT, "dataset_summary.csv"))
    print(summ.T)
    return inv


def plots(inv):
    tr = pd.read_parquet(os.path.join(ROOT, "data", "processed", "features_train.parquet"))
    st = pd.read_csv(os.path.join(OUT, "label_alignment_stats.csv"))

    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    for part, c in (("train", "tab:blue"), ("student_test", "tab:orange")):
        s = inv[inv.partition == part]
        ax[0, 0].hist(s.n_points_clean, bins=30, alpha=.7, color=c, label=part)
        ax[0, 1].hist(s.length_km, bins=30, alpha=.7, color=c, label=part)
        ax[0, 2].hist(np.log10(s.n_arcs), bins=30, alpha=.7, color=c, label=part)
    ax[0, 0].set(title="GPS points per trip", xlabel="points")
    ax[0, 1].set(title="Trip length", xlabel="km")
    ax[0, 2].set(title="Road-network size", xlabel="log10(#arcs)")
    ax[0, 0].legend()
    ax[1, 0].bar(inv.trip_id.astype(str), inv.max_gap_s, color="tab:red")
    ax[1, 0].set_yscale("log"); ax[1, 0].set(title="Largest time gap per trip (s)"); ax[1, 0].set_xticks([])
    ax[1, 1].hist(st.candidate_miss_rate_reliable.dropna(), bins=30, color="tab:purple")
    ax[1, 1].set(title="Candidate miss rate per trip (R=50 m)", xlabel="miss rate")
    ax[1, 2].scatter(inv.lon_min, inv.lat_min, c=(inv.partition == "train"), cmap="coolwarm", s=15)
    ax[1, 2].set(title="Trip locations (train=red, test=blue)", xlabel="lon", ylabel="lat")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "eda_overview.png"), dpi=120); plt.close()

    feats = ["dist_m", "dist_minus_min", "heading_diff", "dist_prev", "dist_next", "rank", "proj_frac", "arc_len"]
    s = tr[tr.label_reliable == 1].sample(300000, random_state=0)
    fig, ax = plt.subplots(2, 4, figsize=(16, 7))
    for a, f in zip(ax.ravel(), feats):
        hi = np.percentile(s[f], 99)
        bins = np.linspace(s[f].min(), hi, 50)
        a.hist(s.loc[s.label == 0, f], bins=bins, alpha=.6, density=True, label="other candidate")
        a.hist(s.loc[s.label == 1, f], bins=bins, alpha=.6, density=True, label="true arc")
        a.set_title(f)
    ax[0, 0].legend()
    plt.suptitle("Feature distributions: true arc vs other candidates (train partition)")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "feature_distributions.png"), dpi=120); plt.close()

    ld = tr[tr.label == 1]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].hist(ld.dist_m, bins=60, color="tab:green")
    ax[0].set(title="GPS-to-true-arc distance (train)", xlabel="metres")
    ax[1].hist(ld["rank"], bins=np.arange(21) - .5, color="tab:green")
    ax[1].set(title="Rank of true arc among candidates", xlabel="rank (0 = nearest)")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "true_arc_distance_rank.png"), dpi=120); plt.close()

    rr = tr[tr.label_reliable == 1].groupby(["trip_id", "point_idx"]).label.max()
    pd.DataFrame({
        "class_balance_pos_frac": [tr.label.mean()],
        "nearest_is_true_frac": [tr[(tr["rank"] == 0) & (tr.label_reliable == 1)].label.mean()],
        "candidate_miss_rate_overall": [1 - rr.mean()],
        "rows": [len(tr)],
    }).round(4).to_csv(os.path.join(OUT, "label_summary_train.csv"), index=False)


def folium_trip(trip_dir, out_html, pred_arcs=None, pad_m=150):
    trip = load_trip(trip_dir)
    ll, nl = trip.track_ll, trip.nodes_ll
    m = folium.Map(location=[np.median(ll[:, 1]), np.median(ll[:, 0])], zoom_start=14, tiles="OpenStreetMap")
    # road network near the track
    from scipy.spatial import cKDTree
    tree = cKDTree(trip.track_xy)
    dn, _ = tree.query(trip.nodes_xy)
    near = dn < pad_m
    net = folium.FeatureGroup("Road network (near track)", show=True)
    for a, b in trip.arcs[near[trip.arcs[:, 0]] & near[trip.arcs[:, 1]]]:
        if a < b or True:
            folium.PolyLine([(nl[a, 1], nl[a, 0]), (nl[b, 1], nl[b, 0])], color="#999999", weight=1, opacity=.6).add_to(net)
    net.add_to(m)
    if trip.route is not None:
        rt = folium.FeatureGroup("Reference route", show=True)
        for e in trip.route:
            a, b = trip.arcs[e]
            folium.PolyLine([(nl[a, 1], nl[a, 0]), (nl[b, 1], nl[b, 0])], color="green", weight=6, opacity=.6).add_to(rt)
        rt.add_to(m)
    if pred_arcs is not None:
        pr = folium.FeatureGroup("Predicted route", show=True)
        for e in pred_arcs:
            a, b = trip.arcs[e]
            folium.PolyLine([(nl[a, 1], nl[a, 0]), (nl[b, 1], nl[b, 0])], color="red", weight=3).add_to(pr)
        pr.add_to(m)
    gps = folium.FeatureGroup("Raw GPS", show=True)
    step = max(1, len(ll) // 1500)
    for lon, lat, t in ll[::step]:
        folium.CircleMarker((lat, lon), radius=2, color="blue", fill=True, opacity=.7, tooltip=f"t={t:.0f}s").add_to(gps)
    gps.add_to(m)
    folium.Marker((ll[0, 1], ll[0, 0]), tooltip="Start", icon=folium.Icon(color="green")).add_to(m)
    folium.Marker((ll[-1, 1], ll[-1, 0]), tooltip="End", icon=folium.Icon(color="red")).add_to(m)
    m.fit_bounds([[ll[:, 1].min(), ll[:, 0].min()], [ll[:, 1].max(), ll[:, 0].max()]])
    folium.LayerControl(collapsed=False).add_to(m)
    legend = ('<div style="position:fixed;bottom:30px;left:30px;z-index:9999;background:white;padding:8px;'
              'border:1px solid #888;font-size:13px"><b>Trip ' + trip.trip_id + '</b><br>'
              '<span style="color:blue">&#9679;</span> Raw GPS<br><span style="color:green">&#9644;</span> Reference route<br>'
              + ('<span style="color:red">&#9644;</span> Predicted route<br>' if pred_arcs is not None else '') +
              '<span style="color:#999">&#9644;</span> Road network</div>')
    m.get_root().html.add_child(folium.Element(legend))
    m.save(out_html)
    return out_html


if __name__ == "__main__":
    inv = inventory()
    plots(inv)
    raw = os.path.join(ROOT, "data", "raw")
    for t in ("train/00000002", "train/00000033", "student_test/00000091"):
        folium_trip(os.path.join(raw, t), os.path.join(OUT, f"map_{t.replace('/', '_')}.html"))
    print("EDA done ->", OUT)
