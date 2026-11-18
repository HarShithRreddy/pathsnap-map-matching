"""
Predicted-vs-reference route maps (Folium) for the final model + HMM/Viterbi.

  python -m eda.predicted_maps                      # 3 student-test trips
  python -m eda.predicted_maps 00000000 00000056    # chosen student-test trips
Outputs -> results/maps/pred_<trip>.html

Layers: raw GPS (blue), road network near the track (grey), reference route (green),
classifier-only prediction (orange, hidden by default), HMM/Viterbi prediction (red).
"""
import os
import sys

import folium
import numpy as np
from scipy.spatial import cKDTree

from data_processing.candidates import LABEL_MAX_DIST_M, align_labels
from data_processing.loader import load_trip
from hmm.pipeline import load_artifacts, predict_trip

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "results", "maps")
DEFAULT_TRIPS = ["00000052", "00000077", "00000007", "00000056"]  # best, hardest, HMM-did-not-help, longest


def arc_lines(group, arcs, nodes_ll, arc_ids, **style):
    for a, b in arcs[arc_ids]:
        folium.PolyLine([(nodes_ll[a, 1], nodes_ll[a, 0]), (nodes_ll[b, 1], nodes_ll[b, 0])],
                        **style).add_to(group)


def dedup(seq):
    return [int(x) for i, x in enumerate(seq) if x >= 0 and (i == 0 or x != seq[i - 1])]


def make_map(trip_dir, model, cfg, out_html, pad_m=150):
    r = predict_trip(trip_dir, model, cfg)          # prediction never reads .route
    gt = load_trip(trip_dir)                         # reference used only for display + score
    true_arc, _, true_d = align_labels(gt)
    rel = true_d <= LABEL_MAX_DIST_M
    acc_h = float((r["hmm_arcs"] == true_arc)[rel].mean())
    acc_c = float((r["clf_arcs"] == true_arc)[rel].mean())

    ll, nl, arcs = gt.track_ll, gt.nodes_ll, gt.arcs
    m = folium.Map(location=[np.median(ll[:, 1]), np.median(ll[:, 0])], zoom_start=14,
                   tiles="OpenStreetMap")
    dn, _ = cKDTree(gt.track_xy).query(gt.nodes_xy)
    near = dn < pad_m
    g = folium.FeatureGroup("Road network (near track)")
    arc_lines(g, arcs, nl, np.where(near[arcs[:, 0]] & near[arcs[:, 1]])[0],
              color="#9e9e9e", weight=1, opacity=.6)
    g.add_to(m)
    g = folium.FeatureGroup("Reference route")
    arc_lines(g, arcs, nl, gt.route, color="green", weight=8, opacity=.5)
    g.add_to(m)
    g = folium.FeatureGroup("Classifier-only prediction", show=False)
    arc_lines(g, arcs, nl, dedup(r["clf_arcs"]), color="orange", weight=3, opacity=.9)
    g.add_to(m)
    g = folium.FeatureGroup("Predicted route (classifier + HMM)")
    arc_lines(g, arcs, nl, r["route_arcs"], color="red", weight=3, opacity=.9)
    g.add_to(m)
    g = folium.FeatureGroup("Raw GPS")
    step = max(1, len(ll) // 1500)
    for lon, lat, t in ll[::step]:
        folium.CircleMarker((lat, lon), radius=2, color="blue", fill=True, opacity=.7,
                            tooltip=f"t={t:.0f}s").add_to(g)
    g.add_to(m)
    folium.Marker((ll[0, 1], ll[0, 0]), tooltip="Start", icon=folium.Icon(color="green")).add_to(m)
    folium.Marker((ll[-1, 1], ll[-1, 0]), tooltip="End", icon=folium.Icon(color="red")).add_to(m)
    m.fit_bounds([[ll[:, 1].min(), ll[:, 0].min()], [ll[:, 1].max(), ll[:, 0].max()]])
    folium.LayerControl(collapsed=False).add_to(m)
    legend = (
        '<div style="position:fixed;bottom:30px;left:30px;z-index:9999;background:white;'
        'padding:8px;border:1px solid #888;font-size:13px">'
        f'<b>Student-test trip {gt.trip_id}</b><br>'
        f'Top-1: classifier {acc_c*100:.1f}% &rarr; +HMM {acc_h*100:.1f}%<br>'
        '<span style="color:blue">&#9679;</span> Raw GPS<br>'
        '<span style="color:green">&#9644;</span> Reference route<br>'
        '<span style="color:red">&#9644;</span> Predicted route (HMM)<br>'
        '<span style="color:orange">&#9644;</span> Classifier-only (toggle)<br>'
        '<span style="color:#9e9e9e">&#9644;</span> Road network</div>')
    m.get_root().html.add_child(folium.Element(legend))
    m.save(out_html)
    return acc_c, acc_h


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    model, cfg = load_artifacts()
    for t in (sys.argv[1:] or DEFAULT_TRIPS):
        d = os.path.join(ROOT, "data", "raw", "student_test", t)
        c, h = make_map(d, model, cfg, os.path.join(OUT, f"pred_{t}.html"))
        print(f"trip {t}: classifier {c:.4f} -> HMM {h:.4f}")
