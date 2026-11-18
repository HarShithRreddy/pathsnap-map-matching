"""
PathSnap evaluator UI (Laptop 2).

Run:
  streamlit run ui/app.py -- --api http://<LAPTOP1_IP>:8000
  (or set PATHSNAP_API env var, or edit the API URL in the sidebar)

Only talks to the API over HTTP; it never touches model files or ground truth.
"""
import argparse
import os
import sys

import folium
import requests
import streamlit as st

p = argparse.ArgumentParser()
p.add_argument("--api", default=os.environ.get("PATHSNAP_API", "http://127.0.0.1:8000"))
args, _ = p.parse_known_args(sys.argv[1:])

st.set_page_config(page_title="PathSnap Evaluator", layout="wide")
st.title("PathSnap: GPS map-matching evaluator")

api = st.sidebar.text_input("API URL (Laptop 1)", args.api).rstrip("/")
try:
    h = requests.get(f"{api}/health", timeout=5).json()
    st.sidebar.success(f"Connected · model: {h.get('model')}")
    trips = requests.get(f"{api}/trips", timeout=10).json()["trips"]
except Exception as e:
    st.sidebar.error(f"Cannot reach API: {e}")
    st.stop()

if not trips:
    st.warning("API returned no trips.")
    st.stop()

trip = st.selectbox("Select an evaluation trip", trips)
if st.button("Run map-matching", type="primary"):
    with st.spinner(f"Sending trip {trip} to the API..."):
        r = requests.post(f"{api}/predict/{trip}", timeout=600)
    if r.status_code != 200:
        st.error(f"API error {r.status_code}: {r.text}")
        st.stop()
    st.session_state["res"] = r.json()

res = st.session_state.get("res")
if res:
    m = res.get("metrics")
    st.subheader(f"Trip {res['trip_id']}: {res['n_gps_points']} GPS points, "
                 f"{res['n_predicted_arcs']} predicted road segments")
    if m:
        h_, c_ = m["classifier_hmm"], m["classifier_only"]
        cols = st.columns(5)
        cols[0].metric("Top-1 accuracy (HMM)", f"{h_['top1_accuracy']*100:.2f}%",
                       f"{(h_['top1_accuracy']-c_['top1_accuracy'])*100:+.2f} vs classifier")
        cols[1].metric("Route F1", f"{h_['route_f1']:.3f}")
        cols[2].metric("Route precision", f"{h_['route_precision']:.3f}")
        cols[3].metric("Route recall", f"{h_['route_recall']:.3f}")
        cols[4].metric("Route continuity", f"{h_['route_continuity']:.3f}")
        with st.expander("Classifier-only vs classifier + HMM"):
            st.table({"classifier_only": c_, "classifier_hmm": h_})
    st.caption(f"Timing: {res['timing_s']} · decoder: {res['decoder_stats']}")

    g = res["gps_points"]
    fm = folium.Map(location=g[len(g) // 2], zoom_start=14, tiles="OpenStreetMap")
    gl = folium.FeatureGroup("Raw GPS")
    for pt in g:
        folium.CircleMarker(pt, radius=2, color="blue", fill=True).add_to(gl)
    gl.add_to(fm)
    pl = folium.FeatureGroup("Predicted route")
    for seg in res["predicted_route_segments"]:
        folium.PolyLine(seg, color="red", weight=4).add_to(pl)
    pl.add_to(fm)
    folium.Marker(g[0], tooltip="Start", icon=folium.Icon(color="green")).add_to(fm)
    folium.Marker(g[-1], tooltip="End", icon=folium.Icon(color="red")).add_to(fm)
    fm.fit_bounds([[min(p[0] for p in g), min(p[1] for p in g)], [max(p[0] for p in g), max(p[1] for p in g)]])
    folium.LayerControl(collapsed=False).add_to(fm)
    html = fm.get_root().render()
    if hasattr(st, "iframe"):                       # newer Streamlit
        st.iframe(html, height=620)
    else:                                           # older Streamlit
        import streamlit.components.v1 as components
        components.html(html, height=620)
