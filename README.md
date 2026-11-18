# PathSnap: Distributed Map-Matching & Route Reconstruction

UE24CS352A Machine Learning hackathon. We match noisy 1 Hz GPS tracks to a road network using a
**candidate classifier** (P(candidate arc is the true arc)) followed by **HMM/Viterbi decoding**,
served by **FastAPI** (Laptop 1) and shown in a **Streamlit** evaluator UI (Laptop 2).

## Repository layout and owners

| Folder | Contents | Owner |
|---|---|---|
| `data_processing/` | Loading, cleaning, UTM projection, label alignment, candidate generation, features | P1 |
| `eda/` | Inventory, data-quality tables, plots, Folium maps | P1 |
| `models/` | `common.py` shared harness (P1), one script per model family (P1–P4), `compare.py` (P4) | All |
| `tuning/` | Optuna (TPE) Bayesian hyperparameter optimisation | P2 |
| `hmm/` | Road graph, transition model, Viterbi decoder, end-to-end `predict_trip` | P3 |
| `evaluation/` | Shared metrics, classifier-only vs HMM comparison, decoder tuning | P3 |
| `api/` | FastAPI backend | P4 |
| `ui/` | Streamlit evaluator interface | P4 |
| `artifacts/` | `final_model.joblib`, `inference_config.json`, all compared models | – |
| `results/` | CSVs, logs, plots, Folium HTML maps | – |

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
Put the provided data at `data/raw/` so that it contains `train/` and `student_test/`
(each trip is a folder `<id>/` with `<id>.track/.nodes/.arcs/.route`).
The data is **not** committed to the repository.

## Reproduce the experiments (run from the repo root)

```bash
python -m data_processing.build_dataset     # candidates + features + labels -> data/processed/
python -m eda.eda                           # EDA tables, plots, Folium maps -> results/eda/
python -m models.train_decision_tree         # DecisionTree + nearest-arc baseline   (P1)
python -m models.train_boosting              # LightGBM + HistGradientBoosting       (P2)
python -m models.train_random_forest         # RandomForest                          (P3)
python -m models.train_mlp                   # MLP + training curve                  (P4)
python -m models.compare                     # merged table + charts -> results/models/ (P4)
python -m tuning.optuna_tune 30             # Bayesian optimisation -> results/tuning/, artifacts/
python -m evaluation.evaluate_hmm tune      # decoder beta/emission weight on TRAIN trips
python -m evaluation.evaluate_hmm           # classifier-only vs +HMM on student-test -> results/hmm/
```

## Two-laptop deployment

**Laptop 1 (model + API)**: needs the repo, `artifacts/` and the trip folders.
```bash
# put the 10 evaluator trips in one folder, e.g. data/raw/evaluator_test/<id>/...
PATHSNAP_TRIPS_DIR=data/raw/evaluator_test uvicorn api.app:app --host 0.0.0.0 --port 8000
ipconfig getifaddr en0          # macOS: note this IP
```
Allow incoming connections if the macOS firewall asks. The request logs appear in this terminal.

**Laptop 2 (UI)**: needs only `ui/app.py` plus `pip install streamlit requests folium`.
```bash
streamlit run ui/app.py -- --api http://<LAPTOP1_IP>:8000
```
Both laptops must be on the same network. Use a phone hotspot if campus Wi-Fi blocks
traffic between devices. Check with `curl http://<LAPTOP1_IP>:8000/health` from Laptop 2.

## Method summary

1. **Projection**: each trip is projected to its local UTM zone (trips span Europe/Russia), so distances are in metres.
2. **Labels**: `.route` is an ordered arc list with no per-point label. Each GPS point is
   assigned to a route position by a monotonic dynamic-programming alignment (non-decreasing
   route index, minimum total distance). Points more than 100 m from their aligned arc are
   flagged unreliable and excluded from training and scoring.
3. **Candidates**: all directed arcs within 50 m (top 20) via an STRtree spatial index.
4. **Features (18)**: distance, distance minus the nearest candidate's distance, distance ratio,
   rank, #candidates, GPS-vs-arc heading difference, speed, heading validity, projection fraction,
   arc length, node degrees, two-way flag, distance and heading difference of the points
   5 steps before and after, time since the previous fix.
5. **Classifier**: binary true/false per candidate, renormalised per GPS point.
6. **HMM**: states = candidate arcs; emission = classifier probability;
   transition = exp(-|network distance - GPS straight-line distance| / beta) (Newson & Krumm);
   the chain restarts on gaps > 60 s, on points with no candidates, and on disconnected steps.

## Data-leakage controls
- Cross-validation uses GroupKFold by `trip_id`.
- The student-test partition is used only for comparison. Evaluator trips are never used in development.
- The prediction path (`hmm/pipeline.py`) loads trips with `with_route=False`. The ground truth is read only in
  `api/app.py::server_side_score` after prediction, and only metrics are returned to the UI.

## AI assistance disclosure
Parts of this code were drafted with AI assistance (Claude). All experiments were run by the team,
all reported numbers come from the scripts and logs in this repository, and each owner has
reviewed and can explain their component.
