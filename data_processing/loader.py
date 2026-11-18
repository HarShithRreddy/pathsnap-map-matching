"""
PathSnap - raw data loading and geographic projection.

Raw files per trip folder <trip_id>/:
  <id>.track : lon lat t          (GPS observations, ~1 Hz)
  <id>.nodes : lon lat            (row index = node_id)
  <id>.arcs  : from_node to_node  (row index = arc_id, directed)
  <id>.route : arc_id             (chronological ground-truth arcs; optional)

Lon/lat are angular, so every trip is projected to its own local UTM zone
(chosen from the trip's centroid) and all distances are in metres.
"""
import os
from dataclasses import dataclass, field

import numpy as np
from pyproj import Transformer

DATA_ROOT = os.environ.get(
    "PATHSNAP_DATA", os.path.join(os.path.dirname(__file__), "..", "data", "raw")
)


@dataclass
class Trip:
    trip_id: str
    track_ll: np.ndarray          # (N,3) lon, lat, t
    nodes_ll: np.ndarray          # (V,2) lon, lat
    arcs: np.ndarray              # (E,2) from, to
    route: np.ndarray | None      # (R,) arc ids, None if not supplied
    epsg: int = 0
    track_xy: np.ndarray = field(default=None)   # (N,2) metres
    nodes_xy: np.ndarray = field(default=None)   # (V,2) metres

    @property
    def t(self):
        return self.track_ll[:, 2]


def utm_epsg(lon, lat):
    zone = int((lon + 180) // 6) + 1
    return (32600 if lat >= 0 else 32700) + zone


def _read(path, dtype=float, ncols=None):
    a = np.loadtxt(path, dtype=dtype, ndmin=2)
    if ncols is not None and a.size == 0:
        a = a.reshape(0, ncols)
    return a


def trip_files(trip_dir):
    tid = os.path.basename(os.path.normpath(trip_dir))
    return tid, {ext: os.path.join(trip_dir, f"{tid}.{ext}") for ext in ("track", "nodes", "arcs", "route")}


def load_trip(trip_dir, with_route=True, clean=True) -> Trip:
    tid, f = trip_files(trip_dir)
    track = _read(f["track"], ncols=3)
    nodes = _read(f["nodes"], ncols=2)
    arcs = _read(f["arcs"], dtype=np.int64, ncols=2)
    route = None
    if with_route and os.path.exists(f["route"]):
        route = _read(f["route"], dtype=np.int64).ravel()

    if clean:
        track = clean_track(track)

    lon0, lat0 = np.median(track[:, 0]), np.median(track[:, 1])
    epsg = utm_epsg(lon0, lat0)
    tf = Transformer.from_crs(4326, epsg, always_xy=True)
    tx, ty = tf.transform(track[:, 0], track[:, 1])
    nx, ny = tf.transform(nodes[:, 0], nodes[:, 1])
    return Trip(tid, track, nodes, arcs, route, epsg,
                np.column_stack([tx, ty]), np.column_stack([nx, ny]))


def clean_track(track):
    """Drop non-finite / implausible coordinates, sort by time, drop duplicate timestamps."""
    ok = np.isfinite(track).all(1)
    ok &= (np.abs(track[:, 0]) <= 180) & (np.abs(track[:, 1]) <= 90)
    ok &= ~((track[:, 0] == 0) & (track[:, 1] == 0))
    track = track[ok]
    track = track[np.argsort(track[:, 2], kind="stable")]
    _, idx = np.unique(track[:, 2], return_index=True)
    return track[np.sort(idx)]


def list_trips(partition, root=DATA_ROOT):
    d = os.path.join(root, partition)
    return sorted(os.path.join(d, x) for x in os.listdir(d)
                  if os.path.isdir(os.path.join(d, x)))
