"""
Shared evaluation metrics (used by models/, tuning/, hmm/ and api/).

Conventions
- A classifier outputs P(candidate is the true arc). Within each GPS observation
  probabilities are renormalised, and the arg-max candidate is the prediction.
- Observation-level Top-1 accuracy: arg-max arc == reference arc (a point whose
  reference arc is not among its candidates counts as wrong -> candidate miss).
- Macro F1 / precision / recall: binary candidate labels, where the predicted
  positive is the arg-max candidate of each observation.
- Route continuity: share of consecutive observation pairs whose predicted arcs
  are identical or directly connected (to_node(a) == from_node(b)).
"""
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, log_loss, precision_score, recall_score

KEYS = ["trip_id", "point_idx"]


def argmax_flags(df, prob_col="prob"):
    """1 for the highest-probability candidate of each observation."""
    order = df.sort_values(KEYS + [prob_col], ascending=[True, True, False], kind="stable")
    first = ~order.duplicated(KEYS)
    flag = pd.Series(0, index=df.index, dtype=np.int8)
    flag.loc[order.index[first.values]] = 1
    return flag.values


def candidate_metrics(df, prob_col="prob"):
    """df needs trip_id, point_idx, label, prob."""
    pred = argmax_flags(df, prob_col)
    y = df["label"].values
    p = np.clip(df[prob_col].values, 1e-7, 1 - 1e-7)
    top1 = pd.Series(pred & y, index=df.index).groupby([df.trip_id, df.point_idx]).max()
    miss = 1 - df.groupby(KEYS)["label"].max()
    return dict(
        top1_acc=float(top1.mean()),
        macro_f1=float(f1_score(y, pred, average="macro")),
        precision_macro=float(precision_score(y, pred, average="macro", zero_division=0)),
        recall_macro=float(recall_score(y, pred, average="macro", zero_division=0)),
        f1_pos=float(f1_score(y, pred)),
        log_loss=float(log_loss(y, p, labels=[0, 1])),
        candidate_miss_rate=float(miss.mean()),
        n_points=int(len(top1)),
    )


def predicted_arcs(df, prob_col="prob"):
    """Arg-max arc per observation -> DataFrame trip_id, point_idx, arc_id."""
    flag = argmax_flags(df, prob_col).astype(bool)
    return df.loc[flag, KEYS + ["arc_id"]].sort_values(KEYS).reset_index(drop=True)


def route_continuity(arc_seq, arcs):
    """arc_seq: predicted arc per observation (time order); arcs: (E,2) array."""
    a = np.asarray(arc_seq)
    if len(a) < 2:
        return 1.0
    same = a[1:] == a[:-1]
    conn = arcs[a[:-1], 1] == arcs[a[1:], 0]
    return float(np.mean(same | conn))


def route_level(pred_seq, true_route, arcs):
    """Sequence-level metrics on the de-duplicated arc sequence."""
    seq = [int(x) for i, x in enumerate(pred_seq) if i == 0 or x != pred_seq[i - 1]]
    ps, ts = set(seq), set(int(x) for x in true_route)
    inter = len(ps & ts)
    prec = inter / max(len(ps), 1)
    rec = inter / max(len(ts), 1)
    return dict(route_precision=prec, route_recall=rec,
                route_f1=2 * prec * rec / max(prec + rec, 1e-9),
                route_continuity=route_continuity(pred_seq, arcs),
                n_pred_arcs=len(ps))
