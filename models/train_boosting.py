"""
Boosting family: LightGBM and HistGradientBoosting candidate classifiers.

  python -m models.train_boosting              # both
  python -m models.train_boosting LightGBM     # one
"""
import sys

from lightgbm import LGBMClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

from models.common import SEED, load_data, run_model


def build_lightgbm():
    return LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=63,
                          subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                          random_state=SEED, n_jobs=2, verbose=-1)


def build_histgb():
    return HistGradientBoostingClassifier(learning_rate=0.1, max_iter=300, max_leaf_nodes=31,
                                          early_stopping=False, random_state=SEED)


MODELS = {
    "HistGradientBoosting": (build_histgb, "lr 0.1, 300 iters, 31 leaves"),
    "LightGBM": (build_lightgbm, "400 trees, lr 0.05, 63 leaves, 0.8 row/col sampling"),
}

if __name__ == "__main__":
    data = load_data()
    for name in (sys.argv[1:] or MODELS):
        build, cfg = MODELS[name]
        run_model(name, "Boosting", build(), cfg, data)
