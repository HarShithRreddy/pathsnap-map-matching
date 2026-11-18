"""
Random Forest candidate classifier (bagging ensemble family).

  python -m models.train_random_forest
"""
from sklearn.ensemble import RandomForestClassifier

from models.common import SEED, load_data, run_model


def build():
    # sqrt feature sampling decorrelates trees; balanced_subsample handles ~11% positives
    return RandomForestClassifier(n_estimators=150, max_depth=18, min_samples_leaf=5,
                                  max_features="sqrt", class_weight="balanced_subsample",
                                  n_jobs=2, random_state=SEED)


if __name__ == "__main__":
    run_model("RandomForest", "Ensemble (bagging)", build(),
              "150 trees, depth 18, min_leaf 5, sqrt features", load_data())
