"""
Decision Tree candidate classifier (tree family) + nearest-arc baseline.

  python -m models.train_decision_tree
"""
from sklearn.tree import DecisionTreeClassifier

from models.common import SEED, load_data, nearest_arc_baseline, run_model


def build():
    # depth/leaf limits keep the single tree from memorising individual trips
    return DecisionTreeClassifier(max_depth=12, min_samples_leaf=50,
                                  class_weight="balanced", random_state=SEED)


if __name__ == "__main__":
    data = load_data()
    print("nearest-arc baseline:", nearest_arc_baseline(data))
    run_model("DecisionTree", "Tree", build(), "max_depth=12, min_leaf=50, balanced", data)
