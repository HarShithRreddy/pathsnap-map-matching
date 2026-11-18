"""
Multi-layer perceptron candidate classifier (neural-network family).

  python -m models.train_mlp
Also saves the real epoch-wise training history (results/models/mlp_training_curve.png).
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from models.common import RES, SEED, load_data, run_model


def build():
    # features have very different scales (metres, degrees, counts) -> standardise first
    return make_pipeline(StandardScaler(), MLPClassifier(
        hidden_layer_sizes=(64, 32), alpha=1e-4, batch_size=1024, learning_rate_init=1e-3,
        max_iter=40, early_stopping=True, n_iter_no_change=5, random_state=SEED))


if __name__ == "__main__":
    m, _ = run_model("MLP", "Neural network", build(), "64-32 ReLU, adam, early stopping", load_data())
    mlp = m.steps[-1][1]
    plt.plot(mlp.loss_curve_, label="training loss (log loss)")
    plt.plot(1 - np.array(mlp.validation_scores_), label="1 - validation accuracy (10% hold-out)")
    plt.xlabel("epoch"); plt.legend(); plt.title("MLP training history")
    plt.savefig(os.path.join(RES, "mlp_training_curve.png"), dpi=110); plt.close()
