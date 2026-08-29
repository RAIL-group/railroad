"""
Quick script for evaluation of a trained GCN model on its trained SceneGraphs.
"""
import os
import matplotlib.pyplot as plt
import numpy as np

from interruption.learning.data import CSVPickleDataset
from interruption.learning.models.gcn import AnticipateGCN
from interruption.learning.utils import get_torch_device
from railroad.environment.procthor.resources import DEFAULT_RESOURCES_BASE, get_procthor_10k_dir


# constants
MODEL_PATH = DEFAULT_RESOURCES_BASE / "models"
# dataset specifications
TRAIN_DATASET_PATH = get_procthor_10k_dir() / "procthor_data_201.csv"

def main():
    """
    Small script to get a graph showcasing the comparison between
    the learned model ev costs and the actual ev costs for the training data.
    """
    # user specifications
    model_name = "best_model_experiment10_val.pt"

    # load dataset
    dataset = CSVPickleDataset(
        os.fspath(TRAIN_DATASET_PATH),
        preprocess_function=None
    )

    # load model
    model=AnticipateGCN.get_net_eval_fn(
        MODEL_PATH / model_name, get_torch_device()
    )

    # store actual expected values and predicted expected values
    actual = []
    predicted = []

    for scene_graph, actual_ev in dataset:
        actual.append(actual_ev)
        predicted_ev = model(scene_graph)
        assert predicted_ev != -1
        predicted.append(predicted_ev.item())

    # generate plot
    plot_predicted_vs_actual(actual, predicted)
    plt.tight_layout()
    # plt.show()
    plt.savefig(f"{model_name.split(".")[0]}_evaluation.jpeg", format="jpeg", dpi=300)
    plt.close()


def plot_predicted_vs_actual(actual, predicted, *, title="Predicted vs. Actual", ax=None):
    """Canonical scatter for regression calibration: predicted vs. actual values."""
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    # y = x reference, drawn first so points sit on top of it
    lo = min(actual.min(), predicted.min())
    hi = max(actual.max(), predicted.max())
    pad = 0.05 * (hi - lo) if hi > lo else 1.0
    lo, hi = lo - pad, hi + pad
    ax.plot([lo, hi], [lo, hi], color="#c3c2b7", linewidth=1.5, linestyle="--", zorder=1)

    ax.scatter(actual, predicted, s=28, color="#2a78d6", alpha=0.6,
               edgecolors="none", zorder=2)

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Actual", color="#52514e")
    ax.set_ylabel("Predicted", color="#52514e")
    ax.set_title(title, fontsize=11)
    ax.grid(True, color="#e1e0d9", linewidth=0.8, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#c3c2b7")
    ax.tick_params(colors="#898781", labelsize=8)

    return ax


if __name__ == "__main__":
    main()
