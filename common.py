"""
Shared settings for all scripts: paths, labels, the candidate models, and the
subject-level train/test split. Keeping the split here guarantees that every script
holds out exactly the same 3 test subjects as Stage 1.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "WESAD_dataset" / "WESAD"
RESULTS_DIR = ROOT / "results"
FIG_DIR = ROOT / "figures"
FEATURES_CSV = RESULTS_DIR / "features.csv"
SELECTION_JSON = RESULTS_DIR / "model_selection.json"

LABELS = {1: "baseline", 2: "stress", 3: "amusement"}  # all other label codes are discarded
CLASS_NAMES = list(LABELS.values())
META_COLS = ["subject", "label", "t_start_s"]

# eda_min and eda_max correlate with eda_mean at r = 1.00 on the development subjects
# (figures/feature_correlation.png), so they add no information and are dropped.
DROPPED_FEATURES = ["eda_min", "eda_max"]

N_TEST_SUBJECTS = 3
SEED = 42

# Each model is StandardScaler + estimator; the scaler is refitted on every training split.
# Ordered from simplest to most flexible: compare_models.py relies on this order when
# breaking near-ties in favour of the simpler model.
MODELS = {
    "logreg": {
        "label": "Logistic regression",
        "estimator": LogisticRegression(max_iter=5000),
        "grid": {"C": [0.001, 0.01, 0.1, 1, 10, 100]},
    },
    "svm_rbf": {
        "label": "SVM (RBF kernel)",
        "estimator": SVC(kernel="rbf"),
        "grid": {"C": [0.1, 1, 10, 100, 1000], "gamma": [0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3]},
    },
}


def build_model(name, params):
    return make_pipeline(StandardScaler(), clone(MODELS[name]["estimator"]).set_params(**params))


def subject_key(subject):
    return int(subject[1:])


def load_dataset():
    """Return (dev, test, feature_cols); test subjects are drawn exactly as in Stage 1."""
    df = pd.read_csv(FEATURES_CSV)
    candidates = [c for c in df.columns if c not in META_COLS]
    df = df[~df[candidates].isna().any(axis=1)].reset_index(drop=True)
    feature_cols = [c for c in candidates if c not in DROPPED_FEATURES]

    subjects = sorted(df["subject"].unique(), key=subject_key)
    test_subjects = np.random.default_rng(SEED).choice(subjects, N_TEST_SUBJECTS, replace=False)
    is_test = df["subject"].isin(test_subjects)
    return df[~is_test].reset_index(drop=True), df[is_test].reset_index(drop=True), feature_cols


def label_counts(frame):
    vc = frame["label"].value_counts()
    return ", ".join(f"{c} {vc.get(c, 0)}" for c in CLASS_NAMES)


def plot_confusion(ax, cm, title):
    norm = cm / cm.sum(axis=1, keepdims=True)
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            ax.text(j, i, f"{cm[i, j]}\n({norm[i, j]:.0%})", ha="center", va="center",
                    color="white" if norm[i, j] > 0.5 else "black", fontsize=9)
    ax.set_xticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    ax.set_yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, fontsize=10)


def save_fig(fig, name):
    FIG_DIR.mkdir(exist_ok=True)
    if fig.get_layout_engine() is None:
        fig.tight_layout()
    fig.savefig(FIG_DIR / name, dpi=150)
    plt.close(fig)
