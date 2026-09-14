"""
WESAD Stage 1: classify baseline / stress / amusement from chest-worn RespiBAN signals.
Follows Stage1_Report_WESAD.md.

Pipeline
  1. Per subject: load the pickle, filter the chest signals, cut 60 s windows with a
     30 s stride *inside* contiguous baseline/stress/amusement segments (so no window
     spans a condition change), and compute summary features per window.
     The feature table is cached in results/features.csv.
  2. Hold out 3 subjects as a test set. They are not used anywhere in this script.
  3. On the remaining 12 subjects: leave-one-subject-out CV of
     StandardScaler + multinomial logistic regression (L2) over a grid of C.
  4. Write every number the report needs to results/summary.txt, plus figures.

Usage
  python wesad_stage1.py             # reuse cached features if they exist
  python wesad_stage1.py --rebuild   # re-extract features from the pickles
"""
import argparse
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal
from scipy.ndimage import uniform_filter1d
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "WESAD_dataset" / "WESAD"
RESULTS_DIR = ROOT / "results"
FIG_DIR = ROOT / "figures"
FEATURES_CSV = RESULTS_DIR / "features.csv"

FS = 700  # RespiBAN sampling rate (Hz)
WINDOW_S, STRIDE_S = 60, 30
LABELS = {1: "baseline", 2: "stress", 3: "amusement"}  # all other label codes are discarded
CLASS_NAMES = list(LABELS.values())

N_TEST_SUBJECTS = 3
SEED = 42
C_GRID = [0.001, 0.01, 0.1, 1, 10, 100]
CLASS_WEIGHT = None  # set to "balanced" to reweight the loss by inverse class frequency


# --------------------------------------------------------------------------------------
# Feature extraction
# --------------------------------------------------------------------------------------

def butter(x, cutoff, btype, order=4):
    sos = signal.butter(order, cutoff, btype=btype, fs=FS, output="sos")
    return signal.sosfiltfilt(sos, x)


def preprocess_chest(chest):
    """Filter the whole recording once, so windows do not suffer from filter edge effects."""
    ecg = butter(chest["ECG"].ravel(), [5, 15], "bandpass")
    # Pan-Tompkins-style QRS envelope: derivative, square, 150 ms moving average.
    # Squaring makes detection independent of ECG polarity.
    ecg_env = uniform_filter1d(np.gradient(ecg) ** 2, size=int(0.15 * FS))
    return {
        "ecg_env": ecg_env,
        "eda": butter(chest["EDA"].ravel(), 1.0, "lowpass"),
        "resp": butter(chest["Resp"].ravel(), [0.1, 0.7], "bandpass", order=2),
        "temp": chest["Temp"].ravel().astype(np.float64),
        "acc": chest["ACC"],
    }


def slope_per_min(x):
    t = np.arange(len(x)) / FS / 60
    return np.polyfit(t, x, 1)[0]


ECG_KEYS = ["ecg_hr_mean", "ecg_hr_std", "ecg_sdnn", "ecg_rmssd", "ecg_pnn50"]


def ecg_features(env):
    peaks, _ = signal.find_peaks(env, height=0.3 * np.percentile(env, 99), distance=int(0.33 * FS))
    rr = np.diff(peaks) / FS
    rr = rr[(rr > 0.33) & (rr < 1.5)]  # 40-180 bpm
    rr = rr[np.abs(rr - np.median(rr)) < 0.2 * np.median(rr)] if len(rr) else rr  # missed/extra beats
    if len(rr) < 20:  # detection failed for this window
        return dict.fromkeys(ECG_KEYS, np.nan)
    hr = 60 / rr
    drr = np.diff(rr)
    return {
        "ecg_hr_mean": hr.mean(),
        "ecg_hr_std": hr.std(),
        "ecg_sdnn": rr.std() * 1000,  # ms
        "ecg_rmssd": np.sqrt(np.mean(drr ** 2)) * 1000,  # ms
        "ecg_pnn50": np.mean(np.abs(drr) > 0.05) * 100,  # %
    }


def eda_features(eda):
    # Skin conductance responses approximated as peaks with >= 0.05 uS prominence, >= 1 s apart
    scr, _ = signal.find_peaks(eda, prominence=0.05, distance=FS)
    return {
        "eda_mean": eda.mean(),
        "eda_std": eda.std(),
        "eda_min": eda.min(),
        "eda_max": eda.max(),
        "eda_slope": slope_per_min(eda),
        "eda_scr_count": len(scr),
    }


def resp_features(resp):
    breaths, _ = signal.find_peaks(resp, distance=int(1.5 * FS), prominence=0.5 * resp.std())
    rate = 60 / np.mean(np.diff(breaths) / FS) if len(breaths) > 2 else np.nan
    return {"resp_rate": rate, "resp_std": resp.std()}


def temp_features(temp):
    return {"temp_mean": temp.mean(), "temp_std": temp.std(), "temp_slope": slope_per_min(temp)}


def acc_features(acc):
    mag = np.linalg.norm(acc, axis=1)
    return {
        "acc_mag_mean": mag.mean(),
        "acc_mag_std": mag.std(),
        "acc_x_std": acc[:, 0].std(),
        "acc_y_std": acc[:, 1].std(),
        "acc_z_std": acc[:, 2].std(),
    }


def label_segments(labels):
    """Yield (label, start, end) for each maximal run of a single label we keep."""
    change = np.flatnonzero(np.diff(labels)) + 1
    for start, end in zip(np.r_[0, change], np.r_[change, len(labels)]):
        if labels[start] in LABELS:
            yield int(labels[start]), start, end


def extract_subject(pkl_path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")
    sig = preprocess_chest(data["signal"]["chest"])
    win, stride = WINDOW_S * FS, STRIDE_S * FS

    rows = []
    for label, start, end in label_segments(data["label"]):
        for w0 in range(start, end - win + 1, stride):
            w = slice(w0, w0 + win)
            rows.append({
                "subject": data["subject"],
                "label": LABELS[label],
                "t_start_s": w0 / FS,
                **ecg_features(sig["ecg_env"][w]),
                **eda_features(sig["eda"][w]),
                **resp_features(sig["resp"][w]),
                **temp_features(sig["temp"][w]),
                **acc_features(sig["acc"][w]),
            })
    return rows


def subject_ids():
    return sorted((p.name for p in DATA_DIR.iterdir() if (p / f"{p.name}.pkl").exists()),
                  key=lambda s: int(s[1:]))


def build_features():
    rows = []
    for subj in subject_ids():
        subj_rows = extract_subject(DATA_DIR / subj / f"{subj}.pkl")
        print(f"  {subj}: {len(subj_rows)} windows")
        rows.extend(subj_rows)
    df = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(exist_ok=True)
    df.to_csv(FEATURES_CSV, index=False)
    return df


# --------------------------------------------------------------------------------------
# Model and validation
# --------------------------------------------------------------------------------------

def make_model(C):
    # The scaler lives inside the pipeline, so it is fitted on the training folds only.
    return make_pipeline(StandardScaler(),
                         LogisticRegression(C=C, class_weight=CLASS_WEIGHT, max_iter=5000))


def macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, labels=CLASS_NAMES, average="macro")


def loso_cv(dev, feature_cols):
    X = dev[feature_cols].to_numpy()
    y = dev["label"].to_numpy()
    groups = dev["subject"].to_numpy()

    records, val_preds = [], {}
    for C in C_GRID:
        pred = np.empty_like(y)
        for subj in pd.unique(groups):
            tr, va = groups != subj, groups == subj
            model = make_model(C).fit(X[tr], y[tr])
            pred[va] = model.predict(X[va])
            y_tr_pred = model.predict(X[tr])
            records.append({
                "C": C, "val_subject": subj, "n_train": int(tr.sum()), "n_val": int(va.sum()),
                "train_acc": accuracy_score(y[tr], y_tr_pred), "train_f1": macro_f1(y[tr], y_tr_pred),
                "val_acc": accuracy_score(y[va], pred[va]), "val_f1": macro_f1(y[va], pred[va]),
            })
        val_preds[C] = pred
    return pd.DataFrame(records), val_preds


# --------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------

def plot_feature_boxplots(dev, feature_cols):
    ncols = 5
    nrows = int(np.ceil(len(feature_cols) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.6 * nrows))
    for ax, col in zip(axes.flat, feature_cols):
        ax.boxplot([dev.loc[dev.label == c, col].dropna() for c in CLASS_NAMES],
                   tick_labels=CLASS_NAMES, showfliers=False)
        ax.set_title(col, fontsize=9)
        ax.tick_params(labelsize=7)
    for ax in axes.flat[len(feature_cols):]:
        ax.axis("off")
    fig.suptitle("Features by condition (development subjects)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "feature_boxplots.png", dpi=150)
    plt.close(fig)


def plot_correlation(dev, feature_cols):
    corr = dev[feature_cols].corr()
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(feature_cols)), feature_cols, rotation=90, fontsize=7)
    ax.set_yticks(range(len(feature_cols)), feature_cols, fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Pearson r")
    ax.set_title("Feature correlation (development subjects)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "feature_correlation.png", dpi=150)
    plt.close(fig)
    return corr


def plot_c_curve(summary):
    fig, ax = plt.subplots(figsize=(5, 3.5))
    for split in ("train", "val"):
        ax.errorbar(summary.index, summary[(f"{split}_f1", "mean")], yerr=summary[(f"{split}_f1", "std")],
                    marker="o", capsize=3, label="training" if split == "train" else "validation")
    ax.set_xscale("log")
    ax.set_xlabel("C (inverse regularisation strength)")
    ax.set_ylabel("Macro-F1 (mean ± sd over folds)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "c_selection.png", dpi=150)
    plt.close(fig)


def plot_confusion(cm):
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    norm = cm / cm.sum(axis=1, keepdims=True)
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            ax.text(j, i, f"{cm[i, j]}\n({norm[i, j]:.0%})", ha="center", va="center",
                    color="white" if norm[i, j] > 0.5 else "black", fontsize=9)
    ax.set_xticks(range(3), CLASS_NAMES)
    ax.set_yticks(range(3), CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("LOSO validation, pooled over folds")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)


def plot_per_subject(folds):
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(folds["val_subject"], folds["val_f1"])
    ax.axhline(folds["val_f1"].mean(), color="black", linestyle="--", linewidth=1, label="mean")
    ax.set_ylabel("Validation macro-F1")
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "per_subject_f1.png", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true", help="re-extract features from the raw pickles")
    args = parser.parse_args()
    RESULTS_DIR.mkdir(exist_ok=True)
    FIG_DIR.mkdir(exist_ok=True)

    if args.rebuild or not FEATURES_CSV.exists():
        print("Extracting features ...")
        df = build_features()
    else:
        print(f"Loading cached features from {FEATURES_CSV.relative_to(ROOT)}")
        df = pd.read_csv(FEATURES_CSV)

    feature_cols = [c for c in df.columns if c not in ("subject", "label", "t_start_s")]
    n_windows_raw = len(df)
    nan_rows = df[feature_cols].isna().any(axis=1)
    df = df[~nan_rows].reset_index(drop=True)

    subjects = sorted(df["subject"].unique(), key=lambda s: int(s[1:]))
    rng = np.random.default_rng(SEED)
    test_subjects = sorted(rng.choice(subjects, N_TEST_SUBJECTS, replace=False), key=lambda s: int(s[1:]))
    dev_subjects = [s for s in subjects if s not in test_subjects]
    dev = df[df["subject"].isin(dev_subjects)].reset_index(drop=True)

    # ---- data inspection (development subjects only, test subjects stay unseen) ----
    plot_feature_boxplots(dev, feature_cols)
    corr = plot_correlation(dev, feature_cols)

    # ---- cross-validation ----
    folds, val_preds = loso_cv(dev, feature_cols)
    folds.to_csv(RESULTS_DIR / "cv_folds.csv", index=False)
    metric_cols = ["train_acc", "train_f1", "val_acc", "val_f1"]
    by_c = folds.groupby("C")[metric_cols].agg(["mean", "std"])
    best_c = by_c[("val_f1", "mean")].idxmax()
    best_folds = folds[folds["C"] == best_c]
    plot_c_curve(by_c)
    plot_per_subject(best_folds)

    cm = confusion_matrix(dev["label"], val_preds[best_c], labels=CLASS_NAMES)
    plot_confusion(cm)

    # Final Stage 1 model: fitted on all 12 development subjects at the selected C
    final = make_model(best_c).fit(dev[feature_cols], dev["label"])
    logreg = final[-1]
    coefs = pd.DataFrame(logreg.coef_, index=logreg.classes_, columns=feature_cols).T
    coefs.to_csv(RESULTS_DIR / "coefficients_standardised.csv")

    # ---- summary for the report ----
    def counts(frame):
        vc = frame["label"].value_counts()
        return ", ".join(f"{c} {vc.get(c, 0)}" for c in CLASS_NAMES)

    strong_pairs = [(a, b, corr.loc[a, b]) for i, a in enumerate(feature_cols)
                    for b in feature_cols[i + 1:] if abs(corr.loc[a, b]) >= 0.8]
    bf = best_folds[metric_cols].agg(["mean", "std"])

    lines = [
        "=== Data (Section 2 / 3.1) ===",
        f"Subjects: {len(subjects)}  ({', '.join(subjects)})",
        f"Window {WINDOW_S}s, stride {STRIDE_S}s, windows never cross a condition change",
        f"Number of features N = {len(feature_cols)}: {', '.join(feature_cols)}",
        f"Windows extracted: {n_windows_raw}; dropped for failed ECG/resp detection: {int(nan_rows.sum())}",
        f"Data points: {len(df)}  ({counts(df)})",
        "",
        "=== Split (Section 3.5) ===",
        f"Test subjects (held out, unused): {', '.join(test_subjects)}  -> {len(df) - len(dev)} windows ({counts(df[df.subject.isin(test_subjects)])})",
        f"Development subjects: {', '.join(dev_subjects)}  -> {len(dev)} windows ({counts(dev)})",
        f"LOSO folds: {len(dev_subjects)}; training points per fold ~{best_folds.n_train.mean():.0f} "
        f"(range {best_folds.n_train.min()}-{best_folds.n_train.max()}), validation points per fold "
        f"~{best_folds.n_val.mean():.0f} (range {best_folds.n_val.min()}-{best_folds.n_val.max()})",
        "",
        "=== Model selection (Section 4) ===",
        f"C grid: {C_GRID}; class_weight: {CLASS_WEIGHT}",
        by_c.round(3).to_string(),
        f"Selected C = {best_c}",
        "",
        "=== Results table at selected C (mean ± sd over folds) ===",
        f"{'':10s}{'Training':>18s}{'Validation':>18s}",
        f"{'Accuracy':10s}{bf.train_acc['mean']:>11.3f} ± {bf.train_acc['std']:.3f}"
        f"{bf.val_acc['mean']:>11.3f} ± {bf.val_acc['std']:.3f}",
        f"{'Macro-F1':10s}{bf.train_f1['mean']:>11.3f} ± {bf.train_f1['std']:.3f}"
        f"{bf.val_f1['mean']:>11.3f} ± {bf.val_f1['std']:.3f}",
        f"Validation error (1 - macro-F1): {1 - bf.val_f1['mean']:.3f}",
        "",
        "Per-subject validation macro-F1:",
        best_folds.set_index("val_subject")[["n_val", "val_acc", "val_f1"]].round(3).to_string(),
        "",
        "Confusion matrix (rows = true, cols = predicted, pooled over LOSO folds):",
        pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_string(),
        "",
        "=== Feature inspection (Section 3.2) ===",
        "Feature pairs with |r| >= 0.8:",
        *[f"  {a} ~ {b}: r = {r:+.2f}" for a, b, r in strong_pairs],
        "",
        f"Standardised coefficients of the final model (all dev subjects, C = {best_c}):",
        coefs.round(3).to_string(),
        "",
        f"Figures written to {FIG_DIR.relative_to(ROOT)}/",
    ]
    summary = "\n".join(lines)
    print(summary)
    (RESULTS_DIR / "summary.txt").write_text(summary, encoding="utf-8")


if __name__ == "__main__":
    main()
