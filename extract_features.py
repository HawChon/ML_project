"""
Raw WESAD pickles -> results/features.csv (run once).

Per subject: load the pickle, filter the chest (RespiBAN) signals over the whole
recording, cut 60 s windows with a 30 s stride *inside* contiguous
baseline/stress/amusement segments (so no window spans a condition change), and
compute summary features per window. EMG and the wrist (E4) device are not used.

Usage
  python extract_features.py           # refuses to overwrite an existing features.csv
  python extract_features.py --force   # re-extract anyway
"""
import argparse
import pickle

import numpy as np
import pandas as pd
from scipy import signal
from scipy.ndimage import uniform_filter1d

from common import DATA_DIR, FEATURES_CSV, LABELS, RESULTS_DIR, subject_key

FS = 700  # RespiBAN sampling rate (Hz)
WINDOW_S, STRIDE_S = 60, 30


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="overwrite an existing features.csv")
    args = parser.parse_args()
    if FEATURES_CSV.exists() and not args.force:
        raise SystemExit(f"{FEATURES_CSV.name} already exists; pass --force to re-extract.")

    subjects = sorted((p.name for p in DATA_DIR.iterdir() if (p / f"{p.name}.pkl").exists()),
                      key=subject_key)
    rows = []
    for subj in subjects:
        subj_rows = extract_subject(DATA_DIR / subj / f"{subj}.pkl")
        print(f"  {subj}: {len(subj_rows)} windows")
        rows.extend(subj_rows)

    RESULTS_DIR.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(FEATURES_CSV, index=False)
    print(f"Wrote {len(rows)} windows to {FEATURES_CSV}")


if __name__ == "__main__":
    main()
