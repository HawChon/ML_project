"""
features.csv -> LOSO validation of every candidate model on the 12 development subjects.

For each model in common.MODELS, every hyperparameter setting in its grid is scored by
leave-one-subject-out cross-validation (12 folds, each validating on one unseen subject).
The best setting per model is picked by mean validation macro-F1, the models are
compared at their best settings, and the winner is written to results/model_selection.json
for evaluate_test.py. The 3 test subjects are never used here.

Outputs: results/cv_grid.csv, results/cv_best_folds.csv, results/cv_summary.txt,
         results/model_selection.json, figures/*.png
"""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import ParameterGrid

from common import (CLASS_NAMES, META_COLS, MODELS, RESULTS_DIR, SELECTION_JSON, build_model,
                    label_counts, load_dataset, plot_confusion, save_fig, subject_key)

METRICS = ["train_acc", "train_f1", "val_acc", "val_f1"]


def macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, labels=CLASS_NAMES, average="macro")


def loso(name, params, X, y, groups):
    """Per-fold metrics and the pooled out-of-fold predictions for one setting."""
    folds, pred = [], np.empty_like(y)
    for subj in sorted(pd.unique(groups), key=subject_key):
        tr, va = groups != subj, groups == subj
        model = build_model(name, params).fit(X[tr], y[tr])
        pred[va] = model.predict(X[va])
        y_tr_pred = model.predict(X[tr])
        folds.append({
            "val_subject": subj, "n_train": int(tr.sum()), "n_val": int(va.sum()),
            "train_acc": accuracy_score(y[tr], y_tr_pred), "train_f1": macro_f1(y[tr], y_tr_pred),
            "val_acc": accuracy_score(y[va], pred[va]), "val_f1": macro_f1(y[va], pred[va]),
        })
    return pd.DataFrame(folds), pred


# --------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------

def plot_feature_inspection(dev):
    """Box plots and correlations of all candidate features (before dropping any)."""
    candidates = [c for c in dev.columns if c not in META_COLS]

    ncols = 5
    nrows = int(np.ceil(len(candidates) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.6 * nrows))
    for ax, col in zip(axes.flat, candidates):
        ax.boxplot([dev.loc[dev.label == c, col] for c in CLASS_NAMES],
                   tick_labels=CLASS_NAMES, showfliers=False)
        ax.set_title(col, fontsize=9)
        ax.tick_params(labelsize=7)
    for ax in axes.flat[len(candidates):]:
        ax.axis("off")
    fig.suptitle("Features by condition (development subjects)")
    save_fig(fig, "feature_boxplots.png")

    corr = dev[candidates].corr()
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(candidates)), candidates, rotation=90, fontsize=7)
    ax.set_yticks(range(len(candidates)), candidates, fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Pearson r")
    ax.set_title("Feature correlation (development subjects)")
    save_fig(fig, "feature_correlation.png")
    return [(a, b, corr.loc[a, b]) for i, a in enumerate(candidates)
            for b in candidates[i + 1:] if abs(corr.loc[a, b]) >= 0.8]


def plot_logreg_curve(grid):
    g = grid[grid.model == "logreg"].sort_values("C")
    fig, ax = plt.subplots(figsize=(5, 3.5))
    for split, label in (("train", "training"), ("val", "validation")):
        ax.errorbar(g["C"], g[f"{split}_f1_mean"], yerr=g[f"{split}_f1_sd"],
                    marker="o", capsize=3, label=label)
    ax.set_xscale("log")
    ax.set_xlabel("C (inverse regularisation strength)")
    ax.set_ylabel("Macro-F1 (mean ± sd over folds)")
    ax.set_title("Logistic regression")
    ax.legend()
    save_fig(fig, "logreg_c_curve.png")


def plot_svm_grid(grid):
    g = grid[grid.model == "svm_rbf"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
    for ax, split, title in zip(axes, ("train", "val"), ("Training", "Validation")):
        table = g.pivot(index="C", columns="gamma", values=f"{split}_f1_mean")
        im = ax.imshow(table, cmap="viridis", vmin=0.4, vmax=1.0, origin="lower")
        for i in range(table.shape[0]):
            for j in range(table.shape[1]):
                ax.text(j, i, f"{table.iloc[i, j]:.2f}", ha="center", va="center",
                        color="white" if table.iloc[i, j] < 0.75 else "black", fontsize=8)
        ax.set_xticks(range(table.shape[1]), table.columns)
        ax.set_yticks(range(table.shape[0]), table.index)
        ax.set_xlabel("gamma (kernel width)")
        ax.set_ylabel("C")
        ax.set_title(f"SVM (RBF): {title} macro-F1")
    fig.colorbar(im, ax=axes, shrink=0.8)
    save_fig(fig, "svm_grid.png")


def plot_per_subject(best_folds):
    table = best_folds.pivot(index="val_subject", columns="model", values="val_f1")
    table = table.loc[sorted(table.index, key=subject_key), list(MODELS)]
    x = np.arange(len(table))
    width = 0.4
    fig, ax = plt.subplots(figsize=(8, 3.2))
    for k, name in enumerate(table.columns):
        ax.bar(x + (k - 0.5) * width, table[name], width, label=MODELS[name]["label"])
    ax.set_xticks(x, table.index)
    ax.set_ylabel("Validation macro-F1")
    ax.set_ylim(0, 1)
    ax.legend(ncols=len(MODELS), loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False)
    save_fig(fig, "per_subject_f1.png")


def plot_val_confusions(cms):
    fig, axes = plt.subplots(1, len(cms), figsize=(4.4 * len(cms), 3.9))
    for ax, (name, cm) in zip(np.atleast_1d(axes), cms.items()):
        plot_confusion(ax, cm, f"{MODELS[name]['label']}\n(LOSO validation, pooled)")
    save_fig(fig, "confusion_val.png")


# --------------------------------------------------------------------------------------

def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    dev, _, feature_cols = load_dataset()  # test subjects deliberately ignored
    X = dev[feature_cols].to_numpy()
    y = dev["label"].to_numpy()
    groups = dev["subject"].to_numpy()
    dev_subjects = sorted(pd.unique(groups), key=subject_key)

    strong_pairs = plot_feature_inspection(dev)

    grid_rows, best = [], {}
    for name, spec in MODELS.items():
        for params in ParameterGrid(spec["grid"]):
            folds, pred = loso(name, params, X, y, groups)
            stats = folds[METRICS].agg(["mean", "std"])
            row = {"model": name, **params}
            for m in METRICS:
                row[f"{m}_mean"], row[f"{m}_sd"] = stats.loc["mean", m], stats.loc["std", m]
            grid_rows.append(row)
            if name not in best or row["val_f1_mean"] > best[name]["row"]["val_f1_mean"]:
                best[name] = {"params": params, "row": row, "folds": folds, "pred": pred}
        print(f"  {spec['label']}: best {best[name]['params']} "
              f"-> val macro-F1 {best[name]['row']['val_f1_mean']:.3f}")

    grid = pd.DataFrame(grid_rows)
    grid.to_csv(RESULTS_DIR / "cv_grid.csv", index=False)
    best_folds = pd.concat([b["folds"].assign(model=n) for n, b in best.items()], ignore_index=True)
    best_folds.to_csv(RESULTS_DIR / "cv_best_folds.csv", index=False)

    per_subject = (best_folds.pivot(index="val_subject", columns="model", values="val_f1")
                   .loc[dev_subjects, list(MODELS)])

    # One-standard-error rule: take the simplest model (MODELS is ordered simplest first)
    # whose mean validation macro-F1 is within one standard error of the top score.
    # The SE is that of the per-subject paired difference to the top model.
    top = max(best, key=lambda n: best[n]["row"]["val_f1_mean"])
    paired = {n: per_subject[top] - per_subject[n] for n in MODELS if n != top}
    paired_se = {n: d.std() / np.sqrt(len(d)) for n, d in paired.items()}
    final = next(n for n in MODELS if n == top or paired[n].mean() <= paired_se[n])

    SELECTION_JSON.write_text(json.dumps({
        "final_model": final,
        "params": best[final]["params"],
        "features": feature_cols,
        "val_f1_mean": best[final]["row"]["val_f1_mean"],
        "selection_rule": "simplest model within one paired standard error of the best mean val macro-F1",
        "top_scoring_model": top,
        "best_params_per_model": {n: b["params"] for n, b in best.items()},
    }, indent=2), encoding="utf-8")

    # Logistic regression weights on standardised features, fitted on all 12 dev subjects
    logreg = build_model("logreg", best["logreg"]["params"]).fit(X, y)[-1]
    pd.DataFrame(logreg.coef_, index=logreg.classes_, columns=feature_cols).T \
        .to_csv(RESULTS_DIR / "coefficients_standardised.csv")

    cms = {n: confusion_matrix(y, b["pred"], labels=CLASS_NAMES) for n, b in best.items()}
    plot_logreg_curve(grid)
    plot_svm_grid(grid)
    plot_per_subject(best_folds)
    plot_val_confusions(cms)

    # ---- summary ----
    def pm(row, metric):
        return f"{row[f'{metric}_mean']:.3f} ± {row[f'{metric}_sd']:.3f}"

    n_train = best_folds["n_train"]
    n_val = best_folds["n_val"]
    lines = [
        "=== Data ===",
        f"Development subjects ({len(dev_subjects)}): {', '.join(dev_subjects)}",
        f"Development windows: {len(dev)}  ({label_counts(dev)})",
        f"Features used ({len(feature_cols)}): {', '.join(feature_cols)}",
        f"Per LOSO fold: ~{n_train.mean():.0f} training windows ({n_train.min()}-{n_train.max()}), "
        f"~{n_val.mean():.0f} validation windows ({n_val.min()}-{n_val.max()})",
        "",
        "Candidate feature pairs with |r| >= 0.8 (all candidates, before dropping):",
        *[f"  {a} ~ {b}: r = {r:+.2f}" for a, b, r in strong_pairs],
        "",
    ]
    for name, spec in MODELS.items():
        g = grid[grid.model == name]
        metric_cols = [f"{m}_{s}" for m in METRICS for s in ("mean", "sd")]
        lines += [f"=== Hyperparameter grid: {spec['label']} ===",
                  g[list(spec["grid"]) + metric_cols].round(dict.fromkeys(metric_cols, 3))
                  .to_string(index=False), ""]

    lines += [
        "=== Comparison at each model's best setting (mean ± sd over 12 LOSO folds) ===",
        f"{'Model':22s}{'Params':28s}{'Train acc':>16s}{'Train F1':>16s}{'Val acc':>16s}{'Val F1':>16s}",
    ]
    for name, b in best.items():
        r = b["row"]
        lines.append(f"{MODELS[name]['label']:22s}{str(b['params']):28s}{pm(r, 'train_acc'):>16s}"
                     f"{pm(r, 'train_f1'):>16s}{pm(r, 'val_acc'):>16s}{pm(r, 'val_f1'):>16s}")
    lines += [
        "",
        "Validation error (1 - macro-F1): " + ", ".join(
            f"{MODELS[n]['label']} {1 - b['row']['val_f1_mean']:.3f}" for n, b in best.items()),
        "",
        "Per-subject validation macro-F1 at best settings:",
        per_subject.round(3).to_string(),
        "",
        f"Top mean validation macro-F1: {MODELS[top]['label']}. "
        "Paired per-subject difference (top minus other):",
        *[f"  vs {MODELS[n]['label']}: mean {d.mean():+.3f}, sd {d.std():.3f}, SE {paired_se[n]:.3f}; "
          f"top model better on {(d > 0).sum()}, worse on {(d < 0).sum()}, tied on {(d == 0).sum()} "
          f"of {len(d)} subjects" for n, d in paired.items()],
        "",
        "Selection rule: simplest model within one paired SE of the top score.",
        f"Final model: {MODELS[final]['label']} {best[final]['params']}",
        "",
    ]
    for name, cm in cms.items():
        lines += [f"Confusion matrix, {MODELS[name]['label']} (rows = true, cols = predicted, pooled):",
                  pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_string(), ""]

    summary = "\n".join(lines)
    print(summary)
    (RESULTS_DIR / "cv_summary.txt").write_text(summary, encoding="utf-8")


if __name__ == "__main__":
    main()
