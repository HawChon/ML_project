"""
Final evaluation on the 3 held-out test subjects (run once).

Reads the chosen model and hyperparameters from results/model_selection.json (written
by compare_models.py), refits it on all 12 development subjects, and evaluates it a
single time on the test subjects. Nothing here feeds back into model choice: if the
test result is disappointing, report it as it is rather than re-tuning.

Usage
  python evaluate_test.py           # refuses to run if test results already exist
  python evaluate_test.py --force   # overwrite them (only if the pipeline was rerun from scratch)
"""
import argparse
import json

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from common import (CLASS_NAMES, MODELS, RESULTS_DIR, SELECTION_JSON, build_model, label_counts,
                    load_dataset, plot_confusion, save_fig, subject_key)

TEST_TXT = RESULTS_DIR / "test_results.txt"


def macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, labels=CLASS_NAMES, average="macro")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="overwrite existing test results")
    args = parser.parse_args()
    if TEST_TXT.exists() and not args.force:
        raise SystemExit(f"{TEST_TXT.name} already exists: the test set has been used. "
                         "Pass --force only if you deliberately want to overwrite it.")
    if not SELECTION_JSON.exists():
        raise SystemExit("Run compare_models.py first to choose the final model.")

    selection = json.loads(SELECTION_JSON.read_text(encoding="utf-8"))
    name, params, features = selection["final_model"], selection["params"], selection["features"]
    dev, test, _ = load_dataset()
    test_subjects = sorted(test["subject"].unique(), key=subject_key)

    model = build_model(name, params).fit(dev[features], dev["label"])
    train_pred = model.predict(dev[features])
    test_pred = model.predict(test[features])
    y_test = test["label"]

    per_subject = pd.DataFrame([
        {"subject": s, "n": int(m.sum()),
         "accuracy": accuracy_score(y_test[m], test_pred[m]),
         "macro_f1": macro_f1(y_test[m], test_pred[m])}
        for s in test_subjects for m in [test["subject"] == s]
    ]).set_index("subject")

    cm = confusion_matrix(y_test, test_pred, labels=CLASS_NAMES)
    fig, ax = plt.subplots(figsize=(4.4, 3.9))
    plot_confusion(ax, cm, f"{MODELS[name]['label']}\n(test subjects {', '.join(test_subjects)})")
    save_fig(fig, "confusion_test.png")

    test_f1 = macro_f1(y_test, test_pred)
    lines = [
        f"Final model: {MODELS[name]['label']} {params}",
        f"Trained on {len(dev)} windows from the 12 development subjects ({label_counts(dev)})",
        f"Tested on {len(test)} windows from {', '.join(test_subjects)} ({label_counts(test)})",
        "",
        f"{'':22s}{'Accuracy':>10s}{'Macro-F1':>10s}",
        f"{'Training (12 dev)':22s}{accuracy_score(dev['label'], train_pred):>10.3f}"
        f"{macro_f1(dev['label'], train_pred):>10.3f}",
        f"{'Test (3 held out)':22s}{accuracy_score(y_test, test_pred):>10.3f}{test_f1:>10.3f}",
        f"Test error (1 - macro-F1): {1 - test_f1:.3f}",
        f"(Mean LOSO validation macro-F1 for comparison: {selection['val_f1_mean']:.3f})",
        "",
        "Per test subject:",
        per_subject.round(3).to_string(),
        "",
        "Per class:",
        classification_report(y_test, test_pred, labels=CLASS_NAMES, digits=3),
        "Confusion matrix (rows = true, cols = predicted):",
        pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_string(),
    ]
    summary = "\n".join(lines)
    print(summary)
    TEST_TXT.write_text(summary, encoding="utf-8")


if __name__ == "__main__":
    main()
