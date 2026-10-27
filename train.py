"""Reproducible 60/20/20 benchmark; selection and cutoff use validation only."""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import platform

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

from data import ROOT, CSV_PATH, TARGET, download_data, load_data
from features import build_features, FEATURES
from modeling import (SEED, MODEL_NAMES, create_models, choose_threshold, evaluate,
                      permutation_explanation)
from statistical_analysis import analyze_payments

ARTIFACTS = ROOT / "artifacts"


def bootstrap_intervals(y, probabilities, repeats=200):
    """Customer-level percentile intervals conditional on this fitted model."""
    y, probabilities = np.asarray(y), np.asarray(probabilities)
    rng = np.random.default_rng(SEED)
    draws = {"average_precision": [], "roc_auc": []}
    for _ in range(repeats):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        draws["average_precision"].append(average_precision_score(y[idx], probabilities[idx]))
        draws["roc_auc"].append(roc_auc_score(y[idx], probabilities[idx]))
    return {name: np.quantile(values, [0.025, 0.975]).tolist() for name, values in draws.items()}


def train():
    df = load_data()
    x, y = build_features(df), df[TARGET]
    train_validation, test = train_test_split(df.index, test_size=0.2,
                                             stratify=y, random_state=SEED)
    train_idx, validation = train_test_split(train_validation, test_size=0.25,
                            stratify=y.loc[train_validation], random_state=SEED)
    split_indices = {"train": train_idx, "validation": validation, "test": test}
    models, scores, thresholds = create_models(), {}, {}
    for name, model in models.items():
        print(f"Training {MODEL_NAMES[name]}...", flush=True)
        model.fit(x.loc[train_idx], y.loc[train_idx])
        validation_prob = model.predict_proba(x.loc[validation])[:, 1]
        thresholds[name] = 0.5 if name == "baseline" else choose_threshold(y.loc[validation], validation_prob)
        scores[name] = {"validation": evaluate(y.loc[validation], validation_prob, thresholds[name])}
    champion = max(["logistic", "forest"], key=lambda name: scores[name]["validation"]["average_precision"])
    # First test evaluation occurs after the model and cutoff are fixed.
    test_predictions = pd.DataFrame({"customer_id": df.loc[test, "ID"].to_numpy(),
                                      "actual": y.loc[test].to_numpy()})
    for name, model in models.items():
        p = model.predict_proba(x.loc[test])[:, 1]
        scores[name]["test"] = evaluate(y.loc[test], p, thresholds[name])
        test_predictions[f"prob_{name}"] = p
    print(f"Selected on validation average precision: {MODEL_NAMES[champion]}", flush=True)
    print("Measuring feature usefulness on a validation subset...", flush=True)
    subset = x.loc[validation].sample(n=min(1500, len(validation)), random_state=SEED)
    importance = permutation_explanation(models[champion], subset, y.loc[subset.index])
    statistics = analyze_payments(df.loc[train_idx])
    ci = bootstrap_intervals(y.loc[test], test_predictions[f"prob_{champion}"])
    split_ids = {name: df.loc[idx, "ID"].astype(int).tolist() for name, idx in split_indices.items()}
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_sha256": sha256(CSV_PATH.read_bytes()).hexdigest(),
        "python": platform.python_version(), "sklearn": sklearn.__version__,
        "random_seed": SEED, "rows": len(df), "features": FEATURES,
        "feature_count": len(FEATURES), "default_rate": float(y.mean()),
        "splits": {name: {"rows": len(idx), "default_rate": float(y.loc[idx].mean())}
                   for name, idx in split_indices.items()},
        "split_method": "Stratified random customer split, 60% train / 20% validation / 20% test.",
        "champion": champion, "champion_name": MODEL_NAMES[champion],
        "selection_metric": "Validation average precision", "thresholds": thresholds,
        "threshold_method": "Maximum validation F1 over cutoffs 0.05–0.80, step 0.01.",
        "metrics": scores, "champion_bootstrap_95_ci": ci,
        "bootstrap_note": "200 test-customer resamples; conditional on trained model; excludes training and deployment uncertainty.",
        "excluded_inputs": ["ID", "SEX", "EDUCATION", "MARRIAGE", "AGE", TARGET],
        "limitations": [
            "One historical 2005 Taiwan snapshot; no later cohort for temporal validation.",
            "Default is the source dataset's binary next-month label, not a newly defined 90-day default rule.",
            "Scores are model estimates; observed calibration is reported, not guaranteed.",
            "No causal conclusions, actual loss reduction, or production lending validation.",
            "Excluding demographics does not remove proxy effects or demonstrate fairness.",
        ],
    }
    ARTIFACTS.mkdir(exist_ok=True)
    joblib.dump({"models": models, "summary": summary}, ARTIFACTS / "models.joblib")
    for name, value in [("summary", summary), ("statistics", statistics), ("split_ids", split_ids)]:
        (ARTIFACTS / f"{name}.json").write_text(json.dumps(value, indent=2))
    test_predictions.to_csv(ARTIFACTS / "test_predictions.csv", index=False)
    importance.to_csv(ARTIFACTS / "feature_importance.csv", index=False)
    metric_rows = [{"model": MODEL_NAMES[name], "split": split,
                    "threshold": thresholds[name], **values}
                   for name, splits in scores.items() for split, values in splits.items()]
    pd.DataFrame(metric_rows).to_csv(ARTIFACTS / "metrics.csv", index=False)
    write_results(summary, statistics)
    print(pd.DataFrame(metric_rows).to_string(index=False), flush=True)
    print(f"Results saved to {ARTIFACTS}", flush=True)
    return summary


def write_results(summary, statistics):
    lines = ["# Measured project results", "", f"Selected model: **{summary['champion_name']}**.",
             "Selection used validation average precision. Test results were measured afterward.", "",
             "| Model | Test AP | Test ROC AUC | Precision | Recall | F1 | Cutoff |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name in MODEL_NAMES:
        m = summary["metrics"][name]["test"]
        lines.append(f"| {MODEL_NAMES[name]} | {m['average_precision']:.3f} | {m['roc_auc']:.3f} | {m['precision']:.1%} | {m['recall']:.1%} | {m['f1']:.3f} | {summary['thresholds'][name]:.2f} |")
    lines += ["", "## Statistical analysis", "",
              "Training customers only. H0: any historical delay and next-month default are independent.", "",
              "| Repayment history | Customers | Default rate | Wilson 95% interval |",
              "|---|---:|---:|---:|"]
    for g in statistics["groups"]:
        lines.append(f"| {g['group']} | {g['customers']:,} | {g['rate']:.1%} | {g['ci_lower']:.1%}–{g['ci_upper']:.1%} |")
    p_text = "< 0.0001" if statistics["p_value"] < 0.0001 else f"= {statistics['p_value']:.4f}"
    lines += ["", f"Chi-square = {statistics['chi_square']:.2f}; df = 1; p {p_text}.",
              f"Observed rate difference: {100*statistics['risk_difference']:.1f} percentage points.",
              "Association does not establish causation. Confidence intervals assume independent customer records.",
              "", "## Scope", "", *[f"- {v}" for v in summary["limitations"]], "",
              "The review cutoff is an educational validation-F1 choice, not a bank-approved policy.",
              "No financial savings, default prevention, or new customer approval benefit was measured.",
              "", "Models use repayment behaviour and amounts; demographic fields and ID are excluded.",
              "These results describe a single reproducible split, not current-market performance.", ""]
    (ROOT / "RESULTS.md").write_text("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true", help="Download official data if absent.")
    options = parser.parse_args()
    if options.download and not CSV_PATH.exists():
        download_data()
    train()
