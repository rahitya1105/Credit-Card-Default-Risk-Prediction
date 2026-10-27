"""Score local records with the saved model; never load user-supplied models."""
import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from data import ROOT
from features import build_features


def score_file(input_path, output_path, model_name=None, cutoff=None):
    artifacts = ROOT / "artifacts"
    summary = json.loads((artifacts / "summary.json").read_text())
    name = model_name or summary["champion"]
    if name not in {"logistic", "forest"}:
        raise ValueError("Choose logistic or forest.")
    threshold = summary["thresholds"][name] if cutoff is None else cutoff
    if not 0 <= threshold <= 1:
        raise ValueError("Cutoff must lie between 0 and 1.")
    frame = pd.read_csv(input_path)
    model = joblib.load(artifacts / "models.joblib")["models"][name]
    probability = model.predict_proba(build_features(frame))[:, 1]
    result = pd.DataFrame({"row": range(1, len(frame)+1), "estimated_default_probability": probability,
                           "flagged_for_review": probability >= threshold})
    if "ID" in frame:
        result.insert(1, "customer_id", frame["ID"].to_numpy())
    result.to_csv(output_path, index=False)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Historical research model: score behaviour-only input rows.")
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "risk_scores.csv")
    parser.add_argument("--model", choices=["logistic", "forest"])
    parser.add_argument("--cutoff", type=float)
    args = parser.parse_args()
    result = score_file(args.input_csv, args.output, args.model, args.cutoff)
    print(f"Saved {len(result):,} model estimates to {args.output}")
