"""Checks for outcome leakage, split integrity, feature semantics, and artifacts."""
import json
import unittest

import joblib
import numpy as np
import pandas as pd

from data import ROOT, TARGET, PAY_STATUS, load_data
from features import build_features, FEATURES, NUMERIC
from modeling import choose_threshold, evaluate
from statistical_analysis import wilson_interval


class CreditRiskChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = load_data()
        cls.artifacts = ROOT / "artifacts"
        cls.split = json.loads((cls.artifacts / "split_ids.json").read_text())
        cls.bundle = joblib.load(cls.artifacts / "models.joblib")
        cls.summary = cls.bundle["summary"]

    def test_outcome_identity_and_demographics_never_enter_features(self):
        original = build_features(self.raw.head(50))
        changed = self.raw.head(50).copy()
        for name in [TARGET, "ID", "SEX", "AGE", "EDUCATION", "MARRIAGE"]:
            changed[name] = 999
        pd.testing.assert_frame_equal(original, build_features(changed))
        self.assertEqual(list(original.columns), FEATURES)

    def test_features_do_not_mix_customer_histories(self):
        all_features = build_features(self.raw.head(100))
        single_features = build_features(self.raw.iloc[[42]])
        pd.testing.assert_frame_equal(all_features.loc[[42]], single_features)

    def test_delay_definition_and_code_normalization(self):
        row = self.raw.iloc[[0]].copy()
        row[PAY_STATUS] = [-2, -1, 0, 1, 2, 0]
        x = build_features(row)
        self.assertEqual(x["months_delayed"].iloc[0], 2)
        self.assertEqual(x["maximum_delay"].iloc[0], 2)
        floats = row.astype({name: float for name in PAY_STATUS})
        pd.testing.assert_frame_equal(x, build_features(floats))
        row["LIMIT_BAL"] = 0
        with self.assertRaises(ValueError):
            build_features(row)

    def test_customer_splits_are_disjoint_and_exhaustive(self):
        sets = [set(self.split[name]) for name in ["train", "validation", "test"]]
        self.assertEqual([len(s) for s in sets], [18000, 6000, 6000])
        self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
        self.assertEqual(set.union(*sets), set(self.raw["ID"]))

    def test_preprocessing_was_fitted_on_training_only(self):
        train = self.raw.set_index("ID").loc[self.split["train"]].reset_index()
        numeric = build_features(train)[NUMERIC]
        pipeline = self.bundle["models"]["logistic"]
        means = pipeline.named_steps["prepare"].named_transformers_["numeric"].named_steps["scale"].mean_
        np.testing.assert_allclose(means, numeric.mean().to_numpy(), rtol=1e-9)

    def test_saved_scores_match_model_and_cutoff_uses_validation(self):
        predictions = pd.read_csv(self.artifacts / "test_predictions.csv")
        raw_test = self.raw.set_index("ID").loc[predictions["customer_id"]].reset_index()
        for name in ["logistic", "forest"]:
            model = self.bundle["models"][name]
            p = model.predict_proba(build_features(raw_test))[:, 1]
            np.testing.assert_allclose(p, predictions[f"prob_{name}"], atol=1e-12)
            val = self.raw.set_index("ID").loc[self.split["validation"]].reset_index()
            threshold = choose_threshold(val[TARGET], model.predict_proba(build_features(val))[:, 1])
            self.assertEqual(threshold, self.summary["thresholds"][name])

    def test_evaluation_counts_and_constant_baseline(self):
        m = evaluate([0, 0, 1, 1], [0.1, 0.8, 0.4, 0.9], 0.5)
        self.assertEqual([m[k] for k in ["true_negative", "false_positive", "false_negative", "true_positive"]], [1, 1, 1, 1])
        self.assertEqual(m["precision"], 0.5)
        self.assertEqual(m["recall"], 0.5)
        baseline = self.summary["metrics"]["baseline"]["test"]
        self.assertAlmostEqual(baseline["roc_auc"], 0.5)
        self.assertAlmostEqual(baseline["average_precision"], self.summary["splits"]["test"]["default_rate"])

    def test_statistical_intervals_and_group_counts(self):
        low, high = wilson_interval(0, 10)
        self.assertAlmostEqual(low, 0)
        self.assertGreater(high, 0)
        stats = json.loads((self.artifacts / "statistics.json").read_text())
        self.assertEqual(sum(g["customers"] for g in stats["groups"]), 18000)
        self.assertEqual(stats["degrees_of_freedom"], 1)
        self.assertGreaterEqual(stats["minimum_expected_count"], 5)
        self.assertTrue(all(0 <= g["ci_lower"] <= g["rate"] <= g["ci_upper"] <= 1 for g in stats["groups"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
