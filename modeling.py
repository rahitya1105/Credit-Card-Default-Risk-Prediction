"""Trainable preprocessing, interpretable baseline, and one tree comparison."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, accuracy_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from features import NUMERIC, CATEGORICAL

SEED = 42
MODEL_NAMES = {
    "baseline": "Constant-risk baseline",
    "logistic": "Logistic regression",
    "forest": "Random forest",
}


def preprocessing():
    return ColumnTransformer([
        ("numeric", Pipeline([
            ("missing", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), NUMERIC),
        ("status", Pipeline([
            ("missing", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), CATEGORICAL),
    ])


def create_models():
    return {
        "baseline": Pipeline([("prepare", preprocessing()),
                              ("model", DummyClassifier(strategy="prior"))]),
        "logistic": Pipeline([("prepare", preprocessing()),
                              ("model", LogisticRegression(C=0.1, max_iter=2000,
                                                           random_state=SEED))]),
        "forest": Pipeline([("prepare", preprocessing()),
                            ("model", RandomForestClassifier(n_estimators=250,
                                     max_depth=10, min_samples_leaf=20,
                                     max_features="sqrt", n_jobs=2, random_state=SEED))]),
    }


def evaluate(y, probability, threshold=0.5):
    prediction = np.asarray(probability) >= threshold
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    return {
        "average_precision": float(average_precision_score(y, probability)),
        "roc_auc": float(roc_auc_score(y, probability)),
        "brier_score": float(brier_score_loss(y, probability)),
        "precision": float(precision_score(y, prediction, zero_division=0)),
        "recall": float(recall_score(y, prediction, zero_division=0)),
        "f1": float(f1_score(y, prediction, zero_division=0)),
        "accuracy": float(accuracy_score(y, prediction)),
        "flagged": int(prediction.sum()),
        "true_positive": int(tp), "false_positive": int(fp),
        "false_negative": int(fn), "true_negative": int(tn),
    }


def choose_threshold(y_validation, p_validation):
    # Choose a teaching cutoff by F1 on validation only. No bank cost assumptions.
    thresholds = np.round(np.arange(0.05, 0.81, 0.01), 2)
    scores = [f1_score(y_validation, p_validation >= t, zero_division=0)
              for t in thresholds]
    return float(thresholds[int(np.argmax(scores))])


def permutation_explanation(model, x_validation, y_validation):
    """Measure the loss in validation AP when each input is shuffled."""
    from sklearn.inspection import permutation_importance
    importance = permutation_importance(model, x_validation, y_validation,
                scoring="average_precision", n_repeats=3, random_state=SEED, n_jobs=1)
    return pd.DataFrame({"feature": x_validation.columns,
                         "importance": importance.importances_mean,
                         "std": importance.importances_std}).sort_values("importance", ascending=False)
