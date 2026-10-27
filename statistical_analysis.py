"""One prespecified hypothesis test, effect size, and binomial uncertainty."""
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, norm

from data import TARGET, PAY_STATUS


def wilson_interval(successes, total, confidence=0.95):
    if total == 0:
        return [None, None]
    z = norm.ppf(1 - (1-confidence)/2)
    p = successes / total
    denominator = 1 + z*z/total
    center = (p + z*z/(2*total)) / denominator
    radius = z * np.sqrt(p*(1-p)/total + z*z/(4*total*total)) / denominator
    return [float(max(0, center-radius)), float(min(1, center+radius))]


def analyze_payments(df):
    """Use development/train customers; keep the test split for model evaluation.

    H0: having any recorded historical delay and next-month default are independent.
    Two groups and a binary outcome form the 2x2 contingency table.
    """
    delayed = (df[PAY_STATUS] > 0).any(axis=1)
    table = pd.crosstab(delayed, df[TARGET]).reindex(index=[False, True], columns=[0, 1], fill_value=0)
    if (table.sum(axis=1) == 0).any() or (table.sum(axis=0) == 0).any():
        raise ValueError("Both repayment groups and both outcomes are required for this test.")
    statistic, p_value, degrees, expected = chi2_contingency(table, correction=False)
    if expected.min() < 5:
        raise ValueError("Expected cell counts below 5; chi-square approximation is unsuitable.")
    groups = []
    for group in [False, True]:
        total = int(table.loc[group].sum())
        defaults = int(table.loc[group, 1])
        low, high = wilson_interval(defaults, total)
        groups.append({"group": "Any past delay" if group else "No recorded past delay",
                       "customers": total, "defaults": defaults,
                       "rate": defaults/total, "ci_lower": low, "ci_upper": high})
    risk_difference = groups[1]["rate"] - groups[0]["rate"]
    # Unpooled normal interval for a difference of two independent proportions.
    se = np.sqrt(sum(g["rate"]*(1-g["rate"])/g["customers"] for g in groups))
    return {
        "hypothesis": "Any historical delay is associated with next-month default.",
        "null": "Historical delay group and next-month default are independent.",
        "scope": "Training customers only; prespecified two-sided test.",
        "groups": groups,
        "chi_square": float(statistic), "degrees_of_freedom": int(degrees),
        "p_value": float(p_value), "minimum_expected_count": float(expected.min()),
        "cramers_v": float(np.sqrt(statistic / table.to_numpy().sum())),
        "risk_difference": float(risk_difference),
        "risk_difference_ci": [float(risk_difference - 1.96*se), float(risk_difference + 1.96*se)],
        "contingency": table.to_numpy().tolist(),
        "interpretation": "Association in this historical sample; not evidence that delay causes future default.",
    }
