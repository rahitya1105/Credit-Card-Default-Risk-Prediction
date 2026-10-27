"""Behaviour-only features, constructed separately for each customer."""
import numpy as np
import pandas as pd

from data import PAY_STATUS, BILLS, PAYMENTS

DERIVED = ["months_delayed", "maximum_delay", "average_bill", "average_payment",
           "latest_bill_to_limit", "bill_change"]
NUMERIC = ["LIMIT_BAL", *BILLS, *PAYMENTS, *DERIVED]
CATEGORICAL = PAY_STATUS
FEATURES = NUMERIC + CATEGORICAL


def build_features(df):
    """Use only the six historical months; exclude ID, demographics, and outcome.

    Repayment codes <=0 are retained as categories, not interpreted as negative
    months of delay. Positive codes denote delay lengths. Statement balances may
    be negative and may exceed the limit, so they are not arbitrarily clipped.
    """
    x = df[["LIMIT_BAL", *PAY_STATUS, *BILLS, *PAYMENTS]].copy()
    if x.isna().any().any() or not np.isfinite(x.to_numpy(dtype=float)).all():
        raise ValueError("Every historical input must contain a finite numeric value.")
    if (x["LIMIT_BAL"] <= 0).any() or (x[PAYMENTS] < 0).any().any():
        raise ValueError("Use positive credit limits and non-negative payment amounts.")
    codes = x[PAY_STATUS].to_numpy(dtype=float)
    if ((codes < -2) | (codes > 9) | (codes != np.floor(codes))).any():
        raise ValueError("Repayment status codes must be integers from -2 through 9.")
    x["months_delayed"] = (df[PAY_STATUS] > 0).sum(axis=1)
    x["maximum_delay"] = df[PAY_STATUS].clip(lower=0).max(axis=1).astype(int)
    x["average_bill"] = df[BILLS].mean(axis=1)
    x["average_payment"] = df[PAYMENTS].mean(axis=1)
    x["latest_bill_to_limit"] = df["BILL_AMT1"] / df["LIMIT_BAL"].replace(0, np.nan)
    x["bill_change"] = df["BILL_AMT1"] - df["BILL_AMT6"]
    for column in CATEGORICAL:
        x[column] = x[column].astype(int).astype(str)
    return x[FEATURES]


def summarize_features():
    return pd.DataFrame([
        {"Feature": "Credit limit", "Meaning": "Available credit limit, in NT$."},
        {"Feature": "Six repayment-status codes", "Meaning": "Historical payment states; positive codes indicate months of delay."},
        {"Feature": "Six bill amounts", "Meaning": "Monthly statements from September back to April 2005."},
        {"Feature": "Six payment amounts", "Meaning": "Historical monthly payments in the same six-month period."},
        {"Feature": "Months delayed", "Meaning": "Number of historical months with a positive delay code, from 0 to 6."},
        {"Feature": "Maximum delay", "Meaning": "Largest positive monthly delay code, or 0."},
        {"Feature": "Average bill / payment", "Meaning": "Mean of each amount across the six months."},
        {"Feature": "Latest bill-to-limit ratio", "Meaning": "Latest statement balance divided by credit limit; not a live balance."},
        {"Feature": "Bill change", "Meaning": "September statement minus April statement."},
    ])
