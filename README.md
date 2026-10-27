# Credit Card Default Risk Prediction and Analysis

A small, explainable machine learning project that asks: **can past repayment
behaviour help identify next-month default risk?**

It uses real historical UCI records, compares logistic regression with Random
Forest, tests one statistical association, and displays the results in a local
Streamlit dashboard called **Credit Risk Studio**.

This is a historical research prototype for human review support. It does not
approve credit, reject applications, determine limits, or demonstrate that an
intervention reduces defaults.

## Quick start

Use **Python 3.12**. This repository includes the historical dataset, fitted
models, and benchmark results, so you can explore the dashboard without retraining.

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock.txt
python -m streamlit run app.py
```

### Windows PowerShell

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.lock.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open **http://127.0.0.1:8502**. Stop the server with Ctrl+C.
Installation requires internet access; afterward the dashboard uses local files.

On macOS, `./run.command` creates the environment if needed and starts the app.
In Antigravity or a compatible editor, the included Run credit risk dashboard
terminal task does the same. The local Python environment is intentionally not
part of the repository.

To reproduce the models and statistical results after installation:

```bash
python train.py
python -m unittest test_project -v
```

If the CSV has been removed, `python train.py --download` retrieves the original
UCI data before training. The fitted models use the pinned dependency versions;
retrain them if you change those versions. Port 8502 keeps this application separate
from other local dashboards.

## The business question

An account-review team has limited capacity. A model can help prioritize records
for inspection or payment-support outreach. A false positive means reviewing an
account that did not default; a false negative means missing a default.

The dashboard makes that trade-off visible. It does not assign monetary savings
or claim that flags prevent default; those require intervention data and testing.

## Real data and provenance

- Source: [UCI Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients).
- Citation: Yeh, I. (2009). DOI [10.24432/C55S3H](https://doi.org/10.24432/C55S3H).
- 30,000 anonymized customer records from Taiwan; currency is NT$.
- Predictors: repayment states, bill statements, and payments for April–September 2005.
- Outcome: the source's binary next-month default label. Do not reinterpret it as a newly defined 90-day delinquency rule.
- License: CC BY 4.0. The downloaded spreadsheet is converted to CSV, and its outcome column is renamed `default_next_month`.
- `data/source_metadata.json` records the source, timestamp, and SHA-256 hashes.

No synthetic training records or dummy-data fallback are used. The interactive
profile editor creates only a hypothetical scoring example, never training data.

## Pipeline

```text
Official UCI dataset
        ↓
Schema validation and per-customer features
        ↓
60% training / 20% validation / 20% test customers
        ├── Fit baseline + logistic regression + Random Forest
        │       ↓
        │   Choose model by validation average precision
        │   Choose each model's cutoff by validation F1
        │       ↓
        │   Evaluate once on test customers; save models and results
        │
        └── Training-only chi-square analysis of delay and default
                        ↓
           Streamlit dashboard and optional local CSV scoring
```

### Features

The model uses credit limit, six repayment-status codes, six bill amounts, six
payment amounts, and six derived features: number of months delayed, maximum
positive delay, average bill, average payment, latest bill-to-limit ratio, and
change between the oldest and newest bill. There are 25 input columns before
one-hot encoding.

Repayment codes are categorical. Positive values indicate a delay in months;
the dataset's non-positive categories are retained separately. Negative bills
and statement-to-limit ratios above 1 are possible and are retained. The ratio
describes a statement, not an independently observed live utilization measure.

ID, outcome, sex, age, education, and marital status are excluded from prediction
inputs. Exclusion does not prove fairness or eliminate proxies.

Numeric features are standardized and repayment states one-hot encoded inside
each model pipeline. All fitted preprocessing uses training customers only.
Random Forest is the single nonlinear comparison. There is no SMOTE,
ensemble weighting, causal inference, or optimization layer to explain.

### Evaluation

The constant-risk baseline estimates the same training prevalence for everyone.
Average precision is the primary selection metric, with ROC AUC, precision,
recall, F1, and Brier score also reported. A reliability chart compares average
predicted risk with observed default frequency. Accuracy is recorded but is not
the selection criterion.

The split is stratified by the outcome, reproducible with seed 42, and disjoint
by customer ID. There is no later cohort for an honest chronological test.
Random customer holdout performance must not be presented as future-period
validation.

The saved cutoff maximizes validation F1 on a grid from 0.05 to 0.80. It is an
educational convention, not an optimal business policy. Changing the dashboard
cutoff explores test results; it does not generate a new validated policy.

The model is trained on the 60% training split only. It is not refitted on
validation/test customers after selection. Feature importance shuffles inputs on
a validation subset. Bootstrap intervals resample test customers and condition
on the fitted model; they exclude training and deployment uncertainty.

See **[RESULTS.md](RESULTS.md)** for actual measured results.

### Statistics

One question is specified in advance: **is any positive historical repayment
delay associated with next-month default?**

A 2×2 contingency table from training customers compares any delay/no delay
against default/no default. Pearson's chi-square test evaluates independence;
95% Wilson intervals describe the two observed default rates. The observed rate
difference and Cramér's V report effect size. A very small p-value is displayed
as `< 0.0001`, including when the numerical calculation underflows to zero.

Independent customer records are assumed, and expected table counts are checked.
The results establish an association in this dataset, not a causal effect.

## Dashboard

1. **Overview:** dataset context, score distributions, and current flag counts.
2. **Model performance:** baseline comparison, precision–recall, errors, calibration, and feature importance.
3. **Statistics:** hypothesis, contingency comparison, group rates, and intervals.
4. **Review queue:** ranked held-out accounts with known outcomes and CSV export.
5. **Try a profile:** edit six months and score a hypothetical account with the saved model.
6. **How it works:** plain-language pipeline, feature dictionary, and limitations.

## Files

| File | Responsibility |
|---|---|
| `data.py` | Official download, provenance, and validation |
| `features.py` | Behaviour-only features and input checks |
| `modeling.py` | Preprocessing, models, cutoffs, metrics, importance |
| `statistical_analysis.py` | Chi-square, rate difference, confidence intervals |
| `train.py` | Splits, training, selection, evaluation, saved artifacts |
| `score.py` | Predict risk for a local CSV using the saved model |
| `app.py` | Dashboard and hypothetical profile scoring |
| `test_project.py` | Leakage, split, feature, inference, and statistics checks |
| `INTERVIEW_GUIDE.md` | Explanation and interview questions |

## Retrain and verify

```bash
.venv/bin/python train.py
.venv/bin/python -m unittest test_project -v
```

## Score a CSV

Supply `LIMIT_BAL`, `PAY_0`, `PAY_2` through `PAY_6`, `BILL_AMT1` through
`BILL_AMT6`, and `PAY_AMT1` through `PAY_AMT6`. ID and outcome are not needed.
Use the historical month ordering of the source; new-period use remains unvalidated.

```bash
.venv/bin/python score.py input.csv --output risk_scores.csv
```

Only locally trained model artifacts are loaded. Never replace `models.joblib`
with an untrusted downloaded file; joblib/pickle artifacts can execute code.

## Limits to discuss in an interview

- A historical 2005 cohort may not represent present-day customers.
- Payment behaviour is predictive information, not proof of causation.
- Raw model probabilities are checked for calibration, not guaranteed calibrated.
- No cost model, actual loss reduction, or intervention benefit is measured.
- No production approval, fairness, legal, or later-period validation is claimed.
- Real operational use requires separate validation and appropriate human oversight.

The original dataset remains under CC BY 4.0. Project code is under the included
MIT license. Keep the UCI attribution when redistributing the data.
