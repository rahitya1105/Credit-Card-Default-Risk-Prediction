# Measured project results

Selected model: **Random forest**.
Selection used validation average precision. Test results were measured afterward.

| Model | Test AP | Test ROC AUC | Precision | Recall | F1 | Cutoff |
|---|---:|---:|---:|---:|---:|---:|
| Constant-risk baseline | 0.221 | 0.500 | 0.0% | 0.0% | 0.000 | 0.50 |
| Logistic regression | 0.526 | 0.759 | 51.5% | 53.1% | 0.523 | 0.27 |
| Random forest | 0.556 | 0.778 | 50.2% | 58.5% | 0.540 | 0.27 |

## Statistical analysis

Training customers only. H0: any historical delay and next-month default are independent.

| Repayment history | Customers | Default rate | Wilson 95% interval |
|---|---:|---:|---:|
| No recorded past delay | 11,888 | 11.4% | 10.8%–12.0% |
| Any past delay | 6,112 | 43.0% | 41.8%–44.3% |

Chi-square = 2348.14; df = 1; p < 0.0001.
Observed rate difference: 31.7 percentage points.
Association does not establish causation. Confidence intervals assume independent customer records.

## Scope

- One historical 2005 Taiwan snapshot; no later cohort for temporal validation.
- Default is the source dataset's binary next-month label, not a newly defined 90-day default rule.
- Scores are model estimates; observed calibration is reported, not guaranteed.
- No causal conclusions, actual loss reduction, or production lending validation.
- Excluding demographics does not remove proxy effects or demonstrate fairness.

The review cutoff is an educational validation-F1 choice, not a bank-approved policy.
No financial savings, default prevention, or new customer approval benefit was measured.

Models use repayment behaviour and amounts; demographic fields and ID are excluded.
These results describe a single reproducible split, not current-market performance.
