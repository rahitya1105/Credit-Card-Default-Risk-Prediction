# Understand and explain this project

## The one-sentence idea

Use a customer's past repayment records to estimate next-month default risk and
help a team understand which historical accounts would merit review.

## A short presentation

“My project predicts credit card default risk using real historical data from
30,000 customers. I use six months of repayment status, bills, payments, and
credit limits, plus simple features such as the number of delayed months.

I compare logistic regression with Random Forest and a constant-risk baseline.
I separate training, validation, and test customers, select the model and cutoff
on validation data, and then evaluate precision, recall, average precision, and
probability reliability on the held-out test set.

For statistics, I test whether previous delays are associated with next-month
default using a chi-square test and report group default rates with confidence
intervals. A Streamlit dashboard shows the results and lets users inspect the
review trade-off and score a hypothetical profile.

The project is a historical prototype. It does not demonstrate current-market
performance or actual financial savings.”

Read RESULTS.md before quoting any numbers. Refer to the cutoffs and denominator
when explaining precision or recall.

## Where the data came from

The real source is UCI's Default of Credit Card Clients dataset, from Taiwan.
The six history months are April–September 2005. UCI supplies the later binary
outcome. Python downloads and prepares these records; it does not invent them.

## What does default mean?

The label records default payment in the next month according to the dataset.
We predict that provided label. We did not independently define a 90-day default
standard, and the result is not a guarantee that a person cannot repay.

## Why logistic regression?

The outcome has two classes: default or no default. Logistic regression learns
weights and transforms their combination into a score between 0 and 1. It is a
simple, understandable starting model. This is not OLS regression.

## Why Random Forest too?

It can learn nonlinear relationships and combinations of features. It combines
many decision trees. Unlike boosting, the trees are not sequential corrections.
We measure whether the extra flexibility improves validation performance rather
than assuming it must be better.

## What is the baseline?

Every customer gets the training default rate. A useful model should rank
customers better than this constant estimate. The baseline can have high
accuracy by predicting no default for everyone, yet miss all defaults.

## What is feature engineering here?

For each customer's six historical months, calculate how many had positive
delay codes, the maximum delay, average bill, average payment, the latest bill
divided by the limit, and the newest-minus-oldest statement change. All inputs
come before the target outcome.

## Why split three ways?

- Training: learn the model and preprocessing.
- Validation: choose the model and score cutoff.
- Test: evaluate decisions made without using those test outcomes.

The split is by customer and stratified so default proportions are similar.
There is only one historical snapshot, so random holdout does not demonstrate
performance on a future cohort.

## Precision and recall, without jargon

Suppose a model flags 100 accounts and 60 actually default: precision is 60%.
Suppose there are 120 actual defaults and the model flags 60: recall is 50%.
These numbers are illustrative; use RESULTS.md for measured project values.

Lowering the cutoff usually flags more customers and catches more defaults,
but also increases unnecessary reviews. A real policy needs costs and capacity.
We use validation F1 as a simple teaching rule, not a business optimum.

## What is average precision?

It summarizes the precision–recall ranking across score thresholds. It is useful
when defaults are the smaller class. Larger is better. A constant-risk model's
average precision equals the default prevalence in the evaluated sample.

## How do you know probabilities are meaningful?

We plot mean predicted risk against the observed default fraction in groups of
test customers and report Brier score. Good ranking does not guarantee accurate
probabilities. We have not externally calibrated or validated this model on
current customers.

## What hypothesis test did you use?

Pearson's chi-square test of independence on a 2×2 table.

- H0: any previous positive delay and next-month default are independent.
- H1: they are associated.

We use training customers for this question, check expected counts, and report
the rate difference, Cramér's V, and 95% Wilson rate intervals. We do not search
through many tests to find a small p-value. A tiny p-value does not establish a
large effect or causation. The customer-independence assumption remains a limit.

## Is it a sales or lending-approval project?

The application explores repayment risk for review support. It does not market
products or make automated approval/denial decisions. Historical demographic
fields are excluded from model inputs. This does not by itself establish fairness.

## What does feature importance mean?

We shuffle each input in a validation subset and measure the drop in average
precision. A larger drop means the model relied more on that input. Correlated
inputs can share importance; these values do not establish causes.

## What would you improve next?

Validate on later customers, assess group-level performance with an appropriate
review process, calibrate scores if needed, and evaluate a realistic review
policy against actual capacity and intervention costs. Actual default prevention
would need an intervention study; this dataset does not provide one.

## Do not claim

- A bank deployed this application or approved its policy.
- The model prevented defaults or saved a measured amount of money.
- The association test proves why customers default.
- Historical test performance guarantees current-customer performance.

There is no PuLP, Prophet, or difference-in-differences layer in this project.
