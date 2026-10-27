"""Streamlit dashboard: load saved results, explain, inspect, and score profiles."""
from hashlib import sha256
import json

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import precision_recall_curve, roc_curve
import streamlit as st

from data import ROOT, CSV_PATH, PAY_STATUS, BILLS, PAYMENTS, load_data
from features import build_features, summarize_features
from modeling import MODEL_NAMES, evaluate

st.set_page_config(page_title="Credit Risk Studio", page_icon="◈", layout="wide")
st.markdown("""<style>
    .block-container {max-width:1320px;padding-top:2rem;padding-bottom:3rem;}
    .stApp {background:#f6f8fb;color:#172a3a;}
    [data-testid="stSidebar"] {background:#edf2f7;}
    [data-testid="stMetric"] {background:white;border:1px solid #dce5ec;border-radius:14px;padding:18px;}
    [data-testid="stMetricLabel"] {color:#52697c;font-size:0.85rem;}
    [data-testid="stMetricValue"] {color:#123349;}
    .hero {background:linear-gradient(115deg,#102d43,#135769);border-radius:20px;
           padding:30px 34px;color:#fff;margin-bottom:22px;}
    .hero h1 {color:#fff!important;font-size:2.3rem!important;padding-top:6px!important;}
    .hero p {color:#d5e7ee;max-width:780px;font-size:1rem;margin:8px 0 0;}
    .eyebrow {font-size:.72rem;letter-spacing:.18em;color:#82d9d0;font-weight:700;}
    div[data-baseweb="tab-list"] {gap:16px;margin-bottom:16px;}
    button[data-baseweb="tab"] {font-weight:600;}
</style>""", unsafe_allow_html=True)

ARTIFACTS = ROOT / "artifacts"
if not (ARTIFACTS / "summary.json").exists():
    st.title("Credit Risk Studio")
    st.info("Train the project first: run .venv/bin/python train.py --download, then reload this page.")
    st.stop()


@st.cache_data
def load_results(version, dataset_version):
    summary = json.loads((ARTIFACTS / "summary.json").read_text())
    if sha256(CSV_PATH.read_bytes()).hexdigest() != summary["dataset_sha256"]:
        raise ValueError("The dataset changed after training. Run train.py again to refresh results.")
    return (summary, json.loads((ARTIFACTS / "statistics.json").read_text()),
            pd.read_csv(ARTIFACTS / "test_predictions.csv"),
            pd.read_csv(ARTIFACTS / "feature_importance.csv"), load_data())


@st.cache_resource
def load_models(version):
    # Only load artifacts produced by this local project, never uploaded joblib files.
    return joblib.load(ARTIFACTS / "models.joblib")["models"]


def chart(fig, height=350):
    fig.update_layout(template="plotly_white", height=height,
                      margin=dict(l=10, r=10, t=35, b=10),
                      font=dict(family="Arial", color="#254358"),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      colorway=["#168b8b", "#de9253", "#395a9a"])
    st.plotly_chart(fig, width="stretch")


try:
    version = (ARTIFACTS / "summary.json").stat().st_mtime_ns
    summary, statistics, predictions, importance, raw = load_results(version, CSV_PATH.stat().st_mtime_ns)
except Exception as error:
    st.error(f"Results could not be loaded: {error}")
    st.stop()

st.sidebar.markdown("### Explore the results")
st.sidebar.caption("REAL DATA · HISTORICAL RESEARCH")
available_models = ["logistic", "forest"]
selected_model = st.sidebar.selectbox("Prediction model", available_models,
                 index=available_models.index(summary["champion"]), format_func=MODEL_NAMES.get)
cutoff = st.sidebar.slider("Review cutoff", min_value=0.05, max_value=0.80, step=0.01,
                          value=summary["thresholds"][selected_model], key=f"cutoff_{selected_model}")
st.sidebar.caption("A score at or above this cutoff is flagged for review. Lower cutoffs catch more defaults and flag more accounts.")
st.sidebar.markdown("---")
st.sidebar.write("**30,000 customers**")
st.sidebar.write("Taiwan · 2005 · NT$ amounts")
st.sidebar.caption("Training 60% · Validation 20% · Test 20%")
st.sidebar.link_button("Original UCI dataset", "https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients")

probabilities = predictions[f"prob_{selected_model}"]
metrics = evaluate(predictions["actual"], probabilities, cutoff)
st.markdown("""<div class="hero"><div class="eyebrow">CREDIT RISK / PAYMENT BEHAVIOUR</div>
<h1>Understand risk. Focus attention.</h1>
<p>Explore how repayment history relates to next-month default, compare two models,
and inspect which historical accounts would be flagged for human review.</p></div>""", unsafe_allow_html=True)
st.caption("Historical research prototype. Predictions support exploration; they do not approve or reject credit.")

overview, performance, stats_tab, queue_tab, profile_tab, guide = st.tabs([
    "Overview", "Model performance", "Statistics", "Review queue", "Try a profile", "How it works"
])

with overview:
    st.subheader("A clear question: who might miss a payment?")
    st.write("Use the previous six months of repayment records to estimate the source dataset’s next-month default outcome.")
    cols = st.columns(4)
    cols[0].metric("Historical customers", f"{len(raw):,}")
    cols[1].metric("Historical default rate", f"{summary['default_rate']:.1%}")
    cols[2].metric("Selected model", MODEL_NAMES[selected_model])
    cols[3].metric("Held-out test customers", f"{len(predictions):,}")
    left, right = st.columns([1.1, 1])
    with left:
        st.markdown("#### Actual vs. estimated risk")
        distribution = pd.DataFrame({"Estimated probability": probabilities,
                     "Observed outcome": predictions["actual"].map({0: "No recorded default", 1: "Default"})})
        chart(px.histogram(distribution, x="Estimated probability", color="Observed outcome", nbins=30,
                           barmode="overlay", opacity=0.65,
                           color_discrete_map={"Default": "#de9253", "No recorded default": "#168b8b"}))
        st.caption("Outcomes are known for these historical test accounts. Overlap shows why predictions cannot be certain.")
    with right:
        st.markdown("#### At the current review cutoff")
        st.metric("Accounts flagged", f"{metrics['flagged']:,} / {len(predictions):,}")
        st.write(f"Of the {int(predictions['actual'].sum()):,} customers who defaulted, the model flags **{metrics['true_positive']:,}**.")
        st.write(f"It also flags **{metrics['false_positive']:,}** customers who did not default.")
        st.info("A risk flag means an account merits review. It is not a guarantee of default.")
    with st.expander("What the data represents"):
        st.write("The original UCI dataset contains six months of 2005 repayment status, statements, and payments. Amounts are New Taiwan dollars. The source supplies a binary next-month default label. This is one historical cohort, with a random customer split; later-period performance is untested.")

with performance:
    st.subheader("Can the model distinguish the two outcomes?")
    st.caption("Model selection and the saved cutoff use validation data only. Charts below evaluate the separate test set.")
    cols = st.columns(4)
    cols[0].metric("Average precision", f"{metrics['average_precision']:.3f}")
    cols[1].metric("ROC AUC", f"{metrics['roc_auc']:.3f}")
    cols[2].metric("Precision at cutoff", f"{metrics['precision']:.1%}")
    cols[3].metric("Recall at cutoff", f"{metrics['recall']:.1%}")
    st.write("**Precision:** among flagged accounts, how many defaulted? **Recall:** among actual defaults, how many were flagged?")
    comparison = []
    for name in MODEL_NAMES:
        saved = summary["metrics"][name]["test"]
        comparison.append({"Model": MODEL_NAMES[name], "Average precision": saved["average_precision"],
            "ROC AUC": saved["roc_auc"], "Brier score (lower is better)": saved["brier_score"],
            "Saved cutoff": summary["thresholds"][name]})
    st.dataframe(pd.DataFrame(comparison).round(3), hide_index=True, width="stretch")
    st.caption(f"Selected on validation average precision: {summary['champion_name']}. The constant-risk baseline assigns everyone the training default rate.")
    left, right = st.columns(2)
    with left:
        precision, recall, _ = precision_recall_curve(predictions["actual"], probabilities)
        fig = go.Figure(go.Scatter(x=recall, y=precision, mode="lines", name="Selected model"))
        fig.add_hline(y=predictions["actual"].mean(), line_dash="dash", line_color="#8699a8")
        fig.update_layout(title="Precision–recall trade-off", xaxis_title="Recall", yaxis_title="Precision")
        chart(fig)
    with right:
        matrix = np.array([[metrics["true_negative"], metrics["false_positive"]],
                           [metrics["false_negative"], metrics["true_positive"]]])
        chart(px.imshow(matrix, text_auto=True, x=["Not flagged", "Flagged"],
              y=["No default", "Default"], color_continuous_scale="Teal",
              labels=dict(x="Review decision", y="Actual outcome", color="Customers"),
              title="Where the model gets it right and wrong"))
    st.caption("Moving the cutoff explores test-set trade-offs. Do not tune a final policy on these test results; use a new holdout for a revised policy.")
    with st.expander("Probability reliability and uncertainty"):
        bins = pd.DataFrame({"score": probabilities, "actual": predictions["actual"]})
        bins["bin"] = pd.qcut(bins["score"], 10, duplicates="drop")
        calibration = bins.groupby("bin", observed=True).agg(mean_score=("score", "mean"),
                                             default_rate=("actual", "mean"), customers=("actual", "size")).reset_index(drop=True)
        fig = go.Figure(go.Scatter(x=calibration["mean_score"], y=calibration["default_rate"],
                                   mode="lines+markers", name="Observed"))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash"), name="Perfect agreement"))
        fig.update_layout(xaxis_title="Mean estimated probability", yaxis_title="Observed default fraction", title="Calibration on held-out customers")
        chart(fig)
        st.write("Points near the diagonal indicate agreement between predicted risk and observed frequency. Scores have not been externally calibrated or validated on current customers.")
        ci = summary["champion_bootstrap_95_ci"]
        st.write(f"For the selected model at training time ({summary['champion_name']}), bootstrap 95% intervals: average precision **{ci['average_precision'][0]:.3f}–{ci['average_precision'][1]:.3f}**; ROC AUC **{ci['roc_auc'][0]:.3f}–{ci['roc_auc'][1]:.3f}**.")
        st.caption(summary["bootstrap_note"])
    st.markdown("#### Which inputs help the selected model?")
    chart(px.bar(importance.head(12).sort_values("importance"), x="importance", y="feature", orientation="h",
                  labels={"importance": "Drop in validation average precision when shuffled", "feature": "Input"},
                  title=f"Validation permutation importance · {summary['champion_name']}"), 400)
    st.caption("These are global predictive associations for the saved selected model, not causes or individual explanations.")

with stats_tab:
    st.subheader("Does previous payment delay relate to next-month default?")
    st.write("Compare training customers with at least one positive delay code against those with none. Test customers are not used for this analysis.")
    groups = pd.DataFrame(statistics["groups"])
    left, right = st.columns([1.3, 1])
    with left:
        graph = groups.assign(rate_pct=groups["rate"]*100,
                    upper=(groups["ci_upper"]-groups["rate"])*100,
                    lower=(groups["rate"]-groups["ci_lower"])*100)
        chart(px.bar(graph, x="group", y="rate_pct", error_y="upper", error_y_minus="lower",
                     labels={"group": "Historical repayment group", "rate_pct": "Next-month default rate (%)"},
                     title="Observed rates with 95% Wilson intervals"))
    with right:
        st.metric("Difference in default rates", f"{100*statistics['risk_difference']:.1f} percentage points")
        p_text = "< 0.0001" if statistics["p_value"] < 0.0001 else f"{statistics['p_value']:.4f}"
        st.metric("Chi-square test p-value", p_text)
        st.write(f"**Null hypothesis:** {statistics['null']}")
        conclusion = "Evidence of an association" if statistics["p_value"] < 0.05 else "Insufficient evidence of an association"
        st.write(f"**At the 5% threshold:** {conclusion.lower()} in this sample.")
    table = groups.rename(columns={"group": "Group", "customers": "Customers", "defaults": "Defaults"})
    table["Default rate"] = groups["rate"].map(lambda x: f"{x:.1%}")
    table["95% interval"] = groups.apply(lambda r: f"{r.ci_lower:.1%} – {r.ci_upper:.1%}", axis=1)
    st.dataframe(table[["Group", "Customers", "Defaults", "Default rate", "95% interval"]], hide_index=True, width="stretch")
    st.info("This is an association, not proof that earlier delays cause future defaults. A small p-value does not measure the size of the relationship or the accuracy of the prediction model.")
    with st.expander("How the statistical calculation works"):
        st.write("We count customers in a 2×2 table: past delay/no past delay versus default/no default. A chi-square test compares those counts with the counts expected if the two variables were independent.")
        st.write(f"Chi-square: {statistics['chi_square']:.2f} · Degrees of freedom: 1 · Cramér’s V: {statistics['cramers_v']:.3f}")
        st.write("The error bars are 95% Wilson intervals for each group's default rate. These calculations assume independent customer records. The question was specified in advance; no search over many tests was performed.")

with queue_tab:
    st.subheader("A review queue you can inspect")
    st.write("These are anonymized held-out historical accounts. Actual outcomes are shown so you can check the model’s mistakes.")
    rows = predictions[["customer_id", "actual"]].copy()
    rows["estimated_risk"] = probabilities
    rows["flagged"] = probabilities >= cutoff
    features = build_features(raw)
    details = pd.DataFrame({"customer_id": raw["ID"], "months_delayed": features["months_delayed"],
                            "credit_limit": raw["LIMIT_BAL"], "latest_statement": raw["BILL_AMT1"]})
    rows = rows.merge(details, on="customer_id", validate="one_to_one")
    flagged_only = st.checkbox("Show only flagged accounts", value=True)
    shown = rows[rows["flagged"]] if flagged_only else rows
    shown = shown.sort_values("estimated_risk", ascending=False)
    st.caption(f"{len(shown):,} accounts match. Credit limits and statements are in NT$.")
    if shown.empty:
        st.info("No accounts meet this cutoff. Lower the cutoff or show all accounts.")
    else:
        display = shown.copy()
        display["estimated_risk"] = display["estimated_risk"].map(lambda v: f"{v:.1%}")
        display["actual"] = display["actual"].map({0: "No default", 1: "Default"})
        st.dataframe(display.head(200).rename(columns={"customer_id": "Customer ID", "estimated_risk": "Estimated risk",
            "actual": "Observed outcome", "flagged": "Flagged", "months_delayed": "Months delayed",
            "credit_limit": "Credit limit (NT$)", "latest_statement": "Latest statement (NT$)"}),
            hide_index=True, width="stretch")
    st.download_button("Download matching historical accounts", shown.to_csv(index=False),
                        file_name="historical_review_queue.csv", mime="text/csv")
    st.caption("The table displays at most 200 rows; the download includes every matching account. Risk ranking alone does not establish the benefit of an intervention.")

with profile_tab:
    st.subheader("Try a hypothetical repayment profile")
    st.write("Enter six historical months and estimate the following month’s risk. This uses the selected saved model; it does not retrain it.")
    with st.form("profile"):
        limit = st.number_input("Credit limit (NT$)", min_value=1000, max_value=1000000, value=50000, step=1000)
        history = pd.DataFrame({"Month": ["September", "August", "July", "June", "May", "April"],
                                "Repayment code": [0]*6, "Bill (NT$)": [20000]*6, "Payment (NT$)": [1500]*6})
        edited = st.data_editor(history, hide_index=True, width="stretch", disabled=["Month"],
                   column_config={"Repayment code": st.column_config.SelectboxColumn(options=list(range(-2, 10)), required=True),
                                  "Bill (NT$)": st.column_config.NumberColumn(required=True),
                                  "Payment (NT$)": st.column_config.NumberColumn(min_value=0, required=True)})
        st.caption("Positive repayment codes indicate months of delay. Codes −2, −1, and 0 are retained as separate source categories and are not counted as positive delays. Bills may be negative or exceed a limit in the source data.")
        submitted = st.form_submit_button("Estimate default risk", type="primary")
    if submitted:
        if edited.isna().any().any() or (edited["Payment (NT$)"] < 0).any():
            st.error("Fill all six months and use non-negative payments.")
        else:
            record = {"LIMIT_BAL": limit}
            for i in range(6):
                record[PAY_STATUS[i]] = edited.iloc[i]["Repayment code"]
                record[BILLS[i]] = edited.iloc[i]["Bill (NT$)"]
                record[PAYMENTS[i]] = edited.iloc[i]["Payment (NT$)"]
            probability = load_models(version)[selected_model].predict_proba(build_features(pd.DataFrame([record])))[:, 1][0]
            st.metric("Estimated next-month default probability", f"{probability:.1%}")
            st.write("**Above the current review cutoff.**" if probability >= cutoff else "**Below the current review cutoff.**")
            st.caption("A hypothetical score from a historical model, not a current borrower assessment. Extreme or unfamiliar inputs may produce unreliable estimates.")

with guide:
    st.subheader("One question, explained from start to finish")
    st.markdown("""
1. **Load real records:** 30,000 customers with six historical months of payment behaviour.
2. **Prepare inputs:** payment states, bills, payments, credit limit, and simple summaries.
3. **Split customers:** 18,000 for learning, 6,000 for model/cutoff selection, 6,000 for final evaluation.
4. **Compare models:** logistic regression learns a weighted relationship; Random Forest combines decision trees. A constant-risk baseline checks whether the models add value.
5. **Test one association:** compare default rates for customers with and without historical delays using a chi-square test.
6. **Inspect results:** examine missed defaults, unnecessary flags, and probability reliability.
""")
    st.dataframe(summarize_features(), hide_index=True, width="stretch")
    st.markdown("#### Keep these distinctions clear")
    st.write("**Machine learning:** estimates an individual record’s outcome risk. **Statistics:** examines a relationship between groups. **Dashboard:** makes both understandable.")
    st.write("The cutoff was selected for validation F1 as a teaching example. Choosing a real operating policy would require verified intervention costs, review capacity, calibration, and testing on later customers.")
    st.markdown("#### What this project has not established")
    for limitation in summary["limitations"]:
        st.write(f"• {limitation}")
    st.caption("Dataset: Yeh (2009), UCI Default of Credit Card Clients, DOI 10.24432/C55S3H. CC BY 4.0. Demographics are excluded from prediction inputs; that alone does not establish fairness.")
