"""
Phase 4 — NetGuardAI Streamlit monitoring dashboard.

Tabs:
  Overview         — KPIs and attack distribution from the test set
  Live Predict     — single-flow prediction from sampled or manual input
  Model Performance — MLflow experiment results and metric comparison
  Anomaly Monitor  — simulated live traffic feed with dual-pipeline scoring
"""

import pickle
import sys
import time
from pathlib import Path

import altair as alt
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import streamlit as st
from mlflow.tracking import MlflowClient
from sklearn.metrics import classification_report
from sklearn.utils import resample

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import predict as predictor

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="NetGuardAI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading models…")
def load_models():
    return predictor._Models.get()


@st.cache_data(show_spinner="Loading test data…")
def load_test_data():
    proc = ROOT / "data" / "processed"
    X_test       = np.load(proc / "X_test.npy")
    y_test_bin   = np.load(proc / "y_test_binary.npy")
    y_test_multi = np.load(proc / "y_test_multi.npy")
    with open(proc / "feature_cols.pkl",    "rb") as f: cols        = pickle.load(f)
    with open(proc / "scaler.pkl",          "rb") as f: scaler      = pickle.load(f)
    with open(proc / "label_encoder.pkl",   "rb") as f: le          = pickle.load(f)
    with open(proc / "multi_class_names.pkl","rb") as f: multi_names = pickle.load(f)
    X_raw = scaler.inverse_transform(X_test)
    return X_raw, y_test_bin, y_test_multi, cols, scaler, le, multi_names


@st.cache_data(show_spinner="Evaluating test set…")
def evaluate_sample(n=3000):
    X_raw, y_bin, _, cols, _, _, _ = load_test_data()
    idx = resample(np.arange(len(X_raw)), n_samples=n,
                   stratify=y_bin, random_state=42)
    flows = [dict(zip(cols, X_raw[i].tolist())) for i in idx]
    results = predictor.predict_batch(flows)
    df = pd.DataFrame(results)
    df["true_label"] = y_bin[idx]
    df["correct"]    = (df["is_attack"].astype(int) == df["true_label"]).astype(int)
    return df


@st.cache_data(show_spinner="Loading MLflow runs…")
def load_mlflow_runs():
    mlflow.set_tracking_uri(str(ROOT / "mlruns"))
    client = MlflowClient()
    rows = []
    for exp_name in ["netguardai_binary", "netguardai_multiclass"]:
        exp = client.get_experiment_by_name(exp_name)
        if not exp:
            continue
        for r in client.search_runs(exp.experiment_id,
                                    filter_string="attributes.status = 'FINISHED'",
                                    order_by=["metrics.f1_weighted DESC"]):
            rows.append({
                "Task":       exp_name.replace("netguardai_", "").title(),
                "Model":      r.data.tags.get("model", "").replace("_", " ").title(),
                "Accuracy":   round(r.data.metrics.get("accuracy", 0), 4),
                "F1 (wtd)":   round(r.data.metrics.get("f1_weighted", 0), 4),
                "Precision":  round(r.data.metrics.get("precision_weighted", 0), 4),
                "Recall":     round(r.data.metrics.get("recall_weighted", 0), 4),
                "ROC-AUC":    round(r.data.metrics.get("roc_auc", 0) or 0, 4),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.image("https://img.shields.io/badge/NetGuardAI-v1.0-blue", use_container_width=False)
st.sidebar.title("🛡️ NetGuardAI")
st.sidebar.caption("AI-powered Network Intrusion Detection")
st.sidebar.divider()

m = load_models()
st.sidebar.markdown("**Active models**")
st.sidebar.markdown(f"- Binary:  `{m.binary_model_name}`")
st.sidebar.markdown(f"- Multi:   `{m.multi_model_name}`")
st.sidebar.markdown(f"- Device:  `{m.device}`")
st.sidebar.markdown(f"- AE threshold: `{m.threshold:.4f}`")
st.sidebar.divider()
st.sidebar.markdown(f"**Features:** {len(m.feature_cols)}  |  **Classes:** {len(m.multi_class_names)}")


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_overview, tab_predict, tab_performance, tab_monitor = st.tabs([
    "📊 Overview", "🔍 Live Predict", "📈 Model Performance", "🚨 Anomaly Monitor"
])


# ── Tab 1: Overview ─────────────────────────────────────────────────────────

with tab_overview:
    st.header("📊 Test Set Overview")
    st.caption("Evaluated on 3,000 stratified flows from the held-out test set.")

    eval_df = evaluate_sample(3000)

    n_total   = len(eval_df)
    n_attacks = eval_df["is_attack"].sum()
    n_anomaly = eval_df["is_anomaly"].sum()
    accuracy  = eval_df["correct"].mean()
    attack_rate = n_attacks / n_total

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Flows Evaluated",  f"{n_total:,}")
    c2.metric("Predicted Attacks", f"{n_attacks:,}",  f"{attack_rate:.1%} of traffic")
    c3.metric("Anomalies Flagged", f"{n_anomaly:,}",  f"{n_anomaly/n_total:.1%} of traffic")
    c4.metric("Binary Accuracy",   f"{accuracy:.2%}")

    st.divider()
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Attack type distribution")
        type_counts = (
            eval_df[eval_df["is_attack"]]
            ["label"].value_counts()
            .reset_index()
            .rename(columns={"label": "Attack Type", "count": "Count"})
        )
        if not type_counts.empty:
            chart = (
                alt.Chart(type_counts)
                .mark_bar()
                .encode(
                    x=alt.X("Count:Q"),
                    y=alt.Y("Attack Type:N", sort="-x"),
                    color=alt.Color("Attack Type:N", legend=None),
                    tooltip=["Attack Type", "Count"],
                )
                .properties(height=300)
            )
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("No attacks predicted in this sample.")

    with col_right:
        st.subheader("Confidence distribution")
        conf_df = pd.DataFrame({
            "Confidence": eval_df["confidence"],
            "Prediction": eval_df["is_attack"].map({True: "Attack", False: "Benign"}),
        })
        chart2 = (
            alt.Chart(conf_df)
            .mark_bar(opacity=0.7)
            .encode(
                x=alt.X("Confidence:Q", bin=alt.Bin(maxbins=40), title="Confidence"),
                y=alt.Y("count():Q", title="Count"),
                color=alt.Color("Prediction:N",
                                scale=alt.Scale(domain=["Benign", "Attack"],
                                                range=["steelblue", "tomato"])),
                tooltip=["Prediction", "count()"],
            )
            .properties(height=300)
        )
        st.altair_chart(chart2, use_container_width=True)

    st.divider()
    st.subheader("Anomaly score — benign vs attack")
    score_df = pd.DataFrame({
        "Anomaly Score": eval_df["anomaly_score"],
        "True Label": eval_df["true_label"].map({0: "Benign", 1: "Attack"}),
    })
    chart3 = (
        alt.Chart(score_df)
        .mark_bar(opacity=0.65)
        .encode(
            x=alt.X("Anomaly Score:Q", bin=alt.Bin(maxbins=60)),
            y=alt.Y("count():Q"),
            color=alt.Color("True Label:N",
                            scale=alt.Scale(domain=["Benign", "Attack"],
                                            range=["steelblue", "tomato"])),
        )
        .properties(height=220)
    )
    thresh_line = (
        alt.Chart(pd.DataFrame({"x": [m.threshold]}))
        .mark_rule(color="black", strokeDash=[6, 3], strokeWidth=2)
        .encode(x="x:Q")
    )
    st.altair_chart(chart3 + thresh_line, use_container_width=True)
    st.caption(f"Vertical dashed line = anomaly threshold ({m.threshold:.4f})")


# ── Tab 2: Live Predict ──────────────────────────────────────────────────────

with tab_predict:
    st.header("🔍 Live Flow Prediction")

    X_raw, y_bin, _, cols, scaler, _, multi_names = load_test_data()

    mode = st.radio("Input mode", ["Sample from test set", "Manual feature entry"],
                    horizontal=True)

    if mode == "Sample from test set":
        st.caption("Pick a random flow from the test set and predict it.")
        col_a, col_b = st.columns([1, 3])
        with col_a:
            label_filter = st.selectbox("Filter by true label",
                                        ["Any", "Benign only", "Attack only"])
        with col_b:
            seed = st.number_input("Random seed", value=42, min_value=0, step=1)

        if st.button("🎲 Sample & predict", type="primary"):
            rng = np.random.default_rng(int(seed))
            if label_filter == "Benign only":
                pool = np.where(y_bin == 0)[0]
            elif label_filter == "Attack only":
                pool = np.where(y_bin == 1)[0]
            else:
                pool = np.arange(len(X_raw))
            idx = int(rng.choice(pool))
            true_label = "BENIGN" if y_bin[idx] == 0 else "ATTACK"
            features = dict(zip(cols, X_raw[idx].tolist()))

            with st.spinner("Predicting…"):
                result = predictor.predict(features)

            st.divider()
            correct = (result["is_attack"] == (true_label == "ATTACK"))
            st.markdown(f"**True label:** `{true_label}` &nbsp; {'✅ Correct' if correct else '❌ Incorrect'}")

            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Predicted Label",  result["label"])
            r2.metric("Confidence",       f"{result['confidence']:.2%}")
            r3.metric("Anomaly Score",    f"{result['anomaly_score']:.4f}")
            r4.metric("Is Anomaly",       "⚠️ Yes" if result["is_anomaly"] else "✅ No")

            with st.expander("Full feature vector"):
                st.dataframe(
                    pd.DataFrame.from_dict(features, orient="index", columns=["Value"]),
                    use_container_width=True,
                )

    else:
        st.caption("Enter feature values manually (47 features, in order).")
        raw_input = st.text_area(
            "Paste 47 comma-separated float values:",
            placeholder="0.0, 0.0, 0.0, …",
            height=100,
        )
        if st.button("Predict", type="primary") and raw_input.strip():
            try:
                values = [float(v.strip()) for v in raw_input.split(",")]
                if len(values) != len(cols):
                    st.error(f"Expected {len(cols)} values, got {len(values)}.")
                else:
                    features = dict(zip(cols, values))
                    with st.spinner("Predicting…"):
                        result = predictor.predict(features)
                    r1, r2, r3, r4 = st.columns(4)
                    r1.metric("Label",         result["label"])
                    r2.metric("Confidence",    f"{result['confidence']:.2%}")
                    r3.metric("Anomaly Score", f"{result['anomaly_score']:.4f}")
                    r4.metric("Is Anomaly",    "⚠️ Yes" if result["is_anomaly"] else "✅ No")
            except ValueError:
                st.error("Could not parse input. Make sure all values are numbers.")


# ── Tab 3: Model Performance ─────────────────────────────────────────────────

with tab_performance:
    st.header("📈 MLflow Experiment Results")

    runs_df = load_mlflow_runs()

    if runs_df.empty:
        st.warning("No MLflow runs found.")
    else:
        st.dataframe(
            runs_df.style
            .highlight_max(subset=["F1 (wtd)", "Accuracy", "ROC-AUC"], color="#d4edda")
            .format({"Accuracy": "{:.4f}", "F1 (wtd)": "{:.4f}",
                     "Precision": "{:.4f}", "Recall": "{:.4f}", "ROC-AUC": "{:.4f}"}),
            use_container_width=True,
            hide_index=True,
        )

        st.divider()
        metric = st.selectbox("Metric to compare", ["F1 (wtd)", "Accuracy", "Precision", "Recall", "ROC-AUC"])

        chart = (
            alt.Chart(runs_df)
            .mark_bar()
            .encode(
                x=alt.X("Model:N"),
                y=alt.Y(f"{metric}:Q", scale=alt.Scale(zero=False)),
                color=alt.Color("Task:N",
                                scale=alt.Scale(domain=["Binary", "Multiclass"],
                                                range=["steelblue", "coral"])),
                column=alt.Column("Task:N"),
                tooltip=["Task", "Model", metric],
            )
            .properties(height=320, width=220)
        )
        st.altair_chart(chart)

        st.divider()
        st.subheader("Autoencoder metrics")
        ae_metrics = {
            "F1 (weighted)": 0.8872,
            "Precision":     0.8214,
            "Recall":        0.8421,
            "ROC-AUC":       0.9247,
            "Threshold (p95)": m.threshold,
        }
        cols_ae = st.columns(len(ae_metrics))
        for col, (k, v) in zip(cols_ae, ae_metrics.items()):
            col.metric(k, f"{v:.4f}")


# ── Tab 4: Anomaly Monitor ───────────────────────────────────────────────────

with tab_monitor:
    st.header("🚨 Live Traffic Simulation")
    st.caption(
        "Streams flows from the test set one by one, applying both detection "
        "pipelines in real time. Alerts fire when either pipeline flags a flow."
    )

    X_raw, y_bin, _, cols, _, _, _ = load_test_data()

    col_cfg1, col_cfg2, col_cfg3 = st.columns(3)
    n_flows   = col_cfg1.slider("Flows to stream", 10, 200, 50, step=10)
    delay_ms  = col_cfg2.slider("Delay per flow (ms)", 0, 500, 100, step=50)
    start_idx = col_cfg3.number_input("Start index in test set", 0,
                                      len(X_raw) - n_flows, 0, step=10)

    if st.button("▶️ Start simulation", type="primary"):
        progress   = st.progress(0.0, text="Streaming…")
        alert_box  = st.empty()
        stats_box  = st.empty()
        chart_box  = st.empty()

        records = []
        alerts  = []

        for i in range(n_flows):
            idx      = int(start_idx) + i
            features = dict(zip(cols, X_raw[idx].tolist()))
            result   = predictor.predict(features)
            result["flow_idx"]   = idx
            result["true_label"] = int(y_bin[idx])
            records.append(result)

            if result["is_attack"] or result["is_anomaly"]:
                alerts.append(result)

            # Progress
            progress.progress((i + 1) / n_flows,
                               text=f"Flow {i+1}/{n_flows} — {result['label']}")

            # Live stats
            df_live = pd.DataFrame(records)
            n_att  = df_live["is_attack"].sum()
            n_anom = df_live["is_anomaly"].sum()
            stats_box.markdown(
                f"**Flows processed:** {i+1} &nbsp;|&nbsp; "
                f"**Attacks detected:** {n_att} &nbsp;|&nbsp; "
                f"**Anomalies flagged:** {n_anom}"
            )

            # Rolling anomaly score chart
            chart_data = df_live[["anomaly_score", "is_attack"]].copy()
            chart_data["flow"] = range(len(chart_data))
            chart_data["is_attack"] = chart_data["is_attack"].map({True: "Attack", False: "Benign"})

            score_chart = (
                alt.Chart(chart_data)
                .mark_line(point=True)
                .encode(
                    x=alt.X("flow:Q", title="Flow #"),
                    y=alt.Y("anomaly_score:Q", title="Anomaly Score"),
                    color=alt.Color("is_attack:N",
                                    scale=alt.Scale(domain=["Benign", "Attack"],
                                                    range=["steelblue", "tomato"])),
                    tooltip=["flow", "anomaly_score", "is_attack"],
                )
                .properties(height=220, title="Reconstruction Error — Rolling Feed")
            )
            thresh_rule = (
                alt.Chart(pd.DataFrame({"y": [m.threshold]}))
                .mark_rule(color="black", strokeDash=[4, 3])
                .encode(y="y:Q")
            )
            chart_box.altair_chart(score_chart + thresh_rule, use_container_width=True)

            if delay_ms > 0:
                time.sleep(delay_ms / 1000)

        progress.progress(1.0, text="Simulation complete.")

        # Alert log
        if alerts:
            st.subheader(f"🔴 {len(alerts)} alert(s) raised")
            alert_df = pd.DataFrame(alerts)[
                ["flow_idx", "label", "is_attack", "confidence",
                 "anomaly_score", "is_anomaly", "true_label"]
            ]
            st.dataframe(alert_df, use_container_width=True, hide_index=True)
        else:
            st.success("No alerts — all flows classified as benign and normal.")
