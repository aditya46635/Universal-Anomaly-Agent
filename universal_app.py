"""
PermitGuard AI - Universal Edition
An AI agent that detects statistical anomalies in ANY tabular dataset and
explains them in plain English using an LLM.

USAGE:
    python -m pip install -U streamlit pandas numpy plotly groq
    streamlit run universal_app.py

Set GROQ_API_KEY as an environment variable before running (see README).
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
import time

st.set_page_config(page_title="Sentry | Anomaly Detection Agent", page_icon="◈", layout="wide")

# ---------- CUSTOM STYLING ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Animated mesh background */
.stApp {
    background-color: #0B0D14;
    background-image:
        radial-gradient(circle at 15% 20%, rgba(217,164,65,0.14) 0%, transparent 40%),
        radial-gradient(circle at 85% 10%, rgba(74,144,164,0.12) 0%, transparent 40%),
        radial-gradient(circle at 50% 90%, rgba(196,69,59,0.10) 0%, transparent 45%);
    background-attachment: fixed;
    animation: meshshift 18s ease-in-out infinite alternate;
}
@keyframes meshshift {
    0%   { background-position: 0% 0%, 100% 0%, 50% 100%; }
    100% { background-position: 10% 10%, 90% 15%, 55% 92%; }
}

/* Radar scan visual */
.radar-wrap {
    position: absolute;
    top: 50%; right: 2.4rem;
    transform: translateY(-50%);
    width: 150px; height: 150px;
    display: none;
}
@media (min-width: 1100px) { .radar-wrap { display: block; } }
.radar-ring {
    position: absolute; border-radius: 50%;
    border: 1px solid rgba(217,164,65,0.35);
    top: 50%; left: 50%; transform: translate(-50%,-50%);
}
.radar-ring.r1 { width: 150px; height: 150px; }
.radar-ring.r2 { width: 105px; height: 105px; border-color: rgba(217,164,65,0.25); }
.radar-ring.r3 { width: 60px; height: 60px; border-color: rgba(217,164,65,0.2); }
.radar-sweep {
    position: absolute; top: 50%; left: 50%;
    width: 75px; height: 75px;
    transform-origin: 0% 0%;
    background: conic-gradient(from 0deg, rgba(217,164,65,0.65), transparent 32%);
    border-radius: 0 100% 0 0;
    animation: sweep 3.2s linear infinite;
}
@keyframes sweep { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
.radar-blip {
    position: absolute; width: 6px; height: 6px; border-radius: 50%;
    background: #C4453B; box-shadow: 0 0 8px 2px rgba(196,69,59,0.7);
    animation: blipfade 3.2s infinite;
}
@keyframes blipfade { 0%,60% { opacity: 0; } 65% { opacity: 1; } 100% { opacity: 0; } }

/* Hero banner — glass panel */
.sentry-hero {
    position: relative;
    background: rgba(27,30,41,0.55);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    border: 1px solid rgba(217,164,65,0.2);
    border-radius: 16px;
    padding: 3rem 2.6rem 2.6rem 2.6rem;
    margin-bottom: 2rem;
    overflow: hidden;
    box-shadow: 0 8px 40px rgba(0,0,0,0.35);
}
.sentry-eyebrow {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    font-family: 'Inter', sans-serif;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #D9A441;
    margin-bottom: 0.9rem;
    background: rgba(217, 164, 65, 0.08);
    border: 1px solid rgba(217, 164, 65, 0.25);
    padding: 0.3rem 0.7rem;
    border-radius: 20px;
}
.pulse-dot {
    width: 6px; height: 6px; border-radius: 50%;
    background: #D9A441;
    box-shadow: 0 0 0 0 rgba(217,164,65,0.6);
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0% { box-shadow: 0 0 0 0 rgba(217,164,65,0.55); }
    70% { box-shadow: 0 0 0 8px rgba(217,164,65,0); }
    100% { box-shadow: 0 0 0 0 rgba(217,164,65,0); }
}
.sentry-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 3rem;
    font-weight: 700;
    margin: 0 0 0.7rem 0;
    letter-spacing: -0.02em;
    line-height: 1.05;
    background: linear-gradient(90deg, #F5F3EE, #D9A441, #F5F3EE);
    background-size: 200% auto;
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: shine 5s linear infinite;
}
@keyframes shine { to { background-position: 200% center; } }
.sentry-title span { color: #D9A441; -webkit-text-fill-color: #D9A441; }
.sentry-subtitle {
    font-family: 'Inter', sans-serif;
    font-size: 1rem;
    color: #9A9DAD;
    max-width: 640px;
    line-height: 1.6;
    margin: 0 0 1.6rem 0;
}
.sentry-stats {
    display: flex;
    gap: 2.2rem;
    margin-top: 1.4rem;
    padding-top: 1.4rem;
    border-top: 1px solid #2A2E3B;
}
.sentry-stat-num {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.5rem;
    font-weight: 700;
    color: #F5F3EE;
}
.sentry-stat-label {
    font-family: 'Inter', sans-serif;
    font-size: 0.75rem;
    color: #74778A;
    letter-spacing: 0.02em;
}

/* Step indicators — numbered badges */
.sentry-step {
    display: flex;
    align-items: center;
    gap: 0.7rem;
    margin: 2.2rem 0 1.1rem 0;
}
.sentry-step-num {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.85rem;
    font-weight: 700;
    color: #12141C;
    background: #D9A441;
    width: 28px; height: 28px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
    box-shadow: 0 0 16px 2px rgba(217,164,65,0.55);
}
.sentry-step-text {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.1rem;
    font-weight: 600;
    color: #F2F0EA;
}

/* Metric cards */
div[data-testid="stMetric"] {
    background-color: #1B1E29;
    border: 1px solid #2A2E3B;
    border-radius: 10px;
    padding: 1.1rem 1.2rem;
}
div[data-testid="stMetricValue"] {
    font-family: 'Space Grotesk', sans-serif;
    color: #D9A441;
}
div[data-testid="stMetricLabel"] {
    font-family: 'Inter', sans-serif;
    color: #9A9DAD;
}

/* Buttons — neon glow */
.stButton > button {
    font-family: 'Inter', sans-serif;
    font-weight: 700;
    border-radius: 8px;
    border: 1px solid #D9A441 !important;
    transition: all 0.25s ease;
}
.stButton > button:hover {
    box-shadow: 0 0 22px 4px rgba(217,164,65,0.55);
    transform: translateY(-1px);
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #D9A441, #C48A2E);
    color: #12141C;
    box-shadow: 0 0 18px 2px rgba(217,164,65,0.35);
}
.stButton > button[kind="primary"]:hover {
    box-shadow: 0 0 28px 6px rgba(217,164,65,0.7);
}

/* Glass panels for select widgets */
div[data-testid="stSelectbox"], div[data-testid="stFileUploader"] {
    background: rgba(27,30,41,0.4);
    border-radius: 10px;
    padding: 0.4rem;
}

/* Dataframe */
div[data-testid="stDataFrame"] {
    border: 1px solid #2A2E3B;
    border-radius: 8px;
}

/* Footer */
.sentry-footer {
    margin-top: 3rem;
    padding-top: 1.2rem;
    border-top: 1px solid #2A2E3B;
    font-family: 'Inter', sans-serif;
    font-size: 0.78rem;
    color: #5D6072;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ---------- SESSION STATE ----------
if "explanations_done" not in st.session_state:
    st.session_state.explanations_done = False
if "report_df" not in st.session_state:
    st.session_state.report_df = None

# ---------- PLOTLY THEME ----------
PLOTLY_TEMPLATE = dict(
    layout=dict(
        paper_bgcolor="#1B1E29",
        plot_bgcolor="#1B1E29",
        font=dict(family="Inter, sans-serif", color="#9A9DAD"),
        title_font=dict(family="Space Grotesk, sans-serif", color="#F2F0EA", size=16),
        colorway=["#D9A441", "#4A90A4", "#C4453B", "#6B7280", "#8B7EC8"],
        xaxis=dict(gridcolor="#2A2E3B", linecolor="#2A2E3B"),
        yaxis=dict(gridcolor="#2A2E3B", linecolor="#2A2E3B"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
)

# ---------- HEADER ----------
st.markdown("""
<div class="sentry-hero">
    <div class="radar-wrap">
        <div class="radar-ring r1"></div>
        <div class="radar-ring r2"></div>
        <div class="radar-ring r3"></div>
        <div class="radar-sweep"></div>
        <div class="radar-blip" style="top:28%; left:65%; animation-delay:0.4s;"></div>
        <div class="radar-blip" style="top:60%; left:35%; animation-delay:1.6s;"></div>
        <div class="radar-blip" style="top:75%; left:70%; animation-delay:2.5s;"></div>
    </div>
    <div class="sentry-eyebrow"><span class="pulse-dot"></span> AI AGENT · LIVE ANOMALY SCANNING</div>
    <div class="sentry-title">◈ Sentry — <span>find what shouldn't be there</span></div>
    <p class="sentry-subtitle">
        Upload any dataset with a numeric value column and a category column.
        Sentry flags statistical outliers using per-group z-score analysis, then
        writes a plain-English explanation for each one using an LLM — turning
        raw numbers into an audit-ready report in minutes, not hours.
    </p>
    <div class="sentry-stats">
        <div><div class="sentry-stat-num">59,173</div><div class="sentry-stat-label">RECORDS SCANNED (BENCHMARK RUN)</div></div>
        <div><div class="sentry-stat-num">347</div><div class="sentry-stat-label">ANOMALIES SURFACED</div></div>
        <div><div class="sentry-stat-num">4</div><div class="sentry-stat-label">FILE FORMATS SUPPORTED</div></div>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------- STEP 1: UPLOAD ----------
st.markdown('<div class="sentry-step"><div class="sentry-step-num">1</div><div class="sentry-step-text">Upload your data</div></div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "Choose a data file",
    type=["csv", "xlsx", "xls", "tsv", "json"]
)

if uploaded_file is None:
    st.info("No file uploaded yet. Supports CSV, Excel (.xlsx/.xls), TSV, and JSON. "
             "Try it with any dataset that has a numeric amount/value column "
             "and a category/type column — transactions, sales, expenses, permits, claims, etc.")
    st.stop()

# Detect file type and load accordingly
file_name = uploaded_file.name.lower()

try:
    if file_name.endswith(".csv"):
        df_raw = pd.read_csv(uploaded_file)
    elif file_name.endswith((".xlsx", ".xls")):
        # If the workbook has multiple sheets, let the user pick one
        xls = pd.ExcelFile(uploaded_file)
        if len(xls.sheet_names) > 1:
            sheet = st.selectbox("This file has multiple sheets — pick one:", xls.sheet_names)
        else:
            sheet = xls.sheet_names[0]
        df_raw = pd.read_excel(xls, sheet_name=sheet)
    elif file_name.endswith(".tsv"):
        df_raw = pd.read_csv(uploaded_file, sep="\t")
    elif file_name.endswith(".json"):
        df_raw = pd.read_json(uploaded_file)
    else:
        st.error("Unsupported file type.")
        st.stop()
except Exception as e:
    st.error(f"Could not read this file: {e}")
    st.stop()
st.success(f"Loaded {len(df_raw):,} rows, {len(df_raw.columns)} columns.")
st.dataframe(df_raw.head(5), use_container_width=True)

# ---------- STEP 2: COLUMN SELECTION ----------
st.markdown('<div class="sentry-step"><div class="sentry-step-num">2</div><div class="sentry-step-text">Tell the agent which columns to use</div></div>', unsafe_allow_html=True)

numeric_cols = df_raw.select_dtypes(include=[np.number]).columns.tolist()
all_cols = df_raw.columns.tolist()

col_a, col_b, col_c = st.columns(3)
with col_a:
    value_col = st.selectbox("Numeric value column (what to check for outliers)", options=numeric_cols)
with col_b:
    category_col = st.selectbox("Category/group column (compare within groups)", options=all_cols)
with col_c:
    label_col = st.selectbox("ID / label column (optional, for display)", options=["(none)"] + all_cols)

z_threshold = st.slider("Anomaly sensitivity (z-score threshold — lower = more flags)", 1.5, 5.0, 3.0, 0.5)

# ---------- STEP 3: DETECT ----------
if st.button("🔎 Detect Anomalies", type="primary"):
    df = df_raw.dropna(subset=[category_col]).copy()
    zero_or_neg = df[df[value_col] <= 0]
    nonzero = df[df[value_col] > 0].copy()

    stats = nonzero.groupby(category_col)[value_col].agg(["mean", "std"]).reset_index()
    stats.columns = [category_col, "Group_Mean", "Group_Std"]
    nonzero = nonzero.merge(stats, on=category_col, how="left")
    nonzero["Z_Score"] = (nonzero[value_col] - nonzero["Group_Mean"]) / nonzero["Group_Std"]
    nonzero["Is_Anomaly"] = nonzero["Z_Score"].abs() > z_threshold
    nonzero["Flag_Type"] = np.where(
        nonzero["Is_Anomaly"],
        np.where(nonzero["Z_Score"] > 0, "Unusually HIGH", "Unusually LOW"),
        "Normal"
    )

    anomalies = nonzero[nonzero["Is_Anomaly"]].sort_values("Z_Score", key=abs, ascending=False).reset_index(drop=True)
    st.session_state.anomalies = anomalies
    st.session_state.value_col = value_col
    st.session_state.category_col = category_col
    st.session_state.label_col = label_col
    st.session_state.explanations_done = False

# ---------- RESULTS ----------
if "anomalies" in st.session_state and st.session_state.anomalies is not None:
    anomalies = st.session_state.anomalies
    value_col = st.session_state.value_col
    category_col = st.session_state.category_col
    label_col = st.session_state.label_col

    st.divider()
    st.markdown('<div class="sentry-step"><div class="sentry-step-num">3</div><div class="sentry-step-text">Results</div></div>', unsafe_allow_html=True)

    m1, m2, m3 = st.columns(3)
    m1.metric("Anomalies Found", f"{len(anomalies):,}")
    m2.metric("% of Dataset", f"{len(anomalies)/len(df_raw)*100:.2f}%")
    m3.metric("Groups Affected", anomalies[category_col].nunique())

    if len(anomalies) > 0:
        fig = px.bar(
            anomalies[category_col].value_counts().reset_index(),
            x=category_col, y="count", color=category_col, title="Anomalies by Group"
        )
        fig.update_layout(**PLOTLY_TEMPLATE["layout"], showlegend=False, height=350)
        st.plotly_chart(fig, use_container_width=True)

        display_cols = [category_col, value_col, "Group_Mean", "Z_Score", "Flag_Type"]
        if label_col != "(none)":
            display_cols = [label_col] + display_cols
        st.dataframe(anomalies[display_cols], use_container_width=True, height=350)

        # ---------- STEP 4: LLM EXPLANATIONS ----------
        st.divider()
        st.markdown('<div class="sentry-step"><div class="sentry-step-num">4</div><div class="sentry-step-text">Generate plain-English explanations</div></div>', unsafe_allow_html=True)
        n_to_explain = st.number_input(
            "How many top anomalies to explain? (uses your Groq API key, 1 call per row)",
            min_value=1, max_value=len(anomalies), value=min(10, len(anomalies))
        )

        if st.button("🤖 Explain with AI"):
            api_key = os.environ.get("GROQ_API_KEY")
            if not api_key:
                st.error("GROQ_API_KEY environment variable not set. See README for setup instructions.")
            else:
                from groq import Groq
                client = Groq(api_key=api_key)
                subset = anomalies.head(int(n_to_explain)).copy()
                explanations = []
                progress = st.progress(0)
                for i, row in subset.reset_index(drop=True).iterrows():
                    label = row[label_col] if label_col != "(none)" else f"Row {i+1}"
                    prompt = f"""You are a data analyst. Explain in 1-2 short, plain-English sentences why this record was flagged as a statistical anomaly. Be specific and factual, no fluff.

Record: {label}
Group: {row[category_col]}
Value: {row[value_col]:,.2f}
Group Average: {row['Group_Mean']:,.2f}
Z-Score: {row['Z_Score']:.2f}
Flag Type: {row['Flag_Type']}

Respond with ONLY the explanation, no preamble."""
                    try:
                        response = client.chat.completions.create(
                            model="openai/gpt-oss-20b",
                            messages=[{"role": "user", "content": prompt}],
                            max_tokens=300,
                            reasoning_effort="low",
                        )
                        explanation = response.choices[0].message.content.strip()
                    except Exception as e:
                        explanation = f"[Error: {e}]"
                    explanations.append(explanation)
                    progress.progress((i + 1) / len(subset))
                    time.sleep(0.3)

                subset["Explanation"] = explanations
                st.session_state.report_df = subset
                st.session_state.explanations_done = True

        if st.session_state.explanations_done and st.session_state.report_df is not None:
            st.success("Explanations generated.")
            report = st.session_state.report_df
            for i, row in report.iterrows():
                label = row[label_col] if label_col != "(none)" else f"Row {i+1}"
                z = abs(row['Z_Score'])
                if z > 10:
                    badge_color, badge_text = "#C4453B", "VERY HIGH"
                elif z > 5:
                    badge_color, badge_text = "#D9A441", "HIGH"
                else:
                    badge_color, badge_text = "#4A90A4", "MODERATE"

                st.markdown(f"""
                <div style="background-color:#1B1E29; border:1px solid #2A2E3B; border-left:3px solid {badge_color};
                            border-radius:6px; padding:0.8rem 1rem; margin-bottom:0.5rem;">
                    <span style="background-color:{badge_color}; color:#12141C; font-family:'Inter',sans-serif;
                                 font-size:0.68rem; font-weight:700; letter-spacing:0.05em; padding:0.15rem 0.5rem;
                                 border-radius:4px;">{badge_text}</span>
                    <span style="color:#F2F0EA; font-family:'Space Grotesk',sans-serif; font-weight:600; margin-left:0.6rem;">
                        {label}
                    </span>
                    <span style="color:#9A9DAD; font-size:0.85rem;"> — {row[category_col]} — {row[value_col]:,.2f} (z={row['Z_Score']:.2f})</span>
                </div>
                """, unsafe_allow_html=True)
                with st.expander("View explanation"):
                    st.write(row["Explanation"])

            csv = report.to_csv(index=False).encode("utf-8")
            st.download_button("Download report as CSV", csv, "anomaly_report.csv", "text/csv")
    else:
        st.warning("No anomalies found at this sensitivity level. Try lowering the z-score threshold.")

st.markdown("""
<div class="sentry-footer">
    ◈ Sentry — built by Aditya Saini · z-score detection engine + Groq LLM explanations · <a href="https://github.com/" style="color:#9A9DAD;">View source on GitHub</a>
</div>
""", unsafe_allow_html=True)
