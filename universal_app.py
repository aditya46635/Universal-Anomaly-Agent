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

st.set_page_config(page_title="Anomaly Detection Agent", page_icon="🔍", layout="wide")

# ---------- SESSION STATE ----------
if "explanations_done" not in st.session_state:
    st.session_state.explanations_done = False
if "report_df" not in st.session_state:
    st.session_state.report_df = None

# ---------- HEADER ----------
st.title("🔍 Anomaly Detection Agent")
st.caption(
    "Upload any dataset with a numeric value column and a category column. "
    "This agent flags statistical outliers and explains each one in plain English using an LLM."
)

# ---------- STEP 1: UPLOAD ----------
st.subheader("1. Upload your data")
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
st.subheader("2. Tell the agent which columns to use")

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
    st.subheader("3. Results")

    m1, m2, m3 = st.columns(3)
    m1.metric("Anomalies Found", f"{len(anomalies):,}")
    m2.metric("% of Dataset", f"{len(anomalies)/len(df_raw)*100:.2f}%")
    m3.metric("Groups Affected", anomalies[category_col].nunique())

    if len(anomalies) > 0:
        fig = px.bar(
            anomalies[category_col].value_counts().reset_index(),
            x=category_col, y="count", color=category_col, title="Anomalies by Group"
        )
        fig.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig, use_container_width=True)

        display_cols = [category_col, value_col, "Group_Mean", "Z_Score", "Flag_Type"]
        if label_col != "(none)":
            display_cols = [label_col] + display_cols
        st.dataframe(anomalies[display_cols], use_container_width=True, height=350)

        # ---------- STEP 4: LLM EXPLANATIONS ----------
        st.divider()
        st.subheader("4. Generate plain-English explanations (optional)")
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
                with st.expander(f"{label} — {row[category_col]} — {row[value_col]:,.2f} (z={row['Z_Score']:.2f})"):
                    st.write(row["Explanation"])

            csv = report.to_csv(index=False).encode("utf-8")
            st.download_button("Download report as CSV", csv, "anomaly_report.csv", "text/csv")
    else:
        st.warning("No anomalies found at this sensitivity level. Try lowering the z-score threshold.")
