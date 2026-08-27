# Universal Anomaly Detection Agent

An AI agent that detects statistical anomalies in any tabular dataset and explains each one in plain English — upload a CSV, pick a value column and a category column, and get flagged outliers with LLM-generated explanations.

## Problem

Anomaly detection is usually built as a one-off script tied to a specific dataset (fraud detection for transactions, outlier detection for sales, etc.). This project generalizes that pattern into a single reusable tool: point it at any dataset with a numeric column and a grouping column, and it finds and explains the outliers — no code changes required.

## What it does

1. **Upload** any CSV through a web UI
2. **Select** which column to check for outliers and which column defines the comparison group (e.g. "Amount" within "Category", or "Claim_Value" within "Claim_Type")
3. **Detect** anomalies using a z-score calculated per group, with an adjustable sensitivity threshold
4. **Explain** flagged records in plain English using an LLM (Groq / Llama)
5. **Export** the flagged records with explanations as a CSV report

## How it works

```
Any CSV
   │
   ▼
Column selection (user picks value + category columns)
   │
   ▼
Per-group z-score calculation → flags |z| > threshold
   │
   ▼
LLM generates a plain-English reason for each flagged record
   │
   ▼
Interactive results table + downloadable CSV report
```

## Tech stack

- **Streamlit** — web UI, file upload, interactive filtering
- **pandas / numpy** — statistical anomaly detection (z-score per group)
- **Groq API** (Llama / GPT-OSS models) — natural language explanation generation
- **Plotly** — result visualizations

## Setup

```bash
pip install -r requirements.txt
```

Get a free API key from [console.groq.com](https://console.groq.com) and set it:

```bash
# Windows PowerShell
$env:GROQ_API_KEY="your-key-here"

# Mac/Linux
export GROQ_API_KEY="your-key-here"
```

## Usage

```bash
streamlit run universal_app.py
```

Upload any CSV with at least one numeric column and one category/grouping column, select the columns, and click "Detect Anomalies."

## Example use cases

- Flagging unusual transaction amounts within a customer segment
- Finding outlier expense claims within a department
- Spotting anomalous sales figures within a product category
- Auditing permit/license values within a jurisdiction type

## Design decisions & limitations

- **Z-score method** assumes roughly normal distribution within each group; works well for symmetric outliers but is a blunter tool on heavily right-skewed data (common in financial/value datasets). A future iteration could add an IQR or isolation-forest option.
- **Groups with too few records** (a handful of rows) will produce unreliable standard deviations — the tool doesn't currently warn about this, which is a known limitation.
- **LLM explanations describe the statistical flag, not a verified cause** — this is a triage tool to prioritize human review, not an automated fraud/error determination system.
- **Cost control** — the UI lets you choose how many flagged records to send to the LLM (rather than explaining all of them automatically), since each explanation is a separate API call.

## Project structure

```
universal-anomaly-agent/
├── universal_app.py      # Streamlit app — upload, detect, explain, export
├── requirements.txt
└── README.md
```

## Future improvements

- Add IQR-based and isolation-forest detection methods as alternatives to z-score
- Let users save/reuse column mappings for datasets they check repeatedly
- Add a human-in-the-loop feedback step (confirm/reject flags) to tune sensitivity over time
