# BlindSpot — AI Data Analyzer

> Upload any dataset. Discover what you never thought to look for.

BlindSpot is an AI-powered data analysis tool that autonomously explores 
your data and surfaces hidden insights you would never have thought to look 
for — no SQL, no code, no dashboards required.

🔗 **Live demo:** https://blindspot-acub95qb7ff499vrq8cnzh.streamlit.app/

---

## The problem it solves

Every data tool answers questions you already know to ask. You open Power BI, 
you build charts for metrics you already care about. You write SQL for columns 
you already chose.

**BlindSpot finds the questions you didn't know to ask.**

---

## How it works

1. Upload any CSV or Excel file
2. Choose your mode:
   - **Find blind spots** — fast statistical scan across all columns
   - **Run AI agent** — GPT-4o autonomously explores the data like a senior analyst
3. Get plain English insights ranked by business impact — with charts

---

## Features

- Auto-detects correlations, anomalies, segment gaps and data quality issues
- AI agent uses tool calling to decide what to investigate
- Interactive charts for every insight
- Works on any dataset — sales, HR, finance, customer data
- No setup, no SQL, no technical knowledge required

---

## Example insights (from Telco churn dataset)

- "Customers with partners stay 1.8x longer — target couples with family plans"
- "Fiber optic customers pay 4.3x more than non-internet customers"
- "Electronic check users churn at 45% — highest of any payment method"

---

## Tech stack

Python · Streamlit · OpenAI GPT-4o · Pandas · Plotly · SciPy

---

## Run locally
```bash
git clone https://github.com/zeyadElshazly1/blindspot.git
cd blindspot
python -m venv venv
source venv/Scripts/activate
pip install -r requirements.txt
streamlit run app.py
```

---

## Project structure
```
blindspot/
├── app.py              ← main Streamlit app
├── utils/
│   ├── analyzer.py     ← statistical analysis engine
│   └── agent.py        ← AI agent with tool calling
└── requirements.txt
```