# BlindSpot — AI Data Analyzer

> Upload any dataset. Discover what you never thought to look for.

BlindSpot is a full-featured AI-powered data analysis tool that autonomously 
explores your data and surfaces hidden insights — no SQL, no code, no dashboards required.

🔗 **Live demo:** https://blindspot-acub95qb7ff499vrq8cnzh.streamlit.app/

---

## What it does

| Feature | Description |
|---------|-------------|
| 🏥 Health score | Instant data quality score out of 100 with breakdown across 5 dimensions |
| 🧹 Smart cleaning | Auto-fixes nulls, duplicates, wrong types, outliers, whitespace — handles messy CSVs |
| 👁️ Before/After comparison | Side by side view of what changed after cleaning with health score improvement |
| 📊 Data profiling | Full column-by-column report with stats, distributions and quality flags |
| 📈 Time series detection | Auto-detects date columns and plots trends over time |
| 👯 Duplicate detector | Finds exact and near-duplicate rows using machine learning |
| 🔎 Outlier explorer | Inspect statistical outliers with distribution charts and downloadable rows |
| 🔗 Correlation matrix | Interactive heatmap of all numeric relationships with ranked table |
| ⚡ Column comparison | Pick any two columns — auto-selects the right chart type |
| 💬 Natural language query | Type "show me churn by contract type" — get a chart back |
| 🔍 Blind spot finder | 10 ranked insights with confidence scores and visualizations |
| 🤖 AI agent | GPT-4o autonomously explores data like a senior analyst |
| 📥 Export | Download cleaned data, insights and agent report |

---

## The problem it solves

Every data tool answers questions you already know to ask. You open Power BI, 
you build charts for metrics you already care about. You write SQL for columns 
you already chose.

**BlindSpot finds the questions you didn't know to ask.**

---

## Example insights (Telco churn dataset)

- "Month-to-month customers churn at 42.7% vs 2.8% on 2-year contracts"
- "Customers with partners stay 1.8x longer than those without"
- "Electronic check users churn at 45% — highest of any payment method"
- "Fiber optic customers pay 4.3x more than non-internet customers"

---

## Tech stack

Python · Streamlit · OpenAI GPT-4o · Pandas · Plotly · SciPy · Scikit-learn · Statsmodels

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
│   ├── agent.py        ← AI agent with OpenAI tool calling
│   ├── cleaner.py      ← smart data cleaning engine
│   └── profiler.py     ← column profiling and health score
└── requirements.txt
```

---

## Roadmap

- [ ] Multi-file comparison — upload two CSVs and compare them
- [ ] PDF export of full report
- [ ] User accounts and saved reports
- [ ] API access for developers