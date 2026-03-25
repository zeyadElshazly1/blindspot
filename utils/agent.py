import pandas as pd
import numpy as np
from scipy import stats
from itertools import combinations
import json
import openai

def get_data_overview(df):
    id_cols = [col for col in df.columns if 'id' in col.lower()]
    df = df.drop(columns=id_cols, errors='ignore')
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [
        col for col in df.select_dtypes(include=['object', 'category']).columns
        if df[col].nunique() < 20
    ]
    
    overview = {
        "shape": f"{df.shape[0]} rows x {df.shape[1]} columns",
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "sample_values": {
            col: df[col].value_counts().head(3).to_dict() 
            for col in categorical_cols[:5]
        },
        "numeric_stats": df[numeric_cols].describe().round(2).to_dict() if numeric_cols else {}
    }
    return df, overview


def tool_scan_correlations(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    results = []
    for col1, col2 in combinations(numeric_cols, 2):
        clean = df[[col1, col2]].dropna()
        if len(clean) < 10:
            continue
        corr, pvalue = stats.pearsonr(clean[col1], clean[col2])
        if abs(corr) > 0.5 and pvalue < 0.05:
            results.append({
                "columns": [col1, col2],
                "correlation": round(corr, 3),
                "pvalue": round(pvalue, 4)
            })
    return results


def tool_detect_anomalies(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    results = []
    for col in numeric_cols:
        col_data = df[col].dropna()
        if len(col_data) < 10:
            continue
        z_scores = np.abs(stats.zscore(col_data))
        outlier_count = int((z_scores > 3).sum())
        if outlier_count > 0:
            results.append({
                "column": col,
                "outlier_count": outlier_count,
                "outlier_pct": round(outlier_count / len(col_data) * 100, 1),
                "mean": round(float(col_data.mean()), 2),
                "std": round(float(col_data.std()), 2)
            })
    return results


def tool_compare_segments(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [
        col for col in df.select_dtypes(include=['object', 'category']).columns
        if 2 <= df[col].nunique() < 20
    ]
    results = []
    for cat_col in categorical_cols[:4]:
        for num_col in numeric_cols[:4]:
            groups = df.groupby(cat_col)[num_col].mean().dropna()
            if len(groups) < 2 or groups.min() == 0:
                continue
            ratio = groups.max() / groups.min()
            if ratio > 1.5:
                results.append({
                    "category": cat_col,
                    "metric": num_col,
                    "highest": {"group": str(groups.idxmax()), "value": round(float(groups.max()), 2)},
                    "lowest": {"group": str(groups.idxmin()), "value": round(float(groups.min()), 2)},
                    "ratio": round(float(ratio), 2)
                })
    return results


def tool_analyze_trends(df):
    date_cols = [col for col in df.columns if any(
        word in col.lower() for word in ['date', 'time', 'year', 'month', 'day']
    )]
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    results = []
    for date_col in date_cols[:2]:
        try:
            df[date_col] = pd.to_datetime(df[date_col])
            df_sorted = df.sort_values(date_col)
            for num_col in numeric_cols[:3]:
                monthly = df_sorted.resample('M', on=date_col)[num_col].mean()
                if len(monthly) > 2:
                    trend = "increasing" if monthly.iloc[-1] > monthly.iloc[0] else "decreasing"
                    change_pct = round((monthly.iloc[-1] - monthly.iloc[0]) / monthly.iloc[0] * 100, 1)
                    results.append({
                        "date_column": date_col,
                        "metric": num_col,
                        "trend": trend,
                        "change_pct": change_pct
                    })
        except:
            continue
    return results


# Tool definitions for OpenAI
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "scan_correlations",
            "description": "Scan all numeric columns for hidden correlations. Use this to find relationships between variables.",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "detect_anomalies",
            "description": "Detect statistical outliers and anomalies in numeric columns.",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "compare_segments",
            "description": "Compare how different categorical groups perform on numeric metrics.",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_trends",
            "description": "Analyze trends over time if date columns exist.",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    }
]


def run_agent(df, api_key, status_callback=None):
    client = openai.OpenAI(api_key=api_key)
    df_clean, overview = get_data_overview(df)

    system_prompt = """You are an expert data analyst. Your job is to autonomously explore a dataset 
and find the most valuable, non-obvious insights a business could act on.

You have access to 4 analysis tools. Use them strategically — not all tools are 
relevant for every dataset. Think like a senior analyst: look at the data overview, 
decide what's worth investigating, run the right tools, then synthesize findings 
into a clear business narrative.

After using the tools, write a final analysis report with:
1. Top 3-5 most actionable insights ranked by business impact
2. For each insight: what you found, why it matters, and what to do about it
3. One overarching recommendation

Be specific with numbers. Write for a business audience, not a technical one."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Analyze this dataset and find the most valuable insights:\n\n{json.dumps(overview, indent=2)}"}
    ]

    tool_map = {
        "scan_correlations": lambda: tool_scan_correlations(df_clean),
        "detect_anomalies": lambda: tool_detect_anomalies(df_clean),
        "compare_segments": lambda: tool_compare_segments(df_clean),
        "analyze_trends": lambda: tool_analyze_trends(df_clean)
    }

    max_iterations = 6
    for i in range(max_iterations):
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=TOOLS,
            tool_choice="auto"
        )

        message = response.choices[0].message
        messages.append(message)

        if message.tool_calls:
            for tool_call in message.tool_calls:
                tool_name = tool_call.function.name
                if status_callback:
                    status_callback(f"Running: {tool_name.replace('_', ' ')}...")
                
                result = tool_map[tool_name]()
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result)
                })
        else:
            return message.content

    return "Analysis complete."