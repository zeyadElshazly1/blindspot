import streamlit as st
import pandas as pd
import plotly.express as px
from io import BytesIO
from utils.analyzer import analyze_dataset, get_dataset_summary
from utils.agent import run_agent
from utils.cleaner import clean_dataset
import anthropic

st.set_page_config(
    page_title="BlindSpot — AI Data Analyzer",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .hero-title {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .hero-sub {
        font-size: 1.2rem;
        color: #888;
        margin-top: 0.5rem;
        margin-bottom: 2rem;
    }
    .step-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-left: 4px solid #667eea;
        padding: 1rem 1.5rem;
        border-radius: 0 8px 8px 0;
        margin: 1.5rem 0 1rem 0;
    }
    .insight-card {
        background: #1a1a2e;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        border: 1px solid #2a2a4e;
    }
    .badge-correlation { background: #1e3a5f; color: #63b3ed; padding: 2px 10px; border-radius: 20px; font-size: 0.75rem; }
    .badge-anomaly { background: #3d1f1f; color: #fc8181; padding: 2px 10px; border-radius: 20px; font-size: 0.75rem; }
    .badge-segment { background: #1f3d2a; color: #68d391; padding: 2px 10px; border-radius: 20px; font-size: 0.75rem; }
    .badge-data_quality { background: #3d3320; color: #f6ad55; padding: 2px 10px; border-radius: 20px; font-size: 0.75rem; }
    .metric-row { display: flex; gap: 1rem; margin: 1rem 0; }
    .divider { border: none; border-top: 1px solid #2a2a4e; margin: 2rem 0; }
</style>
""", unsafe_allow_html=True)

# Hero section
col_hero1, col_hero2 = st.columns([3, 1])
with col_hero1:
    st.markdown('<p class="hero-title">🔍 BlindSpot</p>', unsafe_allow_html=True)
    st.markdown('<p class="hero-sub">Upload any dataset. Discover what you never thought to look for.</p>', unsafe_allow_html=True)
with col_hero2:
    st.markdown("<br><br>", unsafe_allow_html=True)
    

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    api_key = st.text_input("Anthropic API key", type="password")
    st.markdown("---")
    st.markdown("### 🤖 Agent mode")
    openai_key = st.text_input("OpenAI API key", type="password", key="openai")
    st.markdown("---")
    st.markdown("### 📋 How it works")
    st.markdown("**1.** Upload a CSV or Excel file")
    st.markdown("**2.** Auto-clean your data")
    st.markdown("**3.** Profile every column")
    st.markdown("**4.** Correlations matrix")
    st.markdown("**5.** Compare any two columns")
    st.markdown("**6.** Find hidden insights")
    st.markdown("**7.** Export results")
    st.markdown("---")
    st.markdown("### 🔗 Links")
    st.markdown("[GitHub](https://github.com) · [Report bug](https://github.com)")

# File upload
st.markdown('<div class="step-header"><h3 style="margin:0">📁 Upload your dataset</h3></div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Drag and drop any CSV or Excel file",
    type=["csv", "xlsx"],
    help="CSV or Excel files up to 200MB — messy files welcome"
)

if uploaded_file:
    # Load data
    if uploaded_file.name.endswith('.csv'):
        df = None
        error_msg = None

        def try_parse(file, **kwargs):
            file.seek(0)
            try:
                d = pd.read_csv(file, **kwargs)
                if d.shape[1] > 1 and d.shape[0] > 0:
                    return d
            except:
                pass
            return None

        df = try_parse(uploaded_file, encoding='utf-8-sig')

        if df is None or df.shape[1] <= 1:
            uploaded_file.seek(0)
            try:
                lines = uploaded_file.read().decode('utf-8-sig', errors='replace').splitlines()
                max_commas = 0
                header_line = 0
                for idx, line in enumerate(lines):
                    comma_count = line.count(',')
                    if comma_count > max_commas:
                        max_commas = comma_count
                        header_line = idx
                if header_line > 0:
                    from io import StringIO
                    data_str = '\n'.join(lines[header_line:])
                    df = pd.read_csv(StringIO(data_str))
                    if df.shape[1] <= 1 or df.shape[0] == 0:
                        df = None
            except Exception as e:
                error_msg = str(e)
                df = None

        if df is None:
            df = try_parse(uploaded_file, sep=';', encoding='utf-8-sig')
        if df is None:
            df = try_parse(uploaded_file, sep='\t', encoding='utf-8-sig')
        if df is None:
            df = try_parse(uploaded_file, encoding='latin-1', on_bad_lines='skip')
        if df is None:
            df = try_parse(uploaded_file, sep=None, engine='python', encoding='utf-8-sig')

        if df is None:
            st.error("Could not parse this CSV file automatically.")
            st.info("Try opening it in Excel and saving as a fresh CSV, then re-upload.")
            st.stop()

        df.columns = df.columns.str.strip().str.replace('\ufeff', '', regex=False)

    else:
        try:
            df = pd.read_excel(uploaded_file)
        except Exception as e:
            st.error(f"Could not read Excel file: {str(e)}")
            st.stop()

    # Dataset summary metrics
    summary = get_dataset_summary(df)
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("📊 Rows", f"{summary['rows']:,}")
    col2.metric("📋 Columns", f"{summary['columns']}")
    col3.metric("🔢 Numeric", f"{summary['numeric_cols']}")
    col4.metric("🏷️ Categorical", f"{summary['categorical_cols']}")
    col5.metric("❓ Missing", f"{summary['missing_pct']}%")

    # Data health score
    from utils.profiler import calculate_health_score
    health = calculate_health_score(df)

    st.markdown("### 🏥 Dataset health score")
    col_h1, col_h2, col_h3 = st.columns([1, 2, 2])

    with col_h1:
        st.markdown(f"""
        <div style="text-align:center;padding:1rem;background:#1a1a2e;border-radius:12px;border:2px solid {health['color']}">
            <div style="font-size:3rem;font-weight:800;color:{health['color']}">{health['total']}</div>
            <div style="font-size:1.5rem;font-weight:700;color:{health['color']}">{health['grade']}</div>
            <div style="color:#888;font-size:0.9rem">{health['label']}</div>
        </div>
        """, unsafe_allow_html=True)

    with col_h2:
        st.markdown("**Score breakdown:**")
        for dimension, score in health["breakdown"].items():
            max_score = health["max_scores"][dimension]
            pct = score / max_score * 100
            bar_color = "#68d391" if pct >= 80 else "#f6ad55" if pct >= 50 else "#fc8181"
            st.markdown(f"""
            <div style="margin-bottom:8px">
                <div style="display:flex;justify-content:space-between;margin-bottom:2px">
                    <span style="font-size:0.85rem;text-transform:capitalize">{dimension}</span>
                    <span style="font-size:0.85rem;color:#888">{score}/{max_score}</span>
                </div>
                <div style="background:#2a2a4e;border-radius:4px;height:8px">
                    <div style="background:{bar_color};width:{pct}%;height:8px;border-radius:4px"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col_h3:
        if health["deductions"]:
            st.markdown("**Issues found:**")
            for d in health["deductions"]:
                st.markdown(f"🔴 {d}")
        else:
            st.success("No issues found — dataset is clean!")

    # Initialize session state (before cleaning UI; expander needs df_working)
    if "df_working" not in st.session_state:
        st.session_state.df_working = df.copy()
        st.session_state.cleaned = False

    with st.expander("👁️ Before / After cleaning comparison"):
        if not st.session_state.cleaned:
            st.info("Run auto-clean first to see the comparison.")
        else:
            tab1, tab2, tab3 = st.tabs(["Side by side", "What changed", "Health score improvement"])

            with tab1:
                col_before, col_after = st.columns(2)
                with col_before:
                    st.markdown("**Before cleaning**")
                    st.dataframe(df.head(10), use_container_width=True)
                with col_after:
                    st.markdown("**After cleaning**")
                    st.dataframe(st.session_state.df_working.head(10), use_container_width=True)

            with tab2:
                changes = []
                # Shape changes
                if df.shape[0] != st.session_state.df_working.shape[0]:
                    changes.append({
                        "Change": "Rows removed",
                        "Before": df.shape[0],
                        "After": st.session_state.df_working.shape[0],
                        "Difference": df.shape[0] - st.session_state.df_working.shape[0]
                    })
                if df.shape[1] != st.session_state.df_working.shape[1]:
                    changes.append({
                        "Change": "Columns removed",
                        "Before": df.shape[1],
                        "After": st.session_state.df_working.shape[1],
                        "Difference": df.shape[1] - st.session_state.df_working.shape[1]
                    })
                # Missing values
                missing_before = df.isnull().sum().sum()
                missing_after = st.session_state.df_working.isnull().sum().sum()
                changes.append({
                    "Change": "Missing values",
                    "Before": missing_before,
                    "After": missing_after,
                    "Difference": missing_before - missing_after
                })
                # Duplicates
                dupes_before = df.duplicated().sum()
                dupes_after = st.session_state.df_working.duplicated().sum()
                changes.append({
                    "Change": "Duplicate rows",
                    "Before": dupes_before,
                    "After": dupes_after,
                    "Difference": dupes_before - dupes_after
                })
                # Data types
                types_before = df.dtypes.value_counts().to_dict()
                types_after = st.session_state.df_working.dtypes.value_counts().to_dict()
                changes.append({
                    "Change": "Numeric columns",
                    "Before": sum(1 for t in df.dtypes if t in ['int64', 'float64']),
                    "After": sum(1 for t in st.session_state.df_working.dtypes if t in ['int64', 'float64']),
                    "Difference": sum(1 for t in st.session_state.df_working.dtypes if t in ['int64', 'float64']) - sum(1 for t in df.dtypes if t in ['int64', 'float64'])
                })

                changes_df = pd.DataFrame(changes)
                st.dataframe(changes_df, use_container_width=True, hide_index=True)

                # Visual comparison
                fig = px.bar(
                    changes_df,
                    x="Change",
                    y=["Before", "After"],
                    title="Before vs After cleaning",
                    barmode="group",
                    color_discrete_sequence=["#fc8181", "#68d391"]
                )
                fig.update_layout(height=300, margin=dict(t=40, b=20))
                st.plotly_chart(fig, use_container_width=True)

            with tab3:
                from utils.profiler import calculate_health_score
                health_before = calculate_health_score(df)
                health_after = calculate_health_score(st.session_state.df_working)

                col1, col2, col3 = st.columns(3)
                col1.metric(
                    "Score before",
                    f"{health_before['total']} ({health_before['grade']})",
                )
                col2.metric(
                    "Score after",
                    f"{health_after['total']} ({health_after['grade']})",
                    delta=f"+{round(health_after['total'] - health_before['total'], 1)}"
                )
                col3.metric(
                    "Improvement",
                    f"{round(health_after['total'] - health_before['total'], 1)} pts"
                )

                # Side by side breakdown
                breakdown_df = pd.DataFrame({
                    "Dimension": list(health_before["breakdown"].keys()),
                    "Before": list(health_before["breakdown"].values()),
                    "After": list(health_after["breakdown"].values()),
                })
                fig = px.bar(
                    breakdown_df,
                    x="Dimension",
                    y=["Before", "After"],
                    title="Health score breakdown — before vs after",
                    barmode="group",
                    color_discrete_sequence=["#fc8181", "#68d391"]
                )
                fig.update_layout(height=300, margin=dict(t=40, b=20))
                st.plotly_chart(fig, use_container_width=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # Step 2 — Cleaning
    st.markdown('<div class="step-header"><h3 style="margin:0">🧹 Clean your data</h3><p style="margin:0;color:#888;font-size:0.9rem">Auto-fix nulls, duplicates, data types, outliers and whitespace</p></div>', unsafe_allow_html=True)

    col_clean1, col_clean2 = st.columns([2, 1])
    with col_clean1:
        run_cleaning = st.button("🧹 Auto-clean dataset", use_container_width=True, type="primary")
    with col_clean2:
        skip_cleaning = st.button("Skip →", use_container_width=True)

    if run_cleaning:
        with st.spinner("Cleaning your data..."):
            df_clean, clean_report, clean_summary = clean_dataset(df)
            st.session_state.df_working = df_clean
            st.session_state.cleaned = True
            st.session_state.clean_report = clean_report
            st.session_state.clean_summary = clean_summary

    if skip_cleaning:
        st.session_state.df_working = df.copy()
        st.session_state.cleaned = False
        st.info("Skipped cleaning — using raw data.")

    if st.session_state.cleaned and "clean_report" in st.session_state:
        s = st.session_state.clean_summary
        st.success(f"✅ Cleaned in {s['steps']} steps — {s['rows_removed']} rows and {s['cols_removed']} columns removed.")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Before rows", f"{s['original_rows']:,}")
        c2.metric("After rows", f"{s['final_rows']:,}")
        c3.metric("Before cols", f"{s['original_cols']}")
        c4.metric("After cols", f"{s['final_cols']}")

        with st.expander("📋 View cleaning report"):
            for step in st.session_state.clean_report:
                impact_icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(step["impact"], "⚪")
                st.markdown(f"{impact_icon} **{step['step']}** — {step['detail']}")

        with st.expander("👁️ Preview cleaned data"):
            st.dataframe(st.session_state.df_working.head(10), use_container_width=True)

        csv = st.session_state.df_working.to_csv(index=False)
        st.download_button(
            "⬇️ Download cleaned CSV",
            data=csv,
            file_name="blindspot_cleaned.csv",
            mime="text/csv"
        )

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # Step 3 — Profile
    st.markdown('<div class="step-header"><h3 style="margin:0">📊 Data profile report</h3><p style="margin:0;color:#888;font-size:0.9rem">Full stats and distribution for every column</p></div>', unsafe_allow_html=True)

    if st.button("📊 Generate profile report", use_container_width=True):
        from utils.profiler import profile_dataset
        with st.spinner("Profiling your data..."):
            profile = profile_dataset(st.session_state.df_working)

        numeric_count = sum(1 for p in profile if p["type"] == "numeric")
        cat_count = sum(1 for p in profile if p["type"] == "categorical")
        dt_count = sum(1 for p in profile if p["type"] == "datetime")
        flagged_count = sum(1 for p in profile if p["flags"])

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Numeric", numeric_count)
        c2.metric("Categorical", cat_count)
        c3.metric("Datetime", dt_count)
        c4.metric("⚠️ Issues", flagged_count)

        st.markdown("---")

        for p in profile:
            flag_icon = "⚠️" if p["flags"] else "✅"
            with st.expander(f"**{p['column']}** — {p['type']} · {p['missing_pct']}% missing · {p['unique']} unique {flag_icon}"):
                col1, col2, col3 = st.columns(3)
                col1.metric("Type", p["type"])
                col2.metric("Missing", f"{p['missing_pct']}%")
                col3.metric("Unique", p["unique"])

                if p["flags"]:
                    st.warning(f"Issues: {', '.join(p['flags'])}")

                if p["type"] == "numeric":
                    c1, c2, c3, c4, c5 = st.columns(5)
                    c1.metric("Mean", p["mean"])
                    c2.metric("Median", p["median"])
                    c3.metric("Std", p["std"])
                    c4.metric("Min", p["min"])
                    c5.metric("Max", p["max"])
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Skewness", p["skewness"])
                    c2.metric("Outliers", p["outliers"])
                    c3.metric("Zeros", p["zeros"])
                    fig = px.histogram(
                        st.session_state.df_working, x=p["column"],
                        title=f"Distribution of {p['column']}",
                        color_discrete_sequence=["#667eea"]
                    )
                    fig.update_layout(height=250, margin=dict(t=40, b=20))
                    st.plotly_chart(fig, use_container_width=True)

                elif p["type"] == "categorical":
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Most common", p["most_common"])
                        st.metric("Most common %", f"{p['most_common_pct']}%")
                    with col2:
                        st.markdown("**Top values:**")
                        for val, count in list(p["top_values"].items())[:5]:
                            st.markdown(f"- `{val}`: {count:,}")
                    top_df = pd.DataFrame(list(p["top_values"].items()), columns=[p["column"], "count"])
                    fig = px.bar(
                        top_df, x=p["column"], y="count",
                        title=f"Top values in {p['column']}",
                        color="count", color_continuous_scale="Purples"
                    )
                    fig.update_layout(height=250, margin=dict(t=40, b=20))
                    st.plotly_chart(fig, use_container_width=True)

                elif p["type"] == "datetime":
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Earliest", p["min"])
                    col2.metric("Latest", p["max"])
                    col3.metric("Range (days)", p["range_days"])
    df_to_analyze = st.session_state.df_working

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # Auto time series detection
    date_cols = [col for col in df_to_analyze.columns 
                 if pd.api.types.is_datetime64_any_dtype(df_to_analyze[col])
                 or any(word in col.lower() for word in ['date', 'time', 'year', 'month', 'day'])]

    if date_cols:
        st.markdown('<div class="step-header"><h3 style="margin:0">📈 Time series detected</h3><p style="margin:0;color:#888;font-size:0.9rem">Date columns found — auto-plotting trends over time</p></div>', unsafe_allow_html=True)

        numeric_cols_ts = df_to_analyze.select_dtypes(include='number').columns.tolist()

        col_ts1, col_ts2 = st.columns(2)
        with col_ts1:
            date_col_selected = st.selectbox("Date column", date_cols)
        with col_ts2:
            metric_col_selected = st.selectbox("Metric to plot", numeric_cols_ts)

        if st.button("📈 Plot time series", use_container_width=True):
            try:
                df_ts = df_to_analyze.copy()
                df_ts[date_col_selected] = pd.to_datetime(df_ts[date_col_selected], errors='coerce')
                df_ts = df_ts.dropna(subset=[date_col_selected])
                df_ts = df_ts.sort_values(date_col_selected)

                # Determine best frequency
                date_range = (df_ts[date_col_selected].max() - df_ts[date_col_selected].min()).days
                if date_range > 365 * 2:
                    freq = 'M'
                    freq_label = "Monthly"
                elif date_range > 90:
                    freq = 'W'
                    freq_label = "Weekly"
                else:
                    freq = 'D'
                    freq_label = "Daily"

                df_resampled = df_ts.set_index(date_col_selected)[metric_col_selected]\
                    .resample(freq).mean().reset_index()
                df_resampled.columns = [date_col_selected, metric_col_selected]

                # Trend line
                col_ts_chart1, col_ts_chart2 = st.columns(2)

                with col_ts_chart1:
                    fig = px.line(
                        df_resampled,
                        x=date_col_selected,
                        y=metric_col_selected,
                        title=f"{freq_label} average {metric_col_selected} over time",
                        color_discrete_sequence=["#667eea"]
                    )
                    fig.update_traces(line_width=2)
                    fig.update_layout(height=350, margin=dict(t=40, b=20))
                    st.plotly_chart(fig, use_container_width=True)

                with col_ts_chart2:
                    fig = px.bar(
                        df_resampled,
                        x=date_col_selected,
                        y=metric_col_selected,
                        title=f"{freq_label} {metric_col_selected} — bar view",
                        color=metric_col_selected,
                        color_continuous_scale="Purples"
                    )
                    fig.update_layout(height=350, margin=dict(t=40, b=20))
                    st.plotly_chart(fig, use_container_width=True)

                # Stats
                first_val = df_resampled[metric_col_selected].iloc[0]
                last_val = df_resampled[metric_col_selected].iloc[-1]
                change_pct = round((last_val - first_val) / first_val * 100, 1) if first_val != 0 else 0
                trend = "increasing" if change_pct > 0 else "decreasing"

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("First value", round(first_val, 2))
                c2.metric("Last value", round(last_val, 2))
                c3.metric("Overall trend", trend)
                c4.metric("Total change", f"{change_pct}%", delta=f"{change_pct}%")

            except Exception as e:
                st.error(f"Could not plot time series: {str(e)}")

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # Step 4 — Correlation matrix
    st.markdown('<div class="step-header"><h3 style="margin:0">🔗 Correlation matrix</h3><p style="margin:0;color:#888;font-size:0.9rem">See how every numeric column relates to every other</p></div>', unsafe_allow_html=True)

    numeric_cols_list = df_to_analyze.select_dtypes(include='number').columns.tolist()

    if len(numeric_cols_list) < 2:
        st.info("Need at least 2 numeric columns to show a correlation matrix.")
    else:
        if st.button("🔗 Generate correlation matrix", use_container_width=True):
            corr_matrix = df_to_analyze[numeric_cols_list].corr().round(2)

            fig = px.imshow(
                corr_matrix,
                title="Correlation matrix — all numeric columns",
                color_continuous_scale="RdBu_r",
                zmin=-1, zmax=1,
                text_auto=True,
                aspect="auto"
            )
            fig.update_layout(
                height=500,
                margin=dict(t=60, b=20),
                coloraxis_colorbar=dict(title="r value")
            )
            fig.update_traces(textfont_size=11)
            st.plotly_chart(fig, use_container_width=True)

            # Top correlations table
            st.markdown("#### Top correlations")
            pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    col_a = corr_matrix.columns[i]
                    col_b = corr_matrix.columns[j]
                    r = corr_matrix.iloc[i, j]
                    pairs.append({
                        "Column A": col_a,
                        "Column B": col_b,
                        "Correlation (r)": r,
                        "Strength": "Strong" if abs(r) > 0.7 else "Moderate" if abs(r) > 0.4 else "Weak",
                        "Direction": "Positive" if r > 0 else "Negative"
                    })

            pairs_df = pd.DataFrame(pairs).sort_values("Correlation (r)", key=abs, ascending=False)
            st.dataframe(pairs_df, use_container_width=True, hide_index=True)

            # Download
            st.download_button(
                "⬇️ Download correlation table (CSV)",
                data=pairs_df.to_csv(index=False),
                file_name="blindspot_correlations.csv",
                mime="text/csv"
            )
    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # Step 5 — Compare two columns
    st.markdown('<div class="step-header"><h3 style="margin:0">⚡ Compare two columns</h3><p style="margin:0;color:#888;font-size:0.9rem">Pick any two columns for a deep dive comparison</p></div>', unsafe_allow_html=True)

    all_cols = df_to_analyze.columns.tolist()
    col_pick1, col_pick2 = st.columns(2)
    with col_pick1:
        col_a = st.selectbox("Column A", all_cols, index=0)
    with col_pick2:
        col_b = st.selectbox("Column B", all_cols, index=min(1, len(all_cols)-1))

    if st.button("⚡ Compare columns", use_container_width=True):
        type_a = "numeric" if pd.api.types.is_numeric_dtype(df_to_analyze[col_a]) else "categorical"
        type_b = "numeric" if pd.api.types.is_numeric_dtype(df_to_analyze[col_b]) else "categorical"

        st.markdown(f"#### Comparing `{col_a}` ({type_a}) vs `{col_b}` ({type_b})")

        # Numeric vs Numeric
        if type_a == "numeric" and type_b == "numeric":
            from scipy import stats
            corr, pvalue = stats.pearsonr(
                df_to_analyze[col_a].dropna(),
                df_to_analyze[col_b].dropna()
            )
            c1, c2, c3 = st.columns(3)
            c1.metric("Correlation (r)", round(corr, 3))
            c2.metric("P-value", round(pvalue, 4))
            c3.metric("Strength", "Strong" if abs(corr) > 0.7 else "Moderate" if abs(corr) > 0.4 else "Weak")

            col1, col2 = st.columns(2)
            with col1:
                fig = px.scatter(
                    df_to_analyze.sample(min(1000, len(df_to_analyze))),
                    x=col_a, y=col_b,
                    title=f"{col_a} vs {col_b}",
                    opacity=0.5,
                    trendline="ols",
                    color_discrete_sequence=["#667eea"]
                )
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                fig = px.density_contour(
                    df_to_analyze,
                    x=col_a, y=col_b,
                    title=f"Density: {col_a} vs {col_b}",
                    color_discrete_sequence=["#667eea"]
                )
                fig.update_traces(contours_coloring="fill", colorscale="Purples")
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True)

        # Categorical vs Numeric
        elif type_a == "categorical" and type_b == "numeric":
            groups = df_to_analyze.groupby(col_a)[col_b].agg(['mean', 'median', 'std', 'count']).round(2)
            st.dataframe(groups, use_container_width=True)

            col1, col2 = st.columns(2)
            with col1:
                fig = px.box(
                    df_to_analyze, x=col_a, y=col_b,
                    title=f"Distribution of {col_b} by {col_a}",
                    color=col_a,
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                fig.update_layout(height=350, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                fig = px.violin(
                    df_to_analyze, x=col_a, y=col_b,
                    title=f"Violin: {col_b} by {col_a}",
                    color=col_a,
                    color_discrete_sequence=px.colors.qualitative.Set3,
                    box=True
                )
                fig.update_layout(height=350, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

        # Numeric vs Categorical
        elif type_a == "numeric" and type_b == "categorical":
            groups = df_to_analyze.groupby(col_b)[col_a].agg(['mean', 'median', 'std', 'count']).round(2)
            st.dataframe(groups, use_container_width=True)

            col1, col2 = st.columns(2)
            with col1:
                fig = px.box(
                    df_to_analyze, x=col_b, y=col_a,
                    title=f"Distribution of {col_a} by {col_b}",
                    color=col_b,
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                fig.update_layout(height=350, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                fig = px.violin(
                    df_to_analyze, x=col_b, y=col_a,
                    title=f"Violin: {col_a} by {col_b}",
                    color=col_b,
                    color_discrete_sequence=px.colors.qualitative.Set3,
                    box=True
                )
                fig.update_layout(height=350, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

        # Categorical vs Categorical
        else:
            cross_tab = pd.crosstab(
                df_to_analyze[col_a],
                df_to_analyze[col_b],
                normalize='index'
            ).round(3)
            st.markdown("**Cross-tabulation (row %)**")
            st.dataframe(cross_tab, use_container_width=True)

            fig = px.imshow(
                cross_tab,
                title=f"Heatmap: {col_a} vs {col_b}",
                color_continuous_scale="Blues",
                text_auto=True,
                aspect="auto"
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # Step 5 — Analyze
    st.markdown('<div class="step-header"><h3 style="margin:0">🔍 Analyze your data</h3><p style="margin:0;color:#888;font-size:0.9rem">Find hidden patterns, anomalies and segment gaps</p></div>', unsafe_allow_html=True)


    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        run_basic = st.button("🔍 Find blind spots", type="primary", use_container_width=True)
    with col_btn2:
        run_agent_btn = st.button("🤖 Run AI agent", type="secondary", use_container_width=True)

    if run_basic:
        with st.spinner("Scanning for hidden patterns..."):
            insights = analyze_dataset(df_to_analyze)

        st.success(f"✅ Found {len(insights)} insights you might have missed!")
        st.markdown("---")

        type_icons = {
            "correlation": "🔗",
            "anomaly": "⚠️",
            "segment": "👥",
            "data_quality": "🧹"
        }
        type_colors = {
            "correlation": "#63b3ed",
            "anomaly": "#fc8181",
            "segment": "#68d391",
            "data_quality": "#f6ad55"
        }

        # Filter by type
        all_types = list(set(i["type"] for i in insights))
        selected_types = st.multiselect(
            "Filter by insight type:",
            options=all_types,
            default=all_types,
            format_func=lambda x: f"{type_icons.get(x,'🔍')} {x.replace('_',' ').title()}"
        )
        filtered_insights = [i for i in insights if i["type"] in selected_types]

        for i, insight in enumerate(filtered_insights):
            icon = type_icons.get(insight["type"], "🔍")
            with st.container():
                st.markdown(f"### {icon} {insight['title']}")
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**Finding:** {insight['finding']}")
                    st.markdown(f"**Recommended action:** {insight['action']}")
                with col2:
                    st.metric("Confidence", f"{insight['confidence']}%")
                    st.caption(f"Type: {insight['type'].replace('_', ' ').title()}")

                if insight["type"] == "segment" and "vs" in insight["title"]:
                    try:
                        parts = insight["title"].replace("Segment gap: ", "").replace("Rate gap: ", "").split(" vs ")
                        cat_col = parts[0].strip()
                        num_col = parts[1].strip()
                        if cat_col in df_to_analyze.columns and num_col in df_to_analyze.columns:
                            chart_data = df_to_analyze.groupby(cat_col)[num_col].mean().reset_index()
                            chart_data.columns = [cat_col, f"avg_{num_col}"]
                            fig = px.bar(
                                chart_data, x=cat_col, y=f"avg_{num_col}",
                                title=f"Average {num_col} by {cat_col}",
                                color=f"avg_{num_col}",
                                color_continuous_scale="RdYlGn"
                            )
                            fig.update_layout(height=300, margin=dict(t=40, b=20))
                            st.plotly_chart(fig, use_container_width=True)
                    except:
                        pass

                if insight["type"] == "correlation" and ":" in insight["title"]:
                    try:
                        parts = insight["title"].replace("Hidden relationship: ", "").split(" and ")
                        col1_name = parts[0].strip()
                        col2_name = parts[1].strip()
                        if col1_name in df_to_analyze.columns and col2_name in df_to_analyze.columns:
                            fig = px.scatter(
                                df_to_analyze.sample(min(500, len(df_to_analyze))),
                                x=col1_name, y=col2_name,
                                title=f"{col1_name} vs {col2_name}",
                                opacity=0.5, trendline="ols",
                                color_discrete_sequence=["#667eea"]
                            )
                            fig.update_layout(height=300, margin=dict(t=40, b=20))
                            st.plotly_chart(fig, use_container_width=True)
                    except:
                        pass

                if api_key:
                    if st.button(f"🤖 Ask AI to explain this", key=f"ai_{i}"):
                        with st.spinner("Generating AI explanation..."):
                            client = anthropic.Anthropic(api_key=api_key)
                            message = client.messages.create(
                                model="claude-sonnet-4-20250514",
                                max_tokens=300,
                                messages=[{
                                    "role": "user",
                                    "content": f"In 3 sentences, explain this data insight to a business manager with no technical background: {insight['finding']}. Then give one specific action they should take."
                                }]
                            )
                            st.info(message.content[0].text)

                st.markdown("---")

        # Export
        if filtered_insights:
            st.markdown("### 📥 Export results")
            col_exp1, col_exp2 = st.columns(2)
            insights_df = pd.DataFrame([{
                "Title": ins["title"],
                "Type": ins["type"],
                "Finding": ins["finding"],
                "Recommended Action": ins["action"],
                "Confidence %": ins["confidence"]
            } for ins in filtered_insights])

            with col_exp1:
                output = BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    insights_df.to_excel(writer, sheet_name="Insights", index=False)
                    st.session_state.df_working.to_excel(writer, sheet_name="Cleaned Data", index=False)
                output.seek(0)
                st.download_button(
                    "📊 Download insights + data (Excel)",
                    data=output,
                    file_name="blindspot_report.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
            with col_exp2:
                st.download_button(
                    "📄 Download insights (CSV)",
                    data=insights_df.to_csv(index=False),
                    file_name="blindspot_insights.csv",
                    mime="text/csv",
                    use_container_width=True
                )

    if run_agent_btn:
        if not openai_key:
            st.warning("Please enter your OpenAI API key in the sidebar to use agent mode.")
        else:
            st.markdown("### 🤖 AI Agent Analysis")
            status = st.empty()

            def update_status(msg):
                status.info(f"🔍 {msg}")

            with st.spinner("Agent is analyzing your data..."):
                report = run_agent(df_to_analyze, openai_key, status_callback=update_status)

            status.empty()
            st.success("✅ Agent analysis complete!")
            st.markdown("---")
            st.markdown(report)
            st.download_button(
                "📄 Download agent report (TXT)",
                data=report,
                file_name="blindspot_agent_report.txt",
                mime="text/plain",
                use_container_width=True
            )

else:
    # Landing page when no file uploaded
    st.markdown("### 👆 Upload a file above to get started")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("#### 🧹 Smart cleaning")
        st.markdown("Auto-fixes nulls, duplicates, wrong data types, outliers and whitespace — works on any file.")
    with col2:
        st.markdown("#### 🔍 Hidden insights")
        st.markdown("Scans every column combination for patterns, anomalies and segment gaps you'd never think to look for.")
    with col3:
        st.markdown("#### 🤖 AI agent")
        st.markdown("GPT-4o autonomously explores your data like a senior analyst and writes a plain English report.")
