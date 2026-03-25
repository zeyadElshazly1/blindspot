import streamlit as st
import pandas as pd
import plotly.express as px
from utils.analyzer import analyze_dataset, get_dataset_summary
from utils.agent import run_agent
from utils.cleaner import clean_dataset
import anthropic

st.set_page_config(
    page_title="BlindSpot — AI Data Analyzer",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 BlindSpot")
st.subheader("Upload any dataset. Discover what you never thought to look for.")

# Sidebar
with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("Anthropic API key", type="password")
    st.markdown("---")
    st.markdown("**Agent mode:**")
    openai_key = st.text_input("OpenAI API key", type="password", key="openai")
    st.markdown("---")
    st.markdown("**How it works:**")
    st.markdown("1. Upload a CSV or Excel file")
    st.markdown("2. Auto-clean your data")
    st.markdown("3. Profile every column")
    st.markdown("4. AI scans for hidden insights")

# File upload
uploaded_file = st.file_uploader(
    "Upload your dataset",
    type=["csv", "xlsx"],
    help="CSV or Excel files up to 200MB"
)

if uploaded_file:
    # Load data
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

        # Strategy 1: standard
        df = try_parse(uploaded_file, encoding='utf-8-sig')

        # Strategy 2: skip metadata rows — find where real data starts
        if df is None or df.shape[1] <= 1:
            uploaded_file.seek(0)
            try:
                lines = uploaded_file.read().decode('utf-8-sig', errors='replace').splitlines()
                # Find the header row — the line with the most commas
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

        # Strategy 3: semicolon delimiter
        if df is None:
            df = try_parse(uploaded_file, sep=';', encoding='utf-8-sig')

        # Strategy 4: tab delimiter
        if df is None:
            df = try_parse(uploaded_file, sep='\t', encoding='utf-8-sig')

        # Strategy 5: latin-1 encoding + skip bad lines
        if df is None:
            df = try_parse(uploaded_file, encoding='latin-1', on_bad_lines='skip')

        # Strategy 6: auto-detect separator
        if df is None:
            df = try_parse(uploaded_file, sep=None, engine='python', encoding='utf-8-sig')

        if df is None:
            st.error("Could not parse this CSV file automatically.")
            st.info("Try opening it in Excel and saving as a fresh CSV, then re-upload.")
            st.stop()

        # Clean up column names from BOM or whitespace
        df.columns = df.columns.str.strip().str.replace('\ufeff', '', regex=False)

    else:
        try:
            df = pd.read_excel(uploaded_file)
        except Exception as e:
            st.error(f"Could not read Excel file: {str(e)}")
            st.stop()

    # Dataset summary
    summary = get_dataset_summary(df)

    st.markdown("---")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Rows", f"{summary['rows']:,}")
    col2.metric("Columns", f"{summary['columns']}")
    col3.metric("Numeric", f"{summary['numeric_cols']}")
    col4.metric("Categorical", f"{summary['categorical_cols']}")
    col5.metric("Missing data", f"{summary['missing_pct']}%")

    # Preview
    with st.expander("Preview raw data"):
        st.dataframe(df.head(10))

    st.markdown("---")

    # Cleaning section
    st.markdown("## 🧹 Step 1 — Clean your data")
    st.caption("BlindSpot will automatically fix common data issues before analysis.")

    col_clean1, col_clean2 = st.columns([2, 1])
    with col_clean1:
        run_cleaning = st.button("🧹 Auto-clean dataset", use_container_width=True)
    with col_clean2:
        skip_cleaning = st.button("Skip cleaning", use_container_width=True)

    if "df_working" not in st.session_state:
        st.session_state.df_working = df.copy()
        st.session_state.cleaned = False

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
        st.info("Skipped cleaning — using raw data for analysis.")

    if st.session_state.cleaned and "clean_report" in st.session_state:
        s = st.session_state.clean_summary
        st.success(f"Dataset cleaned in {s['steps']} steps — {s['rows_removed']} rows and {s['cols_removed']} columns removed.")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Original rows", f"{s['original_rows']:,}")
        c2.metric("Clean rows", f"{s['final_rows']:,}")
        c3.metric("Original cols", f"{s['original_cols']}")
        c4.metric("Clean cols", f"{s['final_cols']}")

        with st.expander("View cleaning report"):
            for step in st.session_state.clean_report:
                impact_color = {
                    "high": "🔴", "medium": "🟡", "low": "🟢"
                }.get(step["impact"], "⚪")
                st.markdown(f"{impact_color} **{step['step']}** — {step['detail']}")

        with st.expander("Preview cleaned data"):
            st.dataframe(st.session_state.df_working.head(10))

        # Download cleaned data
        csv = st.session_state.df_working.to_csv(index=False)
        st.download_button(
            label="Download cleaned CSV",
            data=csv,
            file_name="blindspot_cleaned.csv",
            mime="text/csv"
        )

    st.markdown("---")
    st.markdown("## 📊 Step 2 — Data profile report")
    st.caption("A full overview of every column in your dataset.")

    if st.button("📊 Generate profile report", use_container_width=True):
        from utils.profiler import profile_dataset
        with st.spinner("Profiling your data..."):
            profile = profile_dataset(st.session_state.df_working)

        st.success(f"Profiled {len(profile)} columns successfully!")

        # Summary row
        numeric_count = sum(1 for p in profile if p["type"] == "numeric")
        cat_count = sum(1 for p in profile if p["type"] == "categorical")
        dt_count = sum(1 for p in profile if p["type"] == "datetime")
        flagged_count = sum(1 for p in profile if p["flags"])

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Numeric columns", numeric_count)
        c2.metric("Categorical columns", cat_count)
        c3.metric("Datetime columns", dt_count)
        c4.metric("Columns with issues", flagged_count)

        st.markdown("---")

        for p in profile:
            with st.expander(f"**{p['column']}** — {p['type']} · {p['missing_pct']}% missing · {p['unique']} unique values {'⚠️' if p['flags'] else '✅'}"):

                col1, col2, col3 = st.columns(3)
                col1.metric("Type", p["type"])
                col2.metric("Missing", f"{p['missing_pct']}%")
                col3.metric("Unique values", p["unique"])

                if p["flags"]:
                    st.warning(f"Issues detected: {', '.join(p['flags'])}")

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

                    # Distribution chart
                    fig = px.histogram(
                        st.session_state.df_working,
                        x=p["column"],
                        title=f"Distribution of {p['column']}",
                        color_discrete_sequence=["#3498db"]
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

                    # Bar chart of top values
                    top_df = pd.DataFrame(
                        list(p["top_values"].items()),
                        columns=[p["column"], "count"]
                    )
                    fig = px.bar(
                        top_df, x=p["column"], y="count",
                        title=f"Top values in {p['column']}",
                        color="count",
                        color_continuous_scale="Blues"
                    )
                    fig.update_layout(height=250, margin=dict(t=40, b=20))
                    st.plotly_chart(fig, use_container_width=True)

                elif p["type"] == "datetime":
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Earliest", p["min"])
                    col2.metric("Latest", p["max"])
                    col3.metric("Range (days)", p["range_days"])

    st.markdown("---")
    st.markdown("## 🔍 Step 3 — Analyze your data")
    

    # Use cleaned df if available
    df_to_analyze = st.session_state.df_working

    # Buttons
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        run_basic = st.button("🔍 Find blind spots", type="primary", use_container_width=True)
    with col_btn2:
        run_agent_btn = st.button("🤖 Run AI agent", type="secondary", use_container_width=True)

    if run_basic:
        with st.spinner("Scanning your data for hidden patterns..."):
            insights = analyze_dataset(df_to_analyze)

        st.success(f"Found {len(insights)} insights you might have missed!")
        st.markdown("---")

        type_icons = {
            "correlation": "🔗",
            "anomaly": "⚠️",
            "segment": "👥",
            "data_quality": "🧹"
        }

        for i, insight in enumerate(insights):
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
                                opacity=0.5,
                                trendline="ols"
                            )
                            fig.update_layout(height=300, margin=dict(t=40, b=20))
                            st.plotly_chart(fig, use_container_width=True)
                    except:
                        pass

                if api_key:
                    if st.button(f"Ask AI to explain this", key=f"ai_{i}"):
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
            st.success("Agent analysis complete!")
            st.markdown("---")
            st.markdown(report)

else:
    st.info("Upload a CSV or Excel file to get started. Try your churn dataset!")
    