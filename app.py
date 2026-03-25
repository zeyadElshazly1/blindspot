import streamlit as st
import pandas as pd
import plotly.express as px
from utils.analyzer import analyze_dataset, get_dataset_summary
from utils.agent import run_agent
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
    st.markdown("2. AI scans every column combination")
    st.markdown("3. Get plain English insights ranked by impact")

# File upload
uploaded_file = st.file_uploader(
    "Upload your dataset",
    type=["csv", "xlsx"],
    help="CSV or Excel files up to 200MB"
)

if uploaded_file:
    # Load data
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

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
    with st.expander("Preview data"):
        st.dataframe(df.head(10))

    st.markdown("---")

    # Buttons
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        run_basic = st.button("🔍 Find blind spots", type="primary", use_container_width=True)
    with col_btn2:
        run_agent_btn = st.button("🤖 Run AI agent", type="secondary", use_container_width=True)

    if run_basic:
        with st.spinner("Scanning your data for hidden patterns..."):
            insights = analyze_dataset(df)

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
                        if cat_col in df.columns and num_col in df.columns:
                            chart_data = df.groupby(cat_col)[num_col].mean().reset_index()
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
                        if col1_name in df.columns and col2_name in df.columns:
                            fig = px.scatter(
                                df.sample(min(500, len(df))),
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
                report = run_agent(df, openai_key, status_callback=update_status)

            status.empty()
            st.success("Agent analysis complete!")
            st.markdown("---")
            st.markdown(report)

else:
    st.info("Upload a CSV or Excel file to get started. Try your churn dataset!")