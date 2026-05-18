import streamlit as st
import pandas as pd
import plotly.express as px
import os
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

# 1. Web Page Configuration
st.set_page_config(page_title="RAG System Analytics", page_icon="🚀", layout="wide")

# 2. Database Connection Wrapper
@st.cache_data # Keeps the app lightning-fast by caching the database data
def load_data_from_mysql():
    user = os.getenv("DB_USER")
    password = os.getenv("DB_PASSWORD")
    host = os.getenv("DB_HOST")
    port = os.getenv("DB_PORT")
    database = os.getenv("DB_NAME")
    try:
        engine = create_engine(f'mysql+mysqlconnector://{user}:{password}@{host}:{port}/{database}')
        # Pull the exploded evaluation master table we engineered
        df = pd.read_sql_table('power_bi', con=engine)
        return df
    except Exception as e:
        st.error(f" Could not connect to MySQL: {e}")
        return None

# Load the data from your local SQL server
df = load_data_from_mysql()

if df is not None:
    # 3. Header Section
    st.title("Automated RAG Production Pipeline Dashboard")
    st.markdown("Live system auditing, semantic evaluation, and diagnostic insights.")
    st.write("---")

    # 4. High-Level KPI Summary Cards 
    # Group by the actual clean query text to get accurate unique question tracking
    unique_df = df.drop_duplicates(subset=['final_query_clean'])
    
    total_queries = unique_df['final_query_clean'].nunique()
    avg_confidence = unique_df['confidence_score'].mean()
    fallback_rate = unique_df['is_fallback'].mean()
    
    # Calculate RAG Accuracy Rate (100% minus the fallback rate)
    rag_accuracy = 1.0 - fallback_rate

    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric(label="Total Queries Audited", value=total_queries)
    kpi2.metric(label="RAG System Accuracy", value=f"{rag_accuracy:.2%}")
    kpi3.metric(label="Average System Confidence", value=f"{avg_confidence:.2%}")
    st.write("---")

    # 5. Interactive Visualizations Layout
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.subheader(" Reliability Baseline (Success vs Fallback)")
        
        # Foolproof approach: Create a temporary display column to avoid mapping errors
        plot_df = unique_df.copy()
        plot_df['Status'] = plot_df['is_fallback'].apply(lambda x: 'Fallback (Failure)' if x == True or str(x).lower() == 'true' else 'Success (True)')
        
        # Let Plotly handle the counting automatically
        fig_pie = px.pie(
            plot_df, 
            names='Status', 
            color='Status',
            color_discrete_map={'Success (True)': '#2E5A88', 'Fallback (Failure)': '#E63946'},
            hole=0.4
        )
        
        st.plotly_chart(fig_pie, use_container_width=True)

    with chart_col2:
        st.subheader("Average AI Honesty by Document Source")
        # Bar chart tracking chunk faithfulness across knowledge bases
        source_faith = df.groupby('source_type')['chunk_faithfulness'].mean().reset_index()
        fig_bar = px.bar(source_faith, x='source_type', y='chunk_faithfulness',
                         labels={'source_type': 'Source Category', 'chunk_faithfulness': 'Avg Faithfulness (0-1)'},
                         color='source_type', color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig_bar, use_container_width=True)

    st.write("---")

    # 6. Advanced Scatter Plot Matrix: The Trust Gap
    st.subheader("The Trust Gap: System Confidence vs. Actual Faithfulness")
    fig_scatter = px.scatter(df, x='confidence_score', y='chunk_faithfulness', 
                             color='is_fallback', color_discrete_map={True: '#E63946', False: '#2E5A88'},
                             hover_data=['final_query_clean'], size='chunk_char_count',
                             labels={'confidence_score': 'AI Confidence (How the AI Felt)', 'chunk_faithfulness': 'ML Faithfulness (How Honest It Was)'})
    # Add the Red Danger Zone Indicator Line matching notebook 6
    fig_scatter.add_hline(y=0.3, line_dash="dash", line_color="red", annotation_text="Hallucination Risk Zone")
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.write("---")

    # 7. Diagnostic Deep-Dive Data Viewer
    st.subheader("Production Diagnostic Engine: Failure Investigation Grid")
    
    # Filter grid to show failures requiring human attention
    failures_only = df[df['is_fallback'] == True][['final_query_clean', 'source_type', 'root_cause', 'llm_output']].drop_duplicates()
    
    if not failures_only.empty:
        st.dataframe(failures_only, use_container_width=True)
    else:
        st.success("Awesome! No system fallbacks or failures detected in the current data batch.")

else:
    st.warning("Please verify your local MySQL server is active to pull pipeline tracking data.")