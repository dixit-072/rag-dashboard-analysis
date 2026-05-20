import os
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from dotenv import load_dotenv

# ==========================================
# 1. PAGE SETUP & STYLING
# ==========================================
st.set_page_config(page_title="RAG Production Dashboard", layout="wide")

st.title("Automated RAG Production Pipeline Dashboard")
st.caption("Live system auditing, semantic evaluation, and LLMOps optimization metrics.")

# ==========================================
# 2. NATIVE REPOSITORY DATA INGESTION
# ==========================================
@st.cache_data
def load_rag_data():
    try:
        # Clean relative mapping pointing straight to your repository outputs folder
        df = pd.read_csv("outputs/RAG_ML_Evaluation_Master.csv")
    except FileNotFoundError:
        st.error("🚨 Missing Data File: 'outputs/RAG_ML_Evaluation_Master.csv' was not found.")
        st.info("Please make sure the outputs folder containing your file is pushed to your GitHub repository.")
        st.stop()
    
    # Safe data type normalization and infrastructure fallbacks
    if 'is_fallback' in df.columns:
        df['is_fallback'] = df['is_fallback'].astype(bool)
    else:
        df['is_fallback'] = False 

    if 'chunk_faithfulness' in df.columns:
        df['chunk_faithfulness'] = pd.to_numeric(df['chunk_faithfulness'], errors='coerce').fillna(0.0)
    else:
        df['chunk_faithfulness'] = 0.0

    if 'confidence_score' in df.columns:
        df['confidence_score'] = pd.to_numeric(df['confidence_score'], errors='coerce').fillna(0.0)
    else:
        df['confidence_score'] = 0.0

    if 'chunk_char_count' in df.columns:
        df['chunk_char_count'] = pd.to_numeric(df['chunk_char_count'], errors='coerce').fillna(0)
    else:
        df['chunk_char_count'] = 0

    if 'Execution_time' in df.columns:
        df['Execution_time'] = pd.to_numeric(df['Execution_time'], errors='coerce').fillna(0.0)
    else:
        df['Execution_time'] = 1.49

    if 'total_tokens' in df.columns:
        df['total_tokens'] = pd.to_numeric(df['total_tokens'], errors='coerce').fillna(0).astype(int)
    else:
        df['total_tokens'] = 2500

    if 'Model_used' not in df.columns:
        df['Model_used'] = 'Claude Haiku'
    
    return df

# Initialize master dataframe
df_master = load_rag_data()

# ==========================================
# 3. INTERFACE TABS ARCHITECTURE
# ==========================================
tab1, tab2, tab3 = st.tabs(["📊 Executive KPIs", "🔬 Technical Deep-Dive", "⚙️ LLMOps Optimization"])

# ==========================================
# TAB 1: EXECUTIVE BUSINESS METRICS
# ==========================================
with tab1:
    st.markdown("### Operational Performance Overview")
    
    # Drop duplicate queries to evaluate unique user interactions
    df_unique_queries = df_master.drop_duplicates(subset=['final_query_clean'])
    total_queries = len(df_unique_queries)
    
    # Count how many unique queries hit a fallback
    fallback_count = int(df_unique_queries[df_unique_queries['is_fallback'] == True]['is_fallback'].count())
    
    # Performance metric aggregations
    accuracy_rate = ((total_queries - fallback_count) / total_queries) * 100 if total_queries > 0 else 0.0
    avg_confidence = df_unique_queries['confidence_score'].mean() * 100 if 'confidence_score' in df_unique_queries.columns else 0.0
    avg_latency = df_master['Execution_time'].mean() if 'Execution_time' in df_master.columns else 1.49
    
    # Executive KPI Metric Cards (Symmetrical 4-column lineup matching your Power BI design)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="Total Unique Queries", value=f"{total_queries}")
    with col2:
        st.metric(label="Avg Inference Latency", value=f"{avg_latency:.2f} sec")
    with col3:
        st.metric(label="RAG System Accuracy", value=f"{accuracy_rate:.2f}%")
    with col4:
        st.metric(label="Average System Confidence", value=f"{avg_confidence:.2f}%")
        
    st.markdown("---")
    
    st.subheader("Reliability Baseline (Success vs Fallback)")
    col_chart, col_text = st.columns([1, 1])
    
    with col_chart:
        fig_pie, ax_pie = plt.subplots(figsize=(4, 4))
        fallback_counts = df_master['is_fallback'].value_counts()
        dynamic_labels = ['Success (False)' if val == False else 'Fallback (True)' for val in fallback_counts.index]
        dynamic_colors = ['#2E5A88' if val == False else '#D9534F' for val in fallback_counts.index]
        
        fallback_counts.plot.pie(
            autopct='%1.1f%%', 
            labels=dynamic_labels, 
            startangle=90, 
            colors=dynamic_colors, 
            ax=ax_pie
        )
        ax_pie.set_ylabel('')
        st.pyplot(fig_pie)
        
    with col_text:
        st.markdown("#### Baseline Reliability Analysis Summary")
        st.write(
            f"This summary evaluation parses our core system baseline stability across tested customer interactions. "
            f"Currently, **{accuracy_rate:.2f}%** of processing user sequences output successfully structured evidence generation workflows. "
            f"The remaining baseline fraction (**{(100 - accuracy_rate):.2f}%**) hits fallback default handlers, highlighting specific contextual documentation blindspots."
        )

# ==========================================
# TAB 2: DATA ANALYST DIAGNOSTIC SUITE
# ==========================================
with tab2:
    st.markdown("### Data Analyst Diagnostic Suite")
    st.write("Deep-dive exploration into system vulnerabilities, trust gaps, and structural thresholds.")
    st.markdown("---")
    
    if 'source_type' not in df_master.columns:
        st.error("Column Missing: 'source_type' was not found in dataset matrix.")
        st.stop()

    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader("Average AI Honesty by Source Type")
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        sns.barplot(data=df_master, x='source_type', y='chunk_faithfulness', estimator=np.mean, errorbar=None, palette="Blues_d", ax=ax1)
        ax1.set_ylabel('Faithfulness Score (0 to 1)')
        ax1.set_xlabel('Source Category')
        plt.xticks(rotation=45)
        st.pyplot(fig1)

    with col_right:
        st.subheader("Does Data Volume Affect Success?")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        sns.boxplot(data=df_master, x='is_fallback', y='chunk_char_count', palette="Set2", ax=ax2)
        ax2.set_xlabel('AI Failed Status (is_fallback)')
        ax2.set_ylabel('Character Count of Evidence Snippet')
        st.pyplot(fig2)

    st.markdown("---")

    st.subheader("The Trust Gap: System Confidence vs. Actual Faithfulness")
    fig_scatter = px.scatter(
        df_master, x='confidence_score', y='chunk_faithfulness', color='is_fallback',
        color_discrete_map={False: '#2E5A88', True: '#D9534F'},
        labels={"confidence_score": "Internal System Confidence", "chunk_faithfulness": "Actual ML Faithfulness Audit Score"}
    )
    fig_scatter.add_hline(y=0.3, line_dash="dash", line_color="red", annotation_text="Hallucination Risk Threshold")
    st.plotly_chart(fig_scatter, use_container_width=True)

# ==========================================
# TAB 3: LLMOPS INFRASTRUCTURE OPTIMIZATION
# ==========================================
with tab3:
    st.markdown("### LLMOps Cloud Resource Optimization Matrix")
    st.write("Inference load metrics evaluating compute costs and runtime parameters live.")
    st.markdown("---")

    col_ops1, col_ops2 = st.columns(2)

    with col_ops1:
        st.subheader("Model Workload Share (Tokens Consumed)")
        fig_donut = px.pie(
            df_master, names='Model_used', values='total_tokens', hole=0.4,
            color_discrete_sequence=['#4CAF50', '#1A237E']
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    with col_ops2:
        st.subheader("Average Processing Latency by Knowledge Source")
        fig_col = px.bar(
            df_master, x='source_type', y='Execution_time', color='Model_used',
            barmode='group', color_discrete_sequence=['#4CAF50', '#1A237E'],
            labels={"Execution_time": "Inference Latency (Seconds)", "source_type": "Knowledge Index"}
        )
        st.plotly_chart(fig_col, use_container_width=True)