# 🚀 Automated RAG Analytics & Observability Pipeline

A modular end-to-end data analytics pipeline designed to monitor, evaluate, and visualize operational behavior inside Retrieval-Augmented Generation (RAG) systems. 

This project processes raw LLM/chatbot logs, engineers operational performance metrics, evaluates response grounding quality using advanced semantic audit metrics, and surfaces interactive insights through live Streamlit and Power BI dashboards.

---

## 🌐 Live Application
👉 **[View Live Interactive Streamlit Dashboard](https://share.streamlit.io/dixit-072/rag-dashboard-analysis/main/app.py)** *(Deployed via Streamlit Cloud)*

---

## 🖥️ Dashboard Analytics Preview

### 🎨 Power BI Visual Canvas Layout:
![Power BI Dashboard Overview](rag_dashboard.png)

---

## 🎯 Problem Statement

Modern RAG systems generate large volumes of operational logs, but most AI applications lack centralized observability tooling to monitor system health. Key challenges include:

- **Hallucination Frequency:** Tracking how often the AI model hallucinates or invents data.
- **Retrieval Failures:** Identifying when the knowledge base fails to provide relevant context chunks.
- **Fallback Behavior:** Auditing system triggers when a user query drops to a safety or fallback layer.
- **Confidence Drift:** Monitoring changes in the distribution of model confidence scores over time.
- **Response Grounding Quality:** Evaluating how faithful the AI's final answer is to the retrieved document sources.

This project transforms raw chatbot logs into actionable operational intelligence for engineering and business stakeholders.

---

## 🏗️ Architecture & Data Flow

```text
       [ Raw CSV Logs ]
              │
              ▼
       [ Data Ingestion ] (src/ingestion.py)
              │
              ▼
 [ Data Cleaning & Processing ] (src/processing.py)
              │
              ▼
     [ Feature Engineering ] (src/features.py)
              │
              ▼
   [ Semantic Audit Metrics ] (src/audit.py)
              │
              ▼
  [ Local MySQL Database Engine ] (src/export.py)
        /                    \
       ▼                      ▼
[ Streamlit App ]      [ Power BI Dashboard ]
   (app.py)             (Rag Dashboard.pbix)