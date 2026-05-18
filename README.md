# 🚀 Automated RAG Production Pipeline Dashboard

An enterprise-grade, end-to-end data engineering and analytics pipeline that scans raw LLM/Chatbot application logs, executes a modular data cleaning and feature engineering workflow, runs semantic text evaluation metrics, and surfaces operational diagnostics via an interactive web dashboard.

---

## 📊 Core Operational Metrics Tracked
* **Total Queries Audited:** Accurate distinct tracking of unique incoming user queries.
* **RAG System Accuracy:** Real-time visibility into successful vs. fallback system triggers.
* **Average System Confidence:** Tracking model confidence distributions across different user prompts.
* **AI Honesty / Faithfulness:** Evaluating how grounded responses are against technical document sources.

---

## 🏗️ Project Architecture & Data Flow
The system is built using a decoupled, modular design separating the core processing engine from the reporting interface:

1. **Data Ingestion & Alignment (`src/ingestion.py`):** Automatically scans raw storage directories, tracks file signatures, aligns column schemas, and merges disparate log sources seamlessly.
2. **Text Processing & Sanitization (`src/processing.py`):** Standardizes textual data types, formats dates, and normalizes categorical system dimensions.
3. **Feature Engineering Engine (`src/features.py`):** Extracts analytical indicators (query lengths, response delays, context chunk counts, and text pattern matching for fallbacks).
4. **Database Pipeline Bridge (`src/export.py`):** Connects securely via SQLAlchemy and drops fully engineered records into a localized MySQL Server instance.
5. **Interactive UI Application (`app.py`):** Built with Streamlit and Plotly Express to pull processed records from MySQL and display real-time interactive analytical visualizations.

---

## 🛠️ Tech Stack & Dependencies
* **Language:** Python 3.14
* **Data Engineering:** Pandas, NumPy
* **Database Management:** MySQL Server, SQLAlchemy, MySQL-Connector-Python
* **Data Visualization & UI:** Streamlit, Plotly Express
* **Environment Security:** Python-Dotenv (Zero hardcoded passwords)
* **Version Control & Hosting:** Git, GitHub, Streamlit Cloud

---

## 🔒 Security & Local Deployment Note
This repository contains the pure processing code and visual structures. To ensure production data privacy and system security:
* All source datasets (`.csv`) are strictly ignored via version control rules.
* Production database access configurations are loaded dynamically using system environment variables (`.env`) and are never exposed to the public codebase.