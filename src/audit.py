import pandas as pd
import numpy as np
import ast
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize the vectorizer
vectorizer = TfidfVectorizer(stop_words='english')

def get_ml_score(text_a, text_b):
    """
    Calculates TF-IDF Cosine Similarity between two pieces of text safely.
    """
    if pd.isna(text_a) or pd.isna(text_b):
        return 0.0

    try:
        vectors = vectorizer.fit_transform([str(text_a), str(text_b)])
        return round(cosine_similarity(vectors[0:1], vectors[1:2])[0][0], 4)
    except Exception:
        return 0.0

def run_technical_audit(df):
    print("Technical Audit & ML Evaluation Matrix...")
    
    if df is None or df.empty:
        print("No data available for auditing.")
        return None

    # Make a copy to protect the pipeline data stream
    df_audit = df.copy()

    # 1. Parse string representations of chunks into lists
    print("Parsing string representations of chunks into lists...")
    def safe_parse_chunks(x):
        try:
            return ast.literal_eval(str(x))
        except Exception:
            return []
    df_audit['chunk_list'] = df_audit['chunks'].apply(safe_parse_chunks)

    # 2. Explode the dataset by chunks
    print("Exploding chunk arrays into individual rows...")
    df_audit = df_audit.explode('chunk_list').reset_index(drop=True)

    # 3. Extract inner text keys safely
    print("Extracting inner chunk text data...")
    def extract_text_key(x):
        if isinstance(x, dict):
            return x.get('text', 'No Context')
        return 'No Context'
    df_audit['chunk_text'] = df_audit['chunk_list'].apply(extract_text_key)

    # 4. Generate metadata calculations (ranks and lengths)
    print("Calculating retrieval rank order sequences...")
    df_audit['chunk_rank'] = df_audit.groupby('timestamp').cumcount() + 1
    df_audit['chunk_char_count'] = df_audit['chunk_text'].astype(str).str.len()

    # 5. Machine Learning Similarity Evaluation Matrix
    print("Computing TF-IDF Cosine Similarity Scores...")
    
    print("   -> Calculating Chunk Faithfulness Matrix...")
    df_audit['chunk_faithfulness'] = df_audit.apply(
        lambda x: get_ml_score(x['chunk_text'], x['llm_output']), axis=1
    )
    
    print("Calculating Query ML Relevance Matrix...")
    df_audit['ml_relevance'] = df_audit.apply(
        lambda x: get_ml_score(x['final_query_clean'], x['llm_output']), axis=1
    )

    # 6. chossing column for power BI
    print("Filtering final column order for the ML Evaluation file...")
    audit_cols = [
        'execution_id', 'timestamp', 'source_type', 'is_fallback', 'root_cause', 
        'chunk_rank', 'chunk_char_count', 'chunk_text', 'final_query_clean', 
        'query_word_count', 'llm_output', 'confidence_score', 
        'chunk_faithfulness', 'ml_relevance'  
    ]
    
    existing_audit_cols = [col for col in audit_cols if col in df_audit.columns]
    df_audit_final = df_audit[existing_audit_cols]

    print(f"Technical Audit Complete! Generated {len(df_audit_final)} rows with ML similarity scores.")
    return df_audit_final