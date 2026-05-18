import pandas as pd
import numpy as np
import ast

def classify_failure(row):
    """
    Helper function to determine the root cause of a fallback response.
    """
    failure_phrases = ["could not find", "couldn't find", "no specific information", "don't have"]
    output = str(row['llm_output']).lower()
    
    if row['is_fallback']:
        if row['num_chunks'] == 0:
            return 'Retrieval Failure: Missing Data'
        elif any(phrase in output for phrase in failure_phrases):
            return 'Generational Failure: AI Refusal/Ambiguity'
        else:
            return 'Potential Hallucination/Logic Error'
    else:
        return 'Success'

def build_advanced_features(df):
    print("Starting Stage 3: Advanced Feature Engineering...")
    
    if df is None or df.empty:
        print("No data available for feature building.")
        return None

    # 1. Cleaning up the user query text first (removes extra hidden spaces)
    print("Standardizing query text formats...")
    if 'user_query' in df.columns:
        df['final_query_clean'] = df['user_query'].str.replace(r'\s+', ' ', regex=True).str.strip()
    else:
        df['final_query_clean'] = "Unknown Query"

    # 2. Extract Date & Time Features
    print("Engineering time components...")
    df['date'] = df['timestamp'].dt.date
    df['hour'] = df['timestamp'].dt.hour
    df['day_name'] = df['timestamp'].dt.day_name()
    df['is_weekend'] = df['day_name'].isin(['Saturday', 'Sunday'])

    # 3. Performance Flagging (Fallback Detection)
    print("Flagging fallback responses via text patterns...")
    failure_keywords = "could not find|couldn't find|no specific information|don't have"
    df['is_fallback'] = df['llm_output'].str.contains(failure_keywords, case=False, na=False)

    # 4. Text Length Profiles (Using our brand new clean column!)
    print("Calculating query and response lengths...")
    df['query_length'] = df['final_query_clean'].str.len()
    df['query_word_count'] = df['final_query_clean'].apply(lambda x: len(str(x).split()))
    df['response_length'] = df['llm_output'].str.len()

    # 5. Categorize Confidence Levels
    print("Segmenting confidence scores...")
    def get_confidence_category(score):
        if score > 0.85: return 'High'
        if score >= 0.60: return 'Medium'
        return 'Low'
    df['Confidence_Category'] = df['confidence_score'].apply(get_confidence_category)

    # 6. Complex Chunks Parsing using ast
    print("Parsing retrieved text chunks safely...")
    df['chunk_character_count'] = df['chunks'].str.len()
    
    def extract_first_chunk(x):
        try:
            return ast.literal_eval(x)[0]['text'] if 'text' in str(x) else 'No source found'
        except Exception:
            return 'No source found'
            
    def count_chunks(x):
        try:
            return len(ast.literal_eval(x)) if 'text' in str(x) else 0
        except Exception:
            return 0

    df['source_evidence'] = df['chunks'].apply(extract_first_chunk)
    df['num_chunks'] = df['chunks'].apply(count_chunks)

    # 7. Diagnostics: Apply Root Cause Analysis
    print("Running Diagnostic Root-Cause Analysis...")
    df['root_cause'] = df.apply(classify_failure, axis=1)

    # 8. Select and Order columns specifically for your Power BI Dashboard layout
    final_cols = [
        'execution_id', 'timestamp', 'date', 'hour', 'day_name', 'is_weekend',
        'source_type', 'confidence_score', 'Confidence_Category', 'is_fallback', 'root_cause',
        'query_word_count', 'query_length', 'response_length', 'final_query_clean',
        'llm_output', 'source_evidence', 'num_chunks'
    ]
    
    existing_final_cols = [col for col in final_cols if col in df.columns]
    df_final = df[existing_final_cols].copy()

    print(f"Feature Engineering Complete! Generated columns layout for Power BI.")
    return df_final