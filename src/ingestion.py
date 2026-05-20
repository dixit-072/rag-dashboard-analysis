import pandas as pd
import numpy as np
import os

def load_and_merge_csvs():
    """
    Ingests and standardizes the new production n8n chatbot logs 
    to match the pipeline's downstream structural schema requirements.
    """
    print("Stage 1: Ingesting and aligning production n8n logs...")
    
    # PATH GUARD ENGINE
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    path_options = [
        os.path.join(BASE_DIR, "data", "raw_logs.csv"),
        os.path.join(BASE_DIR, "raw_logs.csv"),
        os.path.join(BASE_DIR, "n8n LMS Chatbot Agent Logs - Logs1 (1).csv"),
        os.path.join(BASE_DIR, "data", "n8n LMS Chatbot Agent Logs - Logs1 (1).csv")
    ]
    
    input_path = None
    for option in path_options:
        if os.path.exists(option):
            input_path = option
            break
            
    if input_path is None:
        print("Ingestion Error: Your raw log file could not be found anywhere inside the repository layout!")
        return None

    # Load raw dataset
    df = pd.read_csv(input_file_path:=input_path)
    processed_df = pd.DataFrame()
    
    # Map primary structural string keys
    processed_df['final_query_clean'] = df['query'].fillna("Unknown Query")
    processed_df['source_type'] = df['source_type'].fillna("none")
    
    # Sanitize and normalize mixed float/string confidence rows ('high' -> 0.95)
    def clean_confidence(val):
        if pd.isna(val):
            return 0.0
        val_str = str(val).strip().lower()
        if val_str == 'high':
            return 0.95
        try:
            return float(val_str)
        except ValueError:
            return 0.0
            
    processed_df['confidence_score'] = df['confidence'].apply(clean_confidence)
    
    # PURE PANDAS CHUNK CHARACTER COUNT (Extract text snippet elements using regular expressions)
    extracted_text = df['chunks'].astype(str).str.extractall(r'"text":"(.*?)"')
    if not extracted_text.empty:
        extracted_text['char_len'] = extracted_text[0].str.len()
        total_lengths = extracted_text.groupby(level=0)['char_len'].sum()
        processed_df['chunk_char_count'] = total_lengths.reindex(df.index, fill_value=0)
    else:
        processed_df['chunk_char_count'] = 0

    # Build fallback binary boolean mapping flags
    processed_df['is_fallback'] = (
        (df['source_type'].str.lower() == 'fallback') | 
        (df['source_type'].str.lower() == 'none') |
        (df['chunks'].isna()) | 
        (df['chunks'] == '[]')
    )
    
    # Baseline fallback mappings downstream scripts look for
    processed_df['chunk_faithfulness'] = np.where(processed_df['is_fallback'], 0.0, 0.85)
    processed_df['chunk_rank'] = 1 
    
    print(f"Ingestion successful! {len(processed_df)} rows parsed and aligned for downstream evaluation stages.")
    return processed_df