import pandas as pd
import numpy as np

def audit_and_clean_data(df):
    print("Data Processing & Cleaning...")
    
    if df is None or df.empty:
        print("Error: No data received in Stage 2.")
        return None
        
    # 1. Clean text categories
    df['source_type'] = df['source_type'].str.strip()
    
    # 2. Safely parse and preserve timestamp column variations
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    elif 'timestamp_formatted' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp_formatted'], errors='coerce')
    else:
        df['timestamp'] = pd.Timestamp.now()
        
    print("Converting timestamps to datetime format...")
    
    # 3. Handle LLM Output text mapping variations for Stage 3 keyword scanning
    if 'LLM_output' in df.columns:
        df['llm_output'] = df['LLM_output'].fillna("")
    elif 'llm_output' in df.columns:
        df['llm_output'] = df['llm_output'].fillna("")
    else:
        df['llm_output'] = "" # Safe empty string default fallback
    
    # 4. Deduplicate based on query interactions
    initial_rows = len(df)
    df = df.drop_duplicates(subset=['final_query_clean'])
    removed_rows = initial_rows - len(df)
    print(f"Removed {removed_rows} duplicate rows.")
    
    # 5. Arrange columns safely (Ensuring both 'timestamp' and 'llm_output' pass through)
    expected_columns = [
        'final_query_clean', 'source_type', 'confidence_score', 
        'chunk_char_count', 'is_fallback', 'chunk_faithfulness', 
        'chunk_rank', 'timestamp', 'llm_output'  # <-- Added explicitly to pass to Stage 3
    ]
    
    # Fill any missing structural columns with defaults dynamically
    for col in expected_columns:
        if col not in df.columns:
            if col == 'confidence_score' or col == 'chunk_faithfulness':
                df[col] = 0.0
            elif col == 'chunk_char_count' or col == 'chunk_rank':
                df[col] = 0
            elif col == 'is_fallback':
                df[col] = False
            else:
                df[col] = np.nan
                
    # Filter array down to clean expected layout schema
    df_cleaned = df[expected_columns].copy()
    
    # Generate Quick Analytical Summary Report
    print("\n--- PIPELINE DATA QUALITY REPORT ---")
    print(f"Total clean rows processed: {len(df_cleaned)}")
    avg_conf = df_cleaned['confidence_score'].mean() * (100 if df_cleaned['confidence_score'].max() <= 1.0 else 1)
    print(f"Average System Confidence: {avg_conf:.2f}%")
    print("---------------------------------------\n")
    
    return df_cleaned