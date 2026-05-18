import pandas as pd

def audit_and_clean_data(df):
    print("Data Processing & Cleaning...")
    
    if df is None or df.empty:
        print("No data available to process.")
        return None

    # 1. Standardize text values (lowercase & strip accidental spaces)
    print("Cleaning text categories (source_type & status)...")
    if 'source_type' in df.columns:
        df['source_type'] = df['source_type'].astype(str).str.lower().str.strip()
    if 'status' in df.columns:
        df['status'] = df['status'].astype(str).str.lower().str.strip()

    # 2. Convert timestamp strings to proper Datetime objects
    print("Converting timestamps to datetime format...")
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

    # 3. Remove any identical duplicate rows
    initial_rows = len(df)
    df = df.drop_duplicates()
    dropped_rows = initial_rows - len(df)
    if dropped_rows > 0:
        print(f"Removed {dropped_rows} duplicate rows.")
    else:
        print("No duplicate rows found.")

    # 4. Reorder columns: Move 'user_query' right before 'llm_output'
    print("Arranging column order...")
    if 'user_query' in df.columns and 'llm_output' in df.columns:
        cols = list(df.columns)
        cols.remove('user_query')  # Take it out from its current spot
        llm_idx = cols.index('llm_output')  # Find where llm_output is
        cols.insert(llm_idx, 'user_query')  # Insert user_query right before it
        df = df[cols]  # Apply the new order to our dataframe

    # 5. Print a quick pipeline health report to your terminal
    print("\n--- PIPELINE DATA QUALITY REPORT ---")
    print(f"Total clean rows processed: {df.shape[0]}")
    print(f"Average System Confidence: {df['confidence_score'].mean():.2%}")
    
    # Calculate what percentage of answers have low confidence (below 60%)
    low_conf_pct = (df['confidence_score'] < 0.6).mean()
    print(f"Low Confidence Responses (<60%): {low_conf_pct:.2%}")
    print("---------------------------------------\n")

    return df