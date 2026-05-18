import pandas as pd
import glob
import os

def clean_file_columns(df):
    """
    Cleans column names by removing extra spaces, making them lowercase,
    and replacing spaces with underscores.
    """
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    return df

def load_and_merge_csvs(data_folder="data"):
    print("Scanning for CSV files...")
    csv_files = glob.glob(os.path.join(data_folder, "*.csv"))

    if not csv_files:
        print(f"No CSV files found in '{data_folder}' folder.")
        return None

    print(f"Found {len(csv_files)} file(s).")
    dataframes = []

    # Standard column mapping dictionary 
    column_mapping = {
        "query": "user_query",
        "confidence": "confidence_score",
        "recommend_question": "recommended_action"
    }

    for file in csv_files:
        filename = os.path.basename(file)
        
        # Skip the final output file if it exists
        output_files = [
            "merge.csv", 
            "Clean_data.csv", 
            "RAG_Performance_Analysis_Final.csv", 
            "RAG_Technical_Audit_Final.csv",
            "RAG_ML_Evaluation_Master.csv"  
        ]
        if filename in output_files:
            continue

        try:
            # 1. Load the raw file
            df = pd.read_csv(file)
            
            # 2. Immediately clean the spaces and text in column headers
            df = clean_file_columns(df)
            
            # 3. Rename columns dynamically so all files speak the same language
            df = df.rename(columns=column_mapping)
            
            # 4. Drop the 'error' column if it exists (as you did in notebook 1)
            if "error" in df.columns:
                df = df.drop(columns=["error"], errors="ignore")

            print(f"Cleaned & Loaded: {filename} | Rows: {len(df)}")
            dataframes.append(df)
            
        except Exception as e:
            print(f" Error reading {filename}: {e}")

    if not dataframes:
        print("No valid CSV files could be loaded.")
        return None

    print("Merging datasets seamlessly...")
    # Now that headers are identical, pandas will stack them perfectly!
    merged_df = pd.concat(dataframes, ignore_index=True, sort=False)

    print(f"Final Merged Dataset Shape: {merged_df.shape[0]} rows, {merged_df.shape[1]} columns")

    return merged_df