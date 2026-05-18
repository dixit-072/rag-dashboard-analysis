import pandas as pd
from sqlalchemy import create_engine
import os

def export_to_mysql(df, table_name="power_bi_evaluation"):
    """
    Connects to the local MySQL server and uploads the final 
    processed evaluation data automatically.
    """
    print(f"Starting Stage 6: Exporting data to MySQL table '{table_name}'...")
    
    if df is None or df.empty:
        print("No data available to export to MySQL.")
        return False

    # Credentials 
    user = os.getenv("DB_USER")
    password = os.getenv("DB_PASSWORD")
    host = os.getenv("DB_HOST")
    port = os.getenv("DB_PORT")
    database = os.getenv("DB_NAME")

    try:
        # Create the secure pipeline-to-database connection bridge
        print(f"Connecting to MySQL database '{database}'...")
        engine = create_engine(f'mysql+mysqlconnector://{user}:{password}@{host}:{port}/{database}')
        
        # Convert date column to string safely so MySQL doesn't alter its layout
        df_export = df.copy()
        if 'date' in df_export.columns:
            df_export['date'] = df_export['date'].astype(str)

        print("Streaming dataset rows straight to the SQL server...")
        # Push the data. if_exists='replace' means it will automatically overwrite old records with fresh data!
        df_export.to_sql(name=table_name, con=engine, if_exists='replace', index=False)
        
        print(f"SUCCESS! Database table updated perfectly. Total rows uploaded: {len(df_export)}")
        return True
        
    except Exception as e:
        print(f"Database Export Failed: {e}")
        print("Tip: Make sure your MySQL Server workbench is turned on and running!")
        return False
    

