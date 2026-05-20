import os
import pandas as pd
import json
from dotenv import load_dotenv
from sqlalchemy import create_engine

# Load variables from the .env file automatically
load_dotenv()

def export_to_mysql(df, table_name="power_bi"):
    # Grab configuration tokens safely
    host = os.getenv("DB_HOST", "localhost")
    port = os.getenv("DB_PORT")
    user = os.getenv("DB_USER")
    password = os.getenv("DB_PASSWORD")
    db_name = os.getenv("DB_NAME")
    
    if not all([port, user, password, db_name]):
        print(f"Connecting to MySQL database '{db_name}'...")
        print("Database Export Failed: Missing credentials in environmental space.")
        return False
        
    try:
        port = int(port) # Convert port string to integer
        
        # Create a copy to avoid altering the main runtime data frame
        df_export = df.copy()
        
        for col in df_export.columns:
            # If the column contains elements that are lists or dictionaries, convert to strings
            if df_export[col].apply(lambda x: isinstance(x, (dict, list))).any():
                df_export[col] = df_export[col].apply(lambda x: json.dumps(x) if isinstance(x, (dict, list)) else str(x))
        
        print(f"Connecting to MySQL database '{db_name}' on {host}:{port}...")
        
        # Create secure engine connection
        connection_string = f"mysql+pymysql://{user}:{password}@{host}:{port}/{db_name}"
        engine = create_engine(connection_string)
        
        # Write data frame directly to MySQL database
        df_export.to_sql(
            name=table_name, 
            con=engine, 
            if_exists='replace',  # Overwrites old rows completely
            index=False
        )
        
        print(f"Success! Data successfully exported to MySQL table '{table_name}'.")
        return True
        
    except Exception as e:
        print(f" Database Export Failed: {e}")
        return False