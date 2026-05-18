from src.ingestion import load_and_merge_csvs
from src.processing import audit_and_clean_data
from src.features import build_advanced_features
from src.audit import run_technical_audit
from src.export import export_to_mysql
import os

def run_pipeline():
    print("==============================================")
    print("🚀 AUTOMATED RAG SYSTEM PIPELINE RUNNING")
    print("==============================================")
    
    # STAGE 1: Ingest and align raw logs
    raw_data = load_and_merge_csvs()
    
    # STAGE 2: Core Processing & Cleaning
    if raw_data is not None:
        cleaned_data = audit_and_clean_data(raw_data)
        
        if cleaned_data is not None:
            # STAGE 3: Advanced Feature Engineering
            production_data = build_advanced_features(cleaned_data)

            # output folder creation
            os.makedirs("outputs", exist_ok=True)
            
            # Save primary dataset for Power BI backup
            if production_data is not None:
                output_path_main = os.path.join("outputs", "RAG_Performance_Analysis_Final.csv")
                production_data.to_csv(output_path_main, index=False)
                print(f"Success! Performance backup dataset saved to: {output_path_main}")
                
            # STAGE 4 & 5: Technical Evidence Audit & ML Matrix Evaluation
            technical_audit_data = run_technical_audit(cleaned_data)
            
            if technical_audit_data is not None:
                # Save secondary exploded technical dataset for Power BI backup
                output_path_audit = os.path.join("outputs", "RAG_ML_Evaluation_Master.csv")
                technical_audit_data.to_csv(output_path_audit, index=False)
                print(f"Success! ML Evaluation Matrix saved to: {output_path_audit}")
                
                # STAGE 6: Exporting to Mysql
                export_success = export_to_mysql(technical_audit_data, table_name="power_bi")
                
                if export_success:
                    print("\nPRODUCTION PIPELINE RUN COMPLETE! Database live-updated successfully.")
                else:
                    print("\nPipeline finished with warnings: Database export failed.")
                    
    else:
        print("\nPipeline terminated: Ingestion failed.")

if __name__ == "__main__":
    run_pipeline()