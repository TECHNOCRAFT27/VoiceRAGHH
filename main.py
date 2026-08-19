
import polars as pl
import json

def export_sample():
    print("Instantly streaming 50 samples via HTTP range requests...")
    
    # scan_parquet() creates a lazy map of the remote file.
    # .head(50) guarantees it only downloads the exact bytes needed for 50 rows.
    df = (
        pl.scan_parquet("hf://datasets/ai4bharat/MSMARCO-XI/train/hintrain.parquet")
        .head(50)
        .collect()
    )
    
    # Convert the Polars DataFrame to a standard Python dictionary and save it
    samples = df.to_dicts()
    
    with open("sample_data.json", "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
        
    print("Saved successfully!")

if __name__ == "__main__":
    export_sample()
