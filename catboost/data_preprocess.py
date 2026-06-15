import time
import sys
from pathlib import Path
import pandas as pd
import os

sys.path.append(str(Path(__file__).resolve().parents[1]))

from functions import get_config
from embedder import Embedder

def initialize():
    config = get_config()

    ROUND = config["ROUND"]
    AVG = config["AVG"]
    BASE_PATH = config["BASE_PATH"]
    EMBEDDING_MODEL = config["EMBEDDING_MODEL"]
    
    return ROUND, AVG, BASE_PATH, EMBEDDING_MODEL



def process_data():
    start_time = time.time()

    ROUND, AVG, BASE_PATH, EMBEDDING_MODEL = initialize()
    config = get_config()

    splits = [
        ("train", BASE_PATH / f"CTA-SCH-R{ROUND}/gt/sotab_cta_train_round{ROUND}.csv"),
        ("val",   BASE_PATH / f"CTA-SCH-R{ROUND}/gt/sotab_cta_validation_round{ROUND}.csv"),
        ("test",  BASE_PATH / f"CTA-SCH-R{ROUND}/gt/sotab_cta_test_round{ROUND}.csv"),
    ]
    TABLES_URL = BASE_PATH / f"CTA-SCH-R{ROUND}/Round{ROUND}-SOTAB-CTA-SCH-Tables" 

    for name, path in splits:
        print(name, path)
        df = pd.read_csv(path)
        
        df["embedding"] = None

        iteration = 0

        embedder = Embedder(config)

        for gt_row in range(len(df)):
            iteration += 1

            
            embedding, col_data = embedder.get_embedding(df, gt_row)
            has_dups = has_duplicates(col_data)
            #print(has_dups)
 
            embedding = embedding + [has_dups]
            #print(embedding)

            df.at[gt_row, "embedding"] = embedding

        
        output_path = path.parent / "supplementary_files" / f"{name}_{EMBEDDING_MODEL}_AVG:{AVG}.csv"
        df.to_csv(output_path, index=False)

        print(f"Saved: {output_path}")



 


    end_time = time.time()
    print(f"Total execution time for data pre_processing: {end_time - start_time:.2f} seconds")


def has_duplicates(col_data):
    values = col_data.dropna()
    has_dups = values.duplicated().any()
    return int(has_dups)

def main():
    process_data()

if __name__ == "__main__":
    main()
