import tiktoken
import os
import pandas as pd
import numpy as np
import ollama



class Embedder:

    def __init__(self,config):
        self.EMBEDDING_MODEL = config["EMBEDDING_MODEL"]
        self.AVG = config["AVG"]
        self.TABLES_URL = config["BASE_PATH"] / f"CTA-SCH-R{config['ROUND']}/Round{config['ROUND']}-SOTAB-CTA-SCH-Tables"
        self.BOTH_EMBS = config["BOTH_EMBS"]


    def truncate_to_tokens(self, text, max_tokens=7500, model="gpt-4o-mini"):
        encoding = tiktoken.encoding_for_model(model)
        # 1. Convert text → tokens
        tokens = encoding.encode(text)
        # 2. Cut to first N tokens
        truncated_tokens = tokens[:max_tokens]
        # 3. Convert back → text
        return encoding.decode(truncated_tokens)
    
    def get_embedding(self, df, gt_row):

        column_text, col_name, col_data = self.get_essential_info(df, gt_row)
        try:
            response = ollama.embed(
            model=self.EMBEDDING_MODEL,
            input=column_text
            )
            if not self.AVG:
                embedding = response['embeddings'][0]
            else:
                embeddings = response["embeddings"]
                embedding = np.mean(embeddings, axis=0).tolist()
        except Exception as e:
            print(f"⚠️ Embedding failed for column '{col_name}': {e} and gt_row {gt_row}")
            embedding = [0] * 768
            if self.EMBEDDING_MODEL == "snowflake-arctic-embed2:568m":
                embedding = [0] * 1024
            
            #embedding = None


        return embedding, col_data
    


    def get_essential_info(self, df, gt_row):

        table_path = os.path.join(self.TABLES_URL, df["table"][gt_row])
        table = pd.read_json(table_path,compression="gzip",lines=True)

        col_name = df["header"][gt_row]
        col_num = df["col"][gt_row]
        col_data = table.iloc[:40, col_num]
        column_text = col_data.dropna().astype(str).tolist()
        column_text = column_text * 2

        if len(column_text) == 0:
            col_data = table.iloc[:, col_num]
            column_text = col_data.dropna().astype(str).tolist()

        if not self.AVG:
            column_text = " | ".join(column_text)
            column_text = self.truncate_to_tokens(column_text, 7500)
        # else:
        #     column_text = self.truncate_to_tokens(column_text, 7500)
        
        
        return column_text, col_name, col_data
        