import pickle
from pyexpat import model
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

class CatBoostInference:

    def __init__(self,config):
        self.model = CatBoostClassifier() 

        ROUND = config["ROUND"]
        EMBEDDING_MODEL = config["EMBEDDING_MODEL"]
        AVG = config["AVG"]
        
        # Model path is relative to the workspace root (catboost/model/)
        model_dir = Path(__file__).parent / "model"
        
        model_path = model_dir / f"CatBoost_ROUND{ROUND}_{EMBEDDING_MODEL}_AVG-{AVG}.cbm"
        print(model_path)
        label_encoder_path = model_dir / f"CatBoost_ROUND{ROUND}_{EMBEDDING_MODEL}_AVG-{AVG}_label_encoder.pkl"

        self.model.load_model(str(model_path))
        
        with open(label_encoder_path, "rb") as f:
            self.le = pickle.load(f)

        # self.model.load_model("catboost/model/CatBoost_ROUND2_nomic-embed-text-v2-moe:latest_AVG-False.cbm")

        # with open("catboost/model/CatBoost_ROUND2_nomic-embed-text-v2-moe:latest_AVG-False_label_encoder.pkl", "rb") as f:
        #     self.le = pickle.load(f)

    def run_catboost_inference(self, embedding, table_name, col_num, top_k_string):
        X = np.array(embedding, dtype=np.float32).reshape(1, -1) 


        proba = self.model.predict_proba(X)[0]

        pred_idx = self.apply_cosine_reranking_single(
            proba,
            table_name,
            col_num,
            top_k_string,
            self.le
        ) 
        pred_label = self.le.inverse_transform([pred_idx])[0]
        return pred_label
        
        pass

    def apply_cosine_reranking_single(self, proba, table_name, col_num, top_k_string, le):
        # row = cosine_df[
        #     (cosine_df["table"] == table_name) &
        #     (cosine_df["col"] == col_num)
        # ]

        # if row.empty:
        #     return int(np.argmax(proba))

        # cosine_raw = row.iloc[0]["header"]

        # if pd.isna(cosine_raw):
        #     return int(np.argmax(proba))
        if pd.isna(top_k_string) or not top_k_string:
            print("topk problem", top_k_string)
            return int(np.argmax(proba))

        candidates = str(top_k_string).split("+")
        candidate_indices = []

        for c in candidates:
            if c in le.classes_:
                candidate_indices.append(le.transform([c])[0])

        if not candidate_indices:
            return int(np.argmax(proba))

        masked_proba = np.zeros_like(proba)
        masked_proba[candidate_indices] = proba[candidate_indices]

        return int(np.argmax(masked_proba))
    
        