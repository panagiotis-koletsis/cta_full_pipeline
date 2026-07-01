from operator import le
import pickle
import sys
import time

import pandas as pd
import numpy as np
import ast
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, classification_report
from sklearn.utils.class_weight import compute_sample_weight
from catboost import CatBoostClassifier
import pickle
from pathlib import Path

import data_preprocess
sys.path.append(str(Path(__file__).resolve().parents[1]))
from functions import get_config


def parse_embedding(x):
    if pd.isna(x):
        return None
    try:
        #return np.array(ast.literal_eval(x), dtype=np.float32)
        return np.array(ast.literal_eval(x), dtype=np.float32)[:-1] # this removes the last which represents if the column has duplicates or not.
    except Exception as e:
        print(f"Failed to parse: {x}")
        return None


def evaluate_model(y_test, y_pred, le):
    micro_f1 = f1_score(y_test, y_pred, average='micro')
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    precision_micro = precision_score(y_test, y_pred, average='micro')
    precision_macro = precision_score(y_test, y_pred, average='macro')
    recall_micro = recall_score(y_test, y_pred, average='micro')
    recall_macro = recall_score(y_test, y_pred, average='macro')

    print(f"Accuracy:         {accuracy_score(y_test, y_pred):.4f}")
    print(f"Micro F1:         {micro_f1:.4f}")
    print(f"Macro F1:         {macro_f1:.4f}")
    print(f"Micro Precision:  {precision_micro:.4f}")
    print(f"Macro Precision:  {precision_macro:.4f}")
    print(f"Micro Recall:     {recall_micro:.4f}")
    print(f"Macro Recall:     {recall_macro:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_, output_dict=False, zero_division=0))



def combine_emb(BASE_PATH, ROUND, train_df, val_df, test_df):
    emb_mod = "snowflake-arctic-embed2:568m"
    avg = True

    # Load embeddings from the second model
    train_path_aux = build_split_path(BASE_PATH, ROUND, "train", emb_mod, avg)
    val_path_aux   = build_split_path(BASE_PATH, ROUND, "val", emb_mod, avg)
    test_path_aux  = build_split_path(BASE_PATH, ROUND, "test", emb_mod, avg)

    train_df_aux = pd.read_csv(train_path_aux)
    val_df_aux   = pd.read_csv(val_path_aux)
    test_df_aux  = pd.read_csv(test_path_aux)

    # Function to combine two embedding columns
    def combine_embeddings(df_main, df_aux):
        key_cols = ["table", "col", "header"]

        # Keep only rows that exist in both DataFrames
        common_keys = pd.merge(
            df_main[key_cols].drop_duplicates(),
            df_aux[key_cols].drop_duplicates(),
            on=key_cols,
            how="inner"
        )

        # Filter and align the DataFrames
        df_main = (
            df_main.merge(common_keys, on=key_cols, how="inner")
                   .sort_values(key_cols)
                   .reset_index(drop=True)
        )

        df_aux = (
            df_aux.merge(common_keys, on=key_cols, how="inner")
                  .sort_values(key_cols)
                  .reset_index(drop=True)
        )

        # Parse embeddings
        df_main["emb_main"] = df_main["embedding"].apply(parse_embedding)
        df_aux["emb_aux"] = df_aux["embedding"].apply(parse_embedding)

        # Concatenate embeddings
        combined = []
        for e1, e2 in zip(df_main["emb_main"], df_aux["emb_aux"]):
            if e1 is not None and e2 is not None:
                combined.append(np.concatenate([e1, e2]))
            else:
                combined.append(None)

        df_main["combined_embedding"] = combined
        df_main = df_main.drop(columns=["emb_main"])

        return df_main

    # Combine embeddings for all three splits
    train_df = combine_embeddings(train_df, train_df_aux)
    val_df   = combine_embeddings(val_df, val_df_aux)
    test_df  = combine_embeddings(test_df, test_df_aux)


    return train_df, val_df, test_df
# def combine_emb(BASE_PATH, ROUND, train_df, val_df, test_df):
#     emb_mod = "snowflake-arctic-embed2:568m"
#     avg = True
    
#     # Load embeddings from the second model
#     train_path_aux = build_split_path(BASE_PATH, ROUND, "train", emb_mod, avg)
#     val_path_aux   = build_split_path(BASE_PATH, ROUND, "val",   emb_mod, avg)
#     test_path_aux  = build_split_path(BASE_PATH, ROUND, "test",  emb_mod, avg)

#     train_df_aux = pd.read_csv(train_path_aux)
#     val_df_aux   = pd.read_csv(val_path_aux)
#     test_df_aux  = pd.read_csv(test_path_aux)
    
#     # Function to combine two embedding columns
#     def combine_embeddings(df_main, df_aux):
#         df_main["emb_main"] = df_main["embedding"].apply(parse_embedding)
#         df_aux["emb_aux"] = df_aux["embedding"].apply(parse_embedding)
        
#         # Concatenate embeddings
#         combined = []
#         for e1, e2 in zip(df_main["emb_main"], df_aux["emb_aux"]):
#             if e1 is not None and e2 is not None:
#                 combined.append(np.concatenate([e1, e2]))
#             else:
#                 combined.append(None)
        
#         df_main["combined_embedding"] = combined
#         df_main = df_main.drop(columns=["emb_main"])
#         return df_main
    
#     # Combine embeddings for all three splits
#     train_df = combine_embeddings(train_df, train_df_aux)
#     val_df = combine_embeddings(val_df, val_df_aux)
#     test_df = combine_embeddings(test_df, test_df_aux)
    
#     return train_df, val_df, test_df

# def combine_emb(BASE_PATH, ROUND, train_df, val_df, test_df):
#     emb_mod = "snowflake-arctic-embed2:568m"
#     avg = True
#     train_path = build_split_path(BASE_PATH, ROUND, "train", emb_mod, avg)
#     val_path   = build_split_path(BASE_PATH, ROUND, "val",   emb_mod, avg)
#     test_path  = build_split_path(BASE_PATH, ROUND, "test",  emb_mod, avg)

#     train_df1 = pd.read_csv(train_path)
#     val_df1   = pd.read_csv(val_path)
#     test_df1  = pd.read_csv(test_path)
#     print(train_df1.head(5))
#     print(train_df.head(5))


def initialize():
    config = get_config()



    ROUND = config["ROUND"]
    AVG = config["AVG"]
    BASE_PATH = config["BASE_PATH"]
    EMBEDDING_MODEL = config["EMBEDDING_MODEL"]
    PREPROCESS_EMB_FILES_EXISTS = config["PREPROCESS_EMB_FILES_EXISTS"]
    return ROUND, AVG, BASE_PATH, EMBEDDING_MODEL, PREPROCESS_EMB_FILES_EXISTS



def build_split_path(base_path, round_num, split, embedding_model, avg):
    avg_suffix = f"AVG:{avg}"
    return (
        base_path
        / f"CTA-SCH-R{round_num}"
        / "gt"
        / "supplementary_files"
        / f"{split}_{embedding_model}_{avg_suffix}.csv"
    )

def main():
    ROUND, AVG, BASE_PATH, EMBEDDING_MODEL, PREPROCESS_EMB_FILES_EXISTS = initialize()

    if PREPROCESS_EMB_FILES_EXISTS == False:
        data_preprocess.process_data()

    start_time = time.time()
    # Load files
    

    train_path = build_split_path(BASE_PATH, ROUND, "train", EMBEDDING_MODEL, AVG)
    val_path   = build_split_path(BASE_PATH, ROUND, "val",   EMBEDDING_MODEL, AVG)
    test_path  = build_split_path(BASE_PATH, ROUND, "test",  EMBEDDING_MODEL, AVG)

    train_df = pd.read_csv(train_path)
    val_df   = pd.read_csv(val_path)
    test_df  = pd.read_csv(test_path)

    # train_df, val_df, test_df = combine_emb(BASE_PATH, ROUND, train_df, val_df, test_df)
    # # Drop rows with None combined embeddings
    # train_df = train_df.dropna(subset=["combined_embedding"])
    # val_df   = val_df.dropna(subset=["combined_embedding"])
    # test_df  = test_df.dropna(subset=["combined_embedding"])

    # print(train_df.shape, val_df.shape, test_df.shape)

    # # Use combined_embedding directly (already parsed numpy arrays)
    # X_train = np.vstack(train_df["combined_embedding"].values)
    # X_val   = np.vstack(val_df["combined_embedding"].values)
    # X_test  = np.vstack(test_df["combined_embedding"].values)


    #cosine_df = pd.read_csv("topk.csv")

    # #this is for test of concating emb comment out
    for df in [train_df, val_df, test_df]:
        df["parsed_embedding"] = df["embedding"].apply(parse_embedding)

    train_df = train_df.dropna(subset=["parsed_embedding"])
    val_df   = val_df.dropna(subset=["parsed_embedding"])
    test_df  = test_df.dropna(subset=["parsed_embedding"])

    # print(train_df.shape, val_df.shape, test_df.shape)
    # print(type(val_df["embedding"].iloc[0]))
    # print(val_df["embedding"].iloc[0])

    X_train = np.vstack(train_df["parsed_embedding"].values)
    X_val   = np.vstack(val_df["parsed_embedding"].values)
    X_test  = np.vstack(test_df["parsed_embedding"].values)


    print(X_train.shape, X_val.shape, X_test.shape)
    # print(X_val[0])        # first row as numpy array
    # print(X_val[0].shape)

    le = LabelEncoder()
    y_train = le.fit_transform(train_df["header"])
    y_val   = le.transform(val_df["header"])
    y_test  = le.transform(test_df["header"])

    num_classes = len(le.classes_)
    print(f"Num classes: {num_classes}")

    weights = compute_sample_weight("balanced", y_train)

    # Phase 1: find best iteration with early stopping
    # model = CatBoostClassifier(
    #     iterations=500,
    #     learning_rate=0.05,
    #     depth=6,
    #     loss_function="MultiClass",
    #     eval_metric="Accuracy",
    #     random_seed=42,
    #     early_stopping_rounds=20,
    #     task_type="GPU",
    #     verbose=50,
    #     l2_leaf_reg=3.0,
    # )
    model = CatBoostClassifier(
        iterations=500,
        learning_rate=0.0673,
        depth=7,
        loss_function="MultiClass",
        eval_metric="Accuracy",
        random_seed=42,
        task_type="GPU",
        verbose=50,
        l2_leaf_reg=3.176,
        random_strength=6.481,
        bagging_temperature=0.427,
    )
    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        sample_weight=weights,
    )
    best_round = model.best_iteration_
    print(f"Best round: {best_round}")

    # Phase 2: retrain on train+val with best_round, no early stopping
    X_trainval = np.vstack([X_train, X_val])
    y_trainval = np.concatenate([y_train, y_val])
    weights_trainval = compute_sample_weight("balanced", y_trainval)

    # final_model = CatBoostClassifier(
    #     iterations=best_round,          # fixed, no early stopping
    #     learning_rate=0.05,
    #     depth=6,
    #     loss_function="MultiClass",
    #     eval_metric="Accuracy",
    #     random_seed=42,
    #     task_type="GPU",
    #     verbose=50,
    #     l2_leaf_reg=3.0,
    # )
    final_model = CatBoostClassifier(
        iterations=best_round,          # fixed, no early stopping
        learning_rate=0.0673,
        depth=7,
        loss_function="MultiClass",
        eval_metric="Accuracy",
        random_seed=42,
        task_type="GPU",
        verbose=50,
        l2_leaf_reg=3.176,
        random_strength=6.481,
        bagging_temperature=0.427,
    )

    final_model.fit(
        X_trainval, y_trainval,
        sample_weight=weights_trainval,
    )

    #save the model and label encoder 
    model_dir = Path(__file__).resolve().parent / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    model_name = f"CatBoost_ROUND{ROUND}_{EMBEDDING_MODEL}_AVG-{AVG}.cbm"
    label_name = f"CatBoost_ROUND{ROUND}_{EMBEDDING_MODEL}_AVG-{AVG}_label_encoder.pkl"
    model_path = model_dir / model_name


    final_model.save_model(model_path)
    with open(model_dir / label_name, "wb") as f:
        pickle.dump(le, f)

    # Get probabilities
    proba = final_model.predict_proba(X_test)  # shape (n_samples, num_classes)

    # Without reranking
    print("\n=== CatBoost Only ===")
    y_pred_cb = np.argmax(proba, axis=1)
    evaluate_model(y_test, y_pred_cb, le)

    # With cosine reranking
    # print("\n=== CatBoost + Cosine Reranking ===")
    # y_pred_reranked = apply_cosine_reranking(proba, cosine_df, test_df, le)
    # evaluate_model(y_test, y_pred_reranked, le)

    end_time = time.time()
    print(f"Total execution time for catboost training: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main()