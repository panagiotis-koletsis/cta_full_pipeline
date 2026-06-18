from pathlib import Path
import pandas as pd
import json
import os
from sklearn.metrics.pairwise import cosine_similarity
import time

from embedder import Embedder



def get_config():
    config = {
        "TOPK" : False, # whether to use Top-K evaluation or standard single-label evaluation
        "K" : 5,
        "HAS_THRESHOLD" : False, # whether to apply a cosine similarity threshold for accepting predictions, or just take the best match regardless of score
        "THRESHOLD" : 0.95,
        "AVG" : True, # whether to average row embeddings for column representation (True) or concatenate them (False)
        "DOM_RES" : True, # Extract and use domain restrictions
        "ROUND" : 1,
        "BASE_PATH" : Path("/home/kpanag/vscode/cta_full_pipeline/SOTABV2forSemTab2023"),
        #"EMBEDDING_MODEL" : "nomic-embed-text-v2-moe:latest",
        "EMBEDDING_MODEL" : "snowflake-arctic-embed2:568m",
        "PREPROCESS_EMB_FILES_EXISTS" : False, #avoid re running the data preprocessing for extracting the emb.   TRUE or FALSE
        "MODEL_EXISTS" : False, #avoid re running the catboost training. TRUE or FALSE
        "PROTPYPES_EMB_EXISTS" : False, #whether the prototype embeddings file already exists. TRUE or FALSE
        "CELL_DICT_EXISTS" : False, #whether the cell-header dict file already exists. TRUE or FALSE
        "TOP_K_EXISTS" : False, #whether the top-k results file already exists. TRUE or FALSE
        "COS_SIM_TRHRESHOLD" : 0.90,
        #"Train_with_functional_property" : True #whether to train the catboost model with functional property as a feature. TRUE or FALSE
    }
    return config



def get_prot_avg_cols(config, GT_URL_TRAIN):
    print("Computing prototype embeddings for each column...") 
    start_time = time.time()
    ROUND = config["ROUND"]
    GT_URL_VAL = config["BASE_PATH"] / f"CTA-SCH-R{ROUND}/gt/sotab_cta_validation_round{ROUND}.csv"  
    #GT_URL_TRAIN = config["BASE_PATH"] / f"CTA-SCH-R{ROUND}/gt/sotab_cta_train_round{ROUND}.csv"
    TABLES_URL = config["BASE_PATH"] / f"CTA-SCH-R{ROUND}/SOTAB-CTA-SCH-Tables"
    EMBEDDING_MODEL = config["EMBEDDING_MODEL"]
    AVG = config["AVG"]
    BASE_PATH = config["BASE_PATH"]


    prototype_col_embedding_dict = {}

    df_GT = pd.read_csv(GT_URL_VAL)
    if GT_URL_TRAIN is not None:
        df_GT1 = pd.read_csv(GT_URL_TRAIN)
        df_GT = pd.concat([df_GT1, df_GT], ignore_index=True)
    #print(df_GT)

    df_GT = df_GT.sort_values(by="header").reset_index(drop=True)
    print(len(df_GT))
    first_elem = df_GT['header'].iloc[0]
    sum_emb = None
    total_num_elem = 0 
    #print(first_elem)

    embedder = Embedder(config)
    for gt_row in range(len(df_GT)):
        col_name = df_GT["header"][gt_row]

        if col_name == first_elem:
            embedding, _ = embedder.get_embedding(df_GT, gt_row)

            if sum_emb is None:
                sum_emb = embedding.copy()
            else:
                sum_emb = [s + e for s, e in zip(sum_emb, embedding)]
            total_num_elem += 1
        else:
            if sum_emb is not None and total_num_elem > 0:
                avg_emb = [s / total_num_elem for s in sum_emb]
                prototype_col_embedding_dict[first_elem] = avg_emb
                print(f"✓ Saved prototype for '{first_elem}' (averaged {total_num_elem} columns)")

            embedding, _ = embedder.get_embedding(df_GT, gt_row)

            sum_emb = embedding.copy()
            total_num_elem = 1
            first_elem = col_name
    
    # After the loop ends, save the last group
    if sum_emb is not None and total_num_elem > 0:
        avg_emb = [s / total_num_elem for s in sum_emb]
        prototype_col_embedding_dict[first_elem] = avg_emb


    # After computing prototype_col_embedding_dict
    output_path = BASE_PATH / f"CTA-SCH-R{ROUND}/supplementary_files/prototype_embeddings{EMBEDDING_MODEL}_AVG-{AVG}.json"
    with open(output_path, "w") as f:
        json.dump(prototype_col_embedding_dict, f)
        
    end_time = time.time()
    print(f"Total execution time for prototype embeddings: {end_time - start_time:.2f} seconds")
    return prototype_col_embedding_dict


def get_initial_list(df1):
    nameList = []
    for i in range(len(df1)):
        name = df1['table'][i].split('_')[0]
        nameList.append(name)
    nameList = list(set(nameList))    
    return nameList

def get_domains(GT_PATH1,GT_PATH2):
    df1 = pd.read_csv(GT_PATH1)
    df2 = pd.read_csv(GT_PATH2)

    names = get_initial_list(df1)
    dict = {name: [] for name in names}

    for i in range(len(df1)):
        dict[df1['table'][i].split('_')[0]].append(df1['header'][i])

    for i in range(len(df2)):
        dict[df2['table'][i].split('_')[0]].append(df2['header'][i])

    for key in dict:
        dict[key] = list(set(dict[key]))
    
    #print(dict)
    return dict

    #write_to_file()
        

def run_top_k_exp(TOP_K,embedding,prot_dict,df_pred,gt_row,best_attr,DOM_RES,dom_list):
    top_k_results = []

    if embedding is not None:
        for key, value in prot_dict.items():
            cos_sim = cosine_similarity([embedding], [value])[0][0]  # extract scalar
            if DOM_RES:
                if key in dom_list:
                    top_k_results.append((key, cos_sim))
            else:
                top_k_results.append((key, cos_sim))

        # Sort by similarity descending
        top_k_results.sort(key=lambda x: x[1], reverse=True)

        # Keep only Top-K
        top_k_results = top_k_results[:TOP_K]
        #print(f"Top-{TOP_K} results: {top_k_results}")

        # Store best prediction (for evaluator compatibility)
        best_attr = top_k_results[0][0]

        top_k_labels = [attr for attr, score in top_k_results]
        topk_string = "+".join(top_k_labels)
        #print("top k string", topk_string)
    else:
        best_attr = ""


    #df_pred.at[gt_row, "header"] = topk_string
    return topk_string
    #return embedding, prot_dict, df_pred, gt_row, topk_string





def cell_index_matching(cell_header_dict, col_data, tab_dom, dom_dict):

    attr = "nope"

    for i in range(len(col_data)):
        cell = col_data.iloc[i]
        #print("cell", cell)

        entry = cell_header_dict.get(str(cell))
        if entry is not None:
            if len(entry) == 1:
                attr, count = next(iter(entry.items()))
                #attr = next(iter(entry))
                if attr not in dom_dict[tab_dom] or count < 5:
                    attr = "nope"
            else: 
                sum_counts = sum(entry.values())
                max_count = max(entry.values())
                if max_count/sum_counts > 0.9:
                    max_header = max(entry, key=entry.get)
                    if max_header in dom_dict[tab_dom] and max_count > 5:
                        attr = max_header
                    else:
                        attr = "nope"

    return attr

def has_duplicates(col_data):
    values = col_data.dropna()
    has_dups = values.duplicated().any()
    return int(has_dups)