


import json
import time
import pandas as pd


from functions import get_config, get_prot_avg_cols, get_domains, run_top_k_exp, cell_index_matching, has_duplicates
from embedder import Embedder
from functions_for_cell_index_macthing import header_cell_dict
from evaluator import SOTAB_Evaluator, SOTAB_Evaluator_TopK


import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "catboost"))
import train_catboost
from catboost_inferenence import CatBoostInference

def initialize(config):
    TOPK = config["TOPK"]
    K = config["K"]
    HAS_THRESHOLD = config["HAS_THRESHOLD"]
    THRESHOLD = config["THRESHOLD"]
    AVG = config["AVG"]
    DOM_RES = config["DOM_RES"]
    ROUND = config["ROUND"]
    BASE_PATH = config["BASE_PATH"]
    EMBEDDING_MODEL = config["EMBEDDING_MODEL"]
    GT_URL_TRAIN = BASE_PATH / f"CTA-SCH-R{ROUND}/gt/sotab_cta_train_round{ROUND}.csv"
    GT_URL_VAL = BASE_PATH / f"CTA-SCH-R{ROUND}/gt/sotab_cta_validation_round{ROUND}.csv"
    GT_URL_TEST = BASE_PATH / f"CTA-SCH-R{ROUND}/gt/sotab_cta_test_round{ROUND}.csv"
    TABLES_URL = BASE_PATH / f"CTA-SCH-R{ROUND}/Round{ROUND}-SOTAB-CTA-SCH-Tables"
    PREPROCESS_EMB_FILES_EXISTS = config["PREPROCESS_EMB_FILES_EXISTS"]
    MODEL_EXISTS = config["MODEL_EXISTS"]
    PROTPYPES_EMB_EXISTS = config["PROTPYPES_EMB_EXISTS"]
    CELL_DICT_EXISTS = config["CELL_DICT_EXISTS"]
    COS_SIM_THRESHOLD = config["COS_SIM_TRHRESHOLD"]
    TOP_K_EXISTS = config["TOP_K_EXISTS"]



    return TOPK, K, HAS_THRESHOLD, THRESHOLD, AVG, DOM_RES, ROUND, BASE_PATH, EMBEDDING_MODEL, GT_URL_TRAIN, GT_URL_VAL, GT_URL_TEST, TABLES_URL, PREPROCESS_EMB_FILES_EXISTS, MODEL_EXISTS, PROTPYPES_EMB_EXISTS, CELL_DICT_EXISTS, COS_SIM_THRESHOLD, TOP_K_EXISTS

def check_files_exist(config):
    print("Starting Initalization...")
    ROUND = config["ROUND"]
    BASE_PATH = config["BASE_PATH"]
    GT_URL_TRAIN = BASE_PATH / f"CTA-SCH-R{ROUND}/gt/sotab_cta_train_round{ROUND}.csv"
    GT_URL_VAL = BASE_PATH / f"CTA-SCH-R{ROUND}/gt/sotab_cta_validation_round{ROUND}.csv"
    TABLES_URL = BASE_PATH / f"CTA-SCH-R{ROUND}/Round{ROUND}-SOTAB-CTA-SCH-Tables"
    EMBEDDING_MODEL = config["EMBEDDING_MODEL"]
    AVG = config["AVG"]

    PROTPYPES_EMB_EXISTS = config["PROTPYPES_EMB_EXISTS"]
    CELL_DICT_EXISTS = config["CELL_DICT_EXISTS"]
    MODEL_EXISTS = config["MODEL_EXISTS"]
    TOP_K_EXISTS = config["TOP_K_EXISTS"]
    K = config["K"]

    DOM_RES = config["DOM_RES"]

    if PROTPYPES_EMB_EXISTS == False:
        prot_dict = get_prot_avg_cols(config,GT_URL_TRAIN=None)
    else: 
        with open(BASE_PATH / f"CTA-SCH-R{ROUND}/supplementary_files/prototype_embeddings{EMBEDDING_MODEL}_AVG-{AVG}.json" , "r", encoding="utf-8") as f:
            prot_dict = json.load(f)
    
    if CELL_DICT_EXISTS == False:
        cell_header_dict = header_cell_dict(GT_URL_VAL,GT_URL_TRAIN, TABLES_URL, ROUND)
    else:
        with open(BASE_PATH / f"CTA-SCH-R{ROUND}/supplementary_files/inverted_header_dict.json", "r", encoding="utf-8") as f:
            cell_header_dict = json.load(f)
 
    dom_dict = None
    if DOM_RES == True: 
        dom_dict = get_domains(GT_URL_TRAIN,GT_URL_VAL)

    if MODEL_EXISTS == False:
        train_catboost.main()
    
    topk_results = None
    if TOP_K_EXISTS == True:
        with open(BASE_PATH / f"CTA-SCH-R{ROUND}/supplementary_files/top{K}_{EMBEDDING_MODEL}.csv", "r", encoding="utf-8") as f:
            topk_results = json.load(f)

    #load the trained model
    model = CatBoostInference(config)

    return prot_dict, cell_header_dict, dom_dict, model, topk_results


from sklearn.metrics.pairwise import cosine_similarity

def cos_similarity(embedding, prot_dict, dom_dict, tab_dom, DOM_RES):
    best_cosine_sim = -1
    for key, value in prot_dict.items():
        #print(f"{key}: {value}")
        if embedding is not None:
            cos_sim = cosine_similarity([embedding], [value])
            if DOM_RES == True:
                if cos_sim > best_cosine_sim and (key in dom_dict[tab_dom]):
                    best_cosine_sim = cos_sim
                    best_attr = key
            else:
                if cos_sim > best_cosine_sim:
                    best_cosine_sim = cos_sim
                    best_attr = key
    return best_attr, best_cosine_sim




def main(): 
    start_time = time.time()

    config = get_config()
    TOPK, K, HAS_THRESHOLD, THRESHOLD, AVG, DOM_RES, ROUND, BASE_PATH, EMBEDDING_MODEL, GT_URL_TRAIN, GT_URL_VAL, GT_URL_TEST, TABLES_URL, PREPROCESS_EMB_FILES_EXISTS, MODEL_EXISTS, PROTPYPES_EMB_EXISTS, CELL_DICT_EXISTS, COS_SIM_THRESHOLD, TOP_K_EXISTS = initialize(config)


    prot_dict, cell_header_dict, dom_dict, model, topk_results = check_files_exist(config)
    embedder = Embedder(config)


    df_GT = pd.read_csv(GT_URL_TEST)

    iteration = 0
    df_pred = df_GT.copy()
    df_pred["header"] = ""
    df_pred_topk = df_GT.copy()
    df_pred_topk["header"] = ""
    x=0
    for gt_row in range(len(df_GT)):
        iteration += 1
        if iteration % 100 == 0:
            print(f"Processed {iteration}/{len(df_GT)} rows")
        
        tab_name = df_GT["table"][gt_row]
        tab_dom = df_GT["table"][gt_row].split('_')[0]
        col_num = df_GT["col"][gt_row]

        if DOM_RES == True:
            dom_list = dom_dict[tab_dom]
        else:
            dom_list = None
        

        embedding, col_data = embedder.get_embedding(df_GT, gt_row)

        best_attr = ""
        topk_string = run_top_k_exp(K, embedding, prot_dict,df_pred, gt_row, best_attr, DOM_RES, dom_list)
        #print(topk_string)

        
        best_attr, best_cosine_sim = cos_similarity(embedding, prot_dict, dom_dict, tab_dom, DOM_RES)
        if best_cosine_sim >= COS_SIM_THRESHOLD:
            best_attr1 = best_attr
        else: 
            attr = cell_index_matching(cell_header_dict, col_data, tab_dom, dom_dict)
            best_attr1 = attr

        if best_attr1 == "nope" and embedding is not None:
            # has_dups = has_duplicates(col_data)
            # embedding = embedding + [has_dups]

            best_attr1 = best_attr
            pred_label = model.run_catboost_inference(embedding, tab_name, col_num, topk_string)
            best_attr1 = pred_label
                
        best_attr = best_attr1

        if TOPK:
            if DOM_RES:
                topk_string =run_top_k_exp(K, embedding, prot_dict, df_pred_topk, gt_row, best_attr,DOM_RES,dom_list)
            else:
                topk_string =run_top_k_exp(K, embedding, prot_dict, df_pred_topk, gt_row, best_attr,DOM_RES,dom_list)

            df_pred_topk.at[gt_row, "header"] = topk_string
            df_pred_topk.to_csv(BASE_PATH / f"CTA-SCH-R{ROUND}/supplementary_files/top{K}_{EMBEDDING_MODEL}.csv", index=False) 


        


        
        
        if HAS_THRESHOLD:
            if best_cosine_sim > THRESHOLD:
                df_pred.at[gt_row, "header"] = best_attr
            else:
                df_pred.at[gt_row, "header"] = "nope"
        else:
            df_pred.at[gt_row, "header"] = best_attr

    

        df_pred.to_csv("cta_pred.csv", index=False)

    if TOPK:
        evaluator = SOTAB_Evaluator_TopK(GT_URL_TEST, BASE_PATH / f"CTA-SCH-R{ROUND}/supplementary_files/top{K}_{EMBEDDING_MODEL}.csv", k=K)
    else:
        evaluator = SOTAB_Evaluator(GT_URL_TEST, "cta_pred.csv",HAS_THRESHOLD=HAS_THRESHOLD, THRESHOLD=THRESHOLD)
    result = evaluator._evaluate()
    print("Evaluation Results:", result)

    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")





if __name__ == "__main__":
    main()


